"""traceX v0.7 (the v0.6 economy, plus the failure registry): no token. Everything is paid directly in sats, and every
payout is a split of a real payment.

    python node/exchange.py --economy sats ...        # or TRACEX_ECONOMY=sats

One unit
  * Whole millisatoshis (msats; 1 sat = 1,000 msats), paid over Lightning with L402: a call the account can't cover
    answers 402 Payment Required with a Lightning invoice and a macaroon; the client pays and retries with the
    payment's preimage. On the testnet each wallet takes 30,000 test sats once from the faucet, and a 402's invoice is
    a placeholder. There is no token, no pool, no emission and no treasury: nothing is minted, nothing has a price but
    the prices sellers set, and nobody holds anything whose value could move.

Where the money goes
  * Use pays out at once. Each paid use of a learning (calls x the learning's own per_call_msats, set by its seller)
    goes into that payment's own escrow and is split at the epoch's settlement: traces 60 / trainer 25 / checkers 10 /
    validators 5, equal weight per distinct parent (a copy counts as its original), a cited learning passing its slice
    through to its own traces and checkers, 32 deep at most. Trainer and validators are paid then; the parents' part
    waits in the payment's escrow for 4 epochs so a challenge can claw it back (to the payer).
  * The invariant: nothing is ever paid out that a payer didn't pay in. Every payout is a move out of one payment's
    escrow, which holds exactly what its payer paid less the fee; _disburse() refuses (InvariantError) any move that
    would take a payment's payouts past `gross - fee`, and audit() re-checks every payment. So farming loses by
    construction: paying yourself returns at most what you paid, less every share that isn't yours and the fee.
  * One fee: 58 msats a transaction (exchange.TX_FEE_MSATS), paid at once to the operator that served it: its whole
    income. Optional dollar re-peg (Params.fee_repeg_epochs, operator-set bitcoin price), off by default.
  * Licences clear in sats; the buyer's payment waits until its own learnings (or a list it sends) say which traces
    it used, then pays them 85 / 10 / 5 (producer, checker author, validators).
  * Bounties are refundable pledge escrows: free to post, anyone pledges sats, a solve on the poster's hidden eval at
    the target pays trainer 70 / traces 20 / checkers 5 / validators 5 (vesting 4 epochs), and an unsolved bounty gives
    every backer back what it put in. Backers get no token, no share, nothing to trade. A post with the branch,
    failure and model of an open bounty backs it instead of opening a duplicate; a bounty nobody backs expires.

Stakes, in sats
  * Learning bond 5,000 sats, validator minimum stake 10,000 sats, challenge stake 2,000 sats. Forfeits (lost bonds,
    slashed stake, failed challenges) are destroyed: moved to an account nothing can spend (BURN) and recorded per epoch
    in a batch whose digest is destined for a provably unspendable output (mainnet: an OP_RETURN output in the epoch's
    settlement transaction). A validator slashed under the minimum loses its seat at once.
  * Validation (SPEC 4f): random stake-weighted draw after submission, commit then reveal, median minus 2 standard
    errors, decoys, challenges at any time. A decoy miss (a score further from the sealed truth than 4 of the
    validator's own standard errors) is a strike; two strikes inside `strike_window` epochs cost 25% of stake, so one
    unlucky honest measurement never does.
"""
import base64
import decimal
import hashlib
import json
import math
import re
import sqlite3
import statistics
import time
from dataclasses import dataclass

from exchange import (ADDRESS, BOUNTY_SPLIT, MAX_DEPTH, TX_FEE_MSATS, Exchange, PaymentRequired, need_address,
                      msats_in, split_trace_sale, split_usage, leaf, build_tree, proof, canonical, object_id,
                      clear_shared, Bid)
from registry import Registry
from challenges import Challenges
from traceex.trace import Learning

VERSION = "0.7"
UPGRADES_FROM = ("0.6",)       # v0.7 adds the failure registry to a v0.6 database (additive tables, a backfill)
MSATS_PER_BTC = 100_000_000_000
BTC_USD = 85_962               # Coinbase spot, 2026-10-05: only for the approximate dollar figures shown beside sats
BURN = "burn:unspendable"      # forfeits go here; nothing on the node can spend from it
USAGE_SPLIT = dict(Learning.DEFAULT_SPLIT)     # traces 60 / trainer 25 / checkers 10 / validators 5


class InvariantError(RuntimeError):
    """A payout would exceed what its payer paid in (or the burn account would spend): the anti-farming guarantee."""


def fmt_sats(msats):
    """msats for people: '30,000 sats', '0.058 sats', '1.5 sats'."""
    sats = decimal.Decimal(int(msats)) / 1000
    if sats == sats.to_integral_value():
        return f"{int(sats):,} sats"
    return f"{sats.normalize():,f} sats"


def usd_approx(msats, btc_usd=BTC_USD):
    """A dollar figure for people, labelled approximate wherever it is shown: msats at `btc_usd` dollars a bitcoin."""
    usd = decimal.Decimal(int(msats)) * int(btc_usd) / MSATS_PER_BTC
    if abs(usd) >= 100:
        return f"${usd:,.0f}"
    if abs(usd) >= decimal.Decimal("0.01"):
        return f"${usd:,.2f}"
    return f"${usd:.2g}" if usd else "$0"


def _frac(amount, f):
    """amount x f for integers, exactly enough (f to a millionth), rounded down."""
    return int(amount) * round(f * 1_000_000) // 1_000_000


@dataclass
class Params:
    vest_epochs: int = 4                       # parents' shares (and bounty payouts) wait this long, so a challenge can claw back
    keep_epochs: int = 6                       # per-transaction detail is kept this long: the challenge window, plus 2
    quorum: int = 3
    min_gain: float = 0.01
    accept_z: float = 2.0                      # accept only if median gain - 2 standard errors >= min_gain
    overclaim: float = 2.0                     # claiming more than twice the measured gain, beyond the noise: bond forfeit
    z: float = 2.5                             # within 2.5 of its own standard errors of the median: the validator agreed
    min_tol: float = 0.02
    decoy_z: float = 4.0                       # further than this from a decoy's sealed truth: a strike
    decoy_strikes: int = 2                     # strikes inside strike_window epochs before the 25% slash
    strike_window: int = 30
    inconclusive_burn: float = 0.10            # a real-looking but unproven gain: bond back minus 10%, which is destroyed
    audit_min: int = 10                        # every reveal audits at least this many parents (or all of them)
    audit_max_bad: float = 0.10
    pad_burn: float = 0.50                     # parents found padded: their share goes back to the payer, half the bond is destroyed
    learning_bond_msats: int = 5_000_000       # 5,000 sats (about $4.30)
    validator_min_stake_msats: int = 10_000_000    # 10,000 sats (about $8.60)
    challenge_stake_msats: int = 2_000_000     # 2,000 sats (about $1.72)
    noshow_slash: float = 0.05
    fake_slash: float = 0.25                   # vouched for a gain a fresh round refuted, or two decoy strikes
    near_dup: float = 0.3                      # the same distinctive fix, inputs sharing 30% of their words: one trace
    unbacked_epochs: int = 3                   # a bounty nobody pledges to expires after this many epochs
    btc_usd: int = BTC_USD                     # dollars a bitcoin: only for approximate dollar figures (and the re-peg)
    fee_repeg_epochs: int = 0                  # re-peg the fee to fee_target_usd_nanos every N epochs (0: never)
    fee_target_usd_nanos: int = 50_000         # the re-peg's target: $0.00005, at the operator's btc_usd
    fix_bond_msats: int = 2_000_000            # v0.7: 2,000 sats with each fix claim, destroyed if it fixes none of them
    reporter_bond_msats: int = 1_000_000       # v0.7: 1,000 sats makes a reporter verified (counted); destroyed if a
                                               # case it reported does not reproduce
    challenge_sub_bond_msats: int = 1_000_000  # v0.8: a challenge submission validators must measure holds 1,000 sats,
                                               # destroyed if it is invalid or overfits its public instances
    challenge_overfit_steps: float = 10.0      # public score beating the hidden median by more than this many minimum
                                               # steps (or 3 standard errors): overfit
    challenge_bond_share: float = 0.10         # v0.8: a submission's bond is the larger of 1,000 sats and 10% of what
                                               # it would unlock, held through the challenge's prior-art window
    prior_art_reward_share: float = 0.50       # an upheld prior-art claim earns half the bond; the rest is destroyed
    challenge_unbacked_days: float = 30.0      # a challenge nobody pledges to for this many days (wall clock) ends
    escalate_after: int = 6                    # a failure open this many epochs...
    escalate_streak: int = 3                   # ...whose growth stayed positive this many settlements becomes a challenge


SATS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sledger   (epoch INT, account TEXT, msats INT, memo TEXT, payout INT DEFAULT 0);
CREATE INDEX IF NOT EXISTS sledger_account ON sledger(account);
CREATE INDEX IF NOT EXISTS sledger_epoch ON sledger(epoch);
CREATE TABLE IF NOT EXISTS payments  (id INTEGER PRIMARY KEY, epoch INT, kind TEXT, payer TEXT, ref TEXT, gross INT,
                                      fee INT, out INT DEFAULT 0, split INT DEFAULT 0);
CREATE INDEX IF NOT EXISTS payments_kind ON payments(kind, split);
CREATE TABLE IF NOT EXISTS vesting   (id INTEGER PRIMARY KEY, account TEXT, msats INT, payment INT, learning TEXT,
                                      role TEXT, release INT, status TEXT DEFAULT 'vesting');
CREATE INDEX IF NOT EXISTS vesting_learning ON vesting(learning, status);
CREATE TABLE IF NOT EXISTS validators(address TEXT PRIMARY KEY, stake INT, joined INT, slashed INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS verdicts  (learning TEXT PRIMARY KEY, status TEXT, round INT, gain REAL, bond INT DEFAULT 0,
                                      trainer TEXT, registered INT, accepted INT, audit_bad REAL, challenger TEXT,
                                      challenge_stake INT DEFAULT 0, note TEXT, bond_from TEXT);
CREATE TABLE IF NOT EXISTS assignments(learning TEXT, round INT, validator TEXT, epoch INT,
                                       PRIMARY KEY (learning, round, validator));
CREATE TABLE IF NOT EXISTS commits   (learning TEXT, round INT, validator TEXT, digest TEXT, epoch INT,
                                      PRIMARY KEY (learning, round, validator));
CREATE TABLE IF NOT EXISTS reveals   (learning TEXT, round INT, validator TEXT, body TEXT, gain REAL, epoch INT,
                                      agreed INT, PRIMARY KEY (learning, round, validator));
CREATE TABLE IF NOT EXISTS dups      (trace TEXT PRIMARY KEY, canonical TEXT, key TEXT, fix TEXT);
CREATE INDEX IF NOT EXISTS dups_fix ON dups(fix);
CREATE INDEX IF NOT EXISTS dups_key ON dups(key);
CREATE TABLE IF NOT EXISTS licence_escrow(id INTEGER PRIMARY KEY, lot TEXT, buyer TEXT, payment INT, epoch INT,
                                          traces TEXT, paid INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS decoys    (learning TEXT PRIMARY KEY, digest TEXT, funder TEXT, gain REAL,
                                      unsealed INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS strikes   (validator TEXT, learning TEXT, epoch INT);
CREATE TABLE IF NOT EXISTS artifacts (hash TEXT PRIMARY KEY, learning TEXT);
CREATE TABLE IF NOT EXISTS forfeits  (epoch INT, kind TEXT, ref TEXT, msats INT);
CREATE TABLE IF NOT EXISTS burn_batches(epoch INT PRIMARY KEY, msats INT, items INT, digest TEXT);
CREATE TABLE IF NOT EXISTS sats_roots(epoch INT PRIMARY KEY, root TEXT, total INT, claims TEXT);
CREATE TABLE IF NOT EXISTS epochs    (epoch INT PRIMARY KEY, body TEXT);
"""
SLOT = re.compile(r"\{([A-Z]+)_\d+\}")
NO_EARNINGS = ("pending", "rejected", "inconclusive", "clawed back", "decoy")
BURN_DESTINATION = ("testnet: the account 'burn:unspendable', which no call on the node can spend from. Mainnet: the "
                    "epoch's settlement transaction pays the batch to an OP_RETURN output committing to this digest, "
                    "which no key can ever spend")


def _norm(s):
    return re.sub(r"\s+", " ", SLOT.sub(r"{\1}", str(s))).strip().lower()


def dup_key(t):
    """Two traces that differ only in placeholder numbering, whitespace or case are the same fix."""
    body = [t.get("task"), (t.get("base_model") or {}).get("name"), _norm(t.get("input", "")),
            sorted(t.get("fixed_fields", [])), {k: _norm(v) for k, v in sorted((t.get("verified_output") or {}).items())}]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def fix_key(t):
    """The fix itself: task, model and verified output. A trace with the same fix as an earlier one and an input that
    says much the same thing in other words is a reworded copy, and shares the earlier one's slot."""
    body = [t.get("task"), (t.get("base_model") or {}).get("name"),
            {k: _norm(v) for k, v in sorted((t.get("verified_output") or {}).items())}]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def fix_text(t):
    """What a verified output says once placeholders are taken out. Long enough, it identifies the fix; a skeleton's
    output ("{CODE}") says nothing, and its traces are told apart by their inputs alone."""
    return re.sub(r"\{[a-z]+\}", "", " ".join(_norm(v) for _, v in sorted((t.get("verified_output") or {}).items()))).strip()


def words(s):
    return set(re.findall(r"[a-z0-9_{}]+", _norm(s)))


def overlap(a, b):
    """Share of words two inputs have in common (Jaccard)."""
    return len(a & b) / len(a | b) if a | b else 1.0


def attestation_digest(attestation, salt):
    """What a validator commits before revealing: sha256(canonical attestation + salt)."""
    return hashlib.sha256(canonical(attestation) + str(salt).encode()).hexdigest()


def decoy_digest(gain, salt):
    """How the operator seals a decoy's true gain before validators measure it."""
    return hashlib.sha256(f"{float(gain):.4f}|{salt}".encode()).hexdigest()


def standard_error(att):
    """A validator's standard error for its gain: the paired one it reports (from its per-item results), never less
    than a quarter of the unpaired error for its sample size, or the unpaired error when it reports none."""
    clamp = lambda p: min(max(float(p), 0.05), 0.95)
    b, a, n = clamp(att["before"]), clamp(att["after"]), max(int(att.get("n") or 100), 1)
    unpaired = math.sqrt(b * (1 - b) / n + a * (1 - a) / n)
    se = att.get("se")
    return max(float(se), unpaired / 4) if se is not None else unpaired


def tolerance(att, z, floor):
    """How far an honest validator's gain can sit from the median: z standard errors of its own measurement."""
    return max(floor, z * standard_error(att))


class SatsExchange(Challenges, Exchange):
    economy = "sats"
    money = "msats"

    def _fmt(self, msats):
        return f"{fmt_sats(msats)} (about {usd_approx(msats, self.btc_usd)})"

    def __init__(self, path=":memory:", *, params=None, beacon_delay=1, clock=None, reserve_msats=None,
                 tx_fee_msats=None, btc_usd=None, **kw):
        """beacon_delay=1: validators for a learning are drawn from the beacon published at the settlement after it was
        submitted, so nobody can grind a learning's content for friendly validators. 0 assigns at once (tests).
        clock: seconds (time.time by default; simulations pass their own). test_credits: test msats each new wallet
        takes once (30,000,000 = 30,000 sats). reserve_msats: a lot's reserve price per licence. tx_fee_msats: the
        standard fee (TX_FEE_MSATS on a testnet, unless set). btc_usd: dollars a bitcoin for the approximate dollar
        figures (and the fee re-peg); never used in any amount."""
        if "reserve_micros" in kw or "tx_fee_nanos" in kw:
            raise ValueError("a sats node prices everything in sats: reserve_msats and tx_fee_msats, not micros or nanos")
        if reserve_msats is not None:
            kw["reserve_micros"] = int(reserve_msats)          # the base node's reserve, in this node's money
        self.p = params or Params()
        kw.setdefault("unbacked_epochs", self.p.unbacked_epochs)
        super().__init__(path, tx_fee_nanos=0, **kw)
        self.beacon_delay = int(beacon_delay)
        self.clock = clock or time.time
        old = self._meta("coin_version") or ("0.3" if self._meta("pool_coin") else None)
        mine = self._meta("sats_version")
        has_data = self.db.execute("SELECT EXISTS(SELECT 1 FROM traces) OR EXISTS(SELECT 1 FROM ledger)").fetchone()[0]
        if old or (mine and mine != VERSION and mine not in UPGRADES_FROM) or (not mine and has_data):
            self.db.close()
            what = f"a testnet v{old} coin economy (TXC)" if old else (f"a v{mine} node" if mine else "an older node")
            raise ValueError(f"this database holds {what}; v{VERSION} has no token and pays everything in sats: start "
                             "it on an empty database")
        self.db.executescript(SATS_SCHEMA)
        self.tx_fee_msats = int(TX_FEE_MSATS if tx_fee_msats is None and self.test_credits else tx_fee_msats or 0)
        if self._meta("tx_fee_msats") is not None:              # a re-pegged fee outlives a restart
            self.tx_fee_msats = int(self._meta("tx_fee_msats"))
        self.btc_usd = int(self._meta("btc_usd") or btc_usd or self.p.btc_usd)
        if mine is None:                                        # genesis: nothing exists but the rules
            self._set_meta("sats_version", VERSION)
            self._set_meta("beacon", hashlib.sha256(b"traceX v0.6 genesis").hexdigest())
        elif mine in UPGRADES_FROM:                             # v0.6 -> v0.7: the registry is built from its traces
            self._set_meta("sats_version", VERSION)
        self.db.commit()
        Registry._open_registry(self)
        self._open_challenges()                                 # v0.8: challenge bounties (node/challenges.py)
        if mine in UPGRADES_FROM:
            self._event(f"node upgraded from v{mine} to v{VERSION}: every stored trace filed in the failure registry",
                        force=True)
            self.db.commit()

    def _open_registry(self):
        """Deferred until the version checks above pass: a database this node refuses is never touched."""
    # --- the ledger: one unit, every entry a move, so the books always sum to zero ------------------------------------
    def _m(self, k):
        return int(self._meta(k) or 0)

    def _add(self, k, v):
        self._set_meta(k, self._m(k) + int(v))

    def _credit(self, account, msats, memo, payout=False):
        if msats:
            if account == BURN and msats < 0:
                raise InvariantError("the burn account can never spend")
            self.db.execute("INSERT INTO sledger VALUES (?,?,?,?,?)", (self.epoch, account, int(msats), memo, int(payout)))

    def _move(self, src, dst, msats, memo, payout=False):
        if msats:
            self._credit(src, -msats, memo)
            self._credit(dst, msats, memo, payout)

    def _bal(self, account):
        return int(self.db.execute("SELECT COALESCE(SUM(msats), 0) FROM sledger WHERE account=?", (account,)).fetchone()[0])

    def _funds(self, account):
        g = self.db.execute("SELECT micros FROM grants WHERE account=?", (account,)).fetchone()
        return (g[0] if g else 0) + self._bal(account)

    def _bids(self, account):
        return self.db.execute("SELECT COALESCE(SUM(price), 0) FROM bids WHERE bidder=? AND epoch=?",
                               (account, self.epoch)).fetchone()[0]

    def _free(self, account):
        """What an account can spend now: its sats less the bids it has standing this epoch."""
        return self._funds(account) - self._bids(account)

    def _need_funds(self, account, msats, pending=0):
        """On a testnet the account must cover the payment and this transaction's fee. (Mainnet: the L402 challenge.)"""
        need = int(msats) + self.tx_fee_msats
        if self.test_credits and self._free(account) < need:
            room = max(self._free(account) - self.tx_fee_msats, 0)
            raise PaymentRequired(f"not enough test sats: {account[:10]}… can pay {fmt_sats(room)}, this needs "
                                  f"{fmt_sats(msats)} and the {self.tx_fee_msats}-msat fee (POST /v0/faucet opens a "
                                  "wallet)", account, need - self._free(account))

    def _tx_fee(self, account):
        """One transaction, one fee: paid at once, in sats, to the operator that served it."""
        fee = self.tx_fee_msats
        if not fee or not account:
            return
        if self.test_credits and self._free(account) < fee:
            raise PaymentRequired(f"not enough test sats: {account[:10]}… can't cover the {fee}-msat transaction fee "
                                  "(POST /v0/faucet opens a wallet)", account, fee - self._free(account))
        self._move(account, self.fee_to, fee, "transaction fee", payout=True)
        self._add("fees_msats", fee)

    def _bill_fees(self):
        """Fees are paid as each transaction happens; nothing is billed later."""
        return 0

    def _forfeit(self, src, msats, kind, ref=""):
        """Destroy forfeited sats: they move to the burn account, which nothing spends, and join this epoch's batch."""
        if msats > 0:
            self._move(src, BURN, msats, f"forfeit: {kind}")
            self.db.execute("INSERT INTO forfeits VALUES (?,?,?,?)", (self.epoch, kind, ref, int(msats)))
            self._add("forfeited_msats", msats)

    # --- payments: every payout is a move out of one payment's own escrow ---------------------------------------------
    def _pay_in(self, payer, kind, ref, msats, fee=True, check=True):
        """`payer` pays `msats` (and, with `fee`, the transaction fee) for one thing. The msats go into the payment's
        own escrow (pay:<id>); the fee goes to the operator. Returns the payment id."""
        msats = int(msats)
        if msats <= 0:
            raise ValueError("a payment must be more than 0 msats")
        if check:
            self._need_funds(payer, msats)
        f = self.tx_fee_msats if fee else 0
        if fee:
            self._tx_fee(payer)
        pid = self.db.execute("INSERT INTO payments (epoch, kind, payer, ref, gross, fee) VALUES (?,?,?,?,?,?)",
                              (self.epoch, kind, payer, ref, msats + f, f)).lastrowid
        self._move(payer, f"pay:{pid}", msats, f"{kind} {str(ref)[:30]}")
        self._add("paid_in_msats", msats)
        return pid

    def _payment(self, pid):
        r = self.db.execute("SELECT kind, payer, ref, gross, fee, out FROM payments WHERE id=?", (pid,)).fetchone()
        if not r:
            raise KeyError(f"payment {pid}")
        return r

    def _disburse(self, pid, account, msats, memo, refund=False):
        """THE invariant, enforced: a payout leaves one payment's escrow, and never takes that payment's payouts past
        what its payer paid in less the fee."""
        msats = int(msats)
        if msats <= 0:
            return 0
        _, payer, _, gross, fee, out = self._payment(pid)
        held = self._bal(f"pay:{pid}")
        if msats > held or out + msats > gross - fee:
            raise InvariantError(f"payment {pid}: paying out {msats} msats would exceed what its payer paid in "
                                 f"({gross - fee} msats after the fee, {out} already out, {held} held)")
        self._move(f"pay:{pid}", account, msats, memo, payout=True)
        self.db.execute("UPDATE payments SET out=out+? WHERE id=?", (msats, pid))
        self._add("refunded_msats" if refund else "paid_out_msats", msats)
        return msats

    def _refund_payment(self, pid, why):
        """Whatever a payment still holds, unvested and unpaid, back to its payer."""
        live = self.db.execute("SELECT COALESCE(SUM(msats), 0) FROM vesting WHERE payment=? AND status='vesting'",
                               (pid,)).fetchone()[0]
        left = self._bal(f"pay:{pid}") - live
        return self._disburse(pid, self._payment(pid)[1], left, f"refund: {why}", refund=True) if left > 0 else 0

    def _split(self, pid, lid, split, tree, vest=(), withhold=False, amount=None):
        """Split what payment `pid` holds (or `amount` of it: a challenge's tranche) by the protocol's `split` down
        `lid`'s family tree: roles in `vest` wait vest_epochs in the payment's escrow, the others are paid now; what the
        tree can't place (withheld parents, rounding) goes back to the payer. The shares are checked against what the
        payment holds before anything moves."""
        held = self._bal(f"pay:{pid}") - self.db.execute(
            "SELECT COALESCE(SUM(msats), 0) FROM vesting WHERE payment=? AND status='vesting'", (pid,)).fetchone()[0]
        if amount is not None:
            held = min(held, int(amount))
        if held <= 0:
            return {}
        shares = self._shares(lid, held, split, withhold=withhold, tree=tree)
        total = sum(m for payees in shares.values() for m in payees.values())
        if total > held:
            raise InvariantError(f"payment {pid}: shares of {total} msats from {held} held")
        paid = {}
        for role, payees in shares.items():
            for acct, m in payees.items():
                if role in vest or "all" in vest:
                    self._vest(acct, m, pid, lid, role)
                else:
                    self._disburse(pid, acct, m, f"{role}: {lid[:19]}")
                paid[acct] = paid.get(acct, 0) + m
        if held - total > 0:
            self._disburse(pid, self._payment(pid)[1], held - total, "refund: not placed down the tree", refund=True)
        self.db.execute("UPDATE payments SET split=1 WHERE id=?", (pid,))
        return paid

    def fees(self):
        """The standard fee, and what it has paid the operator."""
        return {"per_transaction_msats": self.tx_fee_msats,
                "per_transaction_usd_approx": usd_approx(self.tx_fee_msats, self.btc_usd),
                "usd_per_btc": self.btc_usd, "paid_in": "sats, at once, to the operator that served the transaction",
                "operator": self.fee_to, "paid_msats": self._m("fees_msats"),
                "repeg": {"every_epochs": self.p.fee_repeg_epochs, "target_usd": f"{self.p.fee_target_usd_nanos / 1e9:.5f}"}
                if self.p.fee_repeg_epochs else "off: fixed in sats, so its dollar value floats with bitcoin",
                "how_it_was_set": "examples/fees/measure.py: about 125x the electricity of the dearest transaction, "
                                  "converted at $85,962 a bitcoin"}

    def set_btc_usd(self, usd_per_btc):
        """Operator: the dollars-per-bitcoin reference for the approximate dollar figures and, if it is on, the fee
        re-peg. No amount on the node is ever computed from it otherwise."""
        usd = int(usd_per_btc or 0)
        if usd <= 0:
            raise ValueError("usd_per_btc is a whole number of dollars, more than 0")
        with self.lock:
            self.btc_usd = usd
            self._set_meta("btc_usd", usd)
            self.db.commit()
        return {"usd_per_btc": usd, "fee": self.fees()}

    def _repeg_fee(self, e):
        """Every fee_repeg_epochs epochs (if on): set the fee back to fee_target_usd_nanos at the operator's btc_usd."""
        n = self.p.fee_repeg_epochs
        if not n or not self.tx_fee_msats or e % n:
            return
        fee = max(1, (self.p.fee_target_usd_nanos * 100 + self.btc_usd // 2) // self.btc_usd)   # nanos -> msats
        if fee != self.tx_fee_msats:
            self._event(f"standard fee re-pegged: {self.tx_fee_msats} → {fee} msats (${self.p.fee_target_usd_nanos / 1e9:.5f}"
                        f" at ${self.btc_usd:,} a bitcoin)", force=True)
            self.tx_fee_msats = fee
            self._set_meta("tx_fee_msats", fee)

    def l402(self, account, msats):
        """The 402 challenge for a payment an account can't cover: an invoice for what is missing and a macaroon bound
        to the account and amount. Mainnet: a Lightning invoice; the client pays it and retries with
        `Authorization: L402 <macaroon>:<preimage>`, and the node credits the sats to the account. Testnet: a
        placeholder invoice nobody can pay; the faucet gives test sats instead."""
        msats = max(int(msats or 0), 1)
        nonce = hashlib.sha256(f"{self._meta('beacon')}|{account}|{msats}|{time.time()}".encode()).hexdigest()
        caveats = {"account": account, "amount_msats": msats, "node": "traceX testnet", "payment_hash": nonce}
        mac = base64.urlsafe_b64encode(json.dumps(caveats, sort_keys=True).encode()).decode().rstrip("=")
        testnet = bool(self.test_credits)
        return {"scheme": "L402", "amount_msats": msats, "amount_sats": -(-msats // 1000),
                "invoice": f"lntbs{-(-msats // 1000)}n1testnetplaceholder{nonce[:24]}" if testnet else None,
                "invoice_is_placeholder": testnet, "payment_hash": nonce, "macaroon": mac,
                "then": "pay the invoice, retry with Authorization: L402 <macaroon>:<preimage>" if not testnet
                else "testnet: no Lightning yet; POST /v0/faucet gives each wallet 30,000 test sats once"}

    # --- traces: copies earn nothing extra -----------------------------------------------------------------------------
    def submit_trace(self, t):
        out = super().submit_trace(t)                      # (which charges the standard fee)
        if out.get("duplicate"):
            return out
        key, fix = dup_key(t), fix_key(t)
        with self.lock:
            first = self.db.execute("SELECT canonical FROM dups WHERE key=? LIMIT 1", (key,)).fetchone()
            if not first and len(fix_text(t)) >= 20:               # the same distinctive fix, its input reworded
                mine = words(t.get("input", ""))
                for tid, canon in self.db.execute("SELECT trace, canonical FROM dups WHERE fix=? LIMIT 200", (fix,)).fetchall():
                    body = self.db.execute("SELECT body FROM traces WHERE id=?", (tid,)).fetchone()
                    if body and overlap(mine, words(json.loads(body[0]).get("input", ""))) >= self.p.near_dup:
                        first = (canon,)
                        break
            self.db.execute("INSERT OR IGNORE INTO dups (trace, canonical, key, fix) VALUES (?,?,?,?)",
                            (out["id"], first[0] if first else out["id"], key, fix))
            self.db.commit()
        if first:
            out["near_duplicate_of"] = first[0]
            self._link_copy(out["id"], first[0])       # v0.7: a copy is its original's case in the registry
            self.db.commit()
        return out

    def _canonical(self, tid):
        r = self.db.execute("SELECT canonical FROM dups WHERE trace=?", (tid,)).fetchone()
        return r[0] if r else tid

    # --- bounties: refundable pledge escrows ----------------------------------------------------------------------------
    def _seed_amount(self, b):
        return msats_in(b, "seed") or msats_in(b, "reward")

    def pledge(self, bounty_id, backer, msats):
        """Pledge sats into a bounty's escrow (each pledge is its own payment). A solve pays them to the solver and the
        traces it was built from; if the bounty ends unsolved every backer gets back exactly what it pledged. A
        pledge is not an investment: no token, no share of the solution's revenue, nothing to sell or transfer."""
        need_address(backer, "backer")
        msats = int(msats)
        if msats <= 0:
            raise ValueError("a pledge must be more than 0 msats")
        with self.lock:
            status, pool, _ = self._bounty(bounty_id)
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}")
            pid = self._pay_in(backer, "pledge", f"bounty:{bounty_id}", msats)
            self.db.execute("INSERT INTO pledges (bounty, backer, amount, epoch, payment) VALUES (?,?,?,?,?)",
                            (bounty_id, backer, msats, self.epoch, pid))
            self.db.execute("UPDATE bounties SET pool=pool+?, pledged=pledged+? WHERE id=?", (msats, msats, bounty_id))
            self._event(f"bounty #{bounty_id} backed with {fmt_sats(msats)}; it now holds {fmt_sats(pool + msats)}")
            self.db.commit()
        return {"bounty": bounty_id, "pledged_msats": msats, "pool_msats": pool + msats, "payment": pid,
                "backers": len(self._backers(bounty_id)),
                "refund": "everything you pledged comes back if the bounty ends unsolved"}

    def _pledge_payments(self, bounty_id):
        return [p for (p,) in self.db.execute("SELECT payment FROM pledges WHERE bounty=? ORDER BY id", (bounty_id,))]

    def _refund(self, bounty_id, amount, why):
        """Every backer gets back what is left of its own pledge (all of it, unless a solve already paid some out)."""
        for pid in self._pledge_payments(bounty_id):
            self._refund_payment(pid, f"bounty {bounty_id} {why}")

    def claim_bounty(self, bounty_id, learning_id, attestation=None):
        """A bounty pays when its poster measures the learning on the bounty's hidden eval set (the poster's own failing
        cases) at the target: no validator, however many of them one party controls, can give that for the poster. The
        learning must also be accepted by the federation. Each pledge is then split trainer 70 / traces 20 / checkers 5 /
        validators 5 down the learning's family tree, vesting vest_epochs, so a challenge can still give it back."""
        with self.lock:
            row = self.db.execute("SELECT status, eval_set, target, pool, base_model, poster FROM bounties WHERE id=?",
                                  (bounty_id,)).fetchone()
            if not row:
                raise KeyError(f"bounty {bounty_id}")
            status, eval_set, target, pool, base_model, poster = row
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}")
            v = self.db.execute("SELECT status, audit_bad FROM verdicts WHERE learning=?", (learning_id,)).fetchone()
            if not v or v[0] != "accepted":
                raise ValueError("the learning must be accepted by the validator federation first")
            L = json.loads(self.db.execute("SELECT body FROM learnings WHERE id=?", (learning_id,)).fetchone()[0])
            if not any(a.get("eval_set") == eval_set and a.get("validator") == poster and float(a.get("after", 0)) >= target
                       for a in (L.get("attestation") or {}, attestation or {})):
                raise ValueError("the poster holds this bounty's hidden eval set: a claim needs the poster's own "
                                 "measurement there, at the target")
            if base_model and L["base_model"]["name"] != base_model:
                raise ValueError(f"the bounty is for {base_model}")
            self._tx_fee(L["trainer"])
            padded = v[1] is not None and v[1] > self.p.audit_max_bad
            tree = self._split_tree()
            payout = {}
            for pid in self._pledge_payments(bounty_id):
                for acct, m in self._split(pid, learning_id, BOUNTY_SPLIT, tree, vest=("all",), withhold=padded).items():
                    payout[acct] = payout.get(acct, 0) + m
            self.db.execute("UPDATE bounties SET status='solved', winner=?, learning=?, pool=0 WHERE id=?",
                            (L["trainer"], learning_id, bounty_id))
            self._event(f"bounty #{bounty_id} solved on its poster's own eval: {fmt_sats(pool)} of pledges vest to the "
                        f"solver and the traces over {self.p.vest_epochs} epochs")
            self.db.commit()
        return {"bounty": bounty_id, "status": "solved", "winner": L["trainer"], "pool_msats": pool,
                "vesting_msats": payout}

    # --- licences: paid in sats; the buyer's own learnings say which traces earned them ---------------------------------
    def bid(self, b):
        """A sealed bid for a shared licence on a lot, in sats: `price_msats` or `price_sats`."""
        b = dict(b)
        b["price_micros"] = msats_in(b, "price", required=True)    # the base node's field: this node's money is msats
        return super().bid(b)

    def clear(self):
        out = []
        with self.lock:
            e = self.epoch
            by_lot = {}
            for lot, bidder, price in self.db.execute("SELECT lot, bidder, price FROM bids WHERE epoch=?", (e,)):
                by_lot.setdefault(lot, []).append(Bid(bidder, price))
            self.db.execute("DELETE FROM bids WHERE epoch=?", (e,))        # cleared once: a second call charges nobody
            for lot, bids in sorted(by_lot.items()):
                winners, price = clear_shared(bids, self.k, self.reserve)
                traces = [tid for (tid,) in self.db.execute("SELECT id FROM traces WHERE lot=? ORDER BY id", (lot,))]
                if not winners or not traces:
                    continue
                for w in winners:                          # the bid paid its fee and kept the sats covered
                    pid = self._pay_in(w, "licence", lot, price, fee=False, check=False)
                    self.db.execute("INSERT INTO licences VALUES (?,?,?,?,?)", (lot, w, price, e, json.dumps(traces)))
                    self.db.execute("INSERT INTO licence_escrow (lot, buyer, payment, epoch, traces) VALUES (?,?,?,?,?)",
                                    (lot, w, pid, e, json.dumps(traces)))
                out.append({"lot": lot, "winners": winners, "price_msats": price, "traces": len(traces)})
                self._event(f"lot {lot.split('|')[0]} cleared: {len(winners)} licence{'s' if len(winners) != 1 else ''} at "
                            f"{fmt_sats(price)}; each payment waits for the traces its buyer uses")
            self.db.commit()
        return {"epoch": e, "cleared": out}

    def direct_licence(self, lot, buyer, traces):
        """A buyer names the traces of a lot it used; its licence payment for that lot goes to them. Learnings it
        registers do this for it (their parents are the traces it used). Nobody else can steer it."""
        need_address(buyer, "buyer")
        traces = set(traces or [])
        with self.lock:
            rows = self.db.execute("SELECT id, traces FROM licence_escrow WHERE lot=? AND buyer=? AND paid=0",
                                   (lot, buyer)).fetchall()
            if not rows:
                raise ValueError("none of your licence money is waiting on that lot")
            self._tx_fee(buyer)
            trace_info, _ = self._tree()
            paid = 0
            for rid, tjson in rows:
                used = [t for t in json.loads(tjson) if t in traces]
                if used:
                    paid += self._pay_licence(rid, used, trace_info)
            self.db.commit()
        return {"lot": lot, "buyer": buyer, "paid_msats": paid}

    def _pay_licence(self, rid, used, trace_info):
        lot, pid = self.db.execute("SELECT lot, payment FROM licence_escrow WHERE id=?", (rid,)).fetchone()
        held, vals = self._bal(f"pay:{pid}"), self._validator_set()
        per, dust = divmod(held, len(used))
        for i, tid in enumerate(sorted(used)):
            info = trace_info[tid]
            for acct, m in split_trace_sale(per + (dust if i == 0 else 0), info["producer"], info["checker_author"],
                                            vals).items():
                self._disburse(pid, acct, m, f"licence: {tid[:19]}")
        self._refund_payment(pid, "licence share with no one to pay")   # v0.8: a path trace with no passing trace
        self.db.execute("UPDATE licence_escrow SET paid=1 WHERE id=?", (rid,))
        self._event(f"licence money for lot {lot.split('|')[0]}: {fmt_sats(held)} to the {len(used)} "
                    f"trace{'s' if len(used) != 1 else ''} its buyer used")
        return held

    def _pay_licences(self):
        """At settlement: licence payments go to the traces their buyer's own learnings cite, once validators have
        finished with them and haven't found the parents padded. Junk no buyer uses never earns from them."""
        rows = self.db.execute("SELECT id, buyer, traces FROM licence_escrow WHERE paid=0").fetchall()
        if not rows:
            return
        trace_info, learnings = self._tree()
        state = {lid: (st, bad) for lid, st, bad in self.db.execute("SELECT learning, status, audit_bad FROM verdicts")}
        cited = {}                                       # buyer -> the traces its finished, unpadded learnings cite
        for lid, L in learnings.items():
            st, bad = state.get(lid, ("pending", None))
            if st not in ("pending", "challenged", "decoy") and not (bad is not None and bad > self.p.audit_max_bad):
                cited.setdefault(L.get("trainer"), set()).update(p["trace"] for p in L["parents"])
        for rid, buyer, tjson in rows:
            used = [t for t in json.loads(tjson) if t in cited.get(buyer, ())]
            if used:
                self._pay_licence(rid, used, trace_info)

    # --- validators: staked in sats -------------------------------------------------------------------------------------
    def min_stake_msats(self):
        return self.p.validator_min_stake_msats

    def learning_bond_msats(self):
        return self.p.learning_bond_msats if self.p.quorum > 0 else 0

    def challenge_stake_msats(self):
        return self.p.challenge_stake_msats

    def _stake(self, address):
        r = self.db.execute("SELECT stake FROM validators WHERE address=?", (address,)).fetchone()
        return int(r[0]) if r else 0

    def _stakes(self):
        return [(a, int(s)) for a, s in self.db.execute("SELECT address, stake FROM validators ORDER BY address").fetchall()]

    def _active(self):
        """Validators staked at least the minimum. There is no price to fall, so there is no grace period: a validator
        slashed under the minimum loses its seat at once, and tops up to come back."""
        floor = self.min_stake_msats()
        return [(a, s) for a, s in self._stakes() if s >= floor]

    def register_validator(self, address, stake_msats):
        """Stake sats to join the validator federation: at least 10,000 sats. More stake: drawn more often, earns
        more, loses more."""
        need_address(address, "validator")
        stake = int(stake_msats)
        with self.lock:
            have = self._stake(address)
            if stake <= 0 or have + stake < self.min_stake_msats():
                raise ValueError(f"validators stake at least {fmt_sats(self.min_stake_msats())}")
            self._need_funds(address, stake)
            self._tx_fee(address)
            self._move(address, f"stake:{address}", stake, "validator stake")
            if self.db.execute("SELECT 1 FROM validators WHERE address=?", (address,)).fetchone():
                self.db.execute("UPDATE validators SET stake=? WHERE address=?", (have + stake, address))
            else:
                self.db.execute("INSERT INTO validators (address, stake, joined) VALUES (?,?,?)",
                                (address, stake, self.epoch))
            self._event(f"a validator staked {fmt_sats(stake)}")
            self.db.commit()
        return self.validator(address)

    def validator(self, address):
        r = self.db.execute("SELECT stake, joined, slashed FROM validators WHERE address=?", (address,)).fetchone()
        if not r:
            raise KeyError(address)
        strikes = self.db.execute("SELECT COUNT(*) FROM strikes WHERE validator=? AND epoch > ?",
                                  (address, self.epoch - self.p.strike_window)).fetchone()[0]
        return {"address": address, "stake_msats": int(r[0]), "joined": r[1], "slashed_msats": int(r[2]),
                "active": int(r[0]) >= self.min_stake_msats(), "decoy_strikes": strikes}

    def validators_list(self):
        rows = sorted(self._stakes(), key=lambda v: -v[1])
        return {"validators": [self.validator(a) for a, _ in rows], "quorum": self.p.quorum,
                "min_stake_msats": self.min_stake_msats()}

    def _validator_set(self):
        return [a for a, _ in self._active()] or list(self.validators)

    def _slash(self, address, fraction, why):
        stake = self._stake(address)
        if stake <= 0:
            return 0
        cut = _frac(stake, fraction)
        self.db.execute("UPDATE validators SET stake=stake-?, slashed=slashed+? WHERE address=?", (cut, cut, address))
        self._forfeit(f"stake:{address}", cut, "slash", address)
        return cut

    # --- learnings: bonded, validated by a random federation, commit then reveal --------------------------------------
    def register_learning(self, l, bond_from=None):
        a = l.get("attestation") or {}
        need_address(l.get("trainer"), "trainer")
        if a and not float(a.get("after", 0)) > float(a.get("before", 1)):
            raise ValueError("rejected: the claimed attestation shows no gain")
        for p in l["parents"]:
            known = self.db.execute("SELECT 1 FROM traces WHERE id=? UNION SELECT 1 FROM learnings WHERE id=?",
                                    (p["trace"], p["trace"])).fetchone()
            if not known:
                raise ValueError(f"rejected: unknown parent {p['trace']}")
        if self._too_deep(l["parents"]):
            raise ValueError(f"rejected: learnings nest at most {MAX_DEPTH} deep")
        per_call = (l.get("royalty") or {}).get("per_call_msats")
        if not isinstance(per_call, int) or isinstance(per_call, bool) or per_call < 0:
            raise ValueError("rejected: a sats node prices calls in sats: royalty.per_call_msats, a whole number of "
                             "millisatoshis set by the seller (Learning.build(per_call_msats=...))")
        lid = object_id(l)
        weights = (l.get("artifact") or {}).get("hash")
        payer = bond_from or l["trainer"]
        with self.lock:
            if self.db.execute("SELECT 1 FROM learnings WHERE id=?", (lid,)).fetchone():
                return dict(self.verdict(lid), id=lid)
            if weights:                                  # the same weights under a new name earn nothing new
                first = self.db.execute("SELECT learning FROM artifacts WHERE hash=?", (weights,)).fetchone()
                if first:
                    raise ValueError(f"rejected: these weights are already learning {first[0][:19]}…; a learning built "
                                     "on it has weights of its own")
            bond = self.learning_bond_msats()
            self._need_funds(payer, bond)
            self._tx_fee(payer)
            self._move(payer, f"bond:{lid}", bond, "learning bond")
            self.db.execute("INSERT INTO learnings VALUES (?,?,?)", (lid, canonical(l).decode(), self.epoch))
            self.db.execute("INSERT INTO verdicts (learning, status, round, bond, trainer, registered, bond_from) "
                            "VALUES (?,?,?,?,?,?,?)", (lid, "pending", 0, bond, l["trainer"], self.epoch, payer))
            if weights:
                self.db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?)", (weights, lid))
            if self.p.quorum <= 0:
                gain = float(a.get("after", 0)) - float(a.get("before", 0))
                self._accept(lid, gain, None, gain, 0)
            elif self.beacon_delay == 0:
                self._assign(lid, 0)
            self._event(f"learning submitted for validation: {l['kind']} for {l['base_model']['name']}"
                        + (f", claims {float(a['before']):.1%} → {float(a['after']):.1%}" if a else "")
                        + (f"; bond {fmt_sats(bond)}" if bond else ""))
            self.db.commit()
        return dict(self.verdict(lid), id=lid)

    def _eligible(self, exclude=()):
        """Who can be drawn: validators staked at the minimum, not excluded (all of them if exclusion leaves too few)."""
        act = self._active()
        vals = [(a, s) for a, s in act if a not in exclude]
        return vals if len(vals) >= self.p.quorum else act

    def _score(self, lid, rnd, a, s):
        h = int(hashlib.sha256(f"{self._meta('beacon')}|{lid}|{rnd}|{a}".encode()).hexdigest(), 16)
        return -math.log((h + 1) / (2 ** 256 + 1)) / s

    def _assign(self, lid, rnd, exclude=()):
        """Stake-weighted rendezvous hashing over the current beacon: deterministic, and unknowable when submitted."""
        if self.db.execute("SELECT 1 FROM assignments WHERE learning=? AND round=?", (lid, rnd)).fetchone():
            return
        vals = self._eligible(exclude)
        if len(vals) < self.p.quorum:
            return
        for a, _ in sorted(vals, key=lambda v: self._score(lid, rnd, *v))[:self.p.quorum]:
            self.db.execute("INSERT INTO assignments VALUES (?,?,?,?)", (lid, rnd, a, self.epoch))

    def _redraw(self, lid, rnd):
        """Validators who sat on an assignment for a whole epoch without revealing lose a little stake and are replaced."""
        revealed = {v for (v,) in self.db.execute("SELECT validator FROM reveals WHERE learning=? AND round=?", (lid, rnd))}
        late = [v for v in self.assigned(lid, rnd) if v not in revealed]
        for v in late:
            self._slash(v, self.p.noshow_slash, "assigned but did not reveal")
            self.db.execute("DELETE FROM assignments WHERE learning=? AND round=? AND validator=?", (lid, rnd, v))
            self.db.execute("DELETE FROM commits WHERE learning=? AND round=? AND validator=?", (lid, rnd, v))
        keep = self.assigned(lid, rnd)
        pool = [(a, s) for a, s in self._eligible() if a not in keep and a not in late]
        pool.sort(key=lambda v: self._score(lid, rnd, *v))
        for a, _ in pool[:self.p.quorum - len(keep)]:
            self.db.execute("INSERT INTO assignments VALUES (?,?,?,?)", (lid, rnd, a, self.epoch))

    def assigned(self, lid, rnd=None):
        rnd = self._round(lid) if rnd is None else rnd
        return [a for (a,) in self.db.execute("SELECT validator FROM assignments WHERE learning=? AND round=?", (lid, rnd))]

    def _round(self, lid):
        r = self.db.execute("SELECT round FROM verdicts WHERE learning=?", (lid,)).fetchone()
        if not r:
            raise KeyError(lid)
        return r[0]

    def commit(self, lid, validator, digest, rnd=None):
        rnd = self._round(lid) if rnd is None else int(rnd)
        with self.lock:
            if validator not in self.assigned(lid, rnd):
                raise PermissionError("this validator was not assigned to that learning (assignment is random)")
            if self.db.execute("SELECT 1 FROM commits WHERE learning=? AND round=? AND validator=?",
                               (lid, rnd, validator)).fetchone():
                raise ValueError("already committed")
            self._tx_fee(validator)
            self.db.execute("INSERT INTO commits VALUES (?,?,?,?,?)", (lid, rnd, validator, str(digest), self.epoch))
            self.db.commit()
        return {"learning": lid, "round": rnd, "committed": validator,
                "waiting_for": [v for v in self.assigned(lid, rnd) if v not in self._committed(lid, rnd)]}

    def _committed(self, lid, rnd):
        return [v for (v,) in self.db.execute("SELECT validator FROM commits WHERE learning=? AND round=?", (lid, rnd))]

    def reveal(self, lid, validator, attestation, salt, rnd=None):
        rnd = self._round(lid) if rnd is None else int(rnd)
        with self.lock:
            c = self.db.execute("SELECT digest, epoch FROM commits WHERE learning=? AND round=? AND validator=?",
                                (lid, rnd, validator)).fetchone()
            if not c:
                raise ValueError("commit first")
            assigned, committed = self.assigned(lid, rnd), self._committed(lid, rnd)
            if set(committed) != set(assigned) and self.epoch <= c[1]:
                raise ValueError("reveals open once every assigned validator has committed, or next epoch")
            if attestation_digest(attestation, salt) != c[0]:
                raise ValueError("this attestation does not match the commitment")
            before, after = float(attestation["before"]), float(attestation["after"])
            if not (0 <= before <= 1 and 0 <= after <= 1):
                raise ValueError("before and after are scores between 0 and 1")
            parents = len(json.loads(self.db.execute("SELECT body FROM learnings WHERE id=?", (lid,)).fetchone()[0])["parents"])
            audit = attestation.get("audit") or {}
            if int(audit.get("checked", 0)) < min(self.p.audit_min, parents):
                raise ValueError(f"every reveal audits at least {min(self.p.audit_min, parents)} of the learning's parents "
                                 "(audit: {checked, bad})")
            self._tx_fee(validator)
            self.db.execute("INSERT OR REPLACE INTO reveals VALUES (?,?,?,?,?,?,?)",
                            (lid, rnd, validator, canonical(attestation).decode(), after - before, self.epoch, None))
            done = len(self.db.execute("SELECT validator FROM reveals WHERE learning=? AND round=?", (lid, rnd)).fetchall())
            if done >= len(assigned):
                self._finalize(lid, rnd)
            self.db.commit()
        return dict(self.verdict(lid), revealed=validator)

    def _claim(self, lid):
        """The trainer's own claim: (gain, its standard error), or None when it made none."""
        a = json.loads(self.db.execute("SELECT body FROM learnings WHERE id=?", (lid,)).fetchone()[0]).get("attestation") or {}
        if "after" not in a or "before" not in a:
            return None
        return float(a["after"]) - float(a["before"]), standard_error(a)

    def _claimed_gain(self, lid):
        c = self._claim(lid)
        return c[0] if c else None

    def _finalize(self, lid, rnd):
        rows = [(v, json.loads(b), g) for v, b, g in self.db.execute(
            "SELECT validator, body, gain FROM reveals WHERE learning=? AND round=?", (lid, rnd)).fetchall()]
        if not rows:
            return
        med = statistics.median(g for _, _, g in rows)
        agreed, audits, ses = [], [], []
        for v, a, g in rows:                              # far from the median is no fault: it only shares in less
            ok = abs(g - med) <= tolerance(a, self.p.z, self.p.min_tol)
            self.db.execute("UPDATE reveals SET agreed=? WHERE learning=? AND round=? AND validator=?", (int(ok), lid, rnd, v))
            if ok:
                agreed.append(v)
                ses.append(standard_error(a))
                au = a.get("audit") or {}
                if au.get("checked"):
                    audits.append(float(au.get("bad", 0)) / float(au["checked"]))
        for v in set(self.assigned(lid, rnd)) - {v for v, _, _ in rows}:
            self._slash(v, self.p.noshow_slash, "assigned but did not reveal")
            self.db.execute("DELETE FROM assignments WHERE learning=? AND round=? AND validator=?", (lid, rnd, v))
        se_med = 1.2533 * statistics.median(ses) / math.sqrt(len(ses)) if ses else 1.0
        lower = med - self.p.accept_z * se_med
        audit = statistics.median(audits) if audits else None
        status, prior_bad = self.db.execute("SELECT status, audit_bad FROM verdicts WHERE learning=?", (lid,)).fetchone()
        if rnd == 0:                                      # an overclaim: beyond twice the gain, allowing both sides' noise
            claim = self._claim(lid)
            over = claim is not None and (claim[0] - self.p.accept_z * claim[1]
                                          > self.p.overclaim * max(med, self.p.min_gain) + self.p.accept_z * se_med)
            money = not self._decoy(lid)                   # a decoy looks like any verdict; its money waits for the truth
            if med < self.p.min_gain or over:
                self._reject(lid, med, over, money)
            elif lower < self.p.min_gain:
                self._inconclusive(lid, med, audit, money)
            else:
                self._accept(lid, med, audit, lower, len(agreed), money)
        elif status == "challenged" and not self._decoy(lid):     # a challenged decoy waits for the operator's unseal
            if med < self.p.min_gain:
                self._clawback(lid, med)
            elif audit is not None and audit > self.p.audit_max_bad and (prior_bad is None or prior_bad <= self.p.audit_max_bad):
                self._clawback(lid, med, parents_only=True)
            else:
                self._challenge_failed(lid, med)

    def _bond(self, lid):
        return self._bal(f"bond:{lid}")

    def _reject(self, lid, med, over, money=True):
        """No gain the validators can see, or a claim far beyond what they measured: the bond is forfeit, destroyed."""
        if money:
            self._forfeit(f"bond:{lid}", self._bond(lid), "bond", lid)
        self.db.execute("UPDATE verdicts SET status='rejected', gain=?, note=? WHERE learning=?",
                        (med, "overclaimed" if over else None, lid))
        self._event(f"learning rejected: validators measured {med * 100:+.1f} points"
                    + (", far below what it claimed" if over else "") + "; its bond is destroyed")

    def _inconclusive(self, lid, med, audit, money=True):
        if money:
            bond = self._bond(lid)
            cut = _frac(bond, self.p.inconclusive_burn)
            self._forfeit(f"bond:{lid}", cut, "bond", lid)
            self._release_bond(lid, to=self._bond_payer(lid))
        self.db.execute("UPDATE verdicts SET status='inconclusive', gain=?, audit_bad=? WHERE learning=?", (med, audit, lid))
        self._event(f"learning inconclusive: median gain {med * 100:+.1f} points, but within the noise of the "
                    "validators' eval sets; bond back minus 10%")

    def _accept(self, lid, gain, audit, lower, n, money=True):
        padded = audit is not None and audit > self.p.audit_max_bad
        self.db.execute("UPDATE verdicts SET status='accepted', gain=?, accepted=?, audit_bad=?, note=? WHERE learning=?",
                        (gain, self.epoch, audit, "padded" if padded else None, lid))
        if padded and money:
            self._pad_burn(lid)
        self._event(f"learning accepted by {n} validators: median gain {gain * 100:+.1f} points "
                    f"(at least {lower * 100:+.1f} at 2 standard errors)"
                    + ("; its parents are padding, so their share goes back to payers and half the bond is destroyed"
                       if padded else ""))

    def _pad_burn(self, lid):
        self._forfeit(f"bond:{lid}", _frac(self._bond(lid), self.p.pad_burn), "bond", lid)

    def _bond_payer(self, lid):
        r = self.db.execute("SELECT COALESCE(bond_from, trainer) FROM verdicts WHERE learning=?", (lid,)).fetchone()
        return r[0] if r else None

    def _release_bond(self, lid, to):
        bond = self._bond(lid)
        if bond > 0:
            if to:
                self._move(f"bond:{lid}", to, bond, "bond returned", payout=True)
            else:
                self._forfeit(f"bond:{lid}", bond, "bond", lid)
        self.db.execute("UPDATE verdicts SET bond=0 WHERE learning=?", (lid,))
        return bond

    def verdict(self, lid):
        r = self.db.execute("SELECT status, round, gain, trainer, registered, accepted, audit_bad, challenger, note "
                            "FROM verdicts WHERE learning=?", (lid,)).fetchone()
        if not r:
            raise KeyError(lid)
        reveals = [{"validator": v, "gain": round(g, 4), "agreed": None if ok is None else bool(ok), "round": rd}
                   for v, g, ok, rd in self.db.execute(
                       "SELECT validator, gain, agreed, round FROM reveals WHERE learning=? ORDER BY round, validator", (lid,))]
        return {"learning": lid, "status": r[0], "round": r[1], "median_gain": r[2], "bond_msats": self._bond(lid),
                "trainer": r[3], "registered_epoch": r[4], "accepted_epoch": r[5], "audit_bad": r[6], "challenger": r[7],
                "note": r[8], "assigned": self.assigned(lid, r[1]), "committed": self._committed(lid, r[1]),
                "reveals": reveals, "quorum": self.p.quorum}

    # --- challenges: fraud proofs, any time; they take back whatever hasn't vested yet -----------------------------------
    def challenge(self, lid, challenger):
        """Anyone can challenge an accepted learning at any time by staking 2,000 sats. Fresh validators re-measure it
        on new eval sets (and look at its parents); its escrowed shares wait, so an upheld challenge always has
        something to give back to the people who paid."""
        need_address(challenger, "challenger")
        with self.lock:
            v = self.db.execute("SELECT status, accepted, round FROM verdicts WHERE learning=?", (lid,)).fetchone()
            if not v:
                raise KeyError(lid)
            if v[0] != "accepted":
                raise ValueError("only an accepted learning can be challenged")
            if self.db.execute("SELECT 1 FROM challenge_subs WHERE learning=?", (lid,)).fetchone():
                raise ValueError("a challenge solution was scored by its challenge's verifier (re-run it: it is "
                                 "deterministic) or by drawn validators; it is not re-measured as a learning")
            stake = self.challenge_stake_msats()
            self._need_funds(challenger, stake)
            self._tx_fee(challenger)
            self._move(challenger, f"challenge:{lid}", stake, "challenge stake")
            rnd = v[2] + 1
            self.db.execute("UPDATE verdicts SET status='challenged', round=?, challenger=?, challenge_stake=? WHERE learning=?",
                            (rnd, challenger, stake, lid))
            if self.beacon_delay == 0:
                self._assign(lid, rnd, exclude=self.assigned(lid, v[2]))
            self._event("a learning was challenged: fresh validators re-measure it on new eval sets; its payouts pause")
            self.db.commit()
        return self.verdict(lid)

    def _challenge_row(self, lid):
        challenger, rnd = self.db.execute("SELECT challenger, round FROM verdicts WHERE learning=?", (lid,)).fetchone()
        return challenger, self._bal(f"challenge:{lid}"), rnd

    def _clawback(self, lid, med, parents_only=False):
        """Upheld: escrowed shares go back to whoever paid them (users, bounty backers), the bond is destroyed, the
        challenger gets its stake back, and the validators who vouched for it lose stake. An audit challenge does the
        same to the parents' share and half the bond."""
        challenger, stake, rnd = self._challenge_row(lid)
        for bid in self._claw(lid, roles=("traces", "checkers", "parents") if parents_only else None):
            if not parents_only:
                self.db.execute("UPDATE bounties SET status='clawed back' WHERE id=?", (bid,))
        self._move(f"challenge:{lid}", challenger, stake, "challenge stake back", payout=True)
        first = [v for (v,) in self.db.execute("SELECT validator FROM reveals WHERE learning=? AND round<? AND agreed=1",
                                               (lid, rnd))]
        for v in first:
            self._slash(v, self.p.fake_slash / (2 if parents_only else 1),
                        "passed padded parents" if parents_only else "vouched for a gain fresh validators could not reproduce")
        if parents_only:
            self._pad_burn(lid)
            self.db.execute("UPDATE verdicts SET status='accepted', challenger=NULL, challenge_stake=0, audit_bad=?, "
                            "note='padded' WHERE learning=?", (1.0, lid))
            self._event("audit challenge upheld: the learning's parents were padding; their escrowed share goes back to "
                        "the payers and half the bond is destroyed")
        else:
            self._release_bond(lid, to=None)
            self.db.execute("UPDATE verdicts SET status='clawed back', gain=? WHERE learning=?", (med, lid))
            self._event(f"challenge upheld: fresh validators measured {med * 100:+.1f} points; escrowed shares go back "
                        "to the payers and the bond is destroyed")

    def _claw(self, lid, roles=None):
        """Stop a learning's escrowed shares (only `roles`, if given) and give them back to each payment's payer: a
        bounty's backers, a learning's users. Returns the bounties whose pledges came back."""
        bounties = set()
        for vid, pid, msats, role in self.db.execute(
                "SELECT id, payment, msats, role FROM vesting WHERE learning=? AND status='vesting'", (lid,)).fetchall():
            if roles and role not in roles:
                continue
            self.db.execute("UPDATE vesting SET status='clawed' WHERE id=?", (vid,))
            kind, payer, ref = self._payment(pid)[:3]
            self._disburse(pid, payer, msats, "clawed back: refund", refund=True)
            if kind == "pledge":
                bounties.add(int(str(ref).split(":")[1]))
        return bounties

    def _challenge_failed(self, lid, med):
        challenger, stake, _ = self._challenge_row(lid)
        self._forfeit(f"challenge:{lid}", stake, "challenge", lid)
        self.db.execute("UPDATE verdicts SET status='accepted', challenger=NULL, challenge_stake=0 WHERE learning=?", (lid,))
        self._event(f"challenge rejected: fresh validators reproduced the gain ({med * 100:+.1f} points); the challenge "
                    "stake is destroyed")

    def _agreed(self, lid):
        return [v for (v,) in self.db.execute(
            "SELECT validator FROM reveals WHERE learning=? AND round=0 AND agreed=1", (lid,))]

    def _tree(self):
        """Producers per trace, with every near-duplicate paying the first copy's producer."""
        trace_info, learnings = super()._tree()
        for tid, canon in self.db.execute("SELECT trace, canonical FROM dups WHERE trace != canonical").fetchall():
            if tid in trace_info and canon in trace_info:
                trace_info[tid] = trace_info[canon]
        return trace_info, learnings

    def _nested(self, learnings):
        """Learnings as they count when another learning cites them: each passes its whole slice through to its own
        traces and checkers, equal per distinct parent (copies count as their original). A trainer and its validators
        are paid for their own learning's use, not again whenever another learning cites it, so wrapping someone's
        traces in a learning of one's own, with whatever terms, diverts nothing."""
        canon = dict(self.db.execute("SELECT trace, canonical FROM dups").fetchall())
        through = {"traces": 6 / 7, "checkers": 1 / 7, "trainer": 0.0, "validators": 0.0}
        return {lid: dict(L, royalty=dict(L["royalty"], split=through),
                          parents=[{"trace": p, "weight": 1}
                                   for p in sorted({canon.get(q["trace"], q["trace"]) for q in L["parents"]})])
                for lid, L in learnings.items()}

    def _split_tree(self):
        trace_info, learnings = self._tree()
        return trace_info, self._nested(learnings)

    def _shares(self, lid, amount, split=None, withhold=False, tree=None):
        """Who gets `amount` paid for a learning: by role, on the protocol's split (not the trainer's terms), with equal
        weight per distinct parent (a copy counts as its original), so neither terms nor padding can tilt it; a parent
        learning passes its slice on to its own traces (see _nested). With `withhold` (the parents were found padded)
        the parents' part is left out. Returns {role: {account: msats}}, summing to at most `amount`; the caller gives
        what it leaves out back to the payer."""
        L = json.loads(self.db.execute("SELECT body FROM learnings WHERE id=?", (lid,)).fetchone()[0])
        parents = sorted({self._canonical(p["trace"]) for p in L["parents"]})
        split = dict(split or USAGE_SPLIT)
        trace_info, learnings = tree or self._split_tree()
        validators = self._agreed(lid) or self._validator_set()
        out = {}
        for role, keys in (("trainer", ("trainer",)), ("validators", ("validators",)), ("traces", ("traces",)),
                           ("checkers", ("checkers",))):
            if role in ("traces", "checkers") and (withhold or not parents):
                continue
            part = amount * round(split[role] * 1_000_000) // 1_000_000
            sub = {k: (split[k] if k in keys else 0) for k in split}
            L2 = dict(L, parents=[{"trace": p, "weight": 1 / max(len(parents), 1)} for p in parents],
                      royalty=dict(L["royalty"], split=sub))
            got = split_usage(part, L2, trace_info, validators, learnings)
            if got:
                out[role] = got
        return out

    def _padded(self, lid):
        r = self.db.execute("SELECT audit_bad FROM verdicts WHERE learning=?", (lid,)).fetchone()
        return bool(r and r[0] is not None and r[0] > self.p.audit_max_bad)

    # --- decoys: does a validator measure, or only answer? ------------------------------------------------------------
    def register_decoy(self, learning, digest, funder):
        """Operator: submit a learning whose true gain only the operator knows, sealed as decoy_digest(gain, salt), its
        bond paid by `funder`. Nothing about it differs from other learnings until validators have revealed; then
        unseal_decoy() opens the truth. A validator whose score sits further from it than 4 of its own standard errors
        gets a strike; two strikes inside strike_window epochs cost 25% of its stake (it never measured)."""
        need_address(funder, "funder")
        out = self.register_learning(learning, bond_from=funder)
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO decoys (learning, digest, funder) VALUES (?,?,?)",
                            (out["id"], str(digest), funder))
            self.db.commit()
        return out

    def _decoy(self, lid):
        return self.db.execute("SELECT 1 FROM decoys WHERE learning=? AND unsealed=0", (lid,)).fetchone() is not None

    def unseal_decoy(self, lid, gain, salt):
        with self.lock:
            r = self.db.execute("SELECT digest, funder, unsealed FROM decoys WHERE learning=?", (lid,)).fetchone()
            if not r:
                raise KeyError(lid)
            if r[2]:
                raise ValueError("this decoy is already unsealed")
            if decoy_digest(gain, salt) != r[0]:
                raise ValueError("that is not the truth this decoy was sealed with")
            status = self.db.execute("SELECT status FROM verdicts WHERE learning=?", (lid,)).fetchone()
            if status[0] == "pending":
                raise ValueError("unseal a decoy once its validators have revealed")
            struck, caught = [], []
            for v, body, g in self.db.execute("SELECT validator, body, gain FROM reveals WHERE learning=? AND round=0",
                                              (lid,)).fetchall():
                if abs(g - float(gain)) > tolerance(json.loads(body), self.p.decoy_z, self.p.min_tol):
                    self.db.execute("INSERT INTO strikes VALUES (?,?,?)", (v, lid, self.epoch))
                    struck.append(v)
                    n = self.db.execute("SELECT COUNT(*) FROM strikes WHERE validator=? AND epoch > ?",
                                        (v, self.epoch - self.p.strike_window)).fetchone()[0]
                    if n >= self.p.decoy_strikes:
                        self._slash(v, self.p.fake_slash, "missed decoys twice: never measured them")
                        self.db.execute("DELETE FROM strikes WHERE validator=?", (v,))
                        caught.append(v)
            challenger, stake, _ = self._challenge_row(lid)
            if status[0] == "challenged" and stake:                         # a watchdog caught it first: stake back
                self._move(f"challenge:{lid}", challenger, stake, "challenge stake back", payout=True)
            self._release_bond(lid, to=r[1])
            self._claw(lid)
            self.db.execute("UPDATE verdicts SET status='decoy', challenger=NULL, challenge_stake=0 WHERE learning=?", (lid,))
            self.db.execute("UPDATE decoys SET unsealed=1, gain=? WHERE learning=?", (float(gain), lid))
            self._event(f"a decoy was unsealed (true gain {float(gain) * 100:+.0f} points): "
                        + (f"{len(struck)} validator{'s' if len(struck) != 1 else ''} scored it far from the truth"
                           + (f", {len(caught)} for the second time and lost stake" if caught else " (a strike)")
                           if struck else "every validator measured it"))
            self.db.commit()
        return {"learning": lid, "struck": struck, "caught": caught}

    # --- vesting: the parents' shares (and bounty payouts) wait in their payment's escrow ------------------------------
    def _vest(self, account, msats, pid, learning, role):
        if msats > 0:
            self.db.execute("INSERT INTO vesting (account, msats, payment, learning, role, release) VALUES (?,?,?,?,?,?)",
                            (account, int(msats), pid, learning, role, self.epoch + self.p.vest_epochs))

    def _release(self):
        e = self.epoch
        paused = {lid for (lid,) in self.db.execute("SELECT learning FROM verdicts WHERE status='challenged'")}
        paused |= self._challenge_paused()               # v0.8: a prior-art claim is pending on its challenge
        for vid, account, msats, pid, lid in self.db.execute(
                "SELECT id, account, msats, payment, learning FROM vesting WHERE status='vesting' AND release <= ?",
                (e,)).fetchall():
            if lid in paused:
                continue
            self._disburse(pid, account, msats, "vested")
            self.db.execute("UPDATE vesting SET status='done' WHERE id=?", (vid,))

    # --- usage: paid now, split at settlement ---------------------------------------------------------------------------
    def usage(self, u):
        """A consumer pays for calls of an accepted learning, in sats, now: calls x the learning's own per_call_msats,
        plus the fee. At settlement that payment is split down the learning's family tree."""
        calls = int(u["calls"])
        if calls <= 0:
            raise ValueError("calls must be positive")
        consumer, lid = need_address(u.get("consumer"), "consumer"), u["learning"]
        with self.lock:
            r = self.db.execute("SELECT body FROM learnings WHERE id=?", (lid,)).fetchone()
            if not r:
                raise ValueError(f"unknown learning {lid}")
            st = self.db.execute("SELECT status FROM verdicts WHERE learning=?", (lid,)).fetchone()
            if not st or st[0] != "accepted":
                raise ValueError("only a learning the validator federation accepted can be paid for; this one is "
                                 f"{st[0] if st else 'unknown'}")
            amount = calls * json.loads(r[0])["royalty"]["per_call_msats"]
            if amount <= 0:                               # a free learning: the call is metered, only the fee is paid
                self._tx_fee(consumer)
                pid = None
            else:
                pid = self._pay_in(consumer, "usage", lid, amount)
            self.db.execute("INSERT INTO usage VALUES (?,?,?,?)", (lid, consumer, calls, self.epoch))
            self.db.commit()
        return {"metered": calls, "paid_msats": amount, "fee_msats": self.tx_fee_msats, "payment": pid}

    def _split_usage(self, tree):
        """At settlement: every usage payment not yet split goes down its learning's tree (traces 60 / trainer 25 /
        checkers 10 / validators 5). Trainer and validators are paid now; traces and checkers wait vest_epochs. A
        learning clawed back meanwhile refunds the payer; a challenged one waits."""
        status = dict(self.db.execute("SELECT learning, status FROM verdicts").fetchall())
        for pid, lid in self.db.execute("SELECT id, ref FROM payments WHERE kind='usage' AND split=0 ORDER BY id").fetchall():
            st = status.get(lid)
            if st == "challenged":
                continue
            if st != "accepted":
                self._refund_payment(pid, f"{lid[:19]} is {st}")
                self.db.execute("UPDATE payments SET split=1 WHERE id=?", (pid,))
                continue
            self._split(pid, lid, USAGE_SPLIT, tree, vest=("traces", "checkers"), withhold=self._padded(lid))

    # --- settlement -------------------------------------------------------------------------------------------------
    def settle(self):
        with self.lock:
            e = self.epoch
            self._expire_bounties()
            self._challenge_settle()                       # v0.8: expiries, validator rounds, escalations (SPEC 4k)
            # rounds that ran out of time: settle with the majority that revealed, or re-draw validators
            for lid, rnd in self.db.execute(
                    "SELECT v.learning, v.round FROM verdicts v WHERE v.status IN ('pending','challenged') AND EXISTS "
                    "(SELECT 1 FROM assignments a WHERE a.learning=v.learning AND a.round=v.round AND a.epoch < ?)", (e,)).fetchall():
                n = self.db.execute("SELECT COUNT(*) FROM reveals WHERE learning=? AND round=?", (lid, rnd)).fetchone()[0]
                if n >= self.p.quorum // 2 + 1:
                    self._finalize(lid, rnd)
                else:
                    self._redraw(lid, rnd)
            self._registry_settle()                        # v0.7: fix rounds that ran out of time, payable bounties
            self._pay_licences()
            self._split_usage(self._split_tree())
            self._release()
            for (lid,) in self.db.execute("SELECT learning FROM verdicts WHERE status='accepted' AND bond != 0 AND "
                                          "accepted + ? <= ?", (self.p.vest_epochs, e)).fetchall():
                if not self._decoy(lid):
                    self._release_bond(lid, to=self._bond_payer(lid))
            batch = self._burn_batch(e)
            payouts = {}
            for a, m in self.db.execute("SELECT account, msats FROM sledger WHERE epoch=? AND payout=1 AND msats > 0",
                                        (e,)).fetchall():
                if ADDRESS.fullmatch(str(a)):
                    payouts[a] = payouts.get(a, 0) + int(m)
            leaves = {a: leaf(e, a, m) for a, m in payouts.items()}
            levels = build_tree(list(leaves.values()))
            root = "0x" + levels[-1][0].hex()
            claims = {a: {"amount_msats": m, "proof": ["0x" + h.hex() for h in proof(levels, leaves[a])]}
                      for a, m in payouts.items()}
            self.db.execute("INSERT OR REPLACE INTO sats_roots VALUES (?,?,?,?)", (e, root, sum(payouts.values()),
                                                                                   json.dumps(claims)))
            self.db.execute("INSERT OR REPLACE INTO roots VALUES (?,?,?,?)", (e, root, sum(payouts.values()), "{}"))
            self._set_meta("beacon", hashlib.sha256(f"{self._meta('beacon')}|{root}|{e}".encode()).hexdigest())
            for (lid,) in self.db.execute("SELECT learning FROM verdicts WHERE status='pending' AND registered <= ?", (e,)).fetchall():
                self._assign(lid, 0)
            for lid, rnd in self.db.execute("SELECT learning, round FROM verdicts WHERE status='challenged'").fetchall():
                self._assign(lid, rnd, exclude=self.assigned(lid, rnd - 1))
            self._registry_assign()                        # v0.7: validators for fixes, from the new beacon
            self._challenge_assign()                       # v0.8: and for challenge submissions
            summary = {"epoch": e}
            for k in ("paid_in_msats", "paid_out_msats", "refunded_msats", "fees_msats", "forfeited_msats"):
                summary[k] = self._m(k) - self._m("mark_" + k)          # this epoch's flow
                self._set_meta("mark_" + k, self._m(k))
            summary.update(burn_batch=batch, payout_root=root, payees=len(payouts))
            self.db.execute("INSERT OR REPLACE INTO epochs VALUES (?,?)", (e, json.dumps(summary)))
            self._repeg_fee(e)
            self._event(f"epoch {e} settled: {fmt_sats(summary['paid_out_msats'])} paid out to the work users paid for, "
                        f"{fmt_sats(summary['fees_msats'])} of fees, {fmt_sats(summary['forfeited_msats'])} of "
                        "forfeits destroyed", force=True)
            self._prune(e)
            self._set_meta("epoch", e + 1)
            self.db.commit()
        return {"epoch": e, "root": root, "total_msats": sum(payouts.values()), "claims": claims, "summary": summary}

    def _burn_batch(self, e):
        """This epoch's forfeits, as one batch destined for a provably unspendable output."""
        rows = self.db.execute("SELECT kind, ref, msats FROM forfeits WHERE epoch=? ORDER BY rowid", (e,)).fetchall()
        if not rows:
            return None
        total = sum(m for _, _, m in rows)
        digest = "sha256:" + hashlib.sha256(canonical([e, [list(r) for r in rows]])).hexdigest()
        self.db.execute("INSERT OR REPLACE INTO burn_batches VALUES (?,?,?,?)", (e, total, len(rows), digest))
        return {"msats": total, "items": len(rows), "digest": digest}

    def _prune(self, e):
        """Settlement and storage at machine scale: per-transaction detail is kept through the challenge window
        (`keep_epochs`), then folded away. Each account's older ledger rows become one carried-forward row, so balances
        stay exact; finished payments, usage reports, forfeit rows, finished vesting and paid licence escrow go; each
        epoch keeps its Merkle root, totals and burn batch for good. Traces and learnings stay."""
        cut = e - self.p.keep_epochs + 1                  # rows from epochs before `cut` are past the window
        if cut <= 1:
            return
        old = self.db.execute("SELECT account, SUM(msats) FROM sledger WHERE epoch < ? GROUP BY account", (cut,)).fetchall()
        if old:
            self.db.execute("DELETE FROM sledger WHERE epoch < ?", (cut,))
            for a, v in old:
                if int(v or 0):
                    self.db.execute("INSERT INTO sledger VALUES (?,?,?,?,?)", (cut - 1, a, int(v), "carried forward", 0))
        live = {f"pay:{p}" for (p,) in self.db.execute("SELECT payment FROM vesting WHERE status='vesting'")}
        held = {a for (a,) in self.db.execute("SELECT account FROM sledger WHERE account LIKE 'pay:%' GROUP BY account "
                                              "HAVING SUM(msats) != 0")}
        for (pid,) in self.db.execute("SELECT id FROM payments WHERE epoch < ? AND (split=1 OR kind != 'usage')",
                                      (cut,)).fetchall():
            if f"pay:{pid}" not in live and f"pay:{pid}" not in held:
                self.db.execute("DELETE FROM payments WHERE id=?", (pid,))
        for sql in ("DELETE FROM usage WHERE epoch < ?", "DELETE FROM forfeits WHERE epoch < ?",
                    "DELETE FROM licence_escrow WHERE paid=1 AND epoch < ?",
                    "DELETE FROM vesting WHERE status != 'vesting' AND release < ?",
                    "UPDATE sats_roots SET claims='{}' WHERE epoch < ? AND claims != '{}'"):
            self.db.execute(sql, (cut,))
        self.db.execute("DELETE FROM strikes WHERE epoch <= ?", (e - self.p.strike_window,))

    # --- reads --------------------------------------------------------------------------------------------------------
    def _vesting_of(self, account=None):
        sql = "SELECT COALESCE(SUM(msats), 0) FROM vesting WHERE status='vesting'" + (" AND account=?" if account else "")
        return int(self.db.execute(sql, (account,) if account else ()).fetchone()[0])

    def _sum_like(self, prefix):
        return int(self.db.execute("SELECT COALESCE(SUM(msats), 0) FROM sledger WHERE account LIKE ?",
                                   (prefix + "%",)).fetchone()[0])

    def wallet(self, account):
        w = super().wallet(account)
        return dict(w, vesting_msats=self._vesting_of(account), stake_msats=self._stake(account), unit="msats")

    def balance(self, account):
        """Sats paid out to an account by epoch, with Merkle proofs for the epochs still inside the challenge window."""
        by_epoch, proofs = {}, {}
        for e, claims in self.db.execute("SELECT epoch, claims FROM sats_roots ORDER BY epoch").fetchall():
            c = json.loads(claims).get(account)
            if c:
                by_epoch[str(e)], proofs[str(e)] = c["amount_msats"], c["proof"]
        return {"account": account, "by_epoch_msats": by_epoch, "total_msats": sum(by_epoch.values()), "proofs": proofs,
                "kept": f"per-address claims for {self.p.keep_epochs} epochs; every epoch's root for good"}

    def last_epoch(self):
        r = self.db.execute("SELECT body FROM epochs ORDER BY epoch DESC LIMIT 1").fetchone()
        return json.loads(r[0]) if r else None

    def economy_stats(self):
        """Where the sats are: what payers paid in, what the work they paid for was paid out, what was refunded, the
        fees, what waits in escrow, what is staked, and the forfeits destroyed. No token, no price, no supply."""
        last = self.last_epoch()
        batches = [{"epoch": e, "msats": m, "items": n, "digest": d} for e, m, n, d in self.db.execute(
            "SELECT epoch, msats, items, digest FROM burn_batches ORDER BY epoch DESC LIMIT 5").fetchall()]
        return {"unit": "msats", "token": None, "priced_in": "sats (whole millisatoshis)", "version": VERSION,
                "usd_per_btc": self.btc_usd,
                "paid_in_msats": self._m("paid_in_msats"), "paid_out_msats": self._m("paid_out_msats"),
                "refunded_msats": self._m("refunded_msats"), "fees_msats": self._m("fees_msats"),
                "escrow_msats": {"payments": self._sum_like("pay:"), "vesting": self._vesting_of(),
                                 "bounty_pledges": int(self.db.execute(
                                     "SELECT COALESCE(SUM(pool), 0) FROM bounties WHERE status='open'").fetchone()[0]),
                                 "bonds": self._sum_like("bond:"), "challenge_stakes": self._sum_like("challenge:"),
                                 "fix_bonds": self._sum_like("fixbond:"), "reporter_bonds": self._sum_like("reporter:"),
                                 "challenge_pledges": self.challenge_stats()["escrow_msats"],
                                 "challenge_submission_bonds": self._sum_like("csubbond:"),
                                 "prior_art_stakes": self._sum_like("priorart:")},
                "staked_msats": self._sum_like("stake:"), "validators": len(self._active()),
                "forfeited_msats": self._bal(BURN), "burn": {"account": BURN, "destination": BURN_DESTINATION,
                                                             "batches": batches},
                "learnings": dict(self.db.execute("SELECT status, COUNT(*) FROM verdicts WHERE status != 'decoy' "
                                                  "GROUP BY status")),
                "last_epoch": last,
                "rules": {"tx_fee_msats": self.tx_fee_msats, "tx_fee_usd_approx": usd_approx(self.tx_fee_msats, self.btc_usd),
                          "fee_repeg_epochs": self.p.fee_repeg_epochs, "usage_split": USAGE_SPLIT,
                          "bounty_split": BOUNTY_SPLIT, "licence_split": {"producer": 0.85, "checker": 0.10,
                                                                          "validators": 0.05},
                          "quorum": self.p.quorum, "vest_epochs": self.p.vest_epochs, "keep_epochs": self.p.keep_epochs,
                          "overclaim": self.p.overclaim, "learning_bond_msats": self.p.learning_bond_msats,
                          "validator_min_stake_msats": self.p.validator_min_stake_msats,
                          "challenge_stake_msats": self.p.challenge_stake_msats,
                          "decoy": {"z": self.p.decoy_z, "strikes": self.p.decoy_strikes,
                                    "window_epochs": self.p.strike_window, "slash": self.p.fake_slash},
                          "unbacked_bounty_epochs": self.p.unbacked_epochs,
                          "fix_bond_msats": self.p.fix_bond_msats, "reporter_bond_msats": self.p.reporter_bond_msats,
                          "invariant": "every payout is a split of a real payment: per payment, payouts <= paid - fee"}}

    def stats(self):
        s = super().stats()
        s["economy"] = self.economy_stats()
        s["challenges"] = self.challenge_stats()
        r = self.db.execute("SELECT epoch, root, total FROM sats_roots ORDER BY epoch DESC LIMIT 1").fetchone()
        s["last_root"] = {"epoch": r[0], "root": r[1], "total_msats": int(r[2])} if r else None
        return s

    def describe(self):
        d = super().describe()
        d["settlement"] = {"unit": "msats", "token": None, "network": "testnet" if self.test_credits else "external",
                           "priced_in": "sats", "faucet": "POST /v0/faucet", "test_msats_per_wallet": self.test_credits,
                           "split": "each paid use: traces 60 / trainer 25 / checkers 10 / validators 5"}
        d["payments"] = {"rail": "bitcoin over Lightning", "protocol": "L402",
                         "how": "a payment the account can't cover answers 402 Payment Required with "
                                "WWW-Authenticate: L402 macaroon=..., invoice=...; pay the invoice, retry with "
                                "Authorization: L402 <macaroon>:<preimage>",
                         "testnet": "the invoice is a placeholder; POST /v0/faucet gives 30,000 test sats once"}
        d["fee_per_transaction_msats"] = self.tx_fee_msats
        d.pop("fee_per_transaction_nanos", None)
        d["validation"] = {"quorum": self.p.quorum, "commit": "POST /v0/learnings/{id}/commits",
                           "reveal": "POST /v0/learnings/{id}/reveals", "challenge": "POST /v0/learnings/{id}/challenges"}
        d["registry"].update(reporters="POST /v0/reporters {address}: a 1,000-sat bond makes your reports count",
                             fix_bond_msats=self.p.fix_bond_msats, reporter_bond_msats=self.p.reporter_bond_msats)
        d["challenges"] = {"list": "GET /v0/challenges", "post": "POST /v0/challenges (a challenge/0.1 file, or "
                           "{format: yukon, benchmark})", "pledge": "POST /v0/challenges/{id}/pledges",
                           "submit": "POST /v0/challenges/{id}/submissions", "leaderboard": "GET /v0/challenges/{id}/leaderboard",
                           "export": "GET /v0/challenges/{id}/export?format=yukon",
                           "pays": "per verified improvement, along the challenge's curve: solver 70 / traces 20 / "
                                   "checkers 5 / validators 5; unreleased pledges are refunded at expiry",
                           "submission_bond_msats": self.p.challenge_sub_bond_msats}
        return d

    # --- v0.7: reporters, fix bonds and fix rounds, in sats ------------------------------------------------------------
    def _verified_set(self):
        """A verified reporter holds a reporter bond: counting it costs a sybil 1,000 sats an address, destroyed the
        first time validators find a case it reported does not reproduce."""
        return {a for (a,) in self.db.execute("SELECT address FROM reporters WHERE bond >= ? AND leaving IS NULL",
                                              (self.p.reporter_bond_msats,)).fetchall()}

    def _reporter_failures(self, address):
        return [x for (x,) in self.db.execute(
            "SELECT DISTINCT o.failure FROM occurrences o LEFT JOIN traces t ON t.id = o.canonical "
            "WHERE COALESCE(t.producer, o.reporter) = ?", (address,)).fetchall()]

    def post_reporter_bond(self, address):
        """Hold 1,000 sats in escrow so your reports count as a verified reporter's. Refundable: withdraw, and it comes
        back vest_epochs later. Destroyed if validators find a case you reported does not reproduce."""
        need_address(address, "address")
        bond = self.p.reporter_bond_msats
        with self.lock:
            r = self.db.execute("SELECT bond, leaving FROM reporters WHERE address=?", (address,)).fetchone()
            if r and r[0] >= bond and r[1] is None:
                return self.reporter(address)
            top = bond - (r[0] if r else 0)
            if top > 0:
                self._need_funds(address, top)
            self._tx_fee(address)
            if top > 0:
                self._move(address, f"reporter:{address}", top, "reporter bond")
            self.db.execute("INSERT OR REPLACE INTO reporters VALUES (?,?,?,NULL)",
                            (address, max(bond, r[0] if r else 0), self.epoch))
            for x in self._reporter_failures(address):
                self._index_failure(x)
            self._event(f"a reporter bonded {fmt_sats(bond)}: its reports now count")
            self.db.commit()
        return self.reporter(address)

    def withdraw_reporter(self, address):
        need_address(address, "address")
        with self.lock:
            r = self.db.execute("SELECT bond, leaving FROM reporters WHERE address=?", (address,)).fetchone()
            if not r or r[1] is not None or r[0] <= 0:
                raise ValueError("no reporter bond to withdraw")
            self._tx_fee(address)
            self.db.execute("UPDATE reporters SET leaving=? WHERE address=?", (self.epoch, address))
            for x in self._reporter_failures(address):
                self._index_failure(x)
            self.db.commit()
        return dict(self.reporter(address), returns_epoch=self.epoch + self.p.vest_epochs)

    def reporter(self, address):
        r = self.db.execute("SELECT bond, joined, leaving FROM reporters WHERE address=?", (address,)).fetchone()
        cases = self.db.execute("SELECT COUNT(*), COALESCE(SUM(rejected), 0) FROM occurrences WHERE reporter=?",
                                (address,)).fetchone()
        return {"address": address, "bond_msats": self._bal(f"reporter:{address}"), "joined": r[1] if r else None,
                "leaving": r[2] if r else None, "verified": address in self._verified_set(), "cases": cases[0],
                "refuted": cases[1], "bond_needed_msats": self.p.reporter_bond_msats}

    def _reporter_fabricated(self, address, tid):
        bal = self._bal(f"reporter:{address}")
        if bal > 0:
            self._forfeit(f"reporter:{address}", bal, "reporter bond", address)
            self._event(f"a reporter's bond ({fmt_sats(bal)}) was destroyed: a case it reported did not reproduce")
        self.db.execute("UPDATE reporters SET bond=0 WHERE address=?", (address,))
        for x in self._reporter_failures(address):
            self._index_failure(x)
        return bal

    def _validator_list(self):
        return [a for a, _ in self._active()] or list(self.validators)

    def _repro_quorum(self):
        return max(1, self.p.quorum // 2 + 1)

    def _fix_quorum(self):
        return max(int(self.p.quorum), 0)

    def _commit_required(self):
        return self.p.quorum > 0

    def _fix_can_pay(self, claimant):
        self._need_funds(claimant, self.p.fix_bond_msats)

    def _fix_charge(self, claimant, fix_id):
        self._tx_fee(claimant)
        bond = self.p.fix_bond_msats
        if bond:
            self._move(claimant, f"fixbond:{fix_id}", bond, "fix bond")
        return bond

    def _fix_bond_settle(self, fix_id, keep):
        bal = self._bal(f"fixbond:{fix_id}")
        if bal <= 0:
            return 0
        claimant = self.db.execute("SELECT claimant FROM fixes WHERE id=?", (fix_id,)).fetchone()[0]
        if keep:
            self._move(f"fixbond:{fix_id}", claimant, bal, "fix bond returned", payout=True)
        else:
            self._forfeit(f"fixbond:{fix_id}", bal, "fix bond", fix_id)
        return bal

    def _fix_assign(self, fix_id, claimant):
        """Validators for a fix are drawn like a learning's: stake-weighted rendezvous hashing over a beacon published
        after the claim (beacon_delay), never the claimant itself."""
        if self.p.quorum <= 0 or self.beacon_delay > 0:
            return
        self._draw_fix(fix_id, claimant)

    def _draw_fix(self, fix_id, claimant):
        if self.db.execute("SELECT 1 FROM fix_assign WHERE fix=?", (fix_id,)).fetchone():
            return
        vals = self._eligible(exclude=(claimant,) if claimant else ())
        if len(vals) < self.p.quorum:
            return
        for a, _ in sorted(vals, key=lambda v: self._score("fix:" + fix_id, 0, *v))[:self.p.quorum]:
            self.db.execute("INSERT INTO fix_assign VALUES (?,?,?)", (fix_id, a, self.epoch))

    def _registry_assign(self):
        if self.p.quorum <= 0:
            return
        for fix_id, claimant in self.db.execute("SELECT id, claimant FROM fixes WHERE status='pending'").fetchall():
            self._draw_fix(fix_id, claimant if claimant != "operator" else None)

    def _registry_settle(self):
        """Fix rounds that sat a whole epoch settle with the majority that revealed; leaving reporters get their bonds
        back after vest_epochs (so a reporter can't dodge a re-check by leaving); payable failure bounties pay."""
        e = self.epoch
        for fix_id in [x for (x,) in self.db.execute(
                "SELECT f.id FROM fixes f WHERE f.status='pending' AND EXISTS (SELECT 1 FROM fix_assign a WHERE a.fix=f.id "
                "AND a.epoch < ?)", (e,)).fetchall()]:
            n = self.db.execute("SELECT COUNT(*) FROM fix_reveals WHERE fix=?", (fix_id,)).fetchone()[0]
            if n >= self.p.quorum // 2 + 1:
                self._finalize_fix(fix_id)
        for address, leaving in self.db.execute("SELECT address, leaving FROM reporters WHERE leaving IS NOT NULL AND "
                                                "leaving + ? <= ?", (self.p.vest_epochs, e)).fetchall():
            bal = self._bal(f"reporter:{address}")
            if bal > 0:
                self._move(f"reporter:{address}", address, bal, "reporter bond returned", payout=True)
            self.db.execute("DELETE FROM reporters WHERE address=?", (address,))
        Registry._registry_settle(self)

    def find_learnings(self, path="", model="", kind="", min_gain=0.0, limit=20, include_pending=False):
        """Accepted learnings only (unless include_pending), ranked by the federation's median gain."""
        out = super().find_learnings(path, model, kind, -1.0, 10_000)
        keep = []
        for L in out["learnings"]:
            v = self.verdict(L["id"])
            if v["status"] == "decoy" or (v["status"] != "accepted" and not include_pending):
                continue
            gain = v["median_gain"] if v["median_gain"] is not None else L["gain"]
            if gain < float(min_gain):
                continue
            keep.append(dict(L, status=v["status"], gain=round(gain, 4), claimed_gain=L["gain"],
                             validators=sum(1 for r in v["reveals"] if r["agreed"] and r["round"] == 0)))
        keep.sort(key=lambda x: (-x["gain"], x["id"]))
        return {"learnings": keep[:int(limit)], "count": len(keep)}

    def get_learning(self, lid):
        return dict(super().get_learning(lid), verdict=self.verdict(lid))

    def audit(self):
        """Every msat accounted for. The ledger is double-entry, so it sums to zero; no escrow is negative; the burn
        account only grows; each validator's stake is what its stake account holds; and, per payment, what was paid
        out plus what it still holds is exactly what its payer paid in less the fee, never more."""
        total = int(self.db.execute("SELECT COALESCE(SUM(msats), 0) FROM sledger").fetchone()[0])
        bal = dict(self.db.execute("SELECT account, SUM(msats) FROM sledger WHERE account NOT LIKE '0x%' "
                                   "GROUP BY account").fetchall())
        negative = sorted(a for a, v in bal.items() if v < 0 and a != self.fee_to)
        stakes_ok = all(self._bal(f"stake:{a}") == s for a, s in self._stakes())
        bad = []
        live = dict(self.db.execute("SELECT payment, SUM(msats) FROM vesting WHERE status='vesting' GROUP BY payment"))
        for pid, gross, fee, out in self.db.execute("SELECT id, gross, fee, out FROM payments").fetchall():
            held = int(bal.get(f"pay:{pid}", 0))
            if out > gross - fee or out + held != gross - fee or held < 0 or live.get(pid, 0) > held:
                bad.append(pid)
        return {"ledger_sum": total, "negative_escrows": negative, "burned": self._bal(BURN), "stakes_ok": stakes_ok,
                "payments_checked": self.db.execute("SELECT COUNT(*) FROM payments").fetchone()[0],
                "payments_over": bad,
                "balanced": total == 0 and not negative and not bad and stakes_ok and self._bal(BURN) >= 0}
