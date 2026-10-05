"""The coin economy (testnet v0.5): everything priced in bitcoin. Two units and a burn-and-mint equilibrium, so every
robot and model can pay for what it uses, at any scale, for as long as the network runs.

    python node/exchange.py --economy coin ...        # or TRACEX_ECONOMY=coin

Two units
  * TXC is the asset: rewards, stakes, bonds and bounty pools are TXC, priced in sats by an open constant-product pool
    that pairs it with bitcoin. It has 18 decimals (UNIT = 10**18). A credit is a millisatoshi, so a credit's worth of
    TXC at 10**12 sats a TXC is 10**-15 TXC: 15 decimals is the least that pays every credit there, and 18 leaves a
    thousand times headroom for rounding and higher prices.
  * Credits are what everything is paid in: 1 credit = 1 millisatoshi (msat), whole numbers only, the same at every
    TXC price, so a machine always knows what a call costs. They are made only by burning TXC: sats buy TXC from the
    pool and burn it in the same step (one credit per msat), or a holder burns TXC at the lower of the pool's spot and
    reference prices. Credits can't be moved between accounts and never turn back into TXC or sats. Metered usage,
    licences, trace fees and the standard transaction fee are all paid in credits, and spent credits are gone.
  * Sats arrive over Lightning: L402 (HTTP 402 Payment Required with a Lightning invoice and a macaroon; pay, then
    retry with the macaroon and the payment's preimage) is the mainnet path. On the testnet each wallet takes 30,000
    test sats once from the faucet, and a 402's invoice is a placeholder.

Burn and mint (unchanged from v0.4, now valued in sats)
  * Each epoch mints at most a fixed emission: 50,000 TXC, halving every 180 epochs five times, then 1,562.5 TXC an
    epoch for good. 90% goes to the people whose work was paid for that epoch (traces 60, trainer 25, checkers 10,
    validators 5, down the family tree; a solved bounty's coin holders first take 20% of its learning's usage), 10% to
    the node operator for the transactions it served, each by its share of the credits burned on its work.
  * The anti-farming cap: nobody is ever minted more TXC than the credits burned on their own work were worth, valued
    at the epoch's time-weighted pool price in sats (never the spot), and never more than the TXC those credits burned
    at the pool's price before its spread. Emission nobody earned is never minted and never rolls over. Nothing is
    minted for a verdict, a submission or a stake, and the parents' share vests, so an audit can still claw it back.
  * So burning pushes the price up to where an epoch's burns match its emission (P* = credits burned / emission, in
    msats per TXC) and then mints exactly what is burned: TXC follows the network's usage rate and the halvings, not
    the sum of all that was ever paid. The protocol never pushes the price down (that would take minting TXC nobody
    earned, which is what farming wants); only holders selling does. examples/scaling shows both.

Why farming loses (each rule is a test in tests/test_coin.py and an attack in examples/farming/attacks.py)
  1. Only payments pay, and whoever pays judges: users pay for a learning after measuring it on their own data, a
     bounty's poster measures solutions on its own hidden eval, a licence buyer's own learnings decide which traces get
     its money. Validators' verdicts decide who may earn; they never move money. A federation captured by a stake
     majority can block honest work, but it has nothing to print and nothing to take.
  2. Paying yourself returns at most what you burned: the cap above, per recipient. Wash usage, self-funded bounties
     and buying your own traces lose the pool's spread, the fee, and every share that isn't yours.
  3. Validators are drawn at random from a beacon published after a learning is submitted (stake-weighted rendezvous
     hashing), commit before anyone reveals, and measure on data they each hold privately; the median decides. A claim
     more than twice what they measured, beyond the noise, forfeits the bond.
  4. Forfeits burn: a lost bond, stake or challenge stake goes to nobody.
  5. Validators earn only their share of what the learnings they vouched for go on to earn. Stake is slashed for not
     showing up, for vouching for a gain a fresh round refuted, or for scoring a decoy without measuring it; a
     validator slashed under the minimum loses its seat at once.
  6. Copies and padding earn nothing: copies pay the original's producer; junk no buyer uses gets no licence money;
     the split is the protocol's, equal per distinct parent; a wrapper around someone's traces takes nothing.
  7. Every transaction pays one standard fee in credits, 58 msats (exchange.TX_FEE_MSATS; about $0.00005 at $85,962 a
     bitcoin), about 125 times the electricity of the dearest transaction (examples/fees/measure.py). It is burned like
     any payment and is the operator's claim on its 10% of the emission. It is fixed in sats, so its dollar value
     floats with bitcoin; Params.fee_repeg_epochs re-pegs it to a dollar target every N epochs (off by default).
     Bonds (5,000 sats), the validator minimum stake (10,000 sats), challenge stakes (2,000 sats) and the bounty curve
     (first coin 10 sats, each one sold adds 0.1 sat) are priced in sats at the reference price, so they keep their
     cost in bitcoin whatever TXC does.

Units: sats and credits in msats (1e-3 sat; integer msats everywhere); TXC in base units (1e-18 TXC); prices in msats
per TXC, kept internally times PQ so they stay exact from a billionth of a sat to beyond a trillion sats a TXC. Every
amount key in the API says its unit: `_msats` or `_sats` (integers), `_units` (TXC base units, as decimal strings:
SQLite integers stop at 2**63). Dollars appear only as labelled approximations (`_usd_approx`), at the operator's
reference bitcoin price, and are never used in any amount. Per-transaction detail is kept through the challenge window
(`keep_epochs`); balances, roots, traces and learnings are kept for good.
"""
import base64
import decimal
import hashlib
import json
import math
import re
import statistics
import time
from dataclasses import dataclass

from exchange import (ADDRESS, BOUNTY_SPLIT, MAX_DEPTH, TX_FEE_MSATS, Exchange, PaymentRequired, coin, need_address,
                      msats_in, split_trace_sale, split_usage, leaf, build_tree, proof, canonical, object_id,
                      clear_shared, Bid)
from traceex.trace import Learning

VERSION = "0.5"
DECIMALS = 18
UNIT = 10 ** DECIMALS          # base units in one TXC
PQ = 10 ** 12                  # prices: msats per TXC, times PQ
BPS = 10_000
MSATS_PER_BTC = 100_000_000_000
BTC_USD = 85_962               # Coinbase spot, 2026-10-05: only for the approximate dollar figures shown beside sats


def units_for(credits, price_q):
    """The TXC (base units) that `credits` msats are worth at price_q."""
    return int(credits) * UNIT * PQ // max(int(price_q), 1)


def credits_for(units, price_q):
    """The credits (msats) that `units` of TXC are worth at price_q, rounded down."""
    return int(units) * int(price_q) // (UNIT * PQ)


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


def to_units(amount):
    """An amount of TXC ("12.5", 12.5, 12) in base units, exactly."""
    try:
        return int(decimal.Decimal(str(amount)) * UNIT)
    except (decimal.InvalidOperation, ValueError):
        raise ValueError(f"not an amount of TXC: {amount!r}")


def fmt_txc(units, places=4):
    """Base units for people: '1,234.5'."""
    d = (decimal.Decimal(int(units)) / UNIT).quantize(decimal.Decimal(1).scaleb(-places), decimal.ROUND_DOWN)
    return f"{d:,.{places}f}".rstrip("0").rstrip(".") or "0"


def fmt_price(price_q):
    """A price for people, in sats a TXC: '10 sats', '22,222 sats', '0.000123 sats'."""
    sats = decimal.Decimal(int(price_q)) / PQ / 1000
    if sats >= 100:
        return f"{sats:,.0f} sats"
    if sats >= decimal.Decimal("0.01"):
        return f"{sats:,.3f}".rstrip("0").rstrip(".") + " sats"
    return f"{sats:.3g} sats"


def _frac(amount, f):
    """amount x f for big integers, exactly enough (f to a millionth)."""
    return int(amount) * round(f * 1_000_000) // 1_000_000


def _pro_rata(amount, weights):
    """Split an integer amount by weights (ints exactly; float coin holdings to a billionth of a coin); sums exactly to
    `amount`, dust to the largest holder."""
    w = {k: v for k, v in weights.items() if v and v > 0}
    if not all(isinstance(v, int) for v in w.values()):
        w = {k: v for k, v in ((k, int(round(v * 1e9))) for k, v in w.items()) if v > 0}
    total = sum(w.values())
    if amount <= 0 or total <= 0:
        return {}
    out = {k: amount * v // total for k, v in w.items()}
    dust = amount - sum(out.values())
    if dust:
        out[max(w, key=lambda k: (w[k], k))] += dust
    return {k: m for k, m in out.items() if m}


@dataclass
class Params:
    symbol: str = "TXC"
    genesis_coins: int = 1_000_000 * UNIT      # protocol-owned liquidity at genesis...
    genesis_msats: int = 10_000_000 * 1000     # ...beside 10,000,000 sats (0.1 BTC, about $8,600): TXC opens at 10 sats
    amm_fee_bps: int = 30                      # the pool's spread: 0.3% of a swap stays in the pool, nobody collects it
    emission: int = 50_000 * UNIT              # the most an epoch can mint...
    halving_epochs: int = 180                  # ...halving every 180 epochs...
    max_halvings: int = 5                      # ...five times, then 1,562.5 TXC an epoch for good (see examples/scaling)
    operator_share: float = 0.10               # of each epoch's emission: node operators, by the transactions they served
    vest_epochs: int = 4                       # the parents' share vests, so an audit can claw it back
    keep_epochs: int = 6                       # per-transaction detail is kept this long: the challenge window, plus 2
    quorum: int = 3
    min_gain: float = 0.01
    accept_z: float = 2.0                      # accept only if median gain - 2 standard errors >= min_gain
    overclaim: float = 2.0                     # claiming more than twice the measured gain, beyond the noise: bond forfeit
    z: float = 2.5                             # within 2.5 of its own standard errors of the median: the validator agreed
    min_tol: float = 0.02
    decoy_z: float = 4.0                       # further than this from a decoy's sealed truth: it wasn't measured
    inconclusive_burn: float = 0.10            # a real-looking but unproven gain: bond back minus 10%, no rewards
    audit_min: int = 10                        # every reveal audits at least this many parents (or all of them)
    audit_max_bad: float = 0.10
    pad_burn: float = 0.50                     # parents found padded: their share is withheld and half the bond burns
    learning_bond_msats: int = 5_000_000       # 5,000 sats of TXC at the reference price (about $4.30; v0.4: $5)
    validator_min_stake_msats: int = 10_000_000    # 10,000 sats of TXC (about $8.60; v0.4: $10)
    challenge_stake_msats: int = 2_000_000     # 2,000 sats of TXC (about $1.72; v0.4: $2)
    stake_grace_epochs: int = 4                # a validator the price pushed under the minimum keeps its seat this long
    noshow_slash: float = 0.05
    fake_slash: float = 0.25                   # vouched for a gain a fresh round refuted, or scored a decoy unmeasured
    near_dup: float = 0.3                      # the same distinctive fix, inputs sharing 30% of their words: one trace
    bounty_base: int = 10_000                  # bounty coins are priced in sats: the first costs 10 sats (10,000 msats)...
    bounty_slope: int = 100                    # ...and each one sold adds 0.1 sat (100 msats)
    btc_usd: int = BTC_USD                     # dollars a bitcoin: only for approximate dollar figures (and the re-peg)
    fee_repeg_epochs: int = 0                  # re-peg the fee to fee_target_usd_nanos every N epochs (0: never)
    fee_target_usd_nanos: int = 50_000         # the re-peg's target: $0.00005, at the operator's btc_usd


COIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS coin_ledger (epoch INT, account TEXT, units TEXT, memo TEXT, payout INT DEFAULT 0);
CREATE INDEX IF NOT EXISTS coin_ledger_account ON coin_ledger(account);
CREATE INDEX IF NOT EXISTS coin_ledger_epoch ON coin_ledger(epoch);
CREATE TABLE IF NOT EXISTS vesting   (id INTEGER PRIMARY KEY, account TEXT, units TEXT, released TEXT DEFAULT '0',
                                      start INT, epochs INT, source TEXT, learning TEXT, status TEXT DEFAULT 'vesting',
                                      role TEXT);
CREATE TABLE IF NOT EXISTS validators(address TEXT PRIMARY KEY, stake TEXT, joined INT, slashed TEXT DEFAULT '0',
                                      below_since INT);
CREATE TABLE IF NOT EXISTS verdicts  (learning TEXT PRIMARY KEY, status TEXT, round INT, gain REAL, bond TEXT DEFAULT '0',
                                      trainer TEXT, registered INT, accepted INT, audit_bad REAL, challenger TEXT,
                                      challenge_stake TEXT DEFAULT '0', note TEXT);
CREATE TABLE IF NOT EXISTS assignments(learning TEXT, round INT, validator TEXT, epoch INT,
                                       PRIMARY KEY (learning, round, validator));
CREATE TABLE IF NOT EXISTS commits   (learning TEXT, round INT, validator TEXT, digest TEXT, epoch INT,
                                      PRIMARY KEY (learning, round, validator));
CREATE TABLE IF NOT EXISTS reveals   (learning TEXT, round INT, validator TEXT, body TEXT, gain REAL, epoch INT,
                                      agreed INT, PRIMARY KEY (learning, round, validator));
CREATE TABLE IF NOT EXISTS dups      (trace TEXT PRIMARY KEY, canonical TEXT, key TEXT, fix TEXT);
CREATE INDEX IF NOT EXISTS dups_fix ON dups(fix);
CREATE TABLE IF NOT EXISTS burns     (epoch INT, kind TEXT, ref TEXT, units TEXT);
CREATE TABLE IF NOT EXISTS licence_escrow(id INTEGER PRIMARY KEY, lot TEXT, buyer TEXT, credits INT, basis TEXT,
                                          epoch INT, traces TEXT, paid INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS decoys    (learning TEXT PRIMARY KEY, digest TEXT, funder TEXT, gain REAL,
                                      unsealed INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS artifacts (hash TEXT PRIMARY KEY, learning TEXT);
CREATE TABLE IF NOT EXISTS credits   (account TEXT PRIMARY KEY, credits INT, basis TEXT);
CREATE TABLE IF NOT EXISTS paid      (epoch INT, learning TEXT, credits INT, basis TEXT, PRIMARY KEY (epoch, learning));
CREATE TABLE IF NOT EXISTS work      (epoch INT, account TEXT, role TEXT, ref TEXT, credits INT, basis TEXT);
CREATE INDEX IF NOT EXISTS work_epoch ON work(epoch);
CREATE TABLE IF NOT EXISTS coin_bases(bounty INT, holder TEXT, units TEXT, msats INT, PRIMARY KEY (bounty, holder));
CREATE TABLE IF NOT EXISTS tx_fees   (account TEXT PRIMARY KEY, msats INT);
CREATE TABLE IF NOT EXISTS coin_roots(epoch INT PRIMARY KEY, root TEXT, total TEXT, claims TEXT);
CREATE TABLE IF NOT EXISTS mints     (epoch INT PRIMARY KEY, body TEXT);
"""
SLOT = re.compile(r"\{([A-Z]+)_\d+\}")
NO_EARNINGS = ("pending", "rejected", "inconclusive", "clawed back", "decoy")


class _BigSum:
    """SUM for TXC amounts stored as TEXT: exact, at any size."""

    def __init__(self):
        self.total = 0

    def step(self, v):
        if v is not None:
            self.total += int(v)

    def finalize(self):
        return str(self.total)


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


class CoinExchange(Exchange):
    economy = "coin"
    money = "msats"

    def _fmt(self, msats):
        return f"{fmt_sats(msats)} (about {usd_approx(msats, self.btc_usd)})"

    def __init__(self, path=":memory:", *, params=None, beacon_delay=1, clock=None, reserve_msats=None,
                 tx_fee_msats=None, btc_usd=None, **kw):
        """beacon_delay=1: validators for a learning are drawn from the beacon published at the settlement after it was
        submitted, so nobody can grind a learning's content for friendly validators. 0 assigns at once (tests).
        clock: seconds, for the time-weighted price (time.time by default; simulations pass their own).
        test_credits: test msats each new wallet takes once (30,000,000 = 30,000 sats). reserve_msats: a lot's reserve
        price per licence. tx_fee_msats: the standard fee (TX_FEE_MSATS on a testnet, unless set). btc_usd: dollars a
        bitcoin for the approximate dollar figures (and the fee re-peg); never used in any amount."""
        if "reserve_micros" in kw or "tx_fee_nanos" in kw:
            raise ValueError("a coin node prices everything in sats: reserve_msats and tx_fee_msats, not micros or nanos")
        if reserve_msats is not None:
            kw["reserve_micros"] = int(reserve_msats)          # the base node's reserve, in this node's money
        super().__init__(path, tx_fee_nanos=0, **kw)
        self.p, self.beacon_delay = params or Params(), int(beacon_delay)
        self.clock = clock or time.time
        self.db.conn.create_aggregate("BIGSUM", 1, _BigSum)
        if self._meta("pool_coin") is not None and self._meta("coin_version") != VERSION:
            was = self._meta("coin_version") or "0.3"
            self.db.close()
            raise ValueError(f"this database holds a testnet v{was} coin economy (priced in dollars); v{VERSION} prices "
                             "everything in sats and opens a fresh testnet: start it on an empty database")
        self.db.executescript(COIN_SCHEMA)
        self.tx_fee_msats = int(TX_FEE_MSATS if tx_fee_msats is None and self.test_credits else tx_fee_msats or 0)
        if self._meta("tx_fee_msats") is not None:              # a re-pegged fee outlives a restart
            self.tx_fee_msats = int(self._meta("tx_fee_msats"))
        self.btc_usd = int(self._meta("btc_usd") or btc_usd or self.p.btc_usd)
        if self._meta("pool_coin") is None:                     # genesis
            q0 = self.p.genesis_msats * UNIT * PQ // self.p.genesis_coins
            now = self._now()
            for k, v in (("pool_msats", self.p.genesis_msats), ("pool_coin", self.p.genesis_coins),
                         ("minted", self.p.genesis_coins), ("burned", 0), ("coin_version", VERSION),
                         ("twap_t", now), ("twap_cum", 0), ("twap_last", q0), ("mark_t", now), ("mark_cum", 0),
                         ("mark_q", q0), ("ref_q", q0), ("burned_mark", 0), ("credits_made", 0), ("credits_spent", 0)):
                self._set_meta(k, v)
            self._set_meta("beacon", hashlib.sha256(b"traceX genesis").hexdigest())
        self.db.commit()

    # --- accounting -------------------------------------------------------------------------------------------------
    def _m(self, k):
        return int(self._meta(k) or 0)

    def _add(self, k, v):
        self._set_meta(k, self._m(k) + int(v))

    def _sum(self, sql, *args):
        return int(self.db.execute(sql, args).fetchone()[0] or 0)

    def _coins(self, account):
        return self._sum("SELECT BIGSUM(units) FROM coin_ledger WHERE account=?", account)

    def _coin(self, account, units, memo, payout=False):
        if units:
            self.db.execute("INSERT INTO coin_ledger VALUES (?,?,?,?,?)",
                            (self.epoch, account, str(int(units)), memo, int(payout)))

    def _move(self, src, dst, units, memo, payout=False):
        self._coin(src, -units, memo)
        self._coin(dst, units, memo, payout)

    def _burn(self, units, kind="forfeit", ref=""):
        if units:
            self._add("burned", units)
            self.db.execute("INSERT INTO burns VALUES (?,?,?,?)", (self.epoch, kind, ref, str(int(units))))

    def _need_coins(self, account, units):
        have = self._coins(account)
        if have < units:
            raise ValueError(f"not enough {self.p.symbol}: {account[:10]}… has {fmt_txc(have)}, this needs "
                             f"{fmt_txc(units)} (POST /v0/swap buys {self.p.symbol} with sats)")

    def supply(self):
        return self._m("minted") - self._m("burned")

    # --- the pool, and its time-weighted price ------------------------------------------------------------------------
    def _price_q(self):
        return self._m("pool_msats") * UNIT * PQ // max(self._m("pool_coin"), 1)

    def price(self):
        """msats per whole TXC at the pool's spot (for display; amounts use the exact internal price)."""
        return self._price_q() // PQ

    def _now(self):
        return int(self.clock() * 1000)

    def _observe(self):
        """Before the pool's reserves change: credit the price that held since the last change with the time it held."""
        now, t = self._now(), self._m("twap_t")
        if now > t:
            self._add("twap_cum", self._m("twap_last") * (now - t))
            self._set_meta("twap_t", now)

    def _observed(self):
        self._set_meta("twap_last", self._price_q())

    def twap_q(self):
        """The time-weighted average pool price since this epoch began (or the price it began at, if no time has
        passed). A price someone pushes the pool to counts only for as long as it lasts."""
        now, t0 = self._now(), self._m("mark_t")
        cum = self._m("twap_cum") + self._m("twap_last") * max(now - self._m("twap_t"), 0)
        return (cum - self._m("mark_cum")) // (now - t0) if now > t0 else self._m("mark_q")

    def _mark(self):
        self._observe()
        for k, v in (("mark_t", self._now()), ("mark_cum", self._m("twap_cum")), ("mark_q", self._price_q())):
            self._set_meta(k, v)

    def ref_q(self):
        """The reference price for everything priced in sats (bonds, stakes, the bounty curve, burning held TXC for
        credits): the time-weighted price of the last settled epoch."""
        return self._m("ref_q")

    def _amm_buy(self, msats):
        """Sats into the pool, TXC out. Returns (TXC out, TXC the sats would have bought before the spread)."""
        self._observe()
        x, y = self._m("pool_msats"), self._m("pool_coin")
        dx = msats * (BPS - self.p.amm_fee_bps) // BPS
        dy = y * dx // (x + dx)
        gross = y * msats // (x + msats)
        self._set_meta("pool_msats", x + msats)
        self._set_meta("pool_coin", y - dy)
        self._observed()
        return dy, gross

    def _amm_sell(self, units):
        self._observe()
        x, y = self._m("pool_msats"), self._m("pool_coin")
        dy = units * (BPS - self.p.amm_fee_bps) // BPS
        dx = x * dy // (y + dy)
        self._set_meta("pool_msats", x - dx)
        self._set_meta("pool_coin", y + units)
        self._observed()
        return dx

    def quote(self, side, amount):
        """What a swap would get now: `amount` is msats for a buy, TXC base units for a sell."""
        x, y = self._m("pool_msats"), self._m("pool_coin")
        if side == "buy":
            dx = int(amount) * (BPS - self.p.amm_fee_bps) // BPS
            return {"side": "buy", "pay_msats": int(amount), "get_units": y * dx // (x + dx)}
        dy = int(amount) * (BPS - self.p.amm_fee_bps) // BPS
        return {"side": "sell", "pay_units": int(amount), "get_msats": x * dy // (y + dy)}

    def swap(self, account, side, amount):
        """Sats for TXC or back, at the pool's price; the pool keeps its 0.3% spread. `amount` is msats for a buy, TXC
        base units for a sell. Like every transaction it pays the standard fee (TX_FEE_MSATS), in credits."""
        need_address(account, "account")
        amount = int(amount)
        if amount <= 0 or side not in ("buy", "sell"):
            raise ValueError("side is 'buy' (msats or sats) or 'sell' (units of TXC), the amount more than 0")
        with self.lock:
            if side == "buy":
                self._need_sats(account, amount)
                self._tx_fee(account)
                self._credit(account, -amount, f"swap: buy {self.p.symbol}")
                out, _ = self._amm_buy(amount)
                self._coin(account, out, "swap: bought")
                got = {"bought_units": out}
                self._event(f"{fmt_sats(amount)} bought {fmt_txc(out)} {self.p.symbol}; price "
                            f"{fmt_price(self._price_q())}")
            else:
                self._need_coins(account, amount)
                self._tx_fee(account)
                self._coin(account, -amount, "swap: sold")
                got_msats = self._amm_sell(amount)
                self._credit(account, got_msats, f"swap: sold {self.p.symbol}")
                got = {"paid_msats": got_msats}
                self._event(f"{fmt_txc(amount)} {self.p.symbol} sold for {fmt_sats(got_msats)}; price "
                            f"{fmt_price(self._price_q())}")
            self.db.commit()
        return dict(got, price_msats=self.price(), symbol=self.p.symbol)

    # --- credits: the payment unit -------------------------------------------------------------------------------------
    def _credit_row(self, account):
        r = self.db.execute("SELECT credits, basis FROM credits WHERE account=?", (account,)).fetchone()
        return (r[0], int(r[1])) if r else (0, 0)

    def _set_credits(self, account, credits, basis):
        self.db.execute("INSERT OR REPLACE INTO credits VALUES (?,?,?)", (account, int(credits), str(int(basis))))

    def _make_credits(self, account, credits, basis):
        c, b = self._credit_row(account)
        self._set_credits(account, c + credits, b + basis)
        self._add("credits_made", credits)

    def _onramp(self, account, msats, memo):
        """Sats in, credits out: the sats buy TXC from the pool and it is burned in the same step, one credit per msat.
        The credits remember the TXC their sats bought before the pool's spread (their basis): nobody is ever minted
        more than that for them."""
        self._credit(account, -msats, memo)
        burned, gross = self._amm_buy(msats)
        self._burn(burned, "credits", account)
        self._make_credits(account, msats, gross)
        return burned

    def buy_credits(self, account, msats=0, units=0):
        """Make credits (1 credit = 1 msat). With `msats`: sats (test sats here, Lightning on mainnet) buy TXC from the
        pool and it is burned, one credit per msat. With `units`: burn TXC you hold, at the lower of the pool's spot
        and reference prices. Credits pay for everything on the network; they can't be moved, and never turn back into
        TXC or sats."""
        need_address(account, "account")
        msats, units = int(msats or 0), int(units or 0)
        if (msats > 0) == (units > 0):
            raise ValueError("pay with sats (msats or sats) or burn TXC (units): one of them, more than 0")
        with self.lock:
            if msats:
                self._need_sats(account, msats, spends_cover=False)
                self._tx_fee(account)
                burned, made = self._onramp(account, msats, "credits"), msats
            else:
                self._need_coins(account, units)
                made = credits_for(units, min(self._price_q(), self.ref_q()))
                if made <= 0:
                    raise ValueError("too little TXC to make one credit")
                self._tx_fee(account)
                self._coin(account, -units, "burned for credits")
                self._burn(units, "credits", account)
                self._make_credits(account, made, units)
                burned = units
            self._event(f"{fmt_sats(made)} of credits made: {fmt_txc(burned)} {self.p.symbol} burned")
            self.db.commit()
        return {"account": account, "credits_msats": made, "burned_units": burned,
                "balance_msats": self._credit_row(account)[0]}

    def _pay_credits(self, account, credits, memo):
        """Spend credits; sats top them up through the on-ramp. Returns (credits paid, their TXC basis)."""
        c, b = self._credit_row(account)
        if c < credits:
            short = credits - c
            if self.test_credits:
                short = min(short, max(self._funds(account), 0))
            if short > 0:
                self._onramp(account, short, f"credits for {memo}")
                c, b = self._credit_row(account)
        pay = min(int(credits), c)
        basis = b * pay // c if c else 0
        self._set_credits(account, c - pay, b - basis)
        self._add("credits_spent", pay)
        return pay, basis

    # --- what an account can spend: credits plus the sats that top them up, less what it already owes ------------------
    def _owed_credits(self, account):
        r = self.db.execute("SELECT msats FROM tx_fees WHERE account=?", (account,)).fetchone()
        return r[0] if r else 0

    def _obligations(self, account):
        bids = self.db.execute("SELECT COALESCE(SUM(price), 0) FROM bids WHERE bidder=? AND epoch=?",
                               (account, self.epoch)).fetchone()[0]
        return self._owed_credits(account) + bids

    def _cover(self, account):
        return self._credit_row(account)[0] + self._funds(account)

    def _fee_credits(self):
        return self.tx_fee_msats

    def _short(self, account, msats):
        return self._cover(account) - self._obligations(account) - self._fee_credits() < msats

    def _need_funds(self, account, msats, pending=0):
        """A payment in credits (sats top them up): the account must cover it, the fees and bids it already owes, and
        this transaction's own fee, so nobody can spend first and leave its fees unpaid at settlement."""
        if self.test_credits and self._short(account, msats):
            room = self._cover(account) - self._obligations(account) - self._fee_credits()
            raise PaymentRequired(f"not enough test sats: {account[:10]}… can pay {fmt_sats(max(room, 0))}, this needs "
                                  f"{fmt_sats(msats)} (POST /v0/faucet opens a wallet)", account, msats - max(room, 0))

    def _need_sats(self, account, msats, spends_cover=True):
        """Sats themselves (a swap, backing a bounty, the on-ramp). Turning sats into credits keeps what an account can
        pay with; buying TXC with them doesn't, so it must leave what the account owes covered."""
        if not self.test_credits:
            return
        if self._funds(account) < msats or (spends_cover and self._short(account, msats)):
            free = max(self._funds(account), 0)
            raise PaymentRequired(f"not enough test sats: {account[:10]}… has {fmt_sats(free)} free, this needs "
                                  f"{fmt_sats(msats)} (POST /v0/faucet opens a wallet)", account, msats - free)

    def _tx_fee(self, account):
        """Charge one transaction its standard fee, in msats: accrued, billed at settlement in credits and burned. The
        account must be able to cover all it owes."""
        if not self.tx_fee_msats or not account:
            return
        owed = self._owed_credits(account) + self.tx_fee_msats
        if self.test_credits:
            bids = self._obligations(account) - self._owed_credits(account)
            if self._cover(account) - bids < owed:
                raise PaymentRequired(f"not enough test sats: {account[:10]}… can't cover the transaction fee "
                                      "(POST /v0/faucet opens a wallet)", account, owed - self._cover(account) + bids)
        self.db.execute("INSERT OR REPLACE INTO tx_fees VALUES (?,?)", (account, owed))

    def _bill_fees(self):
        """At settlement: every account pays the transaction fees it owes, in credits; they are burned, and they are the
        operator's claim on its share of the emission."""
        total = basis = 0
        for account, owed in self.db.execute("SELECT account, msats FROM tx_fees WHERE msats > 0").fetchall():
            paid, b = self._pay_credits(account, owed, "transaction fees")
            self.db.execute("UPDATE tx_fees SET msats = msats - ? WHERE account=?", (paid, account))
            total, basis = total + paid, basis + b
        if total:
            self._work(self.fee_to, "operator", "fees", total, basis)
            self._add("fee_credits", total)
        return total

    def fees(self):
        """The standard fee, and what it has collected."""
        accrued = self.db.execute("SELECT COALESCE(SUM(msats), 0) FROM tx_fees").fetchone()[0]
        return {"per_transaction_msats": self.tx_fee_msats,
                "per_transaction_usd_approx": usd_approx(self.tx_fee_msats, self.btc_usd),
                "usd_per_btc": self.btc_usd, "paid_in": "credits (1 credit = 1 msat), burned",
                "billed": "each epoch", "operator": self.fee_to, "operator_share": self.p.operator_share,
                "burned_msats": self._m("fee_credits"), "accrued_msats": accrued,
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
        nonce = hashlib.sha256(f"{self._meta('beacon')}|{account}|{msats}|{self._now()}".encode()).hexdigest()
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
        return out

    def _canonical(self, tid):
        r = self.db.execute("SELECT canonical FROM dups WHERE trace=?", (tid,)).fetchone()
        return r[0] if r else tid

    # --- bounties: a sats curve, a TXC pool ----------------------------------------------------------------------------
    def _curve(self):
        return {"base": self.p.bounty_base, "slope": self.p.bounty_slope}

    def _escrow(self, bounty_id):
        return self._coins(f"escrow:bounty:{bounty_id}")

    def _cbasis(self, bounty_id, holder):
        r = self.db.execute("SELECT units, msats FROM coin_bases WHERE bounty=? AND holder=?", (bounty_id, holder)).fetchone()
        return (int(r[0]), r[1]) if r else (0, 0)

    def _set_cbasis(self, bounty_id, holder, units, msats):
        self.db.execute("INSERT OR REPLACE INTO coin_bases VALUES (?,?,?,?)",
                        (bounty_id, holder, str(max(int(units), 0)), max(int(msats), 0)))

    def _refund_weights(self, bounty_id):
        """Refunds go back by the TXC each holder put in, never by coin count."""
        bases = {h: int(u) for h, u in self.db.execute("SELECT holder, units FROM coin_bases WHERE bounty=?",
                                                        (bounty_id,)).fetchall() if int(u) > 0}
        return bases or dict(self.db.execute("SELECT holder, coins FROM holdings WHERE bounty=?", (bounty_id,)).fetchall())

    def post_bounty(self, b):
        b = dict(b)
        seed_msats = msats_in(b, "seed") or msats_in(b, "reward")
        for k in ("seed_msats", "seed_sats", "reward_msats", "reward_sats"):
            b.pop(k, None)
        seed_units = to_units(b.pop("seed_coins", 0) or 0)
        if seed_msats and b.get("poster"):
            with self.lock:
                self._need_sats(need_address(b["poster"], "poster"), seed_msats)
        out = super().post_bounty(b)
        out["price_msats"] = int(coin.price(0, **self._curve()))
        if seed_msats or seed_units:
            out["seed"] = self.buy_coins(out["id"], b["poster"], msats=seed_msats, units=seed_units)
        return out

    def buy_coins(self, bounty_id, buyer, msats=0, units=0):
        """Back a bounty. The curve is priced in sats, so a sat buys the same coins whatever TXC is worth; the pool
        holds TXC. With `msats`, the sats buy TXC from the pool on the way in (nothing burns) and count for no more than
        that TXC is worth at the reference price, so pushing the pool up first buys no extra coins; with `units`, your
        TXC counts at the lower of the pool's spot and reference prices."""
        need_address(buyer, "buyer")
        msats, units = int(msats or 0), int(units or 0)
        if (msats > 0) == (units > 0):
            raise ValueError("back with sats (msats or sats) or with TXC (units or coins): one of them, more than 0")
        with self.lock:
            status, pool, supply = self._bounty(bounty_id)
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}; buy its coins from a holder")
            if msats:
                self._need_sats(buyer, msats)
                self._tx_fee(buyer)
                self._credit(buyer, -msats, f"bounty {bounty_id} backing")
                units, _ = self._amm_buy(msats)
                value = min(msats, credits_for(units, self.ref_q()))
            else:
                self._need_coins(buyer, units)
                value = credits_for(units, min(self._price_q(), self.ref_q()))
                if value <= 0:
                    raise ValueError("too little TXC to back a bounty with")
                self._tx_fee(buyer)
                self._coin(buyer, -units, f"bounty {bounty_id} backing")
            n = coin.coins_for(supply, value, **self._curve())
            self._coin(f"escrow:bounty:{bounty_id}", units, "pool")
            self._set_holding(bounty_id, buyer, self._holding(bounty_id, buyer) + n)
            bu, bm = self._cbasis(bounty_id, buyer)
            self._set_cbasis(bounty_id, buyer, bu + units, bm + value)
            self.db.execute("UPDATE bounties SET pool=pool+?, supply=supply+? WHERE id=?", (value, n, bounty_id))
            self._event(f"bounty #{bounty_id} backed with {fmt_sats(value)} ({fmt_txc(units)} {self.p.symbol}): "
                        f"{n:,.1f} coins at {value / n / 1000:,.2f} sats; pool {fmt_sats(pool + value)}")
            self.db.commit()
        return {"bounty": bounty_id, "coins": round(n, 6), "spent_units": units, "spent_msats": value,
                "avg_price_msats": round(value / n), "pool_msats": pool + value,
                "pool_units": self._escrow(bounty_id), "next_price_msats": round(coin.price(supply + n, **self._curve()))}

    def sell_coins(self, bounty_id, seller, coins):
        """Sell back while the bounty is open, for the TXC the coins cost and never more: a later backer's money stays
        theirs. (The curve decides how many coins a payment buys; early backers' reward is their share of the
        solution's revenue.)"""
        need_address(seller, "seller")
        with self.lock:
            status, pool, supply = self._bounty(bounty_id)
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}; its coins now earn from the solution")
            have = self._holding(bounty_id, seller)
            coins = min(float(coins), have)
            if coins <= 0:
                raise ValueError("no coins to sell")
            self._tx_fee(seller)
            bu, bm = self._cbasis(bounty_id, seller)
            part = coins / have
            cu, cm = (bu, bm) if part >= 1 else (_frac(bu, part), int(bm * part))
            value = min(cu, self._escrow(bounty_id))
            self._set_cbasis(bounty_id, seller, bu - cu, bm - cm)
            self._set_holding(bounty_id, seller, have - coins)
            self.db.execute("UPDATE bounties SET pool=MAX(pool-?, 0), supply=supply-? WHERE id=?", (cm, coins, bounty_id))
            self._move(f"escrow:bounty:{bounty_id}", seller, value, f"bounty {bounty_id} sell")
            self._event(f"{coins:,.1f} coins of bounty #{bounty_id} sold back for what they cost, "
                        f"{fmt_txc(value)} {self.p.symbol}")
            self.db.commit()
        return {"bounty": bounty_id, "sold": round(coins, 6), "paid_units": value,
                "next_price_msats": round(coin.price(supply - coins, **self._curve()))}

    def transfer_coins(self, bounty_id, sender, to, coins):
        """Move coins between holders, any time; they carry what they cost. (A live network checks the sender's
        signature; the contract does.)"""
        need_address(sender, "from")
        need_address(to, "to")
        with self.lock:
            self._bounty(bounty_id)
            have = self._holding(bounty_id, sender)
            coins = float(coins)
            if coins <= 0 or coins > have + 1e-9:
                raise ValueError(f"{sender} holds {have:.6f} coins")
            self._tx_fee(sender)
            part = min(coins / have, 1.0)
            bu, bm = self._cbasis(bounty_id, sender)
            mu, mm = (bu, bm) if part >= 1 else (_frac(bu, part), int(bm * part))
            self._set_cbasis(bounty_id, sender, bu - mu, bm - mm)
            tu, tm = self._cbasis(bounty_id, to)
            self._set_cbasis(bounty_id, to, tu + mu, tm + mm)
            self._set_holding(bounty_id, sender, have - coins)
            self._set_holding(bounty_id, to, self._holding(bounty_id, to) + coins)
            self.db.commit()
        return {"bounty": bounty_id, "from": sender, "to": to, "coins": coins}

    def holders(self, bounty_id):
        out = super().holders(bounty_id)
        out["pool_units"] = self._escrow(bounty_id)
        out["price_msats"] = round(coin.price(out["supply"], **self._curve()))
        return out

    def bounties(self, path="", status=""):
        out = super().bounties(path, status)
        for b in out["bounties"]:
            b["pool_units"] = self._escrow(b["id"])
            b["price_msats"] = round(coin.price(b["supply"], **self._curve()))
        return out

    def _refund_pool(self, bounty_id, units, memo):
        for h, m in _pro_rata(units, self._refund_weights(bounty_id)).items():   # by what each backer put in
            self._move(f"escrow:bounty:{bounty_id}", h, m, memo, payout=True)

    def _expire_bounties(self):
        for (bid,) in self.db.execute("SELECT id FROM bounties WHERE status='open' AND deadline < ?", (self.epoch,)).fetchall():
            units = self._escrow(bid)
            self._refund_pool(bid, units, f"bounty {bid} refund")
            self.db.execute("UPDATE bounties SET status='expired', pool=0 WHERE id=?", (bid,))
            self._event(f"bounty #{bid} expired unsolved: {fmt_txc(units)} {self.p.symbol} back to its backers")

    def remove(self, kind, oid):
        if kind == "bounty":
            with self.lock:
                r = self.db.execute("SELECT status FROM bounties WHERE id=?", (int(oid),)).fetchone()
                if r and r[0] == "open":
                    self._refund_pool(int(oid), self._escrow(int(oid)), f"bounty {oid} refund")
                    self.db.execute("UPDATE bounties SET pool=0 WHERE id=?", (int(oid),))
                    self.db.commit()
        return super().remove(kind, oid)

    # --- licences: paid in credits and burned; the buyer's own learnings say which traces earned them -------------------
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
            for lot, bids in sorted(by_lot.items()):
                winners, price = clear_shared(bids, self.k, self.reserve)
                traces = [tid for (tid,) in self.db.execute("SELECT id FROM traces WHERE lot=? ORDER BY id", (lot,))]
                if not winners or not traces:
                    continue
                for w in winners:
                    self.db.execute("INSERT INTO licences VALUES (?,?,?,?,?)", (lot, w, price, e, json.dumps(traces)))
                    paid, basis = self._pay_credits(w, price, f"licence {lot}")
                    self.db.execute("INSERT INTO licence_escrow (lot, buyer, credits, basis, epoch, traces) "
                                    "VALUES (?,?,?,?,?,?)", (lot, w, paid, str(basis), e, json.dumps(traces)))
                out.append({"lot": lot, "winners": winners, "price_msats": price, "traces": len(traces)})
                self._event(f"lot {lot.split('|')[0]} cleared: {len(winners)} licence{'s' if len(winners) != 1 else ''} at "
                            f"{fmt_sats(price)}, paid in credits and burned; the traces each buyer uses earn them")
            self.db.execute("DELETE FROM bids WHERE epoch=?", (e,))        # cleared once: a second call charges nobody
            self.db.commit()
        return {"epoch": e, "cleared": out}

    def direct_licence(self, lot, buyer, traces):
        """A buyer names the traces of a lot it used; the credits it paid for that lot count as burned on their work.
        Learnings it registers do this for it (their parents are the traces it used). Nobody else can steer it."""
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
        lot, credits, basis = self.db.execute("SELECT lot, credits, basis FROM licence_escrow WHERE id=?", (rid,)).fetchone()
        basis, vals = int(basis), self._validator_set()
        per, dust = divmod(credits, len(used))
        for i, tid in enumerate(sorted(used)):
            c = per + (dust if i == 0 else 0)
            info = trace_info[tid]
            for acct, cc in split_trace_sale(c, info["producer"], info["checker_author"], vals).items():
                self._work(acct, "licence", tid, cc, basis * cc // credits if credits else 0)
        self.db.execute("UPDATE licence_escrow SET paid=1 WHERE id=?", (rid,))
        self._event(f"licence money for lot {lot.split('|')[0]}: {fmt_sats(credits)} of credits counts for the "
                    f"{len(used)} trace{'s' if len(used) != 1 else ''} its buyer used")
        return credits

    def _pay_licences(self):
        """At settlement: licence credits count for the traces their buyer's own learnings cite, once validators have
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

    # --- validators ---------------------------------------------------------------------------------------------------
    def min_stake_units(self):
        return units_for(self.p.validator_min_stake_msats, self.ref_q())

    def learning_bond_units(self):
        return units_for(self.p.learning_bond_msats, self.ref_q()) if self.p.quorum > 0 else 0

    def challenge_stake_units(self):
        return units_for(self.p.challenge_stake_msats, self.ref_q())

    def _stake(self, address):
        r = self.db.execute("SELECT stake FROM validators WHERE address=?", (address,)).fetchone()
        return int(r[0]) if r else 0

    def _stakes(self):
        return [(a, int(s)) for a, s in self.db.execute("SELECT address, stake FROM validators ORDER BY address").fetchall()]

    def _active(self):
        """Validators whose stake is worth the sats minimum at the reference price, and those the price pushed under
        it less than `stake_grace_epochs` ago (they have that long to top up): a dump can't empty the federation and
        hand every seat to whoever stakes right after it."""
        floor, e = self.min_stake_units(), self.epoch
        rows = self.db.execute("SELECT address, stake, below_since FROM validators ORDER BY address").fetchall()
        return [(a, int(s)) for a, s, since in rows
                if int(s) > 0 and (int(s) >= floor or (since is not None and e - since < self.p.stake_grace_epochs))]

    def _check_floor(self):
        """At settlement, once the new reference price is set: note who fell under the sats minimum, and when."""
        floor = self.min_stake_units()
        for a, s, since in self.db.execute("SELECT address, stake, below_since FROM validators").fetchall():
            if int(s) >= floor and since is not None:
                self.db.execute("UPDATE validators SET below_since=NULL WHERE address=?", (a,))
            elif int(s) < floor and since is None:
                self.db.execute("UPDATE validators SET below_since=? WHERE address=?", (self.epoch + 1, a))

    def register_validator(self, address, stake_units):
        """Stake TXC to join the validator federation: at least 10,000 sats of it at the reference price. More stake:
        drawn more often, earns more, loses more."""
        need_address(address, "validator")
        stake = int(stake_units)
        with self.lock:
            have = self._stake(address)
            if stake <= 0 or have + stake < self.min_stake_units():
                raise ValueError(f"validators stake at least {fmt_sats(self.p.validator_min_stake_msats)} of "
                                 f"{self.p.symbol}: {fmt_txc(self.min_stake_units())} at the reference price")
            self._need_coins(address, stake)
            self._tx_fee(address)
            self._move(address, f"escrow:stake:{address}", stake, "validator stake")
            if self.db.execute("SELECT 1 FROM validators WHERE address=?", (address,)).fetchone():
                self.db.execute("UPDATE validators SET stake=?, below_since=NULL WHERE address=?", (str(have + stake), address))
            else:
                self.db.execute("INSERT INTO validators (address, stake, joined) VALUES (?,?,?)",
                                (address, str(stake), self.epoch))
            self._event(f"a validator staked {fmt_txc(stake)} {self.p.symbol}")
            self.db.commit()
        return self.validator(address)

    def validator(self, address):
        r = self.db.execute("SELECT stake, joined, slashed, below_since FROM validators WHERE address=?", (address,)).fetchone()
        if not r:
            raise KeyError(address)
        return {"address": address, "stake_units": int(r[0]), "joined": r[1], "slashed_units": int(r[2]),
                "active": address in {a for a, _ in self._active()}, "below_minimum_since": r[3]}

    def validators_list(self):
        rows = sorted(self._stakes(), key=lambda v: -v[1])
        return {"validators": [self.validator(a) for a, _ in rows], "quorum": self.p.quorum,
                "min_stake_units": self.min_stake_units(), "min_stake_msats": self.p.validator_min_stake_msats}

    def _validator_set(self):
        return [a for a, _ in self._active()] or list(self.validators)

    def _slash(self, address, fraction, why):
        stake = self._stake(address)
        if stake <= 0:
            return 0
        cut = _frac(stake, fraction)
        slashed = int(self.db.execute("SELECT slashed FROM validators WHERE address=?", (address,)).fetchone()[0])
        self.db.execute("UPDATE validators SET stake=?, slashed=? WHERE address=?",
                        (str(stake - cut), str(slashed + cut), address))
        if stake - cut < self.min_stake_units():
            # The grace period is for validators a price drop pushed under the minimum, not for ones slashed under it:
            # back-date the mark so the seat is lost now.
            self.db.execute("UPDATE validators SET below_since=? WHERE address=?",
                            (self.epoch - self.p.stake_grace_epochs, address))
        self._coin(f"escrow:stake:{address}", -cut, f"slashed: {why}")
        self._burn(cut, "slash", address)
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
            raise ValueError("rejected: a coin node prices calls in sats: royalty.per_call_msats, a whole number of "
                             "millisatoshis (Learning.build(per_call_msats=...))")
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
            bond = self.learning_bond_units()
            if bond:
                self._need_coins(payer, bond)
            self._tx_fee(payer)
            if bond:
                self._move(payer, f"escrow:bond:{lid}", bond, "learning bond")
            self.db.execute("INSERT INTO learnings VALUES (?,?,?)", (lid, canonical(l).decode(), self.epoch))
            self.db.execute("INSERT INTO verdicts (learning, status, round, bond, trainer, registered) VALUES (?,?,?,?,?,?)",
                            (lid, "pending", 0, str(bond), l["trainer"], self.epoch))
            if weights:
                self.db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?)", (weights, lid))
            if self.p.quorum <= 0:
                gain = float(a.get("after", 0)) - float(a.get("before", 0))
                self._accept(lid, gain, None, gain, 0)
            elif self.beacon_delay == 0:
                self._assign(lid, 0)
            self._event(f"learning submitted for validation: {l['kind']} for {l['base_model']['name']}"
                        + (f", claims {float(a['before']):.1%} → {float(a['after']):.1%}" if a else "")
                        + (f"; bond {fmt_sats(self.p.learning_bond_msats)} ({fmt_txc(bond)} {self.p.symbol})" if bond else ""))
            self.db.commit()
        return dict(self.verdict(lid), id=lid)

    def _eligible(self, exclude=()):
        """Who can be drawn: validators staked at the sats minimum, not excluded; if a price crash left fewer than a
        quorum at the minimum, every staked validator (a crash must not stall the federation)."""
        act = self._active()
        vals = [(a, s) for a, s in act if a not in exclude]
        if len(vals) < self.p.quorum:
            vals = act
        if len(vals) < self.p.quorum:
            staked = [(a, s) for a, s in self._stakes() if s > 0]
            vals = [(a, s) for a, s in staked if a not in exclude]
            vals = vals if len(vals) >= self.p.quorum else staked
        return vals

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
        r = self.db.execute("SELECT bond FROM verdicts WHERE learning=?", (lid,)).fetchone()
        return int(r[0] or 0) if r else 0

    def _reject(self, lid, med, over, money=True):
        """No gain the validators can see, or a claim far beyond what they measured: the bond is forfeit, and burned."""
        if money:
            self._burn(self._release_bond(lid, to=None), "bond", lid)
        self.db.execute("UPDATE verdicts SET status='rejected', gain=?, note=? WHERE learning=?",
                        (med, "overclaimed" if over else None, lid))
        self._event(f"learning rejected: validators measured {med * 100:+.1f} points"
                    + (", far below what it claimed" if over else "") + "; its bond is burned")

    def _inconclusive(self, lid, med, audit, money=True):
        if money:
            bond = self._release_bond(lid, to=None)
            burn = _frac(bond, self.p.inconclusive_burn)
            self._burn(burn, "bond", lid)
            trainer = self.db.execute("SELECT trainer FROM verdicts WHERE learning=?", (lid,)).fetchone()[0]
            self._coin(trainer, bond - burn, "bond back: inconclusive", payout=True)
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
                    + ("; its parents are padding, so their share is withheld and half the bond burned" if padded else ""))

    def _pad_burn(self, lid):
        bond = self._bond(lid)
        cut = _frac(bond, self.p.pad_burn)
        if cut:
            self._coin(f"escrow:bond:{lid}", -cut, "padded parents")
            self._burn(cut, "bond", lid)
            self.db.execute("UPDATE verdicts SET bond=? WHERE learning=?", (str(bond - cut), lid))

    def _release_bond(self, lid, to):
        bond = self._bond(lid)
        if not bond:
            return 0
        self._coin(f"escrow:bond:{lid}", -bond, "bond out")
        if to:
            self._coin(to, bond, "bond returned", payout=True)
        self.db.execute("UPDATE verdicts SET bond='0' WHERE learning=?", (lid,))
        return bond

    def verdict(self, lid):
        r = self.db.execute("SELECT status, round, gain, bond, trainer, registered, accepted, audit_bad, challenger, note "
                            "FROM verdicts WHERE learning=?", (lid,)).fetchone()
        if not r:
            raise KeyError(lid)
        reveals = [{"validator": v, "gain": round(g, 4), "agreed": None if ok is None else bool(ok), "round": rd}
                   for v, g, ok, rd in self.db.execute(
                       "SELECT validator, gain, agreed, round FROM reveals WHERE learning=? ORDER BY round, validator", (lid,))]
        return {"learning": lid, "status": r[0], "round": r[1], "median_gain": r[2], "bond_units": int(r[3] or 0),
                "trainer": r[4], "registered_epoch": r[5], "accepted_epoch": r[6], "audit_bad": r[7], "challenger": r[8],
                "note": r[9], "assigned": self.assigned(lid, r[1]), "committed": self._committed(lid, r[1]),
                "reveals": reveals, "quorum": self.p.quorum}

    # --- challenges: fraud proofs, any time; they take back whatever hasn't vested yet -----------------------------------
    def challenge(self, lid, challenger):
        """Anyone can challenge an accepted learning at any time by staking 2,000 sats of TXC. Fresh validators re-measure it on
        new eval sets (and look at its parents); what it earns while challenged waits, and the parents' share vests, so
        an upheld challenge always has something to take back."""
        need_address(challenger, "challenger")
        with self.lock:
            v = self.db.execute("SELECT status, accepted, round FROM verdicts WHERE learning=?", (lid,)).fetchone()
            if not v:
                raise KeyError(lid)
            if v[0] != "accepted":
                raise ValueError("only an accepted learning can be challenged")
            stake = self.challenge_stake_units()
            self._need_coins(challenger, stake)
            self._tx_fee(challenger)
            self._move(challenger, f"escrow:challenge:{lid}", stake, "challenge stake")
            rnd = v[2] + 1
            self.db.execute("UPDATE verdicts SET status='challenged', round=?, challenger=?, challenge_stake=? WHERE learning=?",
                            (rnd, challenger, str(stake), lid))
            if self.beacon_delay == 0:
                self._assign(lid, rnd, exclude=self.assigned(lid, v[2]))
            self._event("a learning was challenged: fresh validators re-measure it on new eval sets; its rewards pause")
            self.db.commit()
        return self.verdict(lid)

    def _challenge_row(self, lid):
        challenger, stake, rnd = self.db.execute("SELECT challenger, challenge_stake, round FROM verdicts WHERE learning=?",
                                                 (lid,)).fetchone()
        return challenger, int(stake or 0), rnd

    def _clawback(self, lid, med, parents_only=False):
        """Upheld: unvested rewards stop (bounty pools go back to their backers), the bond burns, the challenger gets its
        stake back, and the validators who vouched for it lose stake. An audit challenge does the same to the parents'
        share and half the bond."""
        challenger, stake, rnd = self._challenge_row(lid)
        for bid in self._claw(lid, roles=("parents",) if parents_only else None):
            if not parents_only:
                self.db.execute("UPDATE bounties SET status='clawed back' WHERE id=?", (bid,))
        self._move(f"escrow:challenge:{lid}", challenger, stake, "challenge stake back", payout=True)
        first = [v for (v,) in self.db.execute("SELECT validator FROM reveals WHERE learning=? AND round<? AND agreed=1",
                                               (lid, rnd))]
        for v in first:
            self._slash(v, self.p.fake_slash / (2 if parents_only else 1),
                        "passed padded parents" if parents_only else "vouched for a gain fresh validators could not reproduce")
        if parents_only:
            self._pad_burn(lid)
            self.db.execute("UPDATE verdicts SET status='accepted', challenger=NULL, challenge_stake='0', audit_bad=?, "
                            "note='padded' WHERE learning=?", (1.0, lid))
            self._event("audit challenge upheld: the learning's parents were padding; their share is clawed back and half "
                        "the bond burned")
        else:
            self._burn(self._release_bond(lid, to=None), "bond", lid)
            self.db.execute("UPDATE verdicts SET status='clawed back', gain=? WHERE learning=?", (med, lid))
            self._event(f"challenge upheld: fresh validators measured {med * 100:+.1f} points; unvested rewards clawed "
                        "back and the bond burned")

    def _claw(self, lid, roles=None):
        """Stop a learning's unvested rewards (only `roles`, if given). Unminted coins are simply never minted; a bounty
        pool's remainder goes back to its backers; escrowed TXC burns. Returns the bounties refunded."""
        bounties = set()
        for vid, units, released, source, role in self.db.execute(
                "SELECT id, units, released, source, role FROM vesting WHERE learning=? AND status='vesting'", (lid,)).fetchall():
            if roles and role not in roles:
                continue
            self.db.execute("UPDATE vesting SET status='clawed' WHERE id=?", (vid,))
            left = int(units) - int(released)
            if source.startswith("escrow:bounty:"):
                bid = int(source.rsplit(":", 1)[1])
                self._refund_pool(bid, left, f"bounty {bid} clawed back")
                bounties.add(bid)
            elif source.startswith("escrow:") and left:
                self._coin(source, -left, "clawed back")
                self._burn(left, "clawback", lid)
        return bounties

    def _challenge_failed(self, lid, med):
        challenger, stake, _ = self._challenge_row(lid)
        self._coin(f"escrow:challenge:{lid}", -stake, "challenge lost")
        self._burn(stake, "challenge", lid)
        self.db.execute("UPDATE verdicts SET status='accepted', challenger=NULL, challenge_stake='0' WHERE learning=?", (lid,))
        self._event(f"challenge rejected: fresh validators reproduced the gain ({med * 100:+.1f} points); the challenge "
                    "stake is burned")

    # --- bounties pay on the poster's own measurement -----------------------------------------------------------------
    def claim_bounty(self, bounty_id, learning_id, attestation=None):
        """A bounty pays when its poster measures the learning on the bounty's hidden eval set (the poster's own failing
        cases) at the target: no validator, however many of them one party controls, can give that for the poster. The
        learning must also be accepted by the federation. The pool's TXC then vests to the solver and down the
        learning's family tree, so a challenge can still claw it back."""
        with self.lock:
            row = self.db.execute("SELECT status, eval_set, target, pool, base_model, poster FROM bounties WHERE id=?",
                                  (bounty_id,)).fetchone()
            if not row:
                raise KeyError(f"bounty {bounty_id}")
            status, eval_set, target, pool_msats, base_model, poster = row
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
            pool = self._escrow(bounty_id)
            padded = v[1] is not None and v[1] > self.p.audit_max_bad
            shares = self._shares(learning_id, pool, BOUNTY_SPLIT, withhold=padded)
            payout = {}
            for role, payees in shares.items():
                for acct, m in payees.items():
                    self._vest(acct, m, f"escrow:bounty:{bounty_id}", learning_id, role)
                    payout[acct] = payout.get(acct, 0) + m
            rest = pool - sum(payout.values())
            if rest > 0:                                     # padded parents' part, and rounding: back to the backers
                self._refund_pool(bounty_id, rest, f"bounty {bounty_id} refund")
            self.db.execute("UPDATE bounties SET status='solved', winner=?, learning=? WHERE id=?",
                            (L["trainer"], learning_id, bounty_id))
            self._event(f"bounty #{bounty_id} solved on its poster's own eval: {fmt_sats(pool_msats)} "
                        f"({fmt_txc(pool)} {self.p.symbol}) vests to the solver and the traces over {self.p.vest_epochs} "
                        f"epochs; its coins now earn {coin.HOLDER_CUT:.0%} of every use")
            self.db.commit()
        return {"bounty": bounty_id, "status": "solved", "winner": L["trainer"], "pool_units": pool,
                "pool_msats": pool_msats, "vesting_units": payout}

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
        """Who gets `amount` earned by a learning: by role, on the protocol's split (not the trainer's terms), with equal
        weight per distinct parent (a copy counts as its original), so neither terms nor padding can tilt it; a parent
        learning passes its slice on to its own traces (see _nested). With `withhold` (the parents were found padded)
        the parents' part goes to nobody. Returns {role: {account: amount}}; what it leaves out (withheld parts,
        rounding) the caller burns or refunds."""
        L = json.loads(self.db.execute("SELECT body FROM learnings WHERE id=?", (lid,)).fetchone()[0])
        parents = sorted({self._canonical(p["trace"]) for p in L["parents"]})
        split = dict(split or Learning.DEFAULT_SPLIT)
        trace_info, learnings = tree or self._split_tree()
        validators = self._agreed(lid) or self._validator_set()
        out = {}
        for role, keys in (("learner", ("trainer", "validators")), ("parents", ("traces", "checkers"))):
            if role == "parents" and (withhold or not parents):
                continue
            part = amount * round(sum(split[k] for k in keys) * 1_000_000) // 1_000_000
            sub = {k: (split[k] if k in keys else 0) for k in split}
            L2 = dict(L, parents=[{"trace": p, "weight": 1 / max(len(parents), 1)} for p in parents],
                      royalty=dict(L["royalty"], split=sub))
            out[role] = split_usage(part, L2, trace_info, validators, learnings)
        return out

    def _padded(self, lid):
        r = self.db.execute("SELECT audit_bad FROM verdicts WHERE learning=?", (lid,)).fetchone()
        return bool(r and r[0] is not None and r[0] > self.p.audit_max_bad)

    # --- decoys: does a validator measure, or only answer? ------------------------------------------------------------
    def register_decoy(self, learning, digest, funder):
        """Operator: submit a learning whose true gain only the operator knows, sealed as decoy_digest(gain, salt), its
        bond paid by `funder`. Nothing about it differs from other learnings until validators have revealed; then
        unseal_decoy() opens the truth and slashes every validator whose score sits further from it than its own sample
        can explain: it never measured. This replaces slashing for disagreement, which a majority could aim at the
        honest minority."""
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
            caught = []
            for v, body, g in self.db.execute("SELECT validator, body, gain FROM reveals WHERE learning=? AND round=0",
                                              (lid,)).fetchall():
                if abs(g - float(gain)) > tolerance(json.loads(body), self.p.decoy_z, self.p.min_tol):
                    self._slash(v, self.p.fake_slash, "scored a decoy without measuring it")
                    caught.append(v)
            challenger, stake, _ = self._challenge_row(lid)
            if status[0] == "challenged" and stake:                         # a watchdog caught it first: stake back
                self._move(f"escrow:challenge:{lid}", challenger, stake, "challenge stake back", payout=True)
            self._release_bond(lid, to=r[1])
            self._claw(lid)
            self.db.execute("UPDATE verdicts SET status='decoy', challenger=NULL, challenge_stake='0' WHERE learning=?", (lid,))
            self.db.execute("UPDATE decoys SET unsealed=1, gain=? WHERE learning=?", (float(gain), lid))
            self._event(f"a decoy was unsealed (true gain {float(gain) * 100:+.0f} points): "
                        + (f"{len(caught)} validator{'s' if len(caught) != 1 else ''} scored it without measuring and "
                           "lost stake" if caught else "every validator measured it"))
            self.db.commit()
        return {"learning": lid, "caught": caught}

    # --- vesting ----------------------------------------------------------------------------------------------------
    def _vest(self, account, units, source, learning, role=None):
        if units > 0:
            self.db.execute("INSERT INTO vesting (account, units, start, epochs, source, learning, role) "
                            "VALUES (?,?,?,?,?,?,?)", (account, str(int(units)), self.epoch + 1, self.p.vest_epochs,
                                                       source, learning, role))

    def _release(self):
        e = self.epoch
        paused = {lid for (lid,) in self.db.execute("SELECT learning FROM verdicts WHERE status='challenged'")}
        for vid, account, units, released, start, epochs, source, lid in self.db.execute(
                "SELECT id, account, units, released, start, epochs, source, learning FROM vesting "
                "WHERE status='vesting' AND start <= ?", (e,)).fetchall():
            if lid in paused:
                continue
            units, released = int(units), int(released)
            due = units * min(e - start + 1, epochs) // epochs
            delta = due - released
            if delta <= 0:
                continue
            if source == "mint":
                self._add("minted", delta)
                self._coin(account, delta, "vested", payout=True)
            else:
                self._move(source, account, delta, "vested", payout=True)
            self.db.execute("UPDATE vesting SET released=?, status=? WHERE id=?",
                            (str(due), "done" if due >= units else "vesting", vid))

    # --- burn and mint ----------------------------------------------------------------------------------------------
    def emission(self, epoch=None):
        """The most an epoch can mint: 50,000 TXC, halving every 180 epochs five times, then 1,562.5 TXC for good."""
        e = self.epoch if epoch is None else epoch
        return self.p.emission >> min(max(e - 1, 0) // self.p.halving_epochs, self.p.max_halvings)

    def _work(self, account, role, ref, credits, basis):
        """Credits burned on `account`'s work this epoch (its claim on the emission)."""
        if credits > 0:
            self.db.execute("INSERT INTO work VALUES (?,?,?,?,?,?)",
                            (self.epoch, account, role, ref, int(credits), str(int(basis))))

    def _paid(self, lid, credits, basis):
        r = self.db.execute("SELECT credits, basis FROM paid WHERE epoch=? AND learning=?", (self.epoch, lid)).fetchone()
        c, b = (r[0], int(r[1])) if r else (0, 0)
        self.db.execute("INSERT OR REPLACE INTO paid VALUES (?,?,?,?)", (self.epoch, lid, c + credits, str(b + basis)))

    def usage(self, u):
        """A consumer pays for calls of an accepted learning, in credits, now: sats top them up through the on-ramp
        (their TXC is bought and burned). At settlement the credits count as burned on the work of its family tree."""
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
            credits = calls * json.loads(r[0])["royalty"]["per_call_msats"]
            self._need_funds(consumer, credits)
            self._tx_fee(consumer)
            paid, basis = self._pay_credits(consumer, credits, f"usage {lid[:19]}")
            self.db.execute("INSERT INTO usage VALUES (?,?,?,?)", (lid, consumer, calls, self.epoch))
            self._paid(lid, paid, basis)
            self.db.commit()
        return {"metered": calls, "paid_msats": paid}

    def _attribute(self, tree):
        """At settlement: the credits each learning's users burned this epoch, down its family tree (a solved bounty's
        coin holders first take HOLDER_CUT), as work owed a share of the emission."""
        solved = {lid: bid for bid, lid in self.db.execute("SELECT id, learning FROM bounties WHERE status='solved'")}
        for lid, credits, basis in self.db.execute("SELECT learning, credits, basis FROM paid WHERE epoch=?",
                                                   (self.epoch,)).fetchall():
            basis = int(basis)
            if lid in solved:
                holds = dict(self.db.execute("SELECT holder, coins FROM holdings WHERE bounty=?", (solved[lid],)).fetchall())
                hc = credits * round(coin.HOLDER_CUT * 100) // 100
                hb = basis * hc // credits if credits else 0
                for h, c in _pro_rata(hc, holds).items():
                    self._work(h, "holders", lid, c, hb * c // hc)
                credits, basis = credits - hc, basis - hb
            if credits <= 0:
                continue
            for role, payees in self._shares(lid, credits, withhold=self._padded(lid), tree=tree).items():
                for acct, c in payees.items():
                    self._work(acct, role, lid, c, basis * c // credits)

    def _mint(self, twap):
        """Mint this epoch's emission to the work that was paid for: operators (operator_share) by the fees they took,
        everyone else by the credits burned on their work, each never more than those credits were worth at the epoch's
        time-weighted price, nor more than the TXC they burned. Parents' shares vest; what nobody earned isn't minted."""
        status = dict(self.db.execute("SELECT learning, status FROM verdicts").fetchall())
        total = self.emission()
        pools = {"operator": total * round(self.p.operator_share * BPS) // BPS}
        pools["work"] = total - pools["operator"]
        rows = {}
        for a, role, ref, c, b in self.db.execute("SELECT account, role, ref, credits, basis FROM work WHERE epoch=?",
                                                  (self.epoch,)).fetchall():
            got = rows.setdefault((a, role, ref), [0, 0])
            got[0], got[1] = got[0] + c, got[1] + int(b)
        burned = {"work": 0, "operator": 0}
        for (_, role, _), (c, _) in rows.items():
            burned["operator" if role == "operator" else "work"] += c
        minted, capped = {"work": 0, "operator": 0}, {"work": 0, "operator": 0}
        for (a, role, ref), (c, b) in sorted(rows.items()):
            kind = "operator" if role == "operator" else "work"
            st = status.get(ref)
            if st in NO_EARNINGS or not ADDRESS.fullmatch(str(a)):
                continue
            share = pools[kind] * c // burned[kind]
            cap = min(units_for(c, twap), b)
            m = min(share, cap)
            capped[kind] += share > cap
            if m <= 0:
                continue
            if role == "parents" or st == "challenged":        # vests: an audit or a challenge can still stop it
                self._vest(a, m, "mint", ref, role)
            else:
                self._add("minted", m)
                self._coin(a, m, f"minted: {role}", payout=True)
            minted[kind] += m
        return {"emission_units": total, "credits_burned": burned, "minted_units": minted,
                "unminted_units": total - minted["work"] - minted["operator"], "capped": capped}

    # --- settlement -------------------------------------------------------------------------------------------------
    def settle(self):
        with self.lock:
            e = self.epoch
            self._expire_bounties()
            raw = self._tree()
            tree = (raw[0], self._nested(raw[1]))
            self._attribute(tree)
            # rounds that ran out of time: settle with the majority that revealed, or re-draw validators
            for lid, rnd in self.db.execute(
                    "SELECT v.learning, v.round FROM verdicts v WHERE v.status IN ('pending','challenged') AND EXISTS "
                    "(SELECT 1 FROM assignments a WHERE a.learning=v.learning AND a.round=v.round AND a.epoch < ?)", (e,)).fetchall():
                n = self.db.execute("SELECT COUNT(*) FROM reveals WHERE learning=? AND round=?", (lid, rnd)).fetchone()[0]
                if n >= self.p.quorum // 2 + 1:
                    self._finalize(lid, rnd)
                else:
                    self._redraw(lid, rnd)
            self._pay_licences()
            fees = self._bill_fees()                         # transaction fees, in credits, burned
            twap = self.twap_q()
            m = self._mint(twap)
            self._release()
            for lid, trainer in self.db.execute("SELECT learning, trainer FROM verdicts WHERE status='accepted' AND "
                                                "bond != '0' AND accepted + ? <= ?", (self.p.vest_epochs, e)).fetchall():
                if not self._decoy(lid):
                    self._release_bond(lid, to=trainer)
            payouts = {}
            for a, u in self.db.execute("SELECT account, units FROM coin_ledger WHERE epoch=? AND payout=1", (e,)).fetchall():
                if int(u) > 0 and ADDRESS.fullmatch(str(a)):
                    payouts[a] = payouts.get(a, 0) + int(u)
            leaves = {a: leaf(e, a, u) for a, u in payouts.items()}
            levels = build_tree(list(leaves.values()))
            root = "0x" + levels[-1][0].hex()
            claims = {a: {"amount_units": u, "proof": ["0x" + h.hex() for h in proof(levels, leaves[a])]} for a, u in payouts.items()}
            self.db.execute("INSERT OR REPLACE INTO coin_roots VALUES (?,?,?,?)",
                            (e, root, str(sum(payouts.values())), json.dumps(claims)))
            self.db.execute("INSERT OR REPLACE INTO roots VALUES (?,?,?,?)", (e, root, 0, "{}"))
            self._set_meta("beacon", hashlib.sha256(f"{self._meta('beacon')}|{root}|{e}".encode()).hexdigest())
            for (lid,) in self.db.execute("SELECT learning FROM verdicts WHERE status='pending' AND registered <= ?", (e,)).fetchall():
                self._assign(lid, 0)
            for lid, rnd in self.db.execute("SELECT learning, round FROM verdicts WHERE status='challenged'").fetchall():
                self._assign(lid, rnd, exclude=self.assigned(lid, rnd - 1))
            burned = self._m("burned") - self._m("burned_mark")
            work_credits = m["credits_burned"]["work"]
            summary = dict(m, epoch=e, twap_msats=twap // PQ, twap_q=twap, burned_units=burned, fee_credits=fees,
                           equilibrium_msats=work_credits * UNIT // max(m["emission_units"] - m["emission_units"]
                                                                         * round(self.p.operator_share * BPS) // BPS, 1))
            self.db.execute("INSERT OR REPLACE INTO mints VALUES (?,?)", (e, json.dumps(summary)))
            self._set_meta("burned_mark", self._m("burned"))
            self._set_meta("ref_q", twap)
            self._check_floor()
            self._mark()
            self._repeg_fee(e)
            minted = m["minted_units"]["work"] + m["minted_units"]["operator"]
            self._event(f"epoch {e} settled: {fmt_sats(work_credits + fees)} of credits burned ({fmt_txc(burned)} "
                        f"{self.p.symbol}); {fmt_txc(minted)} {self.p.symbol} minted to the work it paid for, of "
                        f"{fmt_txc(m['emission_units'])} on offer; time-weighted price {fmt_price(twap)}", force=True)
            self._prune(e)
            self._set_meta("epoch", e + 1)
            self.db.commit()
        return {"epoch": e, "root": root, "total_units": sum(payouts.values()), "claims": claims, "mint": m,
                "price_msats": self.price(), "twap_msats": twap // PQ, "supply_units": self.supply()}

    def _prune(self, e):
        """Settlement and storage at machine scale: per-transaction detail is kept through the challenge window
        (`keep_epochs`), then folded away. Each account's older ledger rows become one carried-forward row, so balances
        stay exact; usage reports, payments, burns, finished vesting and paid licence escrow go; each epoch keeps its
        Merkle root and total for good, its per-address claims only through the window. Traces and learnings stay."""
        cut = e - self.p.keep_epochs + 1                  # rows from epochs before `cut` are past the window
        if cut <= 1:
            return
        for table, col, agg in (("coin_ledger", "units", "BIGSUM"), ("ledger", "micros", "SUM")):
            old = self.db.execute(f"SELECT account, {agg}({col}) FROM {table} WHERE epoch < ? GROUP BY account",
                                  (cut,)).fetchall()
            if not old:
                continue
            self.db.execute(f"DELETE FROM {table} WHERE epoch < ?", (cut,))
            for a, v in old:
                if int(v or 0):
                    row = (cut - 1, a, str(v), "carried forward", 0) if table == "coin_ledger" else (cut - 1, a, int(v),
                                                                                                       "carried forward")
                    self.db.execute(f"INSERT INTO {table} VALUES ({','.join('?' * len(row))})", row)
        for sql in ("DELETE FROM usage WHERE epoch < ?", "DELETE FROM paid WHERE epoch < ?",
                    "DELETE FROM work WHERE epoch < ?", "DELETE FROM burns WHERE epoch < ?",
                    "DELETE FROM licence_escrow WHERE paid=1 AND epoch < ?",
                    "DELETE FROM vesting WHERE status != 'vesting' AND start + epochs <= ?",
                    "UPDATE coin_roots SET claims='{}' WHERE epoch < ? AND claims != '{}'"):
            self.db.execute(sql, (cut,))

    # --- reads --------------------------------------------------------------------------------------------------------
    def _vesting_of(self, account=None):
        sql = "SELECT units, released FROM vesting WHERE status='vesting'" + (" AND account=?" if account else "")
        return sum(int(u) - int(r) for u, r in self.db.execute(sql, (account,) if account else ()).fetchall())

    def wallet(self, account):
        w = super().wallet(account)
        return dict(w, credits_msats=self._credit_row(account)[0], coin_units=self._coins(account),
                    vesting_units=self._vesting_of(account), stake_units=self._stake(account),
                    owed_fee_msats=self._owed_credits(account), symbol=self.p.symbol, decimals=DECIMALS,
                    price_msats=self.price())

    def balance(self, account):
        """TXC paid to an account by epoch, with Merkle proofs for the epochs still inside the challenge window."""
        by_epoch, proofs = {}, {}
        for e, claims in self.db.execute("SELECT epoch, claims FROM coin_roots ORDER BY epoch").fetchall():
            c = json.loads(claims).get(account)
            if c:
                by_epoch[str(e)], proofs[str(e)] = c["amount_units"], c["proof"]
        return {"account": account, "by_epoch_units": by_epoch, "total_units": sum(by_epoch.values()), "proofs": proofs,
                "kept": f"per-address claims for {self.p.keep_epochs} epochs; every epoch's root for good"}

    def last_mint(self):
        r = self.db.execute("SELECT body FROM mints ORDER BY epoch DESC LIMIT 1").fetchone()
        return json.loads(r[0]) if r else None

    def coin_stats(self):
        q, ref = self._price_q(), self.ref_q()
        stakes = self._stakes()
        last = self.last_mint()
        return {"symbol": self.p.symbol, "decimals": DECIMALS, "priced_in": "sats", "price_msats": self.price(),
                "price": fmt_price(q), "price_usd_approx": usd_approx(q // PQ, self.btc_usd),
                "usd_per_btc": self.btc_usd, "reference_price_msats": ref // PQ,
                "smallest_unit_sats": f"{q / PQ / 1000 / UNIT:.3g}",
                "supply_units": self.supply(), "minted_units": self._m("minted"), "burned_units": self._m("burned"),
                "pool": {"msats": self._m("pool_msats"), "sats": self._m("pool_msats") // 1000,
                         "coin_units": self._m("pool_coin")},
                "emission_units": self.emission(), "halving_epochs": self.p.halving_epochs,
                "max_halvings": self.p.max_halvings, "operator_share": self.p.operator_share,
                "credits": {"made_msats": self._m("credits_made"), "spent_msats": self._m("credits_spent"),
                            "fees_msats": self._m("fee_credits")},
                "last_epoch": last and {"epoch": last["epoch"], "credits_burned_msats": sum(last["credits_burned"].values()),
                                        "burned_units": last["burned_units"],
                                        "minted_units": sum(last["minted_units"].values()),
                                        "emission_units": last["emission_units"], "twap_msats": last["twap_msats"],
                                        "equilibrium_price_msats": last["equilibrium_msats"]},
                "vesting_units": self._vesting_of(),
                "licence_escrow_msats": self.db.execute("SELECT COALESCE(SUM(credits),0) FROM licence_escrow WHERE paid=0").fetchone()[0],
                "validators": len(self._active()),
                "staked_units": sum(s for _, s in stakes),
                "learnings": dict(self.db.execute("SELECT status, COUNT(*) FROM verdicts WHERE status != 'decoy' "
                                                  "GROUP BY status")),
                "rules": {"tx_fee_msats": self.tx_fee_msats, "tx_fee_usd_approx": usd_approx(self.tx_fee_msats, self.btc_usd),
                          "fee_repeg_epochs": self.p.fee_repeg_epochs, "amm_fee_bps": self.p.amm_fee_bps,
                          "quorum": self.p.quorum, "vest_epochs": self.p.vest_epochs, "keep_epochs": self.p.keep_epochs,
                          "overclaim": self.p.overclaim, "learning_bond_msats": self.p.learning_bond_msats,
                          "validator_min_stake_msats": self.p.validator_min_stake_msats,
                          "challenge_stake_msats": self.p.challenge_stake_msats,
                          "learning_bond_units": self.learning_bond_units(), "min_stake_units": self.min_stake_units(),
                          "challenge_stake_units": self.challenge_stake_units(),
                          "bounty_curve_msats": {"first_coin": self.p.bounty_base, "each_coin_adds": self.p.bounty_slope}}}

    def stats(self):
        s = super().stats()
        s["coin"] = self.coin_stats()
        r = self.db.execute("SELECT epoch, root, total FROM coin_roots ORDER BY epoch DESC LIMIT 1").fetchone()
        s["last_root"] = {"epoch": r[0], "root": r[1], "total_units": int(r[2])} if r else None
        return s

    def describe(self):
        d = super().describe()
        d["settlement"] = {"asset": self.p.symbol, "decimals": DECIMALS, "network": "testnet", "priced_in": "sats",
                           "pay_in": "credits (1 credit = 1 msat), made by burning TXC; sats top them up (test sats "
                                     "here, Lightning on mainnet)",
                           "credits": "POST /v0/credits", "faucet": "POST /v0/faucet", "swap": "POST /v0/swap",
                           "test_msats_per_wallet": self.test_credits}
        d["payments"] = {"rail": "bitcoin over Lightning", "protocol": "L402",
                         "how": "a payment the account can't cover answers 402 Payment Required with "
                                "WWW-Authenticate: L402 macaroon=..., invoice=...; pay the invoice, retry with "
                                "Authorization: L402 <macaroon>:<preimage>",
                         "testnet": "the invoice is a placeholder; POST /v0/faucet gives 30,000 test sats once"}
        d["fee_per_transaction_msats"] = self.tx_fee_msats
        d.pop("fee_per_transaction_nanos", None)
        d["validation"] = {"quorum": self.p.quorum, "commit": "POST /v0/learnings/{id}/commits",
                           "reveal": "POST /v0/learnings/{id}/reveals", "challenge": "POST /v0/learnings/{id}/challenges"}
        return d

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
        """Every unit accounted for: minted - burned == held by accounts and escrows + the pool's reserve; and every
        credit made is either spent or still held."""
        held = self._sum("SELECT BIGSUM(units) FROM coin_ledger")
        credits = self.db.execute("SELECT COALESCE(SUM(credits), 0) FROM credits").fetchone()[0]
        made, spent = self._m("credits_made"), self._m("credits_spent")
        return {"minted": self._m("minted"), "burned": self._m("burned"), "held": held, "pool": self._m("pool_coin"),
                "credits_made": made, "credits_spent": spent, "credits_held": credits,
                "balanced": self._m("minted") - self._m("burned") == held + self._m("pool_coin") and made - spent == credits}
