"""The coin economy (testnet v0.3): contributors earn a network coin, users pay in dollars, the market prices the coin.

    python node/exchange.py --economy coin ...        # or TRACEX_ECONOMY=coin

How value moves
  * Users pay dollars (test dollars here, USDC on mainnet). Every payment buys TXC from the open pool; half of what it
    buys is burned, half goes to the contributors whose work was used. That burn is what gives TXC its value.
  * New TXC is minted only to match payments: at most half of what a learning's usage burned, to the same contributors,
    vesting over `vest_epochs` and clawed back if a challenge shows the gain was fake. Nothing is minted for a verdict.
  * Bounties are free to post and backed in TXC on their bonding curve: early backers get more coins, so a bigger share
    of the solution's revenue, but selling back returns only what the coins cost and an unsolved bounty refunds by what
    each backer put in. A bounty pays when its poster, who holds its hidden eval set, measures a validated learning at
    the target there; the pool vests to the solver.
  * Licence money waits until its buyer shows which traces it used (the parents its own learnings cite, or a list it
    sends); those traces share it.
  * The pool (constant product, like Uniswap v2) is the "natural exchange": the protocol never sets a price.

Why farming loses (each rule is a test in tests/test_coin.py and an attack in examples/farming/attacks.py)
  1. Only payments pay, and whoever pays judges: users pay for a learning after measuring it on their own data, a
     bounty's poster measures solutions on its own hidden eval, a licence buyer's own learnings decide which traces get
     its money. Validators' verdicts decide who may earn; they never move money. So a federation captured by a stake
     majority can block honest work, but it has nothing to print and nothing to take.
  2. Paying yourself loses: of what a payment buys, its contributors get back at most three quarters (half kept, plus a
     match of at most half the burn), so wash usage, self-funded bounties and buying your own traces all lose money.
  3. Validators are drawn at random from a beacon published after a learning is submitted (stake-weighted rendezvous
     hashing), commit before anyone reveals (Bittensor's weight-copying problem), and measure on data they each hold
     privately; the median decides (robust aggregation, as federated learning does). A claim more than twice what they
     measured, beyond the noise, forfeits the bond.
  4. Forfeits burn: a lost bond, stake or challenge stake goes to nobody, so no verdict is worth buying for the money it
     moves.
  5. Validators earn only their share of what the learnings they vouched for go on to earn. Disagreeing isn't a fault:
     stake is slashed for not showing up, for vouching for a gain a fresh round refuted, or for scoring a decoy (a
     learning whose true gain only the operator knows) without measuring it.
  6. Copies and padding earn nothing: a trace with the same fix as an earlier one and a reworded input shares the
     original's slot; junk that no buyer uses never receives licence money; royalties and matches follow the protocol's
     split, equal per distinct parent; a learning cited by another passes its slice through to its own traces, so a
     wrapper around someone's traces takes nothing; padded parents lose their share and half the bond.
  7. Every transaction pays one standard fee, about the electricity it uses: $0.0000004 (exchange.TX_FEE_NANOS, measured
     by examples/fees/measure.py), billed each epoch in whole micro-dollars to whoever runs the node. No rule above
     depends on it being large: the attacks that earn nothing lose exactly their fees.

Units: dollars in micros (1e-6 $), TXC in units (1e-6 TXC).
"""
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass

from exchange import (ADDRESS, BOUNTY_SPLIT, MAX_DEPTH, Exchange, coin, need_address, split_trace_sale, split_usage,
                      leaf, build_tree, proof, canonical, object_id, clear_shared, Bid)
from traceex.trace import Learning

UNIT = 1_000_000


@dataclass
class Params:
    symbol: str = "TXC"
    genesis_coins: int = 1_000_000 * UNIT      # protocol-owned liquidity at genesis...
    genesis_usd: int = 10_000 * 1_000_000      # ...beside $10,000: TXC opens at $0.01
    amm_fee_bps: int = 30                      # the pool's spread: 0.3% of a swap stays in the pool, nobody collects it
    burn_share: float = 0.5                    # of the coins a payment buys: half burned, half to contributors
    match: float = 0.5                         # then at most half of what a learning's usage burned is minted back
    emission: int = 50_000 * UNIT              # the most an epoch can mint, halving every `halving_epochs`
    halving_epochs: int = 180
    vest_epochs: int = 4
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
    learning_bond: int = 500 * UNIT
    validator_min_stake: int = 1_000 * UNIT
    challenge_stake: int = 200 * UNIT
    noshow_slash: float = 0.05
    fake_slash: float = 0.25                   # vouched for a gain a fresh round refuted, or scored a decoy unmeasured
    near_dup: float = 0.3                      # the same distinctive fix, inputs sharing 30% of their words: one trace
    bounty_base: int = UNIT                    # bounty coins: the first costs 1 TXC...
    bounty_slope: int = UNIT // 100            # ...and each one sold adds 0.01 TXC


COIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS coin_ledger (epoch INT, account TEXT, units INT, memo TEXT, payout INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS vesting   (id INTEGER PRIMARY KEY, account TEXT, units INT, released INT DEFAULT 0,
                                      start INT, epochs INT, source TEXT, learning TEXT, status TEXT DEFAULT 'vesting',
                                      role TEXT);
CREATE TABLE IF NOT EXISTS validators(address TEXT PRIMARY KEY, stake INT, joined INT, slashed INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS verdicts  (learning TEXT PRIMARY KEY, status TEXT, round INT, gain REAL, bond INT,
                                      trainer TEXT, registered INT, accepted INT, audit_bad REAL, emitted INT DEFAULT 0,
                                      challenger TEXT, challenge_stake INT DEFAULT 0, note TEXT);
CREATE TABLE IF NOT EXISTS assignments(learning TEXT, round INT, validator TEXT, epoch INT,
                                       PRIMARY KEY (learning, round, validator));
CREATE TABLE IF NOT EXISTS commits   (learning TEXT, round INT, validator TEXT, digest TEXT, epoch INT,
                                      PRIMARY KEY (learning, round, validator));
CREATE TABLE IF NOT EXISTS reveals   (learning TEXT, round INT, validator TEXT, body TEXT, gain REAL, epoch INT,
                                      agreed INT, PRIMARY KEY (learning, round, validator));
CREATE TABLE IF NOT EXISTS dups      (trace TEXT PRIMARY KEY, canonical TEXT, key TEXT);
CREATE TABLE IF NOT EXISTS burns     (epoch INT, kind TEXT, ref TEXT, units INT);
CREATE TABLE IF NOT EXISTS licence_escrow(id INTEGER PRIMARY KEY, lot TEXT, buyer TEXT, units INT, epoch INT,
                                          traces TEXT, paid INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS decoys    (learning TEXT PRIMARY KEY, digest TEXT, funder TEXT, gain REAL,
                                      unsealed INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS artifacts (hash TEXT PRIMARY KEY, learning TEXT);
"""
SLOT = re.compile(r"\{([A-Z]+)_\d+\}")


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

    def __init__(self, path=":memory:", *, params=None, beacon_delay=1, **kw):
        """beacon_delay=1: validators for a learning are drawn from the beacon published at the settlement after it was
        submitted, so nobody can grind a learning's content for friendly validators. 0 assigns at once (tests)."""
        super().__init__(path, **kw)
        self.p, self.beacon_delay = params or Params(), int(beacon_delay)
        self.db.executescript(COIN_SCHEMA)
        if "fix" not in {r[1] for r in self.db.execute("PRAGMA table_info(dups)").fetchall()}:
            self.db.execute("ALTER TABLE dups ADD COLUMN fix TEXT")             # databases from testnet v0.2
        self.db.execute("CREATE INDEX IF NOT EXISTS dups_fix ON dups(fix)")
        if self._meta("pool_coin") is None:                     # genesis
            for k, v in (("pool_usd", self.p.genesis_usd), ("pool_coin", self.p.genesis_coins),
                         ("minted", self.p.genesis_coins), ("burned", 0)):
                self._set_meta(k, v)
            self._set_meta("beacon", hashlib.sha256(b"traceX genesis").hexdigest())
        self.db.commit()

    # --- accounting -------------------------------------------------------------------------------------------------
    def _m(self, k):
        return int(self._meta(k) or 0)

    def _add(self, k, v):
        self._set_meta(k, self._m(k) + int(v))

    def _coins(self, account):
        return self.db.execute("SELECT COALESCE(SUM(units),0) FROM coin_ledger WHERE account=?", (account,)).fetchone()[0]

    def _coin(self, account, units, memo, payout=False):
        if units:
            self.db.execute("INSERT INTO coin_ledger VALUES (?,?,?,?,?)", (self.epoch, account, int(units), memo, int(payout)))

    def _move(self, src, dst, units, memo, payout=False):
        self._coin(src, -units, memo)
        self._coin(dst, units, memo, payout)

    def _burn(self, units):
        self._add("burned", units)

    def _need_coins(self, account, units):
        have = self._coins(account)
        if have < units:
            raise ValueError(f"not enough {self.p.symbol}: {account[:10]}… has {have / UNIT:,.2f}, this needs "
                             f"{units / UNIT:,.2f} (POST /v0/swap buys {self.p.symbol} with test dollars)")

    def supply(self):
        return self._m("minted") - self._m("burned")

    def price(self):
        """Micros of dollars per whole TXC, from the pool's reserves."""
        return self._m("pool_usd") * UNIT // max(self._m("pool_coin"), 1)

    # --- the pool ---------------------------------------------------------------------------------------------------
    def _amm_buy(self, micros):
        x, y = self._m("pool_usd"), self._m("pool_coin")
        dx = micros * (10_000 - self.p.amm_fee_bps) // 10_000
        dy = y * dx // (x + dx)
        self._set_meta("pool_usd", x + micros)
        self._set_meta("pool_coin", y - dy)
        return dy

    def _amm_sell(self, units):
        x, y = self._m("pool_usd"), self._m("pool_coin")
        dy = units * (10_000 - self.p.amm_fee_bps) // 10_000
        dx = x * dy // (y + dy)
        self._set_meta("pool_usd", x - dx)
        self._set_meta("pool_coin", y + units)
        return dx

    def _buy_and_burn(self, micros):
        """Protocol buy for a payment: returns (burned, kept) units."""
        bought = self._amm_buy(micros)
        burned = int(bought * self.p.burn_share)
        self._burn(burned)
        return burned, bought - burned

    def quote(self, side, amount):
        x, y = self._m("pool_usd"), self._m("pool_coin")
        if side == "buy":
            dx = int(amount) * (10_000 - self.p.amm_fee_bps) // 10_000
            return {"side": "buy", "pay_micros": int(amount), "get_units": y * dx // (x + dx)}
        dy = int(amount) * (10_000 - self.p.amm_fee_bps) // 10_000
        return {"side": "sell", "pay_units": int(amount), "get_micros": x * dy // (y + dy)}

    def swap(self, account, side, amount):
        """Test dollars for TXC or back, at the pool's price; the pool keeps its 0.3% spread. Like every transaction it
        pays the standard fee (TX_FEE_NANOS)."""
        need_address(account, "account")
        amount = int(amount)
        if amount <= 0 or side not in ("buy", "sell"):
            raise ValueError("side is 'buy' (amount in dollar micros) or 'sell' (amount in TXC units), amount > 0")
        with self.lock:
            if side == "buy":
                self._need_funds(account, amount)
                self._tx_fee(account)
                self._credit(account, -amount, f"swap: buy {self.p.symbol}")
                out = self._amm_buy(amount)
                self._coin(account, out, "swap: bought")
                got = {"bought_units": out}
                self._event(f"${amount / 1e6:,.2f} bought {out / UNIT:,.1f} {self.p.symbol}; "
                            f"price ${self.price() / 1e6:.4f}")
            else:
                self._need_coins(account, amount)
                self._tx_fee(account)
                self._coin(account, -amount, "swap: sold")
                usd = self._amm_sell(amount)
                self._credit(account, usd, f"swap: sold {self.p.symbol}")
                got = {"paid_micros": usd}
                self._event(f"{amount / UNIT:,.1f} {self.p.symbol} sold for ${usd / 1e6:,.2f}; price ${self.price() / 1e6:.4f}")
            self.db.commit()
        return dict(got, price_micros=self.price(), symbol=self.p.symbol)

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

    # --- bounties in TXC ------------------------------------------------------------------------------------------
    def _curve(self):
        return {"base": self.p.bounty_base, "slope": self.p.bounty_slope}

    def post_bounty(self, b):
        b = dict(b)
        seed_micros = int(b.pop("seed_micros", 0) or b.pop("reward_micros", 0) or 0)
        seed_units = int(float(b.pop("seed_coins", 0) or 0) * UNIT)
        out = super().post_bounty(b)
        out["price_units"] = int(coin.price(0, **self._curve()))
        out.pop("price_micros", None)
        if seed_micros or seed_units:
            out["seed"] = self.buy_coins(out["id"], b["poster"], micros=seed_micros, units=seed_units)
        return out

    def buy_coins(self, bounty_id, buyer, micros=0, units=0):
        """Back a bounty with TXC (`units`), or with dollars (`micros`), which buy TXC at the pool's price first."""
        need_address(buyer, "buyer")
        micros, units = int(micros or 0), int(units or 0)
        if micros > 0 and units <= 0:
            units = self.swap(buyer, "buy", micros)["bought_units"]
        if units <= 0:
            raise ValueError("spend must be positive")
        with self.lock:
            status, pool, supply = self._bounty(bounty_id)
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}; buy its coins from a holder")
            self._need_coins(buyer, units)
            self._tx_fee(buyer)
            net = units
            n = coin.coins_for(supply, net, **self._curve())
            self._coin(buyer, -units, f"bounty {bounty_id} backing")
            self._coin(f"escrow:bounty:{bounty_id}", net, "pool")
            self._set_holding(bounty_id, buyer, self._holding(bounty_id, buyer) + n)
            self._set_basis(bounty_id, buyer, self._basis(bounty_id, buyer) + net)
            self.db.execute("UPDATE bounties SET pool=pool+?, supply=supply+? WHERE id=?", (net, n, bounty_id))
            self._event(f"bounty #{bounty_id} backed with {units / UNIT:,.1f} {self.p.symbol}: {n:,.1f} coins; "
                        f"pool {(pool + net) / UNIT:,.1f} {self.p.symbol}")
            self.db.commit()
        return {"bounty": bounty_id, "coins": round(n, 6), "spent_units": units, "pool_units": pool + net,
                "next_price_units": round(coin.price(supply + n, **self._curve()))}

    def sell_coins(self, bounty_id, seller, coins):
        """Sell back while the bounty is open, for what the coins cost and never more: a later backer's money stays
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
            basis = self._basis(bounty_id, seller)
            cost = int(basis * coins / have)
            value = min(cost, pool)
            self._set_basis(bounty_id, seller, basis - cost)
            self._set_holding(bounty_id, seller, have - coins)
            self.db.execute("UPDATE bounties SET pool=pool-?, supply=supply-? WHERE id=?", (value, coins, bounty_id))
            self._move(f"escrow:bounty:{bounty_id}", seller, value, f"bounty {bounty_id} sell")
            self._event(f"{coins:,.1f} coins of bounty #{bounty_id} sold back for what they cost, "
                        f"{value / UNIT:,.1f} {self.p.symbol}")
            self.db.commit()
        return {"bounty": bounty_id, "sold": round(coins, 6), "paid_units": value,
                "next_price_units": round(coin.price(supply - coins, **self._curve()))}

    def holders(self, bounty_id):
        out = super().holders(bounty_id)
        out["pool_units"] = out.pop("pool_micros")
        out["price_units"] = round(coin.price(out["supply"], **self._curve()))
        out.pop("price_micros", None)
        return out

    def bounties(self, path="", status=""):
        out = super().bounties(path, status)
        for b in out["bounties"]:
            b["pool_units"] = b.pop("pool_micros")
            b["price_units"] = round(coin.price(b["supply"], **self._curve()))
            b.pop("price_micros", None)
        return out

    def _refund_pool(self, bounty_id, pool, memo):
        for h, m in coin.pro_rata(pool, self._refund_weights(bounty_id)).items():   # by what each backer put in
            self._move(f"escrow:bounty:{bounty_id}", h, m, memo, payout=True)

    def _expire_bounties(self):
        for bid, pool in self.db.execute(
                "SELECT id, pool FROM bounties WHERE status='open' AND deadline < ?", (self.epoch,)).fetchall():
            self._refund_pool(bid, pool, f"bounty {bid} refund")
            self.db.execute("UPDATE bounties SET status='expired', pool=0 WHERE id=?", (bid,))
            self._event(f"bounty #{bid} expired unsolved: {pool / UNIT:,.1f} {self.p.symbol} back to its backers")

    def remove(self, kind, oid):
        if kind == "bounty":
            with self.lock:
                r = self.db.execute("SELECT status, pool FROM bounties WHERE id=?", (int(oid),)).fetchone()
                if r and r[0] == "open":
                    self._refund_pool(int(oid), r[1], f"bounty {oid} refund")
                    self.db.execute("UPDATE bounties SET pool=0 WHERE id=?", (int(oid),))
                    self.db.commit()
        return super().remove(kind, oid)

    # --- licences: dollars in, half burned, the rest waits until the buyer shows which traces it used ------------------
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
                    self._credit(w, -price, f"licence {lot}")
                    _, kept = self._buy_and_burn(price)
                    rid = self.db.execute("INSERT INTO licence_escrow (lot, buyer, units, epoch, traces) VALUES (?,?,?,?,?)",
                                          (lot, w, kept, e, json.dumps(traces))).lastrowid
                    self._coin(f"escrow:licence:{rid}", kept, "licence")
                out.append({"lot": lot, "winners": winners, "price_micros": price, "traces": len(traces)})
                self._event(f"lot {lot.split('|')[0]} cleared: {len(winners)} licence{'s' if len(winners) != 1 else ''} at "
                            f"${price / 1e6:,.2f}; half the {self.p.symbol} it bought burned, the rest waits for the "
                            "traces each buyer uses")
            self.db.execute("DELETE FROM bids WHERE epoch=?", (e,))        # cleared once: a second call charges nobody
            self.db.commit()
        return {"epoch": e, "cleared": out}

    def direct_licence(self, lot, buyer, traces):
        """A buyer names the traces of a lot it used; its licence money for that lot goes to them. Learnings it
        registers do this for it (their parents are the traces it used). Nobody else can steer a buyer's money."""
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
        return {"lot": lot, "buyer": buyer, "paid_units": paid}

    def _pay_licence(self, rid, used, trace_info):
        lot, units = self.db.execute("SELECT lot, units FROM licence_escrow WHERE id=?", (rid,)).fetchone()
        vals = self._validator_set()
        per, dust = divmod(units, len(used))
        for i, tid in enumerate(sorted(used)):
            info = trace_info[tid]
            for acct, m in split_trace_sale(per + (dust if i == 0 else 0), info["producer"], info["checker_author"],
                                            vals).items():
                self._move(f"escrow:licence:{rid}", acct, m, f"licence {tid[:19]}", payout=True)
        self.db.execute("UPDATE licence_escrow SET paid=1 WHERE id=?", (rid,))
        self._event(f"licence money for lot {lot.split('|')[0]}: {units / UNIT:,.1f} {self.p.symbol} to the "
                    f"{len(used)} trace{'s' if len(used) != 1 else ''} its buyer used")
        return units

    def _pay_licences(self):
        """At settlement: licence money goes to the traces its buyer's own learnings cite, once validators have finished
        with them and haven't found the parents padded. Junk no buyer uses never receives any."""
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
    def register_validator(self, address, stake_units):
        """Stake TXC to join the validator federation. More stake: drawn more often, earns more, loses more."""
        need_address(address, "validator")
        stake = int(stake_units)
        with self.lock:
            have = self.db.execute("SELECT stake FROM validators WHERE address=?", (address,)).fetchone()
            if (have[0] if have else 0) + stake < self.p.validator_min_stake:
                raise ValueError(f"validators stake at least {self.p.validator_min_stake / UNIT:,.0f} {self.p.symbol}")
            self._need_coins(address, stake)
            self._tx_fee(address)
            self._move(address, f"escrow:stake:{address}", stake, "validator stake")
            if have:
                self.db.execute("UPDATE validators SET stake=stake+? WHERE address=?", (stake, address))
            else:
                self.db.execute("INSERT INTO validators (address, stake, joined) VALUES (?,?,?)", (address, stake, self.epoch))
            self._event(f"a validator staked {stake / UNIT:,.0f} {self.p.symbol}")
            self.db.commit()
        return self.validator(address)

    def validator(self, address):
        r = self.db.execute("SELECT stake, joined, slashed FROM validators WHERE address=?", (address,)).fetchone()
        if not r:
            raise KeyError(address)
        return {"address": address, "stake_units": r[0], "joined": r[1], "slashed_units": r[2],
                "active": r[0] >= self.p.validator_min_stake}

    def validators_list(self):
        rows = self.db.execute("SELECT address FROM validators ORDER BY stake DESC").fetchall()
        return {"validators": [self.validator(a) for (a,) in rows], "quorum": self.p.quorum}

    def _validator_set(self):
        rows = self.db.execute("SELECT address FROM validators WHERE stake >= ?", (self.p.validator_min_stake,)).fetchall()
        return [a for (a,) in rows] or list(self.validators)

    def _slash(self, address, fraction, why):
        r = self.db.execute("SELECT stake FROM validators WHERE address=?", (address,)).fetchone()
        if not r or r[0] <= 0:
            return 0
        cut = int(r[0] * fraction)
        self.db.execute("UPDATE validators SET stake=stake-?, slashed=slashed+? WHERE address=?", (cut, cut, address))
        self._coin(f"escrow:stake:{address}", -cut, f"slashed: {why}")
        self._burn(cut)
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
        lid = object_id(l)
        weights = (l.get("artifact") or {}).get("hash")
        with self.lock:
            if self.db.execute("SELECT 1 FROM learnings WHERE id=?", (lid,)).fetchone():
                return dict(self.verdict(lid), id=lid)
            if weights:                                  # the same weights under a new name earn nothing new
                first = self.db.execute("SELECT learning FROM artifacts WHERE hash=?", (weights,)).fetchone()
                if first:
                    raise ValueError(f"rejected: these weights are already learning {first[0][:19]}…; a learning built "
                                     "on it has weights of its own")
            self._tx_fee(bond_from or l["trainer"])
            bond = self.p.learning_bond if self.p.quorum > 0 else 0
            if bond:
                self._need_coins(bond_from or l["trainer"], bond)
                self._move(bond_from or l["trainer"], f"escrow:bond:{lid}", bond, "learning bond")
            self.db.execute("INSERT INTO learnings VALUES (?,?,?)", (lid, canonical(l).decode(), self.epoch))
            self.db.execute("INSERT INTO verdicts (learning, status, round, bond, trainer, registered) VALUES (?,?,?,?,?,?)",
                            (lid, "pending", 0, bond, l["trainer"], self.epoch))
            if weights:
                self.db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?)", (weights, lid))
            if self.p.quorum <= 0:
                gain = float(a.get("after", 0)) - float(a.get("before", 0))
                self._accept(lid, gain, None, gain, 0)
            elif self.beacon_delay == 0:
                self._assign(lid, 0)
            self._event(f"learning submitted for validation: {l['kind']} for {l['base_model']['name']}"
                        + (f", claims {float(a['before']):.1%} → {float(a['after']):.1%}" if a else ""))
            self.db.commit()
        return dict(self.verdict(lid), id=lid)

    def _assign(self, lid, rnd, exclude=()):
        """Stake-weighted rendezvous hashing over the current beacon: deterministic, and unknowable when submitted."""
        if self.db.execute("SELECT 1 FROM assignments WHERE learning=? AND round=?", (lid, rnd)).fetchone():
            return
        vals = [(a, s) for a, s in self.db.execute("SELECT address, stake FROM validators WHERE stake >= ?",
                                                     (self.p.validator_min_stake,)) if a not in exclude]
        if len(vals) < self.p.quorum:
            vals = [(a, s) for a, s in self.db.execute("SELECT address, stake FROM validators WHERE stake >= ?",
                                                         (self.p.validator_min_stake,))]
        if len(vals) < self.p.quorum:
            return
        beacon = self._meta("beacon")

        def score(a, s):
            h = int(hashlib.sha256(f"{beacon}|{lid}|{rnd}|{a}".encode()).hexdigest(), 16)
            return -math.log((h + 1) / (2 ** 256 + 1)) / s
        for a, _ in sorted(vals, key=lambda v: score(*v))[:self.p.quorum]:
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
        pool = [(a, s) for a, s in self.db.execute("SELECT address, stake FROM validators WHERE stake >= ?",
                                                     (self.p.validator_min_stake,)) if a not in keep and a not in late]
        beacon = self._meta("beacon")
        pool.sort(key=lambda v: -math.log((int(hashlib.sha256(f"{beacon}|{lid}|{rnd}|{v[0]}".encode()).hexdigest(), 16) + 1)
                                          / (2 ** 256 + 1)) / v[1])
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

    def _reject(self, lid, med, over, money=True):
        """No gain the validators can see, or a claim far beyond what they measured: the bond is forfeit, and burned."""
        if money:
            self._burn(self._release_bond(lid, to=None))
        self.db.execute("UPDATE verdicts SET status='rejected', gain=?, note=? WHERE learning=?",
                        (med, "overclaimed" if over else None, lid))
        self._event(f"learning rejected: validators measured {med * 100:+.1f} points"
                    + (", far below what it claimed" if over else "") + "; its bond is burned")

    def _inconclusive(self, lid, med, audit, money=True):
        if money:
            bond = self._release_bond(lid, to=None)
            burn = int(bond * self.p.inconclusive_burn)
            self._burn(burn)
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
        bond = self.db.execute("SELECT bond FROM verdicts WHERE learning=?", (lid,)).fetchone()[0] or 0
        cut = int(bond * self.p.pad_burn)
        if cut:
            self._coin(f"escrow:bond:{lid}", -cut, "padded parents")
            self._burn(cut)
            self.db.execute("UPDATE verdicts SET bond=bond-? WHERE learning=?", (cut, lid))

    def _release_bond(self, lid, to):
        r = self.db.execute("SELECT bond, trainer FROM verdicts WHERE learning=?", (lid,)).fetchone()
        if not r or not r[0]:
            return 0
        self._coin(f"escrow:bond:{lid}", -r[0], "bond out")
        if to:
            self._coin(to, r[0], "bond returned", payout=True)
        self.db.execute("UPDATE verdicts SET bond=0 WHERE learning=?", (lid,))
        return r[0]

    def verdict(self, lid):
        r = self.db.execute("SELECT status, round, gain, bond, trainer, registered, accepted, audit_bad, challenger, note "
                            "FROM verdicts WHERE learning=?", (lid,)).fetchone()
        if not r:
            raise KeyError(lid)
        reveals = [{"validator": v, "gain": round(g, 4), "agreed": None if ok is None else bool(ok), "round": rd}
                   for v, g, ok, rd in self.db.execute(
                       "SELECT validator, gain, agreed, round FROM reveals WHERE learning=? ORDER BY round, validator", (lid,))]
        return {"learning": lid, "status": r[0], "round": r[1], "median_gain": r[2], "bond_units": r[3], "trainer": r[4],
                "registered_epoch": r[5], "accepted_epoch": r[6], "audit_bad": r[7], "challenger": r[8], "note": r[9],
                "assigned": self.assigned(lid, r[1]), "committed": self._committed(lid, r[1]), "reveals": reveals,
                "quorum": self.p.quorum}

    # --- challenges: fraud proofs, any time; they take back whatever hasn't vested yet -----------------------------------
    def challenge(self, lid, challenger):
        """Anyone can challenge an accepted learning at any time by staking `challenge_stake`. Fresh validators
        re-measure it on new eval sets (and look at its parents); everything it earns vests, so an upheld challenge
        always has something to take back."""
        need_address(challenger, "challenger")
        with self.lock:
            v = self.db.execute("SELECT status, accepted, round FROM verdicts WHERE learning=?", (lid,)).fetchone()
            if not v:
                raise KeyError(lid)
            if v[0] != "accepted":
                raise ValueError("only an accepted learning can be challenged")
            self._need_coins(challenger, self.p.challenge_stake)
            self._tx_fee(challenger)
            self._move(challenger, f"escrow:challenge:{lid}", self.p.challenge_stake, "challenge stake")
            rnd = v[2] + 1
            self.db.execute("UPDATE verdicts SET status='challenged', round=?, challenger=?, challenge_stake=? WHERE learning=?",
                            (rnd, challenger, self.p.challenge_stake, lid))
            if self.beacon_delay == 0:
                self._assign(lid, rnd, exclude=self.assigned(lid, v[2]))
            self._event("a learning was challenged: fresh validators re-measure it on new eval sets; its rewards pause")
            self.db.commit()
        return self.verdict(lid)

    def _clawback(self, lid, med, parents_only=False):
        """Upheld: unvested rewards stop (bounty pools go back to their backers), the bond burns, the challenger gets its
        stake back, and the validators who vouched for it lose stake. An audit challenge does the same to the parents'
        share and half the bond."""
        challenger, stake, rnd = self.db.execute("SELECT challenger, challenge_stake, round FROM verdicts WHERE learning=?",
                                                 (lid,)).fetchone()
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
            self.db.execute("UPDATE verdicts SET status='accepted', challenger=NULL, challenge_stake=0, audit_bad=?, "
                            "note='padded' WHERE learning=?", (1.0, lid))
            self._event("audit challenge upheld: the learning's parents were padding; their share is clawed back and half "
                        "the bond burned")
        else:
            self._burn(self._release_bond(lid, to=None))
            self.db.execute("UPDATE verdicts SET status='clawed back', gain=? WHERE learning=?", (med, lid))
            self._event(f"challenge upheld: fresh validators measured {med * 100:+.1f} points; unvested rewards clawed "
                        "back and the bond burned")

    def _claw(self, lid, roles=None):
        """Stop a learning's unvested rewards (only `roles`, if given). Unminted coins are simply never minted; a bounty
        pool's remainder goes back to its backers; escrowed royalties are burned. Returns the bounties refunded."""
        bounties = set()
        for vid, units, released, source, role in self.db.execute(
                "SELECT id, units, released, source, role FROM vesting WHERE learning=? AND status='vesting'", (lid,)).fetchall():
            if roles and role not in roles:
                continue
            self.db.execute("UPDATE vesting SET status='clawed' WHERE id=?", (vid,))
            left = units - released
            if source.startswith("escrow:bounty:"):
                bid = int(source.rsplit(":", 1)[1])
                self._refund_pool(bid, left, f"bounty {bid} clawed back")
                bounties.add(bid)
            elif source.startswith("escrow:") and left:
                self._coin(source, -left, "clawed back")
                self._burn(left)
        return bounties

    def _challenge_failed(self, lid, med):
        challenger, stake = self.db.execute("SELECT challenger, challenge_stake FROM verdicts WHERE learning=?",
                                            (lid,)).fetchone()
        self._coin(f"escrow:challenge:{lid}", -stake, "challenge lost")
        self._burn(stake)
        self.db.execute("UPDATE verdicts SET status='accepted', challenger=NULL, challenge_stake=0 WHERE learning=?", (lid,))
        self._event(f"challenge rejected: fresh validators reproduced the gain ({med * 100:+.1f} points); the challenge "
                    "stake is burned")

    # --- bounties pay on the poster's own measurement -----------------------------------------------------------------
    def claim_bounty(self, bounty_id, learning_id, attestation=None):
        """A bounty pays when its poster measures the learning on the bounty's hidden eval set (the poster's own failing
        cases) at the target: no validator, however many of them one party controls, can give that for the poster. The
        learning must also be accepted by the federation. The pool then vests to the solver and down the learning's
        family tree, so a challenge can still claw it back."""
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
            self._event(f"bounty #{bounty_id} solved on its poster's own eval: {pool / UNIT:,.1f} {self.p.symbol} vests "
                        f"to the solver and the traces over {self.p.vest_epochs} epochs; its coins now earn "
                        f"{coin.HOLDER_CUT:.0%} of every use")
            self.db.commit()
        return {"bounty": bounty_id, "status": "solved", "winner": L["trainer"], "pool_units": pool, "vesting": payout}

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
        the parents' part goes to nobody. Returns {role: {account: units}}; what it leaves out (withheld parts,
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
            part = int(amount * sum(split[k] for k in keys))
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
            status = self.db.execute("SELECT status, challenger, challenge_stake FROM verdicts WHERE learning=?",
                                     (lid,)).fetchone()
            if status[0] == "pending":
                raise ValueError("unseal a decoy once its validators have revealed")
            caught = []
            for v, body, g in self.db.execute("SELECT validator, body, gain FROM reveals WHERE learning=? AND round=0",
                                              (lid,)).fetchall():
                if abs(g - float(gain)) > tolerance(json.loads(body), self.p.decoy_z, self.p.min_tol):
                    self._slash(v, self.p.fake_slash, "scored a decoy without measuring it")
                    caught.append(v)
            if status[0] == "challenged" and status[2]:                     # a watchdog caught it first: stake back
                self._move(f"escrow:challenge:{lid}", status[1], status[2], "challenge stake back", payout=True)
            self._release_bond(lid, to=r[1])
            self._claw(lid)
            self.db.execute("UPDATE verdicts SET status='decoy', challenger=NULL, challenge_stake=0 WHERE learning=?", (lid,))
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
                            "VALUES (?,?,?,?,?,?,?)", (account, int(units), self.epoch + 1, self.p.vest_epochs, source,
                                                       learning, role))

    def _release(self):
        e = self.epoch
        paused = {lid for (lid,) in self.db.execute("SELECT learning FROM verdicts WHERE status='challenged'")}
        for vid, account, units, released, start, epochs, source, lid in self.db.execute(
                "SELECT id, account, units, released, start, epochs, source, learning FROM vesting "
                "WHERE status='vesting' AND start <= ?", (e,)).fetchall():
            if lid in paused:
                continue
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
                            (due, "done" if due >= units else "vesting", vid))

    # --- emissions: a match on what payments burned, nothing for a verdict ----------------------------------------------
    def emission(self, epoch=None):
        e = self.epoch if epoch is None else epoch
        return self.p.emission >> ((e - 1) // self.p.halving_epochs)

    def _emit(self, tree=None):
        """Mint only against payments: for each accepted learning, `match` of what its usage burned this epoch, on the
        protocol's split, vesting. An epoch never mints more than its emission cap; what isn't earned isn't minted."""
        ok = {lid for (lid,) in self.db.execute("SELECT learning FROM verdicts WHERE status='accepted'")}
        want = {lid: int(u * self.p.match) for lid, u in self.db.execute(
            "SELECT ref, SUM(units) FROM burns WHERE epoch=? AND kind='usage' GROUP BY ref", (self.epoch,)) if lid in ok}
        total, cap = sum(want.values()), self.emission()
        minted = 0
        for lid, w in sorted(want.items()):
            amount = w if total <= cap else w * cap // total
            for role, payees in self._shares(lid, amount, withhold=self._padded(lid), tree=tree).items():
                for acct, m in payees.items():
                    self._vest(acct, m, "mint", lid, role)
                    minted += m
        return {"match": minted}

    # --- settlement -------------------------------------------------------------------------------------------------
    def settle(self):
        with self.lock:
            e = self.epoch
            self._expire_bounties()
            raw = self._tree()
            tree = (raw[0], self._nested(raw[1]))
            solved = {lid: bid for bid, lid in self.db.execute("SELECT id, learning FROM bounties WHERE status='solved'")}
            for lid, consumer, calls in self.db.execute(
                    "SELECT learning, consumer, SUM(calls) FROM usage WHERE epoch=? GROUP BY learning, consumer", (e,)).fetchall():
                micros = calls * raw[1][lid]["royalty"]["per_call_micros"]
                self._credit(consumer, -micros, f"usage {lid[:19]}")
                burned, kept = self._buy_and_burn(micros)
                self.db.execute("INSERT INTO burns VALUES (?,?,?,?)", (e, "usage", lid, burned))
                if lid in solved:
                    holds = dict(self.db.execute("SELECT holder, coins FROM holdings WHERE bounty=?", (solved[lid],)))
                    cut = coin.pro_rata(int(kept * coin.HOLDER_CUT), holds)
                    for h, m in cut.items():
                        self._coin(h, m, f"bounty {solved[lid]} coin", payout=True)
                    kept -= sum(cut.values())
                paid = 0                                 # royalties on the protocol's split; padded parents' part burns
                for role, payees in self._shares(lid, kept, withhold=self._padded(lid), tree=tree).items():
                    for acct, m in payees.items():
                        if role == "parents":            # vests, so an audit challenge can still take it back
                            self._coin(f"escrow:royalty:{lid}", m, f"royalty {lid[:19]}")
                            self._vest(acct, m, f"escrow:royalty:{lid}", lid, role)
                        else:
                            self._coin(acct, m, f"royalty {lid[:19]}", payout=True)
                        paid += m
                self._burn(kept - paid)
            # rounds that ran out of time: settle with the majority that revealed, or re-draw validators
            for lid, rnd in self.db.execute(
                    "SELECT v.learning, v.round FROM verdicts v WHERE v.status IN ('pending','challenged') AND EXISTS "
                    "(SELECT 1 FROM assignments a WHERE a.learning=v.learning AND a.round=v.round AND a.epoch < ?)", (e,)).fetchall():
                n = self.db.execute("SELECT COUNT(*) FROM reveals WHERE learning=? AND round=?", (lid, rnd)).fetchone()[0]
                if n >= self.p.quorum // 2 + 1:
                    self._finalize(lid, rnd)
                else:
                    self._redraw(lid, rnd)
            minted = self._emit(tree)
            self._release()
            for lid, trainer in self.db.execute("SELECT learning, trainer FROM verdicts WHERE status='accepted' AND bond > 0 "
                                                "AND accepted + ? <= ?", (self.p.vest_epochs, e)).fetchall():
                if not self._decoy(lid):
                    self._release_bond(lid, to=trainer)
            self._pay_licences()
            self._bill_fees()                                # transaction fees, in dollars, to whoever runs the node
            payouts = {}
            for a, u in self.db.execute("SELECT account, SUM(units) FROM coin_ledger WHERE epoch=? AND payout=1 AND units > 0 "
                                        "GROUP BY account", (e,)):
                if ADDRESS.fullmatch(str(a)):
                    payouts[a] = u
            leaves = {a: leaf(e, a, u) for a, u in payouts.items()}
            levels = build_tree(list(leaves.values()))
            root = "0x" + levels[-1][0].hex()
            claims = {a: {"amount_units": u, "proof": ["0x" + h.hex() for h in proof(levels, leaves[a])]} for a, u in payouts.items()}
            self.db.execute("INSERT OR REPLACE INTO roots VALUES (?,?,?,?)", (e, root, sum(payouts.values()), json.dumps(claims)))
            self._set_meta("beacon", hashlib.sha256(f"{self._meta('beacon')}|{root}|{e}".encode()).hexdigest())
            for (lid,) in self.db.execute("SELECT learning FROM verdicts WHERE status='pending' AND registered <= ?", (e,)).fetchall():
                self._assign(lid, 0)
            for lid, rnd in self.db.execute("SELECT learning, round FROM verdicts WHERE status='challenged'").fetchall():
                self._assign(lid, rnd, exclude=self.assigned(lid, rnd - 1))
            self._event(f"epoch {e} settled: {sum(payouts.values()) / UNIT:,.0f} {self.p.symbol} paid out, "
                        f"{sum(minted.values()) / UNIT:,.0f} minted to match usage (vesting), price ${self.price() / 1e6:.4f}",
                        force=True)
            self._set_meta("epoch", e + 1)
            self.db.commit()
        return {"epoch": e, "root": root, "total_units": sum(payouts.values()), "claims": claims, "granted": minted,
                "price_micros": self.price(), "supply_units": self.supply()}

    # --- reads --------------------------------------------------------------------------------------------------------
    def wallet(self, account):
        w = super().wallet(account)
        vest = self.db.execute("SELECT COALESCE(SUM(units - released),0) FROM vesting WHERE account=? AND status='vesting'",
                               (account,)).fetchone()[0]
        stake = self.db.execute("SELECT stake FROM validators WHERE address=?", (account,)).fetchone()
        return dict(w, coin_units=self._coins(account), vesting_units=vest, stake_units=stake[0] if stake else 0,
                    symbol=self.p.symbol, price_micros=self.price())

    def coin_stats(self):
        one = lambda sql, *a: self.db.execute(sql, a).fetchone()[0]
        return {"symbol": self.p.symbol, "price_micros": self.price(), "supply_units": self.supply(),
                "minted_units": self._m("minted"), "burned_units": self._m("burned"),
                "pool": {"usd_micros": self._m("pool_usd"), "coin_units": self._m("pool_coin")},
                "emission_cap_units": self.emission(), "halving_epochs": self.p.halving_epochs,
                "vesting_units": one("SELECT COALESCE(SUM(units - released),0) FROM vesting WHERE status='vesting'"),
                "licence_escrow_units": one("SELECT COALESCE(SUM(units),0) FROM licence_escrow WHERE paid=0"),
                "validators": one("SELECT COUNT(*) FROM validators WHERE stake >= ?", self.p.validator_min_stake),
                "staked_units": one("SELECT COALESCE(SUM(stake),0) FROM validators"),
                "learnings": dict(self.db.execute("SELECT status, COUNT(*) FROM verdicts WHERE status != 'decoy' "
                                                  "GROUP BY status")),
                "rules": {"burn_share": self.p.burn_share, "match": self.p.match,
                          "tx_fee_nanos": self.tx_fee_nanos, "amm_fee_bps": self.p.amm_fee_bps,
                          "quorum": self.p.quorum, "vest_epochs": self.p.vest_epochs, "overclaim": self.p.overclaim,
                          "learning_bond_units": self.p.learning_bond,
                          "validator_min_stake_units": self.p.validator_min_stake}}

    def stats(self):
        s = super().stats()
        s["coin"] = self.coin_stats()
        s["pools_open_units"] = s.pop("pools_open_micros")
        s["pools_paid_units"] = s.pop("pools_paid_micros")
        return s

    def describe(self):
        d = super().describe()
        d["settlement"] = {"asset": self.p.symbol, "network": "testnet", "pay_in": "test dollars (USDC on mainnet)",
                           "faucet": "POST /v0/faucet", "swap": "POST /v0/swap", "credits_micros": self.test_credits}
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
        """Every unit accounted for: minted - burned == held by accounts and escrows + the pool's reserve."""
        held = self.db.execute("SELECT COALESCE(SUM(units),0) FROM coin_ledger").fetchone()[0]
        return {"minted": self._m("minted"), "burned": self._m("burned"), "held": held, "pool": self._m("pool_coin"),
                "balanced": self._m("minted") - self._m("burned") == held + self._m("pool_coin")}
