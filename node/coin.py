"""The coin economy (testnet v0.2): contributors earn a network coin, users pay in dollars, the market prices the coin.

    python node/exchange.py --economy coin ...        # or TRACEX_ECONOMY=coin

How value moves
  * TXC is minted only for proven value: a learning that a federation of validators finds better on eval sets they
    each hold privately. New coins vest over `vest_epochs` and can be clawed back if a challenge shows the gain was fake.
  * Users still pay dollars (test dollars here, USDC on mainnet). Every payment buys TXC from the open pool; half of
    what it buys is burned, half goes to the contributors whose work was used. Usage burns are what give TXC value.
  * Bounties are free to post and backed in TXC on their bonding curve. A solved bounty's pool vests to the solver.
  * The pool (constant product, like Uniswap v2) is the "natural exchange": the protocol never sets a price.

Why farming loses (each rule is a test in tests/test_coin.py and an attack in examples/farming/attacks.py)
  1. Pay for proof, not volume: only accepted learnings mint; a trace earns only through a learning that passed.
  2. Validators are assigned at random from a beacon published after the learning was submitted, so a trainer can't
     pick (or grind for) friendly validators. Assignment is stake-weighted (rendezvous hashing).
  3. Commit, then reveal: every assigned validator commits a hash of its score before anyone reveals, so nobody can
     copy another validator's number (Bittensor's weight-copying problem).
  4. The gain that counts is the median across validators (robust aggregation, as federated learning aggregates
     updates): one bought validator can't move it. A validator further from the median than its own sample size
     explains (z-scored) loses stake; honest noise on a small eval set does not.
  5. Skin in the game: trainers post a bond, validators stake, challengers stake. Bonds stay locked through vesting.
     A successful challenge on fresh eval sets claws back unvested coins, burns half the bond and pays the challenger.
  6. Usage minting is capped below the burn it came from, so paying for your own learning always loses (Ocean
     Protocol's wash-consume lesson: fees must exceed rewards).
  7. Near-duplicate traces share one attribution slot; emission weights are set by the protocol (equal per distinct
     parent), not by the trainer; validators can audit parents, and a failed audit withholds the traces' share.
  8. A minuscule fee on every trace, swap, backing and payout is burned.

Units: dollars in micros (1e-6 $), TXC in units (1e-6 TXC).
"""
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass

from exchange import (ADDRESS, BOUNTY_SPLIT, Exchange, coin, need_address, split_trace_sale, split_usage, leaf,
                      build_tree, proof, canonical, object_id, clear_shared, Bid)
from traceex.trace import Learning

UNIT = 1_000_000


@dataclass
class Params:
    symbol: str = "TXC"
    genesis_coins: int = 1_000_000 * UNIT      # protocol-owned liquidity at genesis...
    genesis_usd: int = 10_000 * 1_000_000      # ...beside $10,000: TXC opens at $0.01
    amm_fee_bps: int = 30                      # 0.3% per swap, stays in the pool
    protocol_fee_bps: int = 10                 # 0.1% of every swap, backing and payout, burned
    trace_fee_micros: int = 500                # $0.0005 per trace, bought and burned
    burn_share: float = 0.5                    # of the coins a payment buys: half burned, half to contributors
    emission: int = 50_000 * UNIT              # minted per epoch at most, halving every `halving_epochs`
    halving_epochs: int = 180
    pool_improve: float = 0.70                 # newly accepted learnings, by their median gain
    pool_validators: float = 0.20              # validators who agreed with the median, by stake
    pool_usage: float = 0.10                   # learnings by usage burned this epoch
    max_share_per_learning: float = 0.25       # of the improvement pool; the rest is simply not minted
    usage_cap: float = 0.5                     # usage minting for a learning <= half of what its usage burned
    vest_epochs: int = 4
    quorum: int = 3
    min_gain: float = 0.01
    accept_z: float = 2.0                      # accept only if median gain - 2 standard errors >= min_gain
    z: float = 2.5                             # consensus tolerance in standard errors of each validator's own sample
    min_tol: float = 0.02
    inconclusive_burn: float = 0.10            # a real-looking but unproven gain: bond back minus 10%, no rewards
    audit_min: int = 10                        # every reveal audits at least this many parents (or all of them)
    learning_bond: int = 500 * UNIT
    validator_min_stake: int = 1_000 * UNIT
    challenge_stake: int = 200 * UNIT
    outlier_slash: float = 0.10
    noshow_slash: float = 0.05
    fake_slash: float = 0.25                   # validators who accepted a gain a challenge round couldn't reproduce
    audit_max_bad: float = 0.10
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
CREATE TABLE IF NOT EXISTS pending_slash(learning TEXT, validator TEXT, fraction REAL, why TEXT);
CREATE TABLE IF NOT EXISTS usage_burns(epoch INT, learning TEXT, units INT);
"""
SLOT = re.compile(r"\{([A-Z]+)_\d+\}")


def dup_key(t):
    """Two traces that differ only in placeholder numbering, whitespace or case are the same fix."""
    norm = lambda s: re.sub(r"\s+", " ", SLOT.sub(r"{\1}", str(s))).strip().lower()
    body = [t.get("task"), (t.get("base_model") or {}).get("name"), norm(t.get("input", "")),
            sorted(t.get("fixed_fields", [])), {k: norm(v) for k, v in sorted((t.get("verified_output") or {}).items())}]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def attestation_digest(attestation, salt):
    """What a validator commits before revealing: sha256(canonical attestation + salt)."""
    return hashlib.sha256(canonical(attestation) + str(salt).encode()).hexdigest()


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

    def _fee(self, units):
        return units * self.p.protocol_fee_bps // 10_000

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
            out = y * dx // (x + dx)
            return {"side": "buy", "pay_micros": int(amount), "get_units": out - self._fee(out)}
        dy = (int(amount) - self._fee(int(amount))) * (10_000 - self.p.amm_fee_bps) // 10_000
        return {"side": "sell", "pay_units": int(amount), "get_micros": x * dy // (y + dy)}

    def swap(self, account, side, amount):
        """Test dollars for TXC or back, at the pool's price. 0.3% stays in the pool; 0.1% of the TXC side is burned."""
        need_address(account, "account")
        amount = int(amount)
        if amount <= 0 or side not in ("buy", "sell"):
            raise ValueError("side is 'buy' (amount in dollar micros) or 'sell' (amount in TXC units), amount > 0")
        with self.lock:
            if side == "buy":
                self._need_funds(account, amount)
                self._credit(account, -amount, f"swap: buy {self.p.symbol}")
                out = self._amm_buy(amount)
                fee = self._fee(out)
                self._burn(fee)
                self._coin(account, out - fee, "swap: bought")
                got = {"bought_units": out - fee}
                self._event(f"${amount / 1e6:,.2f} bought {(out - fee) / UNIT:,.1f} {self.p.symbol}; "
                            f"price ${self.price() / 1e6:.4f}")
            else:
                self._need_coins(account, amount)
                fee = self._fee(amount)
                self._coin(account, -amount, "swap: sold")
                self._burn(fee)
                usd = self._amm_sell(amount - fee)
                self._credit(account, usd, f"swap: sold {self.p.symbol}")
                got = {"paid_micros": usd}
                self._event(f"{amount / UNIT:,.1f} {self.p.symbol} sold for ${usd / 1e6:,.2f}; price ${self.price() / 1e6:.4f}")
            self.db.commit()
        return dict(got, price_micros=self.price(), symbol=self.p.symbol)

    # --- traces: a minuscule fee, and copies earn nothing extra -------------------------------------------------------
    def submit_trace(self, t):
        fee = self.p.trace_fee_micros if self.test_credits else 0
        if fee:
            need_address(t.get("producer"), "producer")
            self._need_funds(t["producer"], fee)
        out = super().submit_trace(t)
        if out.get("duplicate"):
            return out
        key = dup_key(t)
        with self.lock:
            if fee:
                self._credit(t["producer"], -fee, "trace fee")
                self._burn(self._amm_buy(fee))
            first = self.db.execute("SELECT canonical FROM dups WHERE key=? LIMIT 1", (key,)).fetchone()
            self.db.execute("INSERT OR IGNORE INTO dups VALUES (?,?,?)", (out["id"], first[0] if first else out["id"], key))
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
            fee = self._fee(units)
            net = units - fee
            n = coin.coins_for(supply, net, **self._curve())
            self._coin(buyer, -units, f"bounty {bounty_id} backing")
            self._burn(fee)
            self._coin(f"escrow:bounty:{bounty_id}", net, "pool")
            self._set_holding(bounty_id, buyer, self._holding(bounty_id, buyer) + n)
            self.db.execute("UPDATE bounties SET pool=pool+?, supply=supply+? WHERE id=?", (net, n, bounty_id))
            self._event(f"bounty #{bounty_id} backed with {units / UNIT:,.1f} {self.p.symbol}: {n:,.1f} coins; "
                        f"pool {(pool + net) / UNIT:,.1f} {self.p.symbol}")
            self.db.commit()
        return {"bounty": bounty_id, "coins": round(n, 6), "spent_units": units, "pool_units": pool + net,
                "next_price_units": round(coin.price(supply + n, **self._curve()))}

    def sell_coins(self, bounty_id, seller, coins):
        need_address(seller, "seller")
        with self.lock:
            status, pool, supply = self._bounty(bounty_id)
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}; its coins now earn from the solution")
            have = self._holding(bounty_id, seller)
            coins = min(float(coins), have)
            if coins <= 0:
                raise ValueError("no coins to sell")
            value = min(int(coin.sell_value(supply, coins, **self._curve())), pool)
            fee = self._fee(value)
            self._set_holding(bounty_id, seller, have - coins)
            self.db.execute("UPDATE bounties SET pool=pool-?, supply=supply-? WHERE id=?", (value, coins, bounty_id))
            self._coin(f"escrow:bounty:{bounty_id}", -value, "sell back")
            self._burn(fee)
            self._coin(seller, value - fee, f"bounty {bounty_id} sell")
            self._event(f"{coins:,.1f} coins of bounty #{bounty_id} sold back for {(value - fee) / UNIT:,.1f} {self.p.symbol}")
            self.db.commit()
        return {"bounty": bounty_id, "sold": round(coins, 6), "paid_units": value - fee,
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
        holds = dict(self.db.execute("SELECT holder, coins FROM holdings WHERE bounty=?", (bounty_id,)).fetchall())
        for h, m in coin.pro_rata(pool, holds).items():
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

    # --- licences: dollars in, half burned, half to the traces' producers ---------------------------------------------
    def clear(self):
        out = []
        with self.lock:
            e = self.epoch
            by_lot = {}
            for lot, bidder, price in self.db.execute("SELECT lot, bidder, price FROM bids WHERE epoch=?", (e,)):
                by_lot.setdefault(lot, []).append(Bid(bidder, price))
            for lot, bids in sorted(by_lot.items()):
                winners, price = clear_shared(bids, self.k, self.reserve)
                traces = self.db.execute("SELECT id, producer, checker FROM traces WHERE lot=? ORDER BY id", (lot,)).fetchall()
                if not winners or not traces:
                    continue
                vals = self._validator_set()
                for w in winners:
                    self.db.execute("INSERT INTO licences VALUES (?,?,?,?,?)", (lot, w, price, e, json.dumps([t[0] for t in traces])))
                    self._credit(w, -price, f"licence {lot}")
                    _, kept = self._buy_and_burn(price)
                    per, dust = divmod(kept, len(traces))
                    for i, (tid, producer, checker) in enumerate(traces):
                        for acct, m in split_trace_sale(per + (dust if i == 0 else 0), producer,
                                                         self._checker_author(checker), vals).items():
                            self._coin(acct, m, f"sale {tid[:19]}", payout=True)
                out.append({"lot": lot, "winners": winners, "price_micros": price, "traces": len(traces)})
                self._event(f"lot {lot.split('|')[0]} cleared: {len(winners)} licence{'s' if len(winners) != 1 else ''} at "
                            f"${price / 1e6:,.2f}; half the {self.p.symbol} it bought burned, half paid to producers")
            self.db.commit()
        return {"epoch": e, "cleared": out}

    # --- validators ---------------------------------------------------------------------------------------------------
    def register_validator(self, address, stake_units):
        """Stake TXC to join the validator federation. More stake: picked more often, earns more, loses more."""
        need_address(address, "validator")
        stake = int(stake_units)
        with self.lock:
            have = self.db.execute("SELECT stake FROM validators WHERE address=?", (address,)).fetchone()
            if (have[0] if have else 0) + stake < self.p.validator_min_stake:
                raise ValueError(f"validators stake at least {self.p.validator_min_stake / UNIT:,.0f} {self.p.symbol}")
            self._need_coins(address, stake)
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
    def register_learning(self, l):
        a = l.get("attestation") or {}
        need_address(l.get("trainer"), "trainer")
        if a and not float(a.get("after", 0)) > float(a.get("before", 1)):
            raise ValueError("rejected: the claimed attestation shows no gain")
        for p in l["parents"]:
            known = self.db.execute("SELECT 1 FROM traces WHERE id=? UNION SELECT 1 FROM learnings WHERE id=?",
                                    (p["trace"], p["trace"])).fetchone()
            if not known:
                raise ValueError(f"rejected: unknown parent {p['trace']}")
        lid = object_id(l)
        with self.lock:
            if self.db.execute("SELECT 1 FROM learnings WHERE id=?", (lid,)).fetchone():
                return dict(self.verdict(lid), id=lid)
            bond = self.p.learning_bond if self.p.quorum > 0 else 0
            if bond:
                self._need_coins(l["trainer"], bond)
                self._move(l["trainer"], f"escrow:bond:{lid}", bond, "learning bond")
            self.db.execute("INSERT INTO learnings VALUES (?,?,?)", (lid, canonical(l).decode(), self.epoch))
            self.db.execute("INSERT INTO verdicts (learning, status, round, bond, trainer, registered) VALUES (?,?,?,?,?,?)",
                            (lid, "pending", 0, bond, l["trainer"], self.epoch))
            if self.p.quorum <= 0:
                self._accept(lid, float(a.get("after", 0)) - float(a.get("before", 0)), None)
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
            self.db.execute("INSERT OR REPLACE INTO reveals VALUES (?,?,?,?,?,?,?)",
                            (lid, rnd, validator, canonical(attestation).decode(), after - before, self.epoch, None))
            done = len(self.db.execute("SELECT validator FROM reveals WHERE learning=? AND round=?", (lid, rnd)).fetchall())
            if done >= len(assigned):
                self._finalize(lid, rnd)
            self.db.commit()
        return dict(self.verdict(lid), revealed=validator)

    def _finalize(self, lid, rnd):
        rows = [(v, json.loads(b), g) for v, b, g in self.db.execute(
            "SELECT validator, body, gain FROM reveals WHERE learning=? AND round=?", (lid, rnd)).fetchall()]
        if not rows:
            return
        med = statistics.median(g for _, _, g in rows)
        agreed, over, audits, ses = [], [], [], []
        for v, a, g in rows:
            ok = abs(g - med) <= tolerance(a, self.p.z, self.p.min_tol)
            self.db.execute("UPDATE reveals SET agreed=? WHERE learning=? AND round=? AND validator=?", (int(ok), lid, rnd, v))
            if ok:
                agreed.append(v)
                ses.append(standard_error(a))
                au = a.get("audit") or {}
                if au.get("checked"):
                    audits.append(float(au.get("bad", 0)) / float(au["checked"]))
            elif g > med:
                over.append(v)
        for v in set(self.assigned(lid, rnd)) - {v for v, _, _ in rows}:
            self._slash(v, self.p.noshow_slash, "assigned but did not reveal")
            self.db.execute("DELETE FROM assignments WHERE learning=? AND round=? AND validator=?", (lid, rnd, v))
        se_med = 1.2533 * statistics.median(ses) / math.sqrt(len(ses)) if ses else 1.0
        lower = med - self.p.accept_z * se_med
        audit = statistics.median(audits) if audits else None
        status = self.db.execute("SELECT status, audit_bad FROM verdicts WHERE learning=?", (lid,)).fetchone()
        if rnd == 0:
            if med < self.p.min_gain:
                self._reject(lid, med, agreed, over)
            elif lower < self.p.min_gain:
                bond = self._release_bond(lid, to=None)
                burn = int(bond * self.p.inconclusive_burn)
                self._burn(burn)
                trainer = self.db.execute("SELECT trainer FROM verdicts WHERE learning=?", (lid,)).fetchone()[0]
                self._coin(trainer, bond - burn, "bond back: inconclusive", payout=True)
                self.db.execute("UPDATE verdicts SET status='inconclusive', gain=? WHERE learning=?", (med, lid))
                self._event(f"learning inconclusive: median gain {med * 100:+.1f} points, but within the noise of the "
                            "validators' eval sets; no rewards, bond back minus 10%")
            else:
                self._accept(lid, med, audit)
                for v, _, _ in rows:                       # disagreement is settled once the challenge window closes
                    if v not in agreed:
                        self.db.execute("INSERT INTO pending_slash VALUES (?,?,?,?)",
                                        (lid, v, self.p.outlier_slash, "score far from the median"))
                self._event(f"learning accepted by {len(agreed)} validators: median gain {med * 100:+.1f} points "
                            f"(at least {lower * 100:+.1f} at 2 standard errors)")
        elif status[0] == "challenged":
            if med < self.p.min_gain:
                self._clawback(lid, med)
            elif audit is not None and audit > self.p.audit_max_bad and (status[1] is None or status[1] <= self.p.audit_max_bad):
                self._clawback(lid, med, parents_only=True)
            else:
                self._challenge_failed(lid, med)

    def _reject(self, lid, med, agreed, over):
        """No gain: the bond is forfeit, half burned and half to the validators who measured it; validators who
        reported a gain nobody else could see lose stake."""
        bond = self._release_bond(lid, to=None)
        share = (bond - bond // 2) // len(agreed) if bond and agreed else 0
        for v in agreed:
            self._coin(v, share, "rejected learning's bond", payout=True)
        self._burn(bond - share * len(agreed))
        for v in over:
            self._slash(v, self.p.outlier_slash, "claimed a gain the federation could not see")
        self.db.execute("UPDATE verdicts SET status='rejected', gain=? WHERE learning=?", (med, lid))
        self._event(f"learning rejected: validators measured {med * 100:+.1f} points; its bond is forfeit")

    def _accept(self, lid, gain, audit_bad):
        self.db.execute("UPDATE verdicts SET status='accepted', gain=?, accepted=?, audit_bad=? WHERE learning=?",
                        (gain, self.epoch, audit_bad, lid))

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
        r = self.db.execute("SELECT status, round, gain, bond, trainer, registered, accepted, audit_bad, challenger "
                            "FROM verdicts WHERE learning=?", (lid,)).fetchone()
        if not r:
            raise KeyError(lid)
        reveals = [{"validator": v, "gain": round(g, 4), "agreed": None if ok is None else bool(ok), "round": rd}
                   for v, g, ok, rd in self.db.execute(
                       "SELECT validator, gain, agreed, round FROM reveals WHERE learning=? ORDER BY round, validator", (lid,))]
        return {"learning": lid, "status": r[0], "round": r[1], "median_gain": r[2], "bond_units": r[3], "trainer": r[4],
                "registered_epoch": r[5], "accepted_epoch": r[6], "audit_bad": r[7], "challenger": r[8],
                "assigned": self.assigned(lid, r[1]), "committed": self._committed(lid, r[1]), "reveals": reveals,
                "quorum": self.p.quorum}

    # --- challenges: a fraud-proof window while rewards vest -----------------------------------------------------------
    def challenge(self, lid, challenger):
        need_address(challenger, "challenger")
        with self.lock:
            v = self.db.execute("SELECT status, accepted, round FROM verdicts WHERE learning=?", (lid,)).fetchone()
            if not v:
                raise KeyError(lid)
            if v[0] != "accepted" or self.epoch > v[1] + self.p.vest_epochs:
                raise ValueError("only an accepted learning whose rewards are still vesting can be challenged")
            self._need_coins(challenger, self.p.challenge_stake)
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
        r = self.db.execute("SELECT challenger, challenge_stake, round FROM verdicts WHERE learning=?", (lid,)).fetchone()
        challenger, stake, rnd = r
        rows = self.db.execute("SELECT id, account, units, released, source, role FROM vesting WHERE learning=? "
                               "AND status='vesting'", (lid,)).fetchall()
        for vid, account, units, released, source, role in rows:
            if parents_only and role != "parents":
                continue
            if role and role.startswith("validator:") and int(role.split(":")[1]) >= rnd:
                continue                                                # the challenge round's validators were right
            self.db.execute("UPDATE vesting SET status='clawed' WHERE id=?", (vid,))
            if source.startswith("escrow:bounty:"):                     # the backers get their pool back
                bid = int(source.rsplit(":", 1)[1])
                self._refund_pool(bid, units - released, f"bounty {bid} clawed back")
                self.db.execute("UPDATE bounties SET status='clawed back' WHERE id=?", (bid,))
        bond = self._release_bond(lid, to=None)
        if bond:
            self._burn(bond // 2)
            self._coin(challenger, bond - bond // 2, "won a challenge", payout=True)
        self._move(f"escrow:challenge:{lid}", challenger, stake, "challenge stake back", payout=True)
        first = [v for (v,) in self.db.execute("SELECT validator FROM reveals WHERE learning=? AND round<? AND agreed=1",
                                               (lid, rnd))]
        for v in first:
            self._slash(v, self.p.fake_slash / (2 if parents_only else 1),
                        "passed padded parents" if parents_only else "accepted a gain fresh validators could not reproduce")
        if parents_only:
            self.db.execute("UPDATE verdicts SET status='accepted', challenger=NULL, challenge_stake=0, audit_bad=? "
                            "WHERE learning=?", (1.0, lid))
            self._event("audit challenge upheld: the learning's parents were padded; their share is clawed back")
        else:
            self.db.execute("DELETE FROM pending_slash WHERE learning=?", (lid,))      # the dissenters were right
            self.db.execute("UPDATE verdicts SET status='clawed back', gain=? WHERE learning=?", (med, lid))
            self._event(f"challenge upheld: fresh validators measured {med * 100:+.1f} points; unvested rewards clawed back")

    def _challenge_failed(self, lid, med):
        r = self.db.execute("SELECT challenger, challenge_stake, trainer FROM verdicts WHERE learning=?", (lid,)).fetchone()
        challenger, stake, trainer = r
        self._coin(f"escrow:challenge:{lid}", -stake, "challenge lost")
        self._burn(stake // 2)
        self._coin(trainer, stake - stake // 2, "challenge against you failed", payout=True)
        self.db.execute("UPDATE verdicts SET status='accepted', challenger=NULL, challenge_stake=0 WHERE learning=?", (lid,))
        self._event(f"challenge rejected: fresh validators reproduced the gain ({med * 100:+.1f} points)")

    # --- bounty claims vest too ---------------------------------------------------------------------------------------
    def claim_bounty(self, bounty_id, learning_id):
        with self.lock:
            row = self.db.execute("SELECT status, eval_set, target, pool, base_model FROM bounties WHERE id=?",
                                  (bounty_id,)).fetchone()
            if not row:
                raise KeyError(f"bounty {bounty_id}")
            status, eval_set, target, pool, base_model = row
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}")
            v = self.db.execute("SELECT status FROM verdicts WHERE learning=?", (learning_id,)).fetchone()
            if not v or v[0] != "accepted":
                raise ValueError("the learning must be accepted by the validator federation first")
            L = json.loads(self.db.execute("SELECT body FROM learnings WHERE id=?", (learning_id,)).fetchone()[0])
            vals = set(self._validator_set())
            claims = [L.get("attestation") or {}] + [json.loads(b) for (b,) in self.db.execute(
                "SELECT body FROM reveals WHERE learning=?", (learning_id,))]
            ok = [a for a in claims if a.get("eval_set") == eval_set and a.get("validator") in vals
                  and float(a.get("after", 0)) >= target]
            if not ok:
                raise ValueError("no staked validator attested this learning on the bounty's eval set at the target")
            if base_model and L["base_model"]["name"] != base_model:
                raise ValueError(f"the bounty is for {base_model}")
            trace_info, learnings = self._tree()
            payout = split_usage(pool, dict(L, royalty=dict(L["royalty"], split=BOUNTY_SPLIT)), trace_info,
                                 self._agreed(learning_id) or self._validator_set(), learnings)
            for acct, m in payout.items():
                self._vest(acct, m, f"escrow:bounty:{bounty_id}", learning_id)
            self.db.execute("UPDATE bounties SET status='solved', winner=?, learning=? WHERE id=?",
                            (L["trainer"], learning_id, bounty_id))
            self._event(f"bounty #{bounty_id} solved: {pool / UNIT:,.1f} {self.p.symbol} vests to the solver and the traces "
                        f"over {self.p.vest_epochs} epochs; its coins now earn {coin.HOLDER_CUT:.0%} of every use")
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

    # --- emissions ----------------------------------------------------------------------------------------------------
    def emission(self, epoch=None):
        e = self.epoch if epoch is None else epoch
        return self.p.emission >> ((e - 1) // self.p.halving_epochs)

    def _grant_tree(self, lid, amount, withhold_traces=False):
        """Mint (vesting) for one learning on the protocol's split, with equal weight per distinct parent, so a trainer
        can't tilt emissions toward their own traces. The trainer's and validators' part and the parents' part vest as
        separate grants, so a failed parent audit can claw back the parents' part alone."""
        L = json.loads(self.db.execute("SELECT body FROM learnings WHERE id=?", (lid,)).fetchone()[0])
        parents = sorted({self._canonical(p["trace"]) for p in L["parents"]})
        split = dict(Learning.DEFAULT_SPLIT)
        trace_info, learnings = self._tree()
        validators = self._agreed(lid) or self._validator_set()
        granted = 0
        for role, keys in (("learner", ("trainer", "validators")), ("parents", ("traces", "checkers"))):
            if role == "parents" and (withhold_traces or not parents):
                continue
            part = int(amount * sum(split[k] for k in keys))
            sub = {k: (split[k] if k in keys else 0) for k in split}
            L2 = dict(L, parents=[{"trace": p, "weight": 1 / max(len(parents), 1)} for p in parents],
                      royalty=dict(L["royalty"], split=sub))
            for acct, m in split_usage(part, L2, trace_info, validators, learnings).items():
                self._vest(acct, m, "mint", lid, role)
                granted += m
        return granted

    def _emit(self):
        e, E = self.epoch, self.emission()
        minted = {"improve": 0, "validators": 0, "usage": 0}
        fresh = self.db.execute("SELECT learning, gain, audit_bad FROM verdicts WHERE status='accepted' AND emitted=0 "
                                "AND accepted <= ?", (e,)).fetchall()
        pool = int(E * self.p.pool_improve)
        total = sum(max(g, 0) for _, g, _ in fresh)
        for lid, g, bad in fresh:
            share = min(int(pool * max(g, 0) / total) if total else 0, int(pool * self.p.max_share_per_learning))
            minted["improve"] += self._grant_tree(lid, share, withhold_traces=bad is not None and bad > self.p.audit_max_bad)
            self.db.execute("UPDATE verdicts SET emitted=1 WHERE learning=?", (lid,))
        # validators are paid per verdict they agreed with, tied to that learning: a clawback takes it back
        agreed = self.db.execute("SELECT r.validator, r.learning, r.round, v.stake FROM reveals r JOIN validators v "
                                 "ON v.address=r.validator WHERE r.epoch=? AND r.agreed=1", (e,)).fetchall()
        vpool = int(E * self.p.pool_validators)
        wsum = sum(s for _, _, _, s in agreed)
        for v, lid, rnd, s in agreed:
            m = int(vpool * s / wsum) if wsum else 0
            self._vest(v, m, "mint", lid, f"validator:{rnd}")
            minted["validators"] += m
        burns = dict(self.db.execute("SELECT learning, SUM(units) FROM usage_burns WHERE epoch=? GROUP BY learning", (e,)))
        accepted = {lid for (lid,) in self.db.execute("SELECT learning FROM verdicts WHERE status='accepted'")}
        burns = {lid: b for lid, b in burns.items() if lid in accepted}
        upool, bsum = int(E * self.p.pool_usage), sum(burns.values())
        for lid, b in burns.items():
            alloc = min(int(upool * b / bsum), int(b * self.p.usage_cap))
            minted["usage"] += self._grant_tree(lid, alloc)
        return minted

    # --- settlement -------------------------------------------------------------------------------------------------
    def settle(self):
        with self.lock:
            e = self.epoch
            self._expire_bounties()
            trace_info, learnings = self._tree()
            solved = {lid: bid for bid, lid in self.db.execute("SELECT id, learning FROM bounties WHERE status='solved'")}
            for lid, consumer, calls in self.db.execute(
                    "SELECT learning, consumer, SUM(calls) FROM usage WHERE epoch=? GROUP BY learning, consumer", (e,)).fetchall():
                L = learnings[lid]
                micros = calls * L["royalty"]["per_call_micros"]
                self._credit(consumer, -micros, f"usage {lid[:19]}")
                burned, kept = self._buy_and_burn(micros)
                self.db.execute("INSERT INTO usage_burns VALUES (?,?,?)", (e, lid, burned))
                if lid in solved:
                    holds = dict(self.db.execute("SELECT holder, coins FROM holdings WHERE bounty=?", (solved[lid],)))
                    cut = coin.pro_rata(int(kept * coin.HOLDER_CUT), holds)
                    for h, m in cut.items():
                        self._coin(h, m, f"bounty {solved[lid]} coin", payout=True)
                    kept -= sum(cut.values())
                for acct, m in split_usage(kept, L, trace_info, self._agreed(lid) or self._validator_set(), learnings).items():
                    self._coin(acct, m, f"royalty {lid[:19]}", payout=True)
            # rounds that ran out of time: settle with the majority that revealed, or re-draw validators
            for lid, rnd in self.db.execute(
                    "SELECT v.learning, v.round FROM verdicts v WHERE v.status IN ('pending','challenged') AND EXISTS "
                    "(SELECT 1 FROM assignments a WHERE a.learning=v.learning AND a.round=v.round AND a.epoch < ?)", (e,)).fetchall():
                n = self.db.execute("SELECT COUNT(*) FROM reveals WHERE learning=? AND round=?", (lid, rnd)).fetchone()[0]
                if n >= self.p.quorum // 2 + 1:
                    self._finalize(lid, rnd)
                else:
                    self._redraw(lid, rnd)
            minted = self._emit()
            self._release()
            for lid, trainer in self.db.execute("SELECT learning, trainer FROM verdicts WHERE status='accepted' AND bond > 0 "
                                                "AND accepted + ? <= ?", (self.p.vest_epochs, e)).fetchall():
                self._release_bond(lid, to=trainer)
                for v, frac, why in self.db.execute("SELECT validator, fraction, why FROM pending_slash WHERE learning=?",
                                                    (lid,)).fetchall():
                    self._slash(v, frac, why)
                self.db.execute("DELETE FROM pending_slash WHERE learning=?", (lid,))
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
                        f"{sum(minted.values()) / UNIT:,.0f} granted (vesting), price ${self.price() / 1e6:.4f}", force=True)
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
                "emission_units": self.emission(), "halving_epochs": self.p.halving_epochs,
                "vesting_units": one("SELECT COALESCE(SUM(units - released),0) FROM vesting WHERE status='vesting'"),
                "validators": one("SELECT COUNT(*) FROM validators WHERE stake >= ?", self.p.validator_min_stake),
                "staked_units": one("SELECT COALESCE(SUM(stake),0) FROM validators"),
                "learnings": dict(self.db.execute("SELECT status, COUNT(*) FROM verdicts GROUP BY status")),
                "rules": {"burn_share": self.p.burn_share, "protocol_fee_bps": self.p.protocol_fee_bps,
                          "trace_fee_micros": self.p.trace_fee_micros, "quorum": self.p.quorum,
                          "vest_epochs": self.p.vest_epochs, "learning_bond_units": self.p.learning_bond,
                          "validator_min_stake_units": self.p.validator_min_stake, "usage_cap": self.p.usage_cap}}

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
            if v["status"] != "accepted" and not include_pending:
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
