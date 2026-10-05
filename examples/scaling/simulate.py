"""How TXC behaves as usage grows, crashes and the emission halves: v0.3 (burn and match) against v0.4 (credits, burn
and mint), deterministic, no randomness.

    python examples/scaling/simulate.py              # the tables below
    python examples/scaling/simulate.py --hold 0.5   # contributors keep half of what they earn, for good
    python examples/scaling/simulate.py --top-up     # an option the node does NOT implement (see Economy)

The model is the same for both versions:
  * one epoch a day; users pay U dollars a day (a scenario sets U); every paid use is one transaction of $0.001, so the
    standard fee adds 5% (v0.4: $0.00005 in credits, burned; v0.3: $0.0000004 in dollars to the operator, no TXC);
  * the protocol-owned pool from genesis is the only market: 1,000,000 TXC beside $10,000, a 0.3% spread that stays in
    the pool (so it deepens with volume);
  * users' purchases and contributors' sales arrive spread over 24 hours; within an hour they cross at the pool's price
    first and only the difference trades with the pool, as on any exchange with an order book in front of its pool;
  * everyone paid in TXC sells `1 - hold` of what reaches them (hold = 0 by default: machines, model hosts and
    contributors are paid for work and pay their bills in dollars), spread over the next day; nobody else trades.
v0.3: a payment buys TXC; half is burned, half goes to the contributors (30% at once, the parents' 70% over four
epochs); at most half of the epoch's burn is minted back to them, vesting over four epochs, under a cap of 50,000 TXC
an epoch halving every 180 epochs forever. 6 decimals.
v0.4: a payment buys TXC and burns all of it, for credits. Each epoch mints at most the emission (50,000 TXC, halving
every 180 epochs `--halvings` times, then flat): 90% to the work that was paid for, 10% to operators by fees, each
never more than the credits were worth at the epoch's time-weighted price nor than the TXC they burned before the
spread; 30% of the work's share at once, the parents' 70% over four epochs. 18 decimals.
"""
import argparse

HOURS = 24
FEE = 0.003                      # the pool's spread
USE = 0.001                      # dollars per paid use; every use is one transaction
TX_FEE = 0.00005                 # v0.4's standard fee, in dollars
GENESIS_TXC, GENESIS_USD = 1_000_000.0, 10_000.0
EMISSION, HALVING = 50_000.0, 180
VEST = 4
IMMEDIATE = 0.30                 # trainer 25 + validators 5: paid at once; the parents' 60 + 10 vest
GROWTH_MARKS = (180, 360, 540, 720, 900, 1080, 1260, 1290, 1440, 1620, 1800)
YEAR_MARKS = (1, 7, 30, 90, 180, 181, 365)
TAKE_WINDOW = 30


class Pool:
    def __init__(self):
        self.x, self.y = GENESIS_USD, GENESIS_TXC

    @property
    def price(self):
        return self.x / self.y

    def buy(self, usd):
        """Dollars in; returns (TXC out, TXC before the spread)."""
        dx = usd * (1 - FEE)
        out, gross = self.y * dx / (self.x + dx), self.y * usd / (self.x + usd)
        self.x, self.y = self.x + usd, self.y - out
        return out, gross

    def sell(self, txc):
        dy = txc * (1 - FEE)
        out = self.x * dy / (self.y + dy)
        self.x, self.y = self.x - out, self.y + txc
        return out


def trade(pool, usd, txc):
    """One hour: buyers' dollars and sellers' TXC cross at the pool's price; the difference trades with the pool.
    Returns (TXC to the buyers, TXC they'd have had before the spread, dollars to the sellers)."""
    p = pool.price
    if usd >= txc * p:
        got, gross = pool.buy(usd - txc * p) if usd > txc * p else (0.0, 0.0)
        return txc + got, txc + gross, txc * p
    to_buyers = usd / p
    return to_buyers, to_buyers, usd + pool.sell(txc - to_buyers)


def emission(epoch, halvings):
    return EMISSION / 2 ** min((epoch - 1) // HALVING, halvings)


class Economy:
    def __init__(self, version, hold=0.0, halvings=5, top_up=False):
        """top_up (an option, NOT what the node does): the emission nobody earned is minted into the protocol-owned pool
        over the next day instead of not at all. Nobody receives it, so nobody can farm it, but it breaks v0.4's rule
        that unearned emission is never minted, and it dilutes holders."""
        self.v, self.hold, self.halvings, self.top_up = version, hold, halvings, top_up
        self.pool = Pool()
        self.supply, self.burned = GENESIS_TXC, 0.0
        self.vesting = []                          # [txc per epoch, epochs left]
        self.selling = 0.0                         # TXC contributors sell over the coming day
        self.pending = 0.0                         # top_up: unearned emission going into the pool over the coming day
        self.window = []                           # (dollars paid, dollars contributors realised or kept) by epoch
        self.last = {}

    def epoch(self, t, usd):
        fees = usd / USE * TX_FEE if self.v == 4 else 0.0
        pay = usd + fees
        sell_h, bought, gross, prices, realised = self.selling / HOURS, 0.0, 0.0, [], 0.0
        for _ in range(HOURS):
            got, g, cash = trade(self.pool, pay / HOURS, sell_h)
            bought, gross, realised = bought + got, gross + g, realised + cash
            if self.pending:
                self.pool.y += self.pending / HOURS
            prices.append(self.pool.price)
        self.pending = 0.0
        twap = sum(prices) / HOURS
        e = emission(t, self.halvings if self.v == 4 else 10 ** 6)
        if self.v == 3:
            burn, kept = bought / 2, bought / 2
            minted = min(e, burn / 2)                              # the match
            now, vest = kept * IMMEDIATE, [kept * (1 - IMMEDIATE) / VEST, minted / VEST]
        else:
            burn = bought
            work_cap = min(usd / twap, gross * usd / pay)          # credits at the time-weighted price; TXC they burned
            ops_cap = min(fees / twap, gross * fees / pay)
            work, ops = min(e * 0.9, work_cap), min(e * 0.1, ops_cap)
            now, vest = work * IMMEDIATE + ops, [work * (1 - IMMEDIATE) / VEST]
            minted = work + ops
            if self.top_up:
                self.pending, minted = e - minted, e
        self.burned += burn
        self.supply += minted - burn
        self.vesting += [[v, VEST] for v in vest if v]
        released = now
        for v in self.vesting:
            released += v[0]
            v[1] -= 1
        self.vesting = [v for v in self.vesting if v[1] > 0]
        self.selling = released * (1 - self.hold)
        self.window = (self.window + [(pay, realised + released * self.hold * twap)])[-TAKE_WINDOW:]
        self.last = {"usage": usd, "price": self.pool.price, "twap": twap, "supply": self.supply, "burned": burn,
                     "minted": minted, "pool_usd": self.pool.x, "emission": e,
                     "equilibrium": usd / (e * 0.9) if self.v == 4 else None,
                     # dollars contributors realised (sold, or kept at the time-weighted price) per dollar users
                     # paid, over the last TAKE_WINDOW epochs
                     "take": sum(r for _, r in self.window) / sum(p for p, _ in self.window)}
        return self.last


def run(schedule, hold=0.0, halvings=5, checkpoints=(), top_up=False):
    """schedule: usage per epoch (a list). Returns {epoch: (v0.3 row, v0.4 row)} at the checkpoints."""
    a, b = Economy(3, hold), Economy(4, hold, halvings, top_up)
    out, cum = {}, 0.0
    for t, usd in enumerate(schedule, start=1):
        cum += usd
        ra, rb = a.epoch(t, usd), b.epoch(t, usd)
        if t in checkpoints:
            out[t] = (dict(ra, cumulative=cum), dict(rb, cumulative=cum))
    return out


def growth_crash():
    """$10,000 a day for 180 epochs; ten times more every 180 epochs to $1B a day (epoch 1,080); held there for 180
    epochs; a crash to $100M a day for 180 epochs; back to $1B a day for 360. Halvings at epochs 181, 361, 541, 721, 901."""
    s = [1e4] * 180
    for t in range(900):
        s.append(1e4 * 10 ** ((t + 1) / 180))
    s += [1e9] * 180 + [1e8] * 180 + [1e9] * 360
    return s


def money(x):
    if x is None:
        return "-"
    for div, sign in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if x >= div:
            return f"${x / div:,.1f}{sign}"
    return f"${x:,.2f}" if x >= 1 else f"${x:.2g}"


def count(x):
    return f"{x / 1e6:,.2f}M" if x >= 1e5 else f"{x:,.0f}"


def table(title, rows):
    print()
    print(title)
    print(f"{'epoch':>5} {'usage/day':>9} {'all-time':>8} | {'v0.3 price':>10} {'supply':>7} {'pool $':>8} {'take':>4} "
          f"{'1 unit':>8} | {'v0.4 price':>10} {'P*':>8} {'supply':>7} {'pool $':>8} {'burn/mint TXC':>15} {'take':>4} "
          f"{'1 unit':>8}")
    for t, (a, b) in rows.items():
        print(f"{t:>5} {money(a['usage']):>9} {money(a['cumulative']):>8} | {money(a['price']):>10} {count(a['supply']):>7} "
              f"{money(a['pool_usd']):>8} {a['take']:>4.0%} {money(a['price'] * 1e-6):>8} | {money(b['price']):>10} "
              f"{money(b['equilibrium']):>8} {count(b['supply']):>7} {money(b['pool_usd']):>8} "
              f"{count(b['burned']):>7}/{count(b['minted']):<7} {b['take']:>4.0%} {money(b['price'] * 1e-18):>8}")


def sensitivity(halvings):
    """The price at a few moments, for contributors who sell everything or keep half, for v0.3, v0.4 as built, and
    v0.4 with the pool top-up option."""
    s, year = growth_crash(), [1e9] * 365
    print()
    print("4. If contributors keep half of what they earn, for good (price per TXC)")
    print(f"{'':42} {'v0.3':>9} {'v0.4':>9} {'v0.4+top-up':>12} {'P*':>9}")
    for hold in (0.0, 0.5):
        g, gt = run(s, hold, halvings, (1080, 1440, 1800)), run(s, hold, halvings, (1080, 1440, 1800), top_up=True)
        y, yt = run(year, hold, halvings, (365,)), run(year, hold, halvings, (365,), top_up=True)
        for name, r, rt, t in (("growth to $1B a day (epoch 1080)", g, gt, 1080),
                               ("after the crash (epoch 1440)", g, gt, 1440),
                               ("recovered (epoch 1800)", g, gt, 1800),
                               ("machine economy, day 365", y, yt, 365)):
            print(f"  keep {hold:>3.0%}  {name:32} {money(r[t][0]['price']):>9} {money(r[t][1]['price']):>9} "
                  f"{money(rt[t][1]['price']):>12} {money(r[t][1]['equilibrium']):>9}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hold", type=float, default=0.0, help="share of what contributors earn that they keep for good")
    ap.add_argument("--halvings", type=int, default=5, help="v0.4: halvings before the emission stays flat")
    ap.add_argument("--top-up", action="store_true", help="an option, not the node's rule: mint unearned emission "
                                                          "into the protocol-owned pool")
    ap.add_argument("--brief", action="store_true", help="skip the sensitivity table")
    a = ap.parse_args(argv)
    print(__doc__.split("\n\n")[0].replace("\n", " "))
    print(f"Contributors keep {a.hold:.0%} of what they earn; v0.4 halves {a.halvings} times, then stays flat"
          + ("; OPTION, not the node's rule: unearned emission is minted into the protocol-owned pool" if a.top_up else "")
          + ".")
    print("'take' = dollars contributors realised (sold, or kept at the time-weighted price) per dollar users paid, over "
          "the last 30 epochs; "
          "'1 unit' = the smallest TXC amount in dollars (v0.4 payments are whole credits, $0.000001); 'pool $' = users' "
          "dollars held in the pool; P* = usage / the work emission, where v0.4's burns and mints balance.")
    growth = run(growth_crash(), a.hold, a.halvings, GROWTH_MARKS, a.top_up)
    table("1. Growth from $10,000 to $1B a day, a 90% crash, recovery; halvings every 180 epochs", growth)
    year = run([1e9] * 365, a.hold, a.halvings, YEAR_MARKS, a.top_up)
    table("2. The machine economy from day one: 10 billion machines x 100 paid uses a day x $0.001 = $1B a day", year)
    life = {total: run([total / 1000] * 1000, a.hold, a.halvings, (1000,), a.top_up)[1000] for total in (1e6, 1e9)}
    print()
    print("3. Lifetime usage spread over 1,000 days (the v0.3 review's numbers)")
    for total, (r3, r4) in life.items():
        print(f"   {money(total)} in all: v0.3 {money(r3['price'])} a TXC, v0.4 {money(r4['price'])} "
              f"(P* {money(r4['equilibrium'])})")
    if not a.brief:
        sensitivity(a.halvings)
    return {"growth": growth, "machine": year, "lifetime": life}


if __name__ == "__main__":
    main()
