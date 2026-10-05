"""How TXC behaves as usage grows, crashes and the emission halves, everything priced in sats: v0.3's rules (burn and
match) against v0.5 (msat credits, burn and mint), deterministic, no randomness.

    python examples/scaling/simulate.py                  # the tables below
    python examples/scaling/simulate.py --hold 0.5       # contributors keep half of what they earn, for good
    python examples/scaling/simulate.py --btc-usd 50000  # convert the dollar scenarios at another bitcoin price
    python examples/scaling/simulate.py --top-up         # an option the node does NOT implement (see Economy)

The model is the same for both rule sets:
  * one epoch a day; users pay U sats a day. The scenarios are stated in dollars (machines budget in dollars) and
    converted at --btc-usd (default $85,962 a bitcoin, Coinbase spot 2026-10-05): $1B a day is about 1.16 trillion sats
    a day. Every paid use is one transaction of $0.001 (about 1.16 sats), so the standard fee of 58 msats adds about
    5% (v0.5: in credits, burned; v0.3: to the operator, no TXC);
  * the protocol-owned pool from genesis is the only market: 1,000,000 TXC beside 10,000,000 sats (0.1 BTC), a 0.3%
    spread that stays in the pool (so it deepens with volume);
  * users' purchases and contributors' sales arrive spread over 24 hours; within an hour they cross at the pool's price
    first and only the difference trades with the pool, as on any exchange with an order book in front of its pool;
  * everyone paid in TXC sells `1 - hold` of what reaches them (hold = 0 by default: machines, model hosts and
    contributors are paid for work and pay their bills), spread over the next day; nobody else trades.
v0.3: a payment buys TXC; half is burned, half goes to the contributors (30% at once, the parents' 70% over four
epochs); at most half of the epoch's burn is minted back to them, vesting over four epochs, under a cap of 50,000 TXC
an epoch halving every 180 epochs forever.
v0.5 (v0.4's rules, valued in sats): a payment buys TXC and burns all of it, for credits. Each epoch mints at most the
emission (50,000 TXC, halving every 180 epochs `--halvings` times, then flat): 90% to the work that was paid for, 10% to
operators by fees, each never more than the credits were worth at the epoch's time-weighted price nor than the TXC
they burned before the spread; 30% of the work's share at once, the parents' 70% over four epochs. 18 decimals.
"""
import argparse

HOURS = 24
FEE = 0.003                      # the pool's spread
BTC_USD = 85_962                 # dollars a bitcoin: converts the dollar-stated scenarios into sats
SATS_PER_BTC = 100_000_000
USE_USD = 0.001                  # a paid use, in dollars (converted to sats at the bitcoin price)
TX_FEE_SATS = 0.058              # the standard fee: 58 msats
GENESIS_TXC, GENESIS_SATS = 1_000_000.0, 10_000_000.0
EMISSION, HALVING = 50_000.0, 180
VEST = 4
IMMEDIATE = 0.30                 # trainer 25 + validators 5: paid at once; the parents' 60 + 10 vest
GROWTH_MARKS = (180, 360, 540, 720, 900, 1080, 1260, 1290, 1440, 1620, 1800)
YEAR_MARKS = (1, 7, 30, 90, 180, 181, 365)
TAKE_WINDOW = 30


class Pool:
    def __init__(self):
        self.x, self.y = GENESIS_SATS, GENESIS_TXC

    @property
    def price(self):
        return self.x / self.y

    def buy(self, sats):
        """Sats in; returns (TXC out, TXC before the spread)."""
        dx = sats * (1 - FEE)
        out, gross = self.y * dx / (self.x + dx), self.y * sats / (self.x + sats)
        self.x, self.y = self.x + sats, self.y - out
        return out, gross

    def sell(self, txc):
        dy = txc * (1 - FEE)
        out = self.x * dy / (self.y + dy)
        self.x, self.y = self.x - out, self.y + txc
        return out


def trade(pool, sats, txc):
    """One hour: buyers' sats and sellers' TXC cross at the pool's price; the difference trades with the pool.
    Returns (TXC to the buyers, TXC they'd have had before the spread, sats to the sellers)."""
    p = pool.price
    if sats >= txc * p:
        got, gross = pool.buy(sats - txc * p) if sats > txc * p else (0.0, 0.0)
        return txc + got, txc + gross, txc * p
    to_buyers = sats / p
    return to_buyers, to_buyers, sats + pool.sell(txc - to_buyers)


def emission(epoch, halvings):
    return EMISSION / 2 ** min((epoch - 1) // HALVING, halvings)


class Economy:
    def __init__(self, version, hold=0.0, halvings=5, top_up=False, btc_usd=BTC_USD):
        """top_up (an option, NOT what the node does): the emission nobody earned is minted into the protocol-owned pool
        over the next day instead of not at all. Nobody receives it, so nobody can farm it, but it breaks the rule that
        unearned emission is never minted, and it dilutes holders."""
        self.v, self.hold, self.halvings, self.top_up = version, hold, halvings, top_up
        self.use = USE_USD / btc_usd * SATS_PER_BTC          # sats a paid use
        self.pool = Pool()
        self.supply, self.burned = GENESIS_TXC, 0.0
        self.vesting = []                          # [txc per epoch, epochs left]
        self.selling = 0.0                         # TXC contributors sell over the coming day
        self.pending = 0.0                         # top_up: unearned emission going into the pool over the coming day
        self.window = []                           # (sats paid, sats contributors realised or kept) by epoch
        self.last = {}

    def epoch(self, t, sats):
        fees = sats / self.use * TX_FEE_SATS if self.v == 5 else 0.0
        pay = sats + fees
        sell_h, bought, gross, prices, realised = self.selling / HOURS, 0.0, 0.0, [], 0.0
        for _ in range(HOURS):
            got, g, cash = trade(self.pool, pay / HOURS, sell_h)
            bought, gross, realised = bought + got, gross + g, realised + cash
            if self.pending:
                self.pool.y += self.pending / HOURS
            prices.append(self.pool.price)
        self.pending = 0.0
        twap = sum(prices) / HOURS
        e = emission(t, self.halvings if self.v == 5 else 10 ** 6)
        if self.v == 3:
            burn, kept = bought / 2, bought / 2
            minted = min(e, burn / 2)                              # the match
            now, vest = kept * IMMEDIATE, [kept * (1 - IMMEDIATE) / VEST, minted / VEST]
        else:
            burn = bought
            work_cap = min(sats / twap, gross * sats / pay)        # credits at the time-weighted price; TXC they burned
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
        self.last = {"usage": sats, "price": self.pool.price, "twap": twap, "supply": self.supply, "burned": burn,
                     "minted": minted, "pool_sats": self.pool.x, "emission": e,
                     "equilibrium": sats / (e * 0.9) if self.v == 5 else None,
                     # sats contributors realised (sold, or kept at the time-weighted price) per sat users paid, over
                     # the last TAKE_WINDOW epochs
                     "take": sum(r for _, r in self.window) / sum(p for p, _ in self.window)}
        return self.last


def run(schedule, hold=0.0, halvings=5, checkpoints=(), top_up=False, btc_usd=BTC_USD):
    """schedule: usage per epoch in sats (a list). Returns {epoch: (v0.3 row, v0.5 row)} at the checkpoints."""
    a, b = Economy(3, hold, btc_usd=btc_usd), Economy(5, hold, halvings, top_up, btc_usd=btc_usd)
    out, cum = {}, 0.0
    for t, sats in enumerate(schedule, start=1):
        cum += sats
        ra, rb = a.epoch(t, sats), b.epoch(t, sats)
        if t in checkpoints:
            out[t] = (dict(ra, cumulative=cum), dict(rb, cumulative=cum))
    return out


def to_sats(usd, btc_usd=BTC_USD):
    return usd / btc_usd * SATS_PER_BTC


def growth_crash(btc_usd=BTC_USD):
    """$10,000 a day (about 11.6M sats) for 180 epochs; ten times more every 180 epochs to $1B a day (about 1.16T
    sats, epoch 1,080); held there for 180 epochs; a crash to $100M a day for 180 epochs; back to $1B a day for 360.
    Halvings at epochs 181, 361, 541, 721, 901."""
    s = [1e4] * 180
    for t in range(900):
        s.append(1e4 * 10 ** ((t + 1) / 180))
    s += [1e9] * 180 + [1e8] * 180 + [1e9] * 360
    return [to_sats(u, btc_usd) for u in s]


def sats_str(x):
    """Sats for people: '1.2T sats', '827M sats', '10.0 sats', '0.0012 sats'."""
    if x is None:
        return "-"
    for div, sign in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if x >= div:
            return f"{x / div:,.1f}{sign}"
    return f"{x:,.1f}" if x >= 1 else f"{x:.2g}"


def usd_str(sats, btc_usd=BTC_USD):
    x = sats * btc_usd / SATS_PER_BTC
    for div, sign in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if x >= div:
            return f"${x / div:,.1f}{sign}"
    return f"${x:,.2f}" if x >= 1 else f"${x:.2g}"


def count(x):
    return f"{x / 1e6:,.2f}M" if x >= 1e5 else f"{x:,.0f}"


def table(title, rows, btc_usd=BTC_USD):
    print()
    print(title)
    print(f"{'epoch':>5} {'sats/day':>9} {'~$/day':>8} | {'v0.3 price':>10} {'supply':>7} {'pool':>7} {'take':>4} | "
          f"{'v0.5 price':>10} {'~$':>8} {'P*':>8} {'supply':>7} {'pool':>7} {'burn/mint TXC':>15} {'take':>4} "
          f"{'1 unit':>8}")
    for t, (a, b) in rows.items():
        print(f"{t:>5} {sats_str(a['usage']):>9} {usd_str(a['usage'], btc_usd):>8} | {sats_str(a['price']):>10} "
              f"{count(a['supply']):>7} {sats_str(a['pool_sats']):>7} {a['take']:>4.0%} | {sats_str(b['price']):>10} "
              f"{usd_str(b['price'], btc_usd):>8} {sats_str(b['equilibrium']):>8} {count(b['supply']):>7} "
              f"{sats_str(b['pool_sats']):>7} {count(b['burned']):>7}/{count(b['minted']):<7} {b['take']:>4.0%} "
              f"{sats_str(b['price'] * 1e-18):>8}")


def sensitivity(halvings, btc_usd):
    """The price at a few moments, for contributors who sell everything or keep half, for v0.3, v0.5 as built, and
    v0.5 with the pool top-up option."""
    s, year = growth_crash(btc_usd), [to_sats(1e9, btc_usd)] * 365
    print()
    print("4. If contributors keep half of what they earn, for good (sats per TXC)")
    print(f"{'':42} {'v0.3':>9} {'v0.5':>9} {'v0.5+top-up':>12} {'P*':>9}")
    for hold in (0.0, 0.5):
        g = run(s, hold, halvings, (1080, 1440, 1800), btc_usd=btc_usd)
        gt = run(s, hold, halvings, (1080, 1440, 1800), top_up=True, btc_usd=btc_usd)
        y, yt = run(year, hold, halvings, (365,), btc_usd=btc_usd), run(year, hold, halvings, (365,), top_up=True,
                                                                        btc_usd=btc_usd)
        for name, r, rt, t in (("growth to $1B a day (epoch 1080)", g, gt, 1080),
                               ("after the crash (epoch 1440)", g, gt, 1440),
                               ("recovered (epoch 1800)", g, gt, 1800),
                               ("machine economy, day 365", y, yt, 365)):
            print(f"  keep {hold:>3.0%}  {name:32} {sats_str(r[t][0]['price']):>9} {sats_str(r[t][1]['price']):>9} "
                  f"{sats_str(rt[t][1]['price']):>12} {sats_str(r[t][1]['equilibrium']):>9}")


def bitcoin_moves(halvings):
    """The machine economy's dollar demand ($1B a day) at three bitcoin prices: the sats it pays, what TXC settles at,
    and what the fixed 58-msat fee is worth in dollars."""
    print()
    print("5. The same $1B a day of machine demand if bitcoin is worth half, the same, or twice as much (day 365)")
    print(f"{'$ per BTC':>10} {'sats/day':>9} {'v0.5 price':>10} {'~$':>8} {'P*':>8} {'fee ~$':>10} {'fee x electricity':>18}")
    for btc in (BTC_USD // 2, BTC_USD, BTC_USD * 2):
        r = run([to_sats(1e9, btc)] * 365, 0.0, halvings, (365,), btc_usd=btc)[365][1]
        fee_usd = TX_FEE_SATS * btc / SATS_PER_BTC
        print(f"{btc:>10,} {sats_str(r['usage']):>9} {sats_str(r['price']):>10} {usd_str(r['price'], btc):>8} "
              f"{sats_str(r['equilibrium']):>8} {'$' + format(fee_usd, '.6f'):>10} {fee_usd / 4.0e-7:>17.0f}x")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hold", type=float, default=0.0, help="share of what contributors earn that they keep for good")
    ap.add_argument("--halvings", type=int, default=5, help="v0.5: halvings before the emission stays flat")
    ap.add_argument("--btc-usd", type=int, default=BTC_USD, help="dollars a bitcoin, to state the scenarios in sats")
    ap.add_argument("--top-up", action="store_true", help="an option, not the node's rule: mint unearned emission "
                                                          "into the protocol-owned pool")
    ap.add_argument("--brief", action="store_true", help="skip the sensitivity tables")
    a = ap.parse_args(argv)
    print(__doc__.split("\n\n")[0].replace("\n", " "))
    print(f"Contributors keep {a.hold:.0%} of what they earn; v0.5 halves {a.halvings} times, then stays flat; dollar "
          f"figures (~$) at ${a.btc_usd:,} a bitcoin, approximate"
          + ("; OPTION, not the node's rule: unearned emission is minted into the protocol-owned pool" if a.top_up else "")
          + ".")
    print("Prices are sats per TXC. 'take' = sats contributors realised (sold, or kept at the time-weighted price) per "
          "sat users paid, over the last 30 epochs; '1 unit' = the smallest TXC amount in sats (v0.5 payments are whole "
          "msats, 0.001 sat); 'pool' = users' sats held in the pool; P* = usage / the work emission, where v0.5's burns "
          "and mints balance.")
    growth = run(growth_crash(a.btc_usd), a.hold, a.halvings, GROWTH_MARKS, a.top_up, a.btc_usd)
    table("1. Growth from $10,000 to $1B a day (11.6M to 1.16T sats), a 90% crash, recovery; halvings every 180 epochs",
          growth, a.btc_usd)
    machine = to_sats(1e9, a.btc_usd)
    year = run([machine] * 365, a.hold, a.halvings, YEAR_MARKS, a.top_up, a.btc_usd)
    table(f"2. The machine economy from day one: 10 billion machines x 100 paid uses a day x $0.001 = $1B a day = "
          f"{machine:,.0f} sats a day ({machine / SATS_PER_BTC:,.0f} BTC)", year, a.btc_usd)
    life = {total: run([to_sats(total / 1000, a.btc_usd)] * 1000, a.hold, a.halvings, (1000,), a.top_up,
                       a.btc_usd)[1000] for total in (1e6, 1e9)}
    print()
    print("3. Lifetime usage spread over 1,000 days (the v0.3 review's numbers)")
    for total, (r3, r5) in life.items():
        print(f"   ${total:,.0f} in all ({sats_str(to_sats(total, a.btc_usd))} sats): v0.3 {sats_str(r3['price'])} sats "
              f"a TXC, v0.5 {sats_str(r5['price'])} (P* {sats_str(r5['equilibrium'])})")
    if not a.brief:
        sensitivity(a.halvings, a.btc_usd)
        bitcoin_moves(a.halvings)
    return {"growth": growth, "machine": year, "lifetime": life, "machine_sats_per_day": machine}


if __name__ == "__main__":
    main()
