"""Bounty coins: posting a bounty is free and mints a coin on a linear bonding curve. Buying coins funds the bounty;
the price rises with supply, so early backers get more coins for their money. While the bounty is open, holders can
sell coins back for what they paid (pro rata), never more: a profit there could only come out of later backers'
money. If it expires unsolved, the pool goes back to the backers by what each put in. Coins transfer freely. When a
learning claims the bounty, the pool pays the solver and the coin becomes a share of the solution: HOLDER_CUT of
every metered use of that learning is paid to whoever holds coins at settlement. That share is the early backers'
reward.

Price of the next coin at supply s:  p(s) = BASE + SLOPE * s     (micros of USDC per coin; a coin-economy node
                                                               prices the curve in its network coin instead)
Cost of n coins from supply s:       BASE*n + SLOPE*(s*n + n*n/2)
Same maths as contracts/BountyMarket.sol.
"""
import math

BASE = 10_000          # the first coin costs $0.01
SLOPE = 100            # each coin sold raises the price by $0.0001
HOLDER_CUT = 0.20      # share of a winning learning's metered revenue paid to coin holders, forever


def price(s, base=BASE, slope=SLOPE):
    return base + slope * s


def cost(s, n, base=BASE, slope=SLOPE):
    """Amount to buy n coins starting at supply s (micros of USDC, or coin units on a coin-economy node)."""
    return base * n + slope * (s * n + n * n / 2)


def coins_for(s, amount, base=BASE, slope=SLOPE):
    """Coins that `amount` buys starting at supply s (inverse of cost)."""
    b = base + slope * s
    return (-b + math.sqrt(b * b + 2 * slope * amount)) / slope


def sell_value(s, n, base=BASE, slope=SLOPE):
    """Amount returned for selling n coins back when supply is s."""
    return cost(s - n, n, base, slope)


def pro_rata(amount, holdings):
    """Split an integer amount across {holder: coins}; sums exactly to amount (dust to the largest holder)."""
    total = sum(v for v in holdings.values() if v > 0)
    if amount <= 0 or total <= 0:
        return {}
    out = {h: int(amount * v / total) for h, v in holdings.items() if v > 0}
    dust = amount - sum(out.values())
    if dust:
        out[max(out, key=lambda h: (holdings[h], h))] += dust
    return {h: m for h, m in out.items() if m}
