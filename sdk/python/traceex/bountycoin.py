"""Bounty coins: posting a bounty is free and mints a coin on a linear bonding curve. Buying coins funds the bounty;
the price rises with supply, so early backers pay less. While the bounty is open, holders can sell back to the curve.
Coins transfer freely. When a learning claims the bounty, the pool pays the solver and the coin becomes a share of
the solution: HOLDER_CUT of every metered use of that learning is paid to whoever holds coins at settlement.

Price of the next coin at supply s:  p(s) = BASE + SLOPE * s     (micros of USDC per coin)
Cost of n coins from supply s:       BASE*n + SLOPE*(s*n + n*n/2)
The pool always equals the area under the curve up to the current supply, so selling everything back empties it
exactly. Same maths as contracts/BountyMarket.sol.
"""
import math

BASE = 10_000          # the first coin costs $0.01
SLOPE = 100            # each coin sold raises the price by $0.0001
HOLDER_CUT = 0.20      # share of a winning learning's metered revenue paid to coin holders, forever


def price(s):
    return BASE + SLOPE * s


def cost(s, n):
    """Micros to buy n coins starting at supply s."""
    return BASE * n + SLOPE * (s * n + n * n / 2)


def coins_for(s, micros):
    """Coins that `micros` buys starting at supply s (inverse of cost)."""
    b = BASE + SLOPE * s
    return (-b + math.sqrt(b * b + 2 * SLOPE * micros)) / SLOPE


def sell_value(s, n):
    """Micros returned for selling n coins back when supply is s."""
    return cost(s - n, n)


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
