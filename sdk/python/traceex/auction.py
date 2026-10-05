"""Sealed-bid batch auctions (spec section 3). Prices are integers in the node's money (msats on a
coin node, where everything is priced in sats; micro-dollars on the retired v0.1 dollar node).

Shared (non-rival) licences: a uniform-price k-unit Vickrey auction. The top k bids at or above the reserve win and
each pays the (k+1)-th highest bid, or the reserve if there is no (k+1)-th. For unit demand, bidding your true value
is a dominant strategy. Exclusive windows: k = 1, which is a plain second-price auction.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Bid:
    bidder: str
    price_micros: int


def _rank(bids):
    # highest first; ties broken by bidder id so clearing is deterministic and anyone can recompute it
    return sorted(bids, key=lambda b: (-b.price_micros, b.bidder))


def clear_shared(bids, k, reserve_micros=0):
    """Returns (winners, clearing_price_micros). One bid per bidder (the highest) counts."""
    best = {}
    for b in bids:
        if b.price_micros >= reserve_micros and b.price_micros > best.get(b.bidder, Bid(b.bidder, -1)).price_micros:
            best[b.bidder] = b
    ranked = _rank(best.values())
    winners = ranked[:k]
    if not winners:
        return [], None
    price = ranked[k].price_micros if len(ranked) > k else reserve_micros
    return [w.bidder for w in winners], max(price, reserve_micros)


def clear_exclusive(bids, reserve_micros=0):
    winners, price = clear_shared(bids, 1, reserve_micros)
    return (winners[0] if winners else None), price
