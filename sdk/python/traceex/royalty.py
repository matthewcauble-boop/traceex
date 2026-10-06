"""Who gets paid what (spec sections 3-4). All amounts are integers in the node's money (msats on a sats node; micros
on the retired v0.1 dollar node); rounding dust goes to the first payee so every split sums exactly to the amount
paid. Every payout is a split of a real payment: nothing here creates money."""
from collections import defaultdict

TRACE_SALE_SPLIT = {"producer": 0.85, "checker": 0.10, "validators": 0.05}
MAX_DEPTH = 32          # learnings nest at most this deep: nodes refuse deeper chains, and payouts never recurse further


def _apportion(amount, weights):
    """Split an integer amount by weights; returns {key: micros} summing exactly to amount."""
    total = sum(weights.values())
    if amount <= 0 or total <= 0:
        return {}
    out = {k: int(amount * w / total) for k, w in weights.items()}
    dust = amount - sum(out.values())
    if dust:
        first = max(weights, key=lambda k: (weights[k], k))
        out[first] += dust
    return {k: v for k, v in out.items() if v}


def split_trace_sale(amount, producer, checker_author, validators):
    """One trace's share of an auction payment."""
    parts = _apportion(amount, TRACE_SALE_SPLIT)
    out = defaultdict(int)
    out[producer] += parts.get("producer", 0)
    out[checker_author] += parts.get("checker", 0)
    for v, m in _apportion(parts.get("validators", 0), {v: 1 for v in validators}).items():
        out[v] += m
    return dict(out)


def split_usage(amount, learning, trace_info, validators, learning_info=None, _depth=0):
    """Royalties for metered use of a learning, propagated down the family tree.

    trace_info: {trace_id: {"producer": addr, "checker_author": addr}}
    learning_info: {learning_id: learning dict} for learnings that cite other learnings as parents.
    A part that would go deeper than MAX_DEPTH learnings is left out (the caller sees the shortfall).
    """
    split = learning["royalty"]["split"]
    parts = _apportion(amount, split)
    out = defaultdict(int)
    out[learning["trainer"]] += parts.get("trainer", 0)
    for v, m in _apportion(parts.get("validators", 0), {v: 1 for v in validators}).items():
        out[v] += m
    parents = {p["trace"]: p["weight"] for p in learning["parents"]}
    # the traces share is divided by contribution weight; the checkers share by the same weights, to each trace's checker
    for share, role in (("traces", "producer"), ("checkers", "checker_author")):
        for pid, m in _apportion(parts.get(share, 0), parents).items():
            if pid in trace_info:
                out[trace_info[pid][role]] += m
            elif learning_info and pid in learning_info and _depth < MAX_DEPTH:      # a parent learning: recurse
                for a, mm in split_usage(m, learning_info[pid], trace_info, validators, learning_info, _depth + 1).items():
                    out[a] += mm
    return dict(out)


def pro_rata(amount, weights):
    """Split an integer amount by integer (or float) weights; sums exactly to amount, dust to the largest weight. Used
    to refund a bounty's backers by what each pledged."""
    w = {k: v for k, v in weights.items() if v and v > 0}
    total = sum(w.values())
    if amount <= 0 or total <= 0:
        return {}
    if all(isinstance(v, int) for v in w.values()):
        out = {k: amount * v // total for k, v in w.items()}
    else:
        out = {k: int(amount * v / total) for k, v in w.items()}
    dust = amount - sum(out.values())
    if dust:
        out[max(w, key=lambda k: (w[k], k))] += dust
    return {k: m for k, m in out.items() if m}
