"""What one transaction costs in electricity on the reference node, and so the standard transaction fee.

    python examples/fees/measure.py

Runs each kind of transaction many times on a real coin node (an on-disk SQLite database, as hosted) and measures the
CPU time it takes, the bytes it moves over the network and the bytes it leaves on disk. Then it prices the energy:

    energy = CPU seconds x watts per busy core x PUE
           + bytes moved x network energy per GB
           + bytes stored x copies kept x storage watts per GB x time kept x PUE
    cost   = energy in kWh x electricity price

Time kept follows the node's storage rule (SPEC 4e): traces and learnings are kept for good (priced here as ten years);
everything else is per-transaction detail, kept only through the challenge window (`keep_epochs`, a day each on the
hosted node) and then folded into balances and one Merkle root per epoch. The assumptions are printed with the result:
change them and run it again. The standard fee (exchange.TX_FEE_NANOS) is set at about 125 times the dearest
transaction here, so it funds the network and prices out spam at machine scale, not just the electricity.
"""
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node")]
from traceex import Trace, Learning, attest  # noqa: E402
from coin import CoinExchange, Params, UNIT, attestation_digest  # noqa: E402
from exchange import TX_FEE_NANOS  # noqa: E402

KWH_USD = 0.15              # $ per kWh: the US commercial average is about $0.13-0.14 (EIA); data centres often pay less
WATTS_PER_CORE = 10         # a busy server core with its share of memory, board and fans
PUE = 1.4                   # data-centre overhead: cooling and power conversion on top of the servers themselves
NET_KWH_PER_GB = 0.02       # moving data across the internet; published estimates run from about 0.006 to 0.06
STORE_WATTS_PER_GB = 0.0012  # an enterprise SSD draws about 5 W for 4 TB
COPIES = 3                  # every byte kept in three copies...
FOR_GOOD_YEARS = 10         # ...traces and learnings priced for ten years ("for good")...
WINDOW_DAYS = Params().keep_epochs   # ...everything else through the challenge window, one day an epoch
HTTP_BYTES = 800            # request and response headers, both ways
N = 300                     # repetitions of each transaction

J_PER_KWH = 3.6e6
A = lambda c: "0x" + c * 40


def db_bytes(ex):
    size, = ex.db.execute("PRAGMA page_count").fetchone()
    page, = ex.db.execute("PRAGMA page_size").fetchone()
    free, = ex.db.execute("PRAGMA freelist_count").fetchone()
    return (size - free) * page


def node(path):
    ex = CoinExchange(path, test_credits=25_000_000, beacon_delay=0, params=Params(quorum=3))
    for who in [A(c) for c in "0123456789abcdef"]:
        ex.db.execute("INSERT OR REPLACE INTO grants VALUES (?,?,?,?)", (who, 10**12, "measure", "measure"))
    vals = [A(c) for c in "abc"]
    for v in vals:
        ex.swap(v, "buy", 20_000_000)
        ex.register_validator(v, 1_500 * UNIT)
    ex.register_checker("unit-tests", A("1"))
    ex.db.commit()
    return ex, vals


def trace(i):
    return Trace.from_fix(task="code.python", base_model="Qwen/Qwen2.5-0.5B-Instruct",
                          input=f"Write a function to find the {i}th triangular number.\nassert tri({i}) == {i * (i + 1) // 2}",
                          model_output={"code": f"def tri(n):\n    return n * n // 2  # {i}"},
                          verified_output={"code": f"def tri(n):\n    return n * (n + 1) // 2  # {i}"},
                          checker="unit-tests@1", producer=A("1"), created="2026-10-04T00:00:00Z", privacy="open",
                          failure_modes={"code": "wrong_answer"})


def measure(ex, label, make, run, kept="window"):
    """Each transaction: CPU seconds, bytes over the wire, bytes left on disk, and how long they are kept."""
    bodies = [make(i) for i in range(N)]
    disk0, cpu0, wire = db_bytes(ex), time.process_time(), 0
    for body in bodies:
        out = run(body)
        ex.db.commit()
        wire += len(json.dumps(body, default=str)) + len(json.dumps(out, default=str)) + HTTP_BYTES
    cpu = (time.process_time() - cpu0) / N
    return {"transaction": label, "cpu_s": cpu, "wire_bytes": wire / N, "disk_bytes": max(db_bytes(ex) - disk0, 0) / N,
            "kept": kept}


def price(row):
    seconds = FOR_GOOD_YEARS * 365.25 * 86400 if row["kept"] == "for good" else WINDOW_DAYS * 86400
    cpu_j = row["cpu_s"] * WATTS_PER_CORE * PUE
    net_j = row["wire_bytes"] / 1e9 * NET_KWH_PER_GB * J_PER_KWH
    disk_j = row["disk_bytes"] / 1e9 * COPIES * STORE_WATTS_PER_GB * seconds * PUE
    joules = cpu_j + net_j + disk_j
    return dict(row, joules=joules, nanos=joules / J_PER_KWH * KWH_USD * 1e9,
                parts={"cpu": cpu_j / joules, "network": net_j / joules, "storage": disk_j / joules})


def main():
    folder = tempfile.mkdtemp(prefix="tracex-fees-")
    try:
        return _main(folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)            # leave nothing behind


def _main(folder):
    ex, vals = node(os.path.join(folder, "exchange.db"))
    rows, ids = [], []

    def submit(t):
        r = ex.submit_trace(t)
        ids.append(r["id"])
        return r
    rows.append(measure(ex, "file a trace (about 1 KB)", lambda i: dict(trace(i)), submit, kept="for good"))
    rows.append(measure(ex, "swap dollars for TXC", lambda i: ("buy", 10_000), lambda b: ex.swap(A("2"), *b)))
    rows.append(measure(ex, "buy credits (TXC burned)", lambda i: 10_000, lambda m: ex.buy_credits(A("7"), micros=m)))
    bounty = ex.post_bounty({"poster": A("3"), "path": "code/generate", "eval_set": "sha256:x", "target": 0.5,
                             "title": "measure"})["id"]
    ex.swap(A("3"), "buy", 100_000_000)
    rows.append(measure(ex, "back a bounty", lambda i: UNIT, lambda u: ex.buy_coins(bounty, A("3"), units=u)))
    lot = ex.lots()["lots"][0]["lot"]
    rows.append(measure(ex, "bid on a lot", lambda i: {"lot": lot, "bidder": A("4"), "price_micros": 1_000 + i},
                        ex.bid))
    ex.db.execute("DELETE FROM bids")
    learning_ids = []

    def register(L):
        r = ex.register_learning(L)
        learning_ids.append(r["id"])
        return r
    ex.swap(A("5"), "buy", 3_000_000_000)
    rows.append(measure(ex, "register a learning (bonded)",
                        lambda i: Learning.build(kind="lora", task="code.python", base_model="Qwen/Qwen2.5-0.5B-Instruct",
                                                 artifact={"uri": f"weights-{i}", "hash": None},
                                                 parents=[(t, 1) for t in ids[i % 200:i % 200 + 20]], trainer=A("5"),
                                                 attestation=attest(vals[0], "sha256:c", "pass@1", .3, .4),
                                                 per_call_micros=50),
                        register, kept="for good"))

    def commit(lid):
        v = ex.verdict(lid)["assigned"][0]
        att = {"validator": v, "eval_set": "sha256:p", "metric": "pass@1", "before": .3, "after": .4, "n": 500,
               "se": .02, "audit": {"checked": 10, "bad": 0}}
        return ex.commit(lid, v, attestation_digest(att, "salt"))
    rows.append(measure(ex, "a validator's commitment", lambda i: learning_ids[i], commit))
    # Paying for usage needs accepted learnings: this harness accepts them directly (validation is measured above).
    ex.db.execute("UPDATE verdicts SET status='accepted', accepted=1")
    rows.append(measure(ex, "pay for metered usage", lambda i: {"learning": learning_ids[i], "consumer": A("6"),
                                                                  "calls": 1_000}, ex.usage))
    rows = [price(r) for r in rows]
    print(__doc__.split("\n\n")[0], "\n")
    print(f"assumptions: ${KWH_USD}/kWh, {WATTS_PER_CORE} W a busy core, PUE {PUE}, {NET_KWH_PER_GB} kWh/GB moved, "
          f"{STORE_WATTS_PER_GB} W/GB stored, {COPIES} copies; traces and learnings kept {FOR_GOOD_YEARS} years, "
          f"everything else {WINDOW_DAYS} days (the challenge window); {N} runs each\n")
    print(f"{'transaction':32} {'kept':>8} {'CPU':>8} {'wire':>8} {'disk':>8} {'energy':>9} {'cost':>13}   where it goes")
    for r in rows:
        parts = ", ".join(f"{k} {v:.0%}" for k, v in sorted(r["parts"].items(), key=lambda kv: -kv[1]) if v >= 0.01)
        print(f"{r['transaction']:32} {r['kept']:>8} {r['cpu_s'] * 1e3:6.2f}ms {r['wire_bytes'] / 1e3:6.1f}KB "
              f"{r['disk_bytes'] / 1e3:6.1f}KB {r['joules']:7.2f} J  ${r['nanos'] / 1e9:.9f}   {parts}")
    top, low = max(r["nanos"] for r in rows), min(r["nanos"] for r in rows)
    print(f"\ndearest: {top:.0f} nano-dollars (${top / 1e9:.9f}); cheapest: {low:.0f}")
    print(f"the standard fee: {TX_FEE_NANOS:,} nano-dollars (${TX_FEE_NANOS / 1e9:.5f}), {TX_FEE_NANOS / top:,.0f}x the "
          f"dearest transaction's electricity and {TX_FEE_NANOS / low:,.0f}x the cheapest's")
    ex.db.close()
    return rows


if __name__ == "__main__":
    main()
