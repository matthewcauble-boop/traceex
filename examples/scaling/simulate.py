"""Payouts scale linearly with paid usage, from a few dollars a day to a machine economy of $1B a day (v0.6, no token).

    python examples/scaling/simulate.py              # the table, on a real node (node/sats.py)
    python examples/scaling/simulate.py --btc-usd 120000

v0.5 needed a simulation of a token's price against its emission (and showed the coin was a pass-through: contributors
got back 99-100% of what users burned). v0.6 has nothing to simulate: every payout is a split of a real payment, so the
work a learning is built from earns exactly its share of what users pay, at every scale, with no price in between.
This runs the same learning (20 traces from two producers, one trainer, one checker author, three validators) on a
fresh node at five usage levels, one epoch of usage each, settles until every escrowed share has been paid, and prints
who got what. The last row is a machine economy at $1B a day: 10 billion machines x 100 paid uses x $0.001, converted at
--btc-usd. Fees are per transaction (58 msats to the operator that served it): a machine that batches its day's calls
into one usage report pays one fee a day; one that reports every call pays a fee per call.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node")]
from traceex import Trace, Learning, attest  # noqa: E402
from sats import SatsExchange, Params, attestation_digest, MSATS_PER_BTC, BTC_USD  # noqa: E402
from exchange import TX_FEE_MSATS  # noqa: E402

A = lambda c: "0x" + c * 40
V = lambda i: "0x" + f"{0xa0 + i:02x}" * 20
PRODUCERS, TRAINER, AUTHOR, USER = (A("a"), A("b")), A("c"), A("d"), A("e")
MACHINES, USES, USD_PER_USE = 10_000_000_000, 100, 0.001
SCALES_USD = (10, 10_000, 1_000_000, 100_000_000, 1_000_000_000)      # usage a day, in dollars


def one_day(usd_per_day, btc_usd):
    """A fresh node; one epoch of usage worth `usd_per_day`; settled until nothing is left in escrow."""
    msats = round(usd_per_day / btc_usd * MSATS_PER_BTC)
    ex = SatsExchange(":memory:", test_credits=30_000_000, beacon_delay=0, params=Params(quorum=3))
    for i in range(3):
        ex.faucet(V(i))
        ex.register_validator(V(i), 15_000_000)
    for who in (*PRODUCERS, TRAINER, AUTHOR):
        ex.faucet(who)
    ex.db.execute("INSERT INTO grants VALUES (?,?,?,?)", (USER, msats + 10 ** 6, "sim", "sim"))
    ex.register_checker("unit-tests", AUTHOR)
    parents = [ex.submit_trace(dict(Trace.from_fix(
        task="code.python", base_model="qwen", input=f"Write a function number {i} that adds two numbers.",
        model_output={"code": "bad"}, verified_output={"code": f"fixed {i}"}, checker="unit-tests@1",
        producer=PRODUCERS[i % 2], created="2026-10-05T00:00:00Z", privacy="open",
        failure_modes={"code": "wrong_answer"})))["id"] for i in range(20)]
    L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": "w", "hash": None},
                       parents=[(p, 1) for p in parents], trainer=TRAINER, per_call_msats=1,
                       attestation=attest(V(0), "sha256:c", "pass@1", .3, .4))
    lid = ex.register_learning(L)["id"]
    for v in ex.verdict(lid)["assigned"]:
        att = {"validator": v, "eval_set": "sha256:" + v[-4:], "metric": "pass@1", "before": .3, "after": .4, "n": 500,
               "se": .02, "audit": {"checked": 10, "bad": 0}}
        ex.commit(lid, v, attestation_digest(att, "s"))
        ex.__dict__.setdefault("atts", {})[v] = att
    for v, att in ex.atts.items():
        ex.reveal(lid, v, att, "s")
    for _ in range(ex.p.vest_epochs + 1):
        ex.settle()                                                     # the bond is home; the books start clean
    start = {w: ex.wallet(w)["balance_msats"] for w in (*PRODUCERS, TRAINER, AUTHOR, *[V(i) for i in range(3)])}
    e0 = ex.economy_stats()
    ex.usage({"learning": lid, "consumer": USER, "calls": msats})       # one batched report: a day's calls at 1 msat
    for _ in range(ex.p.vest_epochs + 1):
        ex.settle()
    e1 = ex.economy_stats()
    got = {w: ex.wallet(w)["balance_msats"] - start[w] for w in start}
    row = {"usd_per_day": usd_per_day, "paid_in_msats": e1["paid_in_msats"] - e0["paid_in_msats"],
           "paid_out_msats": e1["paid_out_msats"] - e0["paid_out_msats"],
           "refunded_msats": e1["refunded_msats"] - e0["refunded_msats"],
           "fees_msats": e1["fees_msats"] - e0["fees_msats"],
           "traces": sum(got[p] for p in PRODUCERS), "trainer": got[TRAINER], "checkers": got[AUTHOR],
           "validators": sum(got[V(i)] for i in range(3)), "balanced": ex.audit()["balanced"],
           "escrow_left": e1["escrow_msats"]["payments"]}
    ex.db.close()
    return row


def fmt(msats):
    sats = msats / 1000
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if sats >= div:
            return f"{sats / div:,.3f}{unit}"
    return f"{sats:,.0f}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    btc_usd = float(argv[argv.index("--btc-usd") + 1]) if "--btc-usd" in argv else BTC_USD
    brief = "--brief" in argv
    rows = [one_day(u, btc_usd) for u in SCALES_USD]
    machine_usd = MACHINES * USES * USD_PER_USE
    machine_sats = round(machine_usd / btc_usd * 1e8)
    if not brief:
        print(__doc__.split("\n\n")[1].strip(), "\n")
    print(f"{'usage a day':>16} {'paid in, sats':>15} {'traces 60%':>12} {'trainer 25%':>12} {'checkers 10%':>13} "
          f"{'validators 5%':>14} {'paid out / in':>14}")
    for r in rows:
        print(f"{'$' + format(r['usd_per_day'], ','):>16} {fmt(r['paid_in_msats']):>15} {fmt(r['traces']):>12} "
              f"{fmt(r['trainer']):>12} {fmt(r['checkers']):>13} {fmt(r['validators']):>14} "
              f"{r['paid_out_msats'] / r['paid_in_msats']:>14.9f}")
    per_call = MACHINES * USES * TX_FEE_MSATS // 1000
    batched = MACHINES * TX_FEE_MSATS // 1000
    print(f"\nthe machine economy: {MACHINES:,} machines x {USES} paid uses x ${USD_PER_USE} = ${machine_usd:,.0f} a day = "
          f"{machine_sats:,} sats a day ({machine_sats / 1e8:,.0f} BTC) at ${btc_usd:,.0f} a bitcoin")
    print(f"  paid out a day: traces {machine_sats * 60 // 100:,} sats, trainers {machine_sats * 25 // 100:,}, "
          f"checkers {machine_sats * 10 // 100:,}, validators {machine_sats * 5 // 100:,}")
    print(f"  operators' fees a day: {batched:,} sats with one batched report per machine a day, "
          f"{per_call:,} sats if every call were its own transaction")
    print("\nPayouts are the same share of what users pay at every scale (the ratio above; the remainder is a few msats "
          "of rounding refunded to the payer): nothing is minted, nothing has a price to move, and nothing is paid out "
          "that a payer didn't pay in.")
    return {"rows": rows, "machine_sats_per_day": machine_sats, "fees_batched_sats": batched,
            "fees_per_call_sats": per_call}


if __name__ == "__main__":
    main()
