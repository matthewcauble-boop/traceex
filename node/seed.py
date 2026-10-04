"""Seed an empty node with the repo's two worked examples, so a new public exchange opens on real data.

  examples/flight_emails  two agents' skeleton traces from a 26M on-device model, a bounty that a routing learning
                          solved (first-pass 63.3% -> 73.3% on unseen airlines), and an autopilot's follow-up bounty.
  examples/code_repair    244 verified Python fixes from Qwen2.5-0.5B-Instruct (MBPP train split, CC BY 4.0), the
                          maintainer's bounty and its backers, and the attested LoRA built from those fixes.

Everything is replayed from the recorded runs through the node's own HTTP API (a private loopback listener, so the
operator-only calls work); nothing is made up. The demo accounts are the examples' fixed addresses (0xaaaa…, 0x7777…).
Epoch 1 is then settled, so the node opens on epoch 2 with a published payout root. A database that already holds
traces is left alone.

    python node/seed.py exchange.db        # seed a database file directly
"""
import datetime as dt
import json
import os
import sys
import threading
from collections import Counter
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FLIGHT, CODE = os.path.join(ROOT, "examples", "flight_emails"), os.path.join(ROOT, "examples", "code_repair")
for p in (HERE, os.path.join(ROOT, "sdk", "python")):
    if p not in sys.path:
        sys.path.insert(0, p)
for p in (FLIGHT, CODE):                        # appended: the examples' module names (demo, tasks) never shadow others
    if p not in sys.path:
        sys.path.append(p)

A = lambda c: "0x" + c * 40
VALIDATOR, TRAINER, CHECKER_AUTHOR, MAINTAINER = A("5"), A("c"), A("f"), A("d")
KIM, RAJ, LEE = A("7"), A("8"), A("6")
CONSUMER, AUTO, HOST, BIDDER2, BIDDER3, BIDDER4, BIDDER5 = A("e"), A("4"), A("9"), A("3"), A("2"), A("1"), A("0")
CODE_PRODUCERS = [A("a"), A("b"), A("e")]
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS = 14                                     # seeded bounties stay open two weeks at one epoch a day
REPO = "https://github.com/matthewcauble-boop/traceex/tree/main/examples"


def _load(*parts):
    path = os.path.join(CODE, "runs", *parts)
    with open(path, encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else [json.loads(line) for line in f]


def _grant(ex, accounts, micros=25_000_000):
    """On a testnet node every spend needs test credits; the demo accounts get theirs the way anyone does."""
    if not ex.test_credits:
        return
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with ex.lock:
        for a in accounts:
            ex.db.execute("INSERT OR IGNORE INTO grants VALUES (?,?,?,?)", (a, micros, now, "seed"))
        ex.db.commit()


def seed_if_empty(ex):
    if ex.db.execute("SELECT COUNT(*) FROM traces").fetchone()[0]:
        return "seed: database already has traces; left alone"
    from exchange import make_handler
    from traceex.classify import RulesEngine
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ex))       # loopback only, never exposed
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    # The seed files with the keyword engine: offline, quick, the same on every boot. A node with a hosted engine
    # (Jev) re-files these traces in the background once it is serving.
    ex.quiet, engine, ex.engine = True, ex.engine, RulesEngine()
    try:
        story = seed(ex, url)
    finally:
        ex.quiet, ex.engine = False, engine
        srv.shutdown()
        srv.server_close()
    for line in story:
        ex._event(line, force=True)
    s = ex.settle()
    return (f"seed: {ex.stats()['traces']} traces, {len(ex.bounties()['bounties'])} bounties, "
            f"{ex.stats()['learnings']} learnings; epoch {s['epoch']} settled, root {s['root'][:18]}…")


def seed(ex, url):
    from traceex import AdaptiveAgent, Client, Learning, apply_routing, attest, first_pass_score, routing_from_traces
    from traceex.autopilot import Autopilot, Policy
    from traceex.classify import RulesEngine
    import flight
    from model import Model
    import tasks

    _grant(ex, [KIM, RAJ, LEE, CONSUMER, AUTO, HOST, TRAINER, MAINTAINER, BIDDER2, BIDDER3, BIDDER4, BIDDER5,
                *CODE_PRODUCERS])
    node = Client(url)
    node._call("POST", "/v0/checkers", {"id": "flight-rules", "author": CHECKER_AUTHOR})
    node._call("POST", "/v0/checkers", {"id": "mbpp-tests", "author": CHECKER_AUTHOR})
    story = ["seeded from the repo's two examples: flight emails on a 26M on-device model, Python on "
             "Qwen2.5-0.5B-Instruct"]

    # --- flight emails: a bounty, skeleton traces, an auction, a routing learning that solves the bounty -------------
    model = Model(live=False)

    def agent(addr):
        return AdaptiveAgent(model, flight.check, fields=flight.FIELDS, task=flight.TASK, base_model=Model.name,
                             checker=flight.CHECKER, producer=addr, clean=flight.clean, narrow=flight.context_for,
                             evidence=flight.evidence)

    eval_hash = attest(VALIDATOR, flight.EVAL, "", 0, 0)["eval_set"]
    fb = Client(url, CONSUMER).post_bounty(title="Flight extraction: 70% first-pass on unseen airlines",
                                           path="extract/travel/flight", eval_set=eval_hash, target=0.70,
                                           base_model=Model.name, epochs=EPOCHS)
    k = Client(url, KIM).buy_coins(fb["id"], 2_000_000)
    Client(url, RAJ).buy_coins(fb["id"], 3_000_000)
    Client(url, KIM).transfer_coins(fb["id"], LEE, k["coins"] / 2)
    traces = []
    for who, emails in ((A("a"), ["southwest", "united"]), (A("b"), ["delta"])):
        ag = agent(who)
        for name in emails:
            t = ag.run(flight.TRAIN[name], created="2026-10-03T00:00:00Z")["trace"]
            Client(url, who).submit(t)
            traces.append(t)
    lot = next(x["lot"] for x in node.lots()["lots"] if x["lot"].startswith(flight.TASK + "|"))
    for who, price in ((TRAINER, 900_000), (BIDDER2, 600_000), (HOST, 250_000)):
        Client(url, who).bid(lot, price)
    node.clear()
    artifact, parents = routing_from_traces(traces)
    artifact["name"] = "Field routing for flight emails"
    before, _ = first_pass_score(flight.EVAL, model, flight.check, fields=flight.FIELDS, clean=flight.clean)
    after, _ = first_pass_score(flight.EVAL, apply_routing(model, artifact, flight.context_for), flight.check,
                                fields=flight.FIELDS, clean=flight.clean)
    att = attest(VALIDATOR, flight.EVAL, "first-pass field accuracy, 3 unseen airlines", before, after)
    L = Learning.build(kind="routing", task=flight.TASK, base_model=Model.name, artifact=artifact, parents=parents,
                       trainer=TRAINER, attestation=att, per_call_micros=200)
    lid = Client(url, TRAINER).register_learning(L)["id"]
    won = Client(url, TRAINER).claim_bounty(fb["id"], lid)
    Client(url, CONSUMER).report_usage(lid, 5_000)
    story += [f"{len(traces)} skeleton traces filed under extract/travel/flight by 2 agents: no names, codes, dates "
              "or prices left their devices",
              f"bounty #{fb['id']} solved by a routing learning: first-pass {before:.1%} → {after:.1%} on unseen "
              f"airlines; pool ${won['pool_micros'] / 1e6:,.2f} paid, coin holders now earn 20% of every use"]

    # --- open weights: the maintainer's bounty, 244 verified Python fixes, the LoRA built from them ------------------
    base, rule, rep = _load("eval-base.json"), _load("bounty.json"), _load("eval-repair.json")
    lot1, lot2, prod1 = _load("traces-r1.jsonl"), _load("traces-r2.jsonl"), _load("produce-r1.json")
    cb = Client(url, MAINTAINER).post_bounty(title=f"+3 points pass@1 for {QWEN} on a hidden code eval",
                                             path="code/generate", base_model=QWEN, eval_set=base["eval_set"],
                                             target=rule["target"], epochs=EPOCHS)
    for who, spend in ((KIM, 4_000_000), (RAJ, 3_000_000), (LEE, 3_000_000)):
        Client(url, who).buy_coins(cb["id"], spend)
    for t in lot1:
        Client(url, t["producer"]).submit(t)
    prompts = {t["task_id"]: tasks.prompt(t) for t in tasks.load("train")}
    pilots = {p: Autopilot(Client(url, p), task=tasks.TASK, base_model=QWEN, checker=tasks.CHECKER, engine=RulesEngine(),
                           policy=Policy(privacy="open", bounty_after=3, back_micros=500_000, budget_micros=500_000,
                                         bounty_epochs=EPOCHS))
              for p in CODE_PRODUCERS}
    acts = Counter()
    for u in (p for p in prod1 if p["how"] == "unsolved"):
        for a in pilots[u["producer"]].on_result(prompts[u["task_id"]], {"failing": ["code"], "result": {}},
                                                 failure=u["first_mode"]):
            acts[a["action"]] += 1
    code_lot = next(x["lot"] for x in node.lots()["lots"] if x["lot"].startswith(tasks.TASK + "|"))
    for who, price in ((TRAINER, 2_000_000), (BIDDER4, 1_200_000), (BIDDER3, 900_000)):
        Client(url, who).bid(code_lot, price)
    node.clear()
    for t in lot2:
        Client(url, t["producer"]).submit(t)
    ev, r, info = _load("eval-lora-v2.json"), rep["lora-v2"], _load("lora-v2", "training.json")
    att = attest(VALIDATOR, base["eval_set"], "pass@1 with one round of checker feedback, 500 held-out problems",
                 round(rep["base"]["rate"], 4), round(r["rate"], 4))
    att.update(p_value=r["vs_base"]["p_value"], n=base["n"],
               first_try={"before": base["rate"], "after": ev["rate"], "p_value": ev["vs_base"]["p_value"]})
    artifact = {"name": f"LoRA v2 · Python from {info['traces']} shared fixes", "uri": f"{REPO}/code_repair/runs/lora-v2",
                "hash": ev["adapter"], "body": {"kind": "lora", "rank": info["config"]["rank"], "base_model": QWEN}}
    L = Learning.build(kind="lora", task=tasks.TASK, base_model=QWEN, artifact=artifact,
                       parents=[(json_id(t), 1) for t in lot1 + lot2], trainer=TRAINER, attestation=att,
                       per_call_micros=50, release="open")
    lid2 = Client(url, TRAINER).register_learning(L)["id"]
    Client(url, HOST).report_usage(lid2, 400_000)

    # --- an agent on autopilot meets a failure nobody has fixed, and posts a bounty for it ---------------------------
    pilot = Autopilot(Client(url, AUTO), task=flight.TASK, base_model=Model.name, checker=flight.CHECKER,
                      engine=RulesEngine(), policy=Policy(bounty_after=2, back_micros=1_000_000, budget_micros=1_000_000, bounty_epochs=EPOCHS))
    auto = agent(AUTO)
    auto.autopilot = pilot
    posted = []
    for email in flight.LIVE.values():
        for a in auto.run(email).get("autopilot", []):
            acts[a["action"]] += 1
            if a["action"] == "posted_bounty":
                posted.append(a["bounty"])

    backed = acts.get("backed_existing_bounty", 0)
    story += [f"bounty #{cb['id']} posted free: +3 points first-try pass@1 for {QWEN}; kim, raj and lee back it with $10",
              f"{len(lot1)} open traces filed under code/generate by 3 agents' unit tests (round 1)"
              + (f"; their autopilots back bounty #{cb['id']} {backed} times for the failures nobody fixed" if backed else ""),
              f"{len(lot2)} more traces from the adapted model's own failures (round 2)",
              f"LoRA v2 attested: {rep['base']['rate']:.1%} → {r['rate']:.1%} with one round of checker feedback "
              f"(p = {r['vs_base']['p_value']:.1g}), released as open weights",
              f"LoRA v2 first try: {base['rate']:.1%} → {ev['rate']:.1%}, not significant; bounty #{cb['id']} stays open",
              "a host serves the open weights: 400,000 calls metered at $0.00005"]
    if posted:
        story.append(f"an agent on autopilot hit the same flight failure twice and posted bounty #{posted[0]} free, "
                     "backed with $1 of its budget")
    return story


def json_id(t):
    from traceex import object_id
    return object_id(t)


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="backslashreplace")
    from exchange import Exchange
    db = sys.argv[1] if len(sys.argv) > 1 else "exchange.db"
    print(seed_if_empty(Exchange(db, test_credits=int(os.environ.get("TRACEX_TEST_CREDITS", 0)))))
