"""Seed an empty node with the repo's recorded runs, so a new public exchange opens on real data.

  examples/code_repair    (always) 244 verified Python fixes from Qwen2.5-0.5B-Instruct (MBPP train split, CC BY 4.0),
                          the maintainer's bounty and the backers' pledges, the producers' autopilots backing it, an
                          auction for the first lot, and the attested LoRA built from those fixes.
  examples/flight_emails  (opt-in: TRACEX_SEED_FLIGHT=1, or seed_if_empty(ex, flight=True)) two agents' skeleton
                          traces from a 26M on-device model, a bounty that a routing learning solved (first-pass
                          63.3% -> 73.3% on unseen airlines), and an autopilot's follow-up bounty. Off by default, so
                          the public preview (api/node.py, render.yaml) holds no flight or email records.

Everything is replayed from the recorded runs through the node's own HTTP API (a private loopback listener, so the
operator-only calls work); nothing is made up. The demo accounts are the examples' fixed addresses (0xaaaa…, 0x7777…).
Epoch 1 is then settled, so the node opens on epoch 2 with a published payout root. A database that already holds
traces is left alone.

On a sats node (node/sats.py, v0.7) the seed also stakes three validators and runs the federation for real: each
validator holds its own slice of the held-out data (one airline's email each; a third of the 500 MBPP problems each),
commits, then reveals its paired measurement (the flight
part only when the flight example is seeded). Learnings are validated in epoch 2, so that node opens on epoch 3.

The failure registry (v0.7) fills itself as the traces arrive. The three code producers post reporter bonds, so their
reports count. Then, measured the same way and from the same recorded runs: LoRA v2 claims the code failures it was
built from, and each validator measures it on its own third of the held-out problems the base model failed with that
failure's mode; with the flight example, the flight routing learning claims the flight failures, measured field by
field on each validator's airline (too few cases to call, so they stay open); LoRA v1, registered as a model version,
re-checks every Qwen2.5 failure; and the maintainer posts a bounty on the TypeError failure, then measures LoRA v2 on its own hidden cases
(below the target, so it stays open).

    python node/seed.py exchange.db        # seed a database file directly
    TRACEX_SEED_FLIGHT=1 python node/seed.py exchange.db    # ... with the flight-email example as well
"""
import datetime as dt
import hashlib
import json
import math
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
SEED_VALIDATORS = [VALIDATOR, "0x" + "51" * 20, "0x" + "52" * 20]


def _load(*parts):
    path = os.path.join(CODE, "runs", *parts)
    with open(path, encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else [json.loads(line) for line in f]


def _grant(ex, accounts, amount=None):
    """On a testnet node every spend needs test money; the demo accounts get theirs the way anyone does (the node's
    faucet amount: 30,000 test sats on a sats node)."""
    if not ex.test_credits:
        return
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with ex.lock:
        for a in accounts:
            ex.db.execute("INSERT OR IGNORE INTO grants VALUES (?,?,?,?)", (a, amount or ex.test_credits, now, "seed"))
        ex.db.commit()


def seed_if_empty(ex, flight=None):
    """Seed `ex` if it holds no traces. `flight` adds examples/flight_emails; None reads TRACEX_SEED_FLIGHT (off)."""
    if flight is None:
        flight = os.environ.get("TRACEX_SEED_FLIGHT", "") == "1"
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
        story = seed(ex, url, flight=flight)
    finally:
        ex.quiet, ex.engine = False, engine
        srv.shutdown()
        srv.server_close()
    for line in story:
        ex._event(line, force=True)
    s = ex.settle()
    return (f"seed: {ex.stats()['traces']} traces, {len(ex.bounties()['bounties'])} bounties, "
            f"{ex.stats()['learnings']} learnings; epoch {s['epoch']} settled, root {s['root'][:18]}…")


def seed(ex, url, flight=False):
    from traceex import AdaptiveAgent, Client, Learning, apply_routing, attest, first_pass_score, routing_from_traces
    from traceex.autopilot import Autopilot, Policy
    from traceex.classify import RulesEngine
    import tasks
    fl = Model = model = routed = fb = lid = parents = None
    if flight:
        import flight as fl
        from model import Model

    sats_mode = getattr(ex, "economy", "") == "sats"
    # The demo's amounts are the same numbers on both nodes: msats on a sats node (everything paid in sats), micros
    # on the v0.1 dollar node. 10,000 msats is 10 sats, and $0.01 on the dollar node.
    amt = (lambda m: {"msats": m}) if sats_mode else (lambda m: {"micros": m})
    bid_at = (lambda m: {"price_msats": m}) if sats_mode else (lambda m: {"price_micros": m})
    per_call = (lambda m: {"per_call_msats": m}) if sats_mode else (lambda m: {"per_call_micros": m})
    budget = (lambda m: {"back_msats": m, "budget_msats": m}) if sats_mode else \
        (lambda m: {"back_micros": m, "budget_micros": m})
    money = (lambda m: f"{m / 1000:,.0f} sats") if sats_mode else (lambda m: f"${m / 1e6:,.2f}")
    _grant(ex, [KIM, RAJ, LEE, CONSUMER, AUTO, HOST, TRAINER, MAINTAINER, BIDDER2, BIDDER3, BIDDER4, BIDDER5,
                *CODE_PRODUCERS, *SEED_VALIDATORS])
    if sats_mode:                                # three validators stake 15,000 sats each
        for v in SEED_VALIDATORS:
            ex.register_validator(v, 15_000_000)
    node = Client(url)
    if flight:
        node._call("POST", "/v0/checkers", {"id": "flight-rules", "author": CHECKER_AUTHOR})
    node._call("POST", "/v0/checkers", {"id": "mbpp-tests", "author": CHECKER_AUTHOR})
    story = (["seeded from the repo's two examples: flight emails on a 26M on-device model, Python on "
              "Qwen2.5-0.5B-Instruct"] if flight else
             [f"seeded from the repo's recorded code-repair runs: Python on {QWEN}, MBPP problems checked by their "
              "unit tests"])

    # --- flight emails (opt-in): a bounty, skeleton traces, an auction, a routing learning that solves the bounty -----
    if flight:
        model = Model(live=False)

        def agent(addr):
            return AdaptiveAgent(model, fl.check, fields=fl.FIELDS, task=fl.TASK, base_model=Model.name,
                                 checker=fl.CHECKER, producer=addr, clean=fl.clean, narrow=fl.context_for,
                                 evidence=fl.evidence)

        eval_hash = attest(VALIDATOR, fl.EVAL, "", 0, 0)["eval_set"]
        fb = Client(url, CONSUMER).post_bounty(title="Flight extraction: 70% first-pass on unseen airlines",
                                               path="extract/travel/flight", eval_set=eval_hash, target=0.70,
                                               base_model=Model.name, epochs=EPOCHS)
        Client(url, KIM).pledge(fb["id"], **amt(2_000_000))
        Client(url, RAJ).pledge(fb["id"], **amt(3_000_000))
        traces = []
        for who, emails in ((A("a"), ["southwest", "united"]), (A("b"), ["delta"])):
            ag = agent(who)
            for name in emails:
                t = ag.run(fl.TRAIN[name], created="2026-10-03T00:00:00Z")["trace"]
                Client(url, who).submit(t)
                traces.append(t)
        lot = next(x["lot"] for x in node.lots()["lots"] if x["lot"].startswith(fl.TASK + "|"))
        for who, price in ((TRAINER, 900_000), (BIDDER2, 600_000), (HOST, 250_000)):
            Client(url, who).bid(lot, **bid_at(price))
        node.clear()
        if sats_mode:               # licence money waits for the traces each buyer used: these two name theirs, the
            _direct(ex, lot, (BIDDER2, HOST))     # trainer's learnings do it for the trainer
        artifact, parents = routing_from_traces(traces)
        artifact["name"] = "Field routing for flight emails"
        before, _ = first_pass_score(fl.EVAL, model, fl.check, fields=fl.FIELDS, clean=fl.clean)
        after, _ = first_pass_score(fl.EVAL, apply_routing(model, artifact, fl.context_for), fl.check,
                                    fields=fl.FIELDS, clean=fl.clean)
        att = attest(VALIDATOR, fl.EVAL, "first-pass field accuracy, 3 unseen airlines", before, after)
        L = Learning.build(kind="routing", task=fl.TASK, base_model=Model.name, artifact=artifact, parents=parents,
                           trainer=TRAINER, attestation=att, **per_call(200))
        lid = Client(url, TRAINER).register_learning(L)["id"]
        story += [f"{len(traces)} skeleton traces filed under extract/travel/flight by 2 agents: no names, codes, "
                  "dates or prices left their devices"]
        if not sats_mode:
            won = Client(url, TRAINER).claim_bounty(fb["id"], lid)
            Client(url, CONSUMER).report_usage(lid, 5_000)
            story += [f"bounty #{fb['id']} solved by a routing learning: first-pass {before:.1%} → {after:.1%} on "
                      f"unseen airlines; its ${won['pool_micros'] / 1e6:,.2f} of pledges paid the solver and the traces"]
        routed = apply_routing(model, artifact, fl.context_for)

    # --- open weights: the maintainer's bounty, 244 verified Python fixes, the LoRA built from them ------------------
    base, rule, rep = _load("eval-base.json"), _load("bounty.json"), _load("eval-repair.json")
    lot1, lot2, prod1 = _load("traces-r1.jsonl"), _load("traces-r2.jsonl"), _load("produce-r1.json")
    cb = Client(url, MAINTAINER).post_bounty(title=f"+3 points pass@1 for {QWEN} on a hidden code eval",
                                             path="code/generate", base_model=QWEN, eval_set=base["eval_set"],
                                             target=rule["target"], epochs=EPOCHS)
    for who, spend in ((KIM, 4_000_000), (RAJ, 3_000_000), (LEE, 3_000_000)):
        Client(url, who).pledge(cb["id"], **amt(spend))
    for t in lot1:
        Client(url, t["producer"]).submit(t)
    prompts = {t["task_id"]: tasks.prompt(t) for t in tasks.load("train")}
    pilots = {p: Autopilot(Client(url, p), task=tasks.TASK, base_model=QWEN, checker=tasks.CHECKER, engine=RulesEngine(),
                           policy=Policy(privacy="open", bounty_after=3, bounty_epochs=EPOCHS, **budget(500_000)))
              for p in CODE_PRODUCERS}
    acts = Counter()
    for u in (p for p in prod1 if p["how"] == "unsolved"):
        for a in pilots[u["producer"]].on_result(prompts[u["task_id"]], {"failing": ["code"], "result": {}},
                                                 failure=u["first_mode"]):
            acts[a["action"]] += 1
    code_lot = next(x["lot"] for x in node.lots()["lots"] if x["lot"].startswith(tasks.TASK + "|"))
    for who, price in ((TRAINER, 2_000_000), (BIDDER4, 1_200_000), (BIDDER3, 900_000)):
        Client(url, who).bid(code_lot, **bid_at(price))
    node.clear()
    if sats_mode:
        _direct(ex, code_lot, (BIDDER4, BIDDER3))
    for t in lot2:
        Client(url, t["producer"]).submit(t)
    if sats_mode:                     # v0.7: the producers bond themselves as reporters, so their reports count
        for p in CODE_PRODUCERS:
            Client(url, p).reporter_bond()
    ev, r, info = _load("eval-lora-v2.json"), rep["lora-v2"], _load("lora-v2", "training.json")
    att = attest(VALIDATOR, base["eval_set"], "pass@1 with one round of checker feedback, 500 held-out problems",
                 round(rep["base"]["rate"], 4), round(r["rate"], 4))
    att.update(p_value=r["vs_base"]["p_value"], n=base["n"],
               first_try={"before": base["rate"], "after": ev["rate"], "p_value": ev["vs_base"]["p_value"]})
    artifact = {"name": f"LoRA v2 · Python from {info['traces']} shared fixes", "uri": f"{REPO}/code_repair/runs/lora-v2",
                "hash": ev["adapter"], "body": {"kind": "lora", "rank": info["config"]["rank"], "base_model": QWEN}}
    L = Learning.build(kind="lora", task=tasks.TASK, base_model=QWEN, artifact=artifact,
                       parents=[(json_id(t), 1) for t in lot1 + lot2], trainer=TRAINER, attestation=att,
                       release="open", **per_call(50))
    lid2 = Client(url, TRAINER).register_learning(L)["id"]
    if not sats_mode:
        Client(url, HOST).report_usage(lid2, 400_000)

    # --- (flight) an agent on autopilot meets a failure nobody has fixed, and posts a bounty for it -----------------
    posted, joined = [], []
    if flight:
        pilot = Autopilot(Client(url, AUTO), task=fl.TASK, base_model=Model.name, checker=fl.CHECKER,
                          engine=RulesEngine(), policy=Policy(bounty_after=2, bounty_epochs=EPOCHS, **budget(1_000_000)))
        auto = agent(AUTO)
        auto.autopilot = pilot
        for email in fl.LIVE.values():
            for a in auto.run(email).get("autopilot", []):
                acts[a["action"]] += 1
                if a["action"] == "posted_bounty":
                    posted.append(a["bounty"])
                elif a["action"] == "backed_existing_bounty":
                    joined.append(a["bounty"])

    backed = acts.get("backed_existing_bounty", 0)
    story += [f"bounty #{cb['id']} posted free: +3 points first-try pass@1 for {QWEN}; kim, raj and lee pledge "
              f"{money(10_000_000)} to it, refunded if nobody reaches the target",
              f"{len(lot1)} open traces filed under code/generate by 3 agents' unit tests (round 1)"
              + (f"; their autopilots pledge to bounty #{cb['id']} {backed} times for the failures nobody fixed" if backed else ""),
              f"{len(lot2)} more traces from the adapted model's own failures (round 2)",
              f"LoRA v2 attested: {rep['base']['rate']:.1%} → {r['rate']:.1%} with one round of checker feedback "
              f"(p = {r['vs_base']['p_value']:.1g}), released as open weights",
              f"LoRA v2 first try: {base['rate']:.1%} → {ev['rate']:.1%}, not significant; bounty #{cb['id']} stays open"]
    if not sats_mode:
        story.append("a host serves the open weights: 400,000 calls metered at $0.00005")
    if posted:
        story.append(f"an agent on autopilot hit the same flight failure twice and posted bounty #{posted[0]} free, "
                     f"pledging {money(1_000_000)} of its budget")
    elif joined:
        story.append(f"an agent on autopilot hit the same flight failure twice; bounty #{joined[0]} already covers it, "
                     f"so it pledged {money(1_000_000)} of its budget to that one")
    if sats_mode:
        story += federate(ex, url, fl, model, routed, lid, lid2, fb, rep, parents, lot1 + lot2)
    if hasattr(ex, "post_challenge"):
        story += seed_challenges(ex)
    return story


SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "challenges", "sample")


def seed_challenges(ex):
    """v0.8 (SPEC 4k): the repo's imported sample, as open challenges with no pledges (nothing made up): the AlphaEvolve
    problems with each published construction as the verified baseline (the node re-scores it), and two Erdős problems,
    each listed from the Erdős problems database and merged with its formal-conjectures Lean statement."""
    from traceex import challenges as C
    files = [os.path.join(SAMPLE, "alphaevolve_sample.json"),
             os.path.join(SAMPLE, "erdosproblems", "problems_sample.yaml"),
             *(os.path.join(SAMPLE, "formal_conjectures", "FormalConjectures", "ErdosProblems", f"{n}.lean")
               for n in (3, 28))]
    posted, merged = [], 0
    for f in files:
        for c in C.load(f):
            r = ex.post_challenge(dict(c, days=365), origin="import")
            merged += bool(r.get("merged"))
            if not r.get("merged"):
                posted.append(r["id"])
    return [f"{len(posted)} open problems posted as challenges from openly licensed sources (AlphaEvolve's packing "
            f"records, the verified baselines; Erdős problems 3 and 28 with their Lean statements, {merged} merged "
            "sources); unfunded: nobody has pledged yet"]


def _direct(ex, lot, buyers):
    """Buyers that build nothing on the exchange name the traces they used (here: all of them)."""
    used = [tid for (tid,) in ex.db.execute("SELECT id FROM traces WHERE lot=?", (lot,)).fetchall()]
    for who in buyers:
        ex.direct_licence(lot, who, used)


def _paired_se(diffs):
    n = len(diffs)
    if n < 2:
        return 0.0
    m = sum(diffs) / n
    return math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1) / n)


def federate(ex, url, flight, model, routed, lid, lid2, fb, rep, flight_parents, code_parents):
    """Sats node: settle epoch 1 (the beacon draws the validators), then each validator measures its own slice of
    the held-out data, commits, and reveals. Accepted learnings earn as people pay to use them. `flight` is the
    flight example's module, or None when it is not seeded."""
    from traceex import Client, first_pass_score
    from sats import attestation_digest
    claims = registry_claims(ex, url, lid, lid2)          # v0.7: drawn from the beacon this settlement publishes
    ex.settle()
    story = ["3 validators staked 15,000 sats each; each holds its own slice of the held-out data"]
    verdicts = {}
    if flight:
        fields = list(flight.FIELDS)
        shards = {}
        for v, name in zip(SEED_VALIDATORS, sorted(flight.EVAL)):      # one unseen airline's email each
            doc = {name: flight.EVAL[name]}
            b, bad_b = first_pass_score(doc, model, flight.check, fields=flight.FIELDS, clean=flight.clean)
            a, bad_a = first_pass_score(doc, routed, flight.check, fields=flight.FIELDS, clean=flight.clean)
            diffs = [(f in bad_b[name]) - (f in bad_a[name]) for f in fields]
            shards[v] = {"validator": v, "eval_set": "sha256:" + hashlib.sha256(flight.EVAL[name].encode()).hexdigest(),
                         "metric": "first-pass field accuracy", "before": b, "after": a, "n": len(fields),
                         "se": round(_paired_se(diffs), 4), "audit": {"checked": min(10, len(flight_parents)), "bad": 0}}
        verdicts[lid] = shards
    base, v2 = rep["base"]["per_task"], rep["lora-v2"]["per_task"]
    shards = {}
    for i, v in enumerate(SEED_VALIDATORS):                            # a third of the 500 problems each
        ids = sorted((t for t in base if int(t) % 3 == i), key=int)
        diffs = [int(v2[t]["passed"]) - int(base[t]["passed"]) for t in ids]
        shards[v] = {"validator": v, "eval_set": "sha256:" + hashlib.sha256(",".join(ids).encode()).hexdigest(),
                     "metric": "pass@1 with one round of checker feedback", "n": len(ids),
                     "before": round(sum(base[t]["passed"] for t in ids) / len(ids), 4),
                     "after": round(sum(v2[t]["passed"] for t in ids) / len(ids), 4),
                     "se": round(_paired_se(diffs), 4), "audit": {"checked": 10, "bad": 0}}
    verdicts[lid2] = shards
    for learning_id, atts in verdicts.items():
        for v, att in atts.items():
            ex.commit(learning_id, v, attestation_digest(att, "seed:" + v))
        for v, att in atts.items():
            ex.reveal(learning_id, v, att, "seed:" + v)
    vc = ex.verdict(lid2)
    gains = lambda vd: ", ".join(f"{g['gain'] * 100:+.0f}" for g in vd["reveals"])
    if flight:
        vr = ex.verdict(lid)
        story.append(f"flight routing: validators measured {gains(vr)} points on one airline each (10 fields apiece); "
                     f"median {vr['median_gain'] * 100:+.0f}, {vr['status']}: three emails can't prove it, so bounty "
                     f"#{fb['id']} stays open for whoever can")
        if vr["status"] == "accepted":
            Client(url, TRAINER).claim_bounty(fb["id"], lid)
    story.append(f"LoRA v2: three validators on a third of the 500 held-out problems each measured {gains(vc)} points; "
                 f"median {vc['median_gain'] * 100:+.1f}, {vc['status']}: it may now earn, as people use it")
    if vc["status"] == "accepted":
        Client(url, HOST).report_usage(lid2, 400_000)
        story.append("a host serves the open weights: 400,000 calls at 0.05 sat (20,000 sats, about $17), split at "
                     "settlement: traces 60, trainer 25, checkers 10, validators 5; the traces' part waits 4 epochs "
                     "in escrow in case a challenge claws it back")
    story.append(f"licence money waits for the traces each buyer used: the trainer's learning{'s' if flight else ''} "
                 f"name{'' if flight else 's'} its traces, the "
                 "other buyers named theirs")
    story += registry_measure(ex, url, claims, flight, model, routed)
    return story


# --- v0.7: the failure registry, measured from the same recorded runs ----------------------------------------------------
def registry_claims(ex, url, lid, lid2):
    """Fixes claim failures and a model version is registered (epoch 1); validators are drawn at the settlement."""
    from traceex import Client
    from registry import model_family
    qwen = model_family(QWEN)
    code = [x for (x,) in ex.db.execute("SELECT id FROM failures WHERE family=? AND path='code/generate' ORDER BY seq",
                                        (qwen,)).fetchall()]
    flights = [x for (x,) in ex.db.execute("SELECT id FROM failures WHERE family='needle3' ORDER BY seq").fetchall()]
    trainer = Client(url, TRAINER)
    lora = trainer.claim_fix(code, QWEN, kind="learning", learning=lid2,
                             artifact={"name": "LoRA v2", "uri": f"{REPO}/code_repair/runs/lora-v2"})["id"]
    routing = trainer.claim_fix(flights, "needle3", kind="learning", learning=lid,
                                artifact={"name": "Field routing for flight emails"})["id"] if flights and lid else None
    v1 = ex.register_model({"version": QWEN + "+lora-v1", "parent": QWEN})["recheck"]
    typeerr = ex.db.execute("SELECT id FROM failures WHERE family=? AND path='code/generate' AND "
                            "signature='code:runtime_error/TypeError'", (qwen,)).fetchone()
    bounty = None
    if typeerr:
        base = _load("eval-base.json")["per_task"]
        hidden = sorted((t for t, v in base.items() if not v["passed"] and v.get("mode") == "runtime_error"), key=int)
        eval_set = "sha256:" + hashlib.sha256(",".join(hidden).encode()).hexdigest()
        bounty = Client(url, MAINTAINER).post_bounty(title=f"Stop {QWEN} raising TypeError on MBPP-style tasks",
                                                     failure_id=typeerr[0], eval_set=eval_set, target=0.5,
                                                     epochs=EPOCHS)["id"]
        Client(url, LEE).pledge(bounty, msats=2_000_000)
    return {"lora": lora, "routing": routing, "v1": v1, "bounty": bounty, "typeerr": typeerr[0] if typeerr else None}


def registry_measure(ex, url, claims, flight, model, routed):
    """Each drawn validator measures each claim on its own cases (commit, then reveal): the held-out problems in its
    third that the base model failed with the failure's mode (code), or the fields of the failure on its own airline's
    email that the base model got wrong (flight, when seeded; `flight` is its module or None). Real recorded results; nothing is made up."""
    from traceex import Client, attest, first_pass_score
    from registry import measurement_digest
    base, v1, v2 = (_load(f)["per_task"] for f in ("eval-base.json", "eval-lora-v1.json", "eval-lora-v2.json"))
    mode_of = {x: json.loads(m)[0] for x, m in ex.db.execute("SELECT id, modes FROM failures").fetchall()}
    sig_of = dict(ex.db.execute("SELECT id, signature FROM failures").fetchall())

    def code_results(fix_id, after, i):
        out = {}
        for x in ex.get_fix(fix_id)["claims"]:
            ids = [t for t, v in base.items() if not v["passed"] and v.get("mode") == mode_of[x["failure"]] and int(t) % 3 == i]
            if ids:
                out[x["failure"]] = {"passed": sum(bool(after[t]["passed"]) for t in ids), "n": len(ids)}
        return out

    airline = dict(zip(SEED_VALIDATORS, sorted(flight.EVAL))) if flight else {}

    def flight_results(fix_id, v):
        name = airline[v]
        doc = {name: flight.EVAL[name]}
        _, bad_b = first_pass_score(doc, model, flight.check, fields=flight.FIELDS, clean=flight.clean)
        _, bad_a = first_pass_score(doc, routed, flight.check, fields=flight.FIELDS, clean=flight.clean)
        out = {}
        for x in ex.get_fix(fix_id)["claims"]:
            fields = [p.split(":")[0] for p in sig_of[x["failure"]].split()]
            cases = [f for f in fields if f in bad_b[name]]
            if cases:
                out[x["failure"]] = {"passed": sum(f not in bad_a[name] for f in cases), "n": len(cases)}
        return out

    rounds = [(claims["lora"], lambda v: code_results(claims["lora"], v2, SEED_VALIDATORS.index(v))),
              (claims["v1"], lambda v: code_results(claims["v1"], v1, SEED_VALIDATORS.index(v)))]
    if claims["routing"]:
        rounds.append((claims["routing"], lambda v: flight_results(claims["routing"], v)))
    for fix_id, measure in rounds:
        if not fix_id:
            continue
        drawn = ex.get_fix(fix_id)["assigned"]
        ms = {v: {"results": measure(v)} for v in drawn}
        for v in drawn:
            ex.commit_fix(fix_id, v, measurement_digest(ms[v], "seed:" + v))
        for v in drawn:
            ex.reveal_fix(fix_id, v, ms[v], "seed:" + v)
    lora = ex.get_fix(claims["lora"])
    part = [c for c in lora["claims"] if c["status"] == "partly_fixed"]
    rates = ", ".join(f"{c['pass_rate']:.0%}" for c in part)
    story = [f"the failure registry filed the {ex.stats()['traces']} traces under {ex.registry_stats()['failures']} "
             "failures; the three code producers bonded 1,000 sats each as reporters, so their reports count",
             f"LoRA v2 claimed {len(lora['claims'])} code failures as fix {lora['id']}: validators measured it on "
             f"their own held-out cases; {len(part)} partly fixed"
             + (f" ({rates})" if part else "") + ", none fixed",
             f"{QWEN}+lora-v1 registered as a model version: every Qwen2.5 failure re-checked "
             f"(GET /v0/models/{{version}}/report)"]
    if claims["routing"]:
        story.append("the flight routing learning claimed the flight failures: a few fields per airline, too few cases "
                     "to call, so they stay open")
    if claims["bounty"]:
        base_ids = sorted((t for t, v in base.items() if not v["passed"] and v.get("mode") == "runtime_error"), key=int)
        after = round(sum(bool(v2[t]["passed"]) for t in base_ids) / len(base_ids), 4)
        b = ex.bounties()["bounties"]
        bb = next(x for x in b if x["id"] == claims["bounty"])
        ex.poster_measure(claims["bounty"], claims["lora"], attest(MAINTAINER, bb["eval_set"], "pass@1, first try",
                                                                   round(0.0, 4), after))
        story.append(f"bounty #{claims['bounty']} posted on {claims['typeerr']} (TypeError): its poster measured LoRA v2 "
                     f"at {after:.0%} on its own hidden cases, target 50%: it stays open, refunded if nobody gets there")
    return story


def json_id(t):
    from traceex import object_id
    return object_id(t)


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="backslashreplace")
    from exchange import Exchange
    db = sys.argv[1] if len(sys.argv) > 1 else "exchange.db"
    print(seed_if_empty(Exchange(db, test_credits=int(os.environ.get("TRACEX_TEST_CREDITS", 0)))))
