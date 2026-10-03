"""The whole loop, end to end, against a real exchange node over HTTP.

  1. Two agents run a 26M on-device model (Needle) on their own flight emails. The checker catches its mistakes, the
     loop fixes them, and each fix becomes a skeleton trace: no names, codes, dates or prices leave the device.
  2. They submit the traces. Two trainers bid in a sealed-bid batch auction; producers are paid 85/10/5.
  3. A trainer turns the traces into a learning. A validator measures it on emails nobody trained on and attests
     before/after. The node only accepts a learning that made things better.
  4. A third agent that never saw any of these emails adopts the learning and gets better. Its usage is metered.
  5. Settlement pays royalties back down the family tree and publishes one Merkle root; anyone claims with a proof.

    python examples/flight_emails/demo.py         # replays recorded Needle output; live if cactus-needle is installed
"""
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "sdk", "python"), os.path.join(HERE, "..", "..", "node")]
from traceex import AdaptiveAgent, Client, Learning, routing_from_traces, first_pass_score, attest, apply_routing, merkle
from traceex.autopilot import Autopilot, Policy
from exchange import serve
import flight
from model import Model

A = lambda c: "0x" + c * 40
PRODUCERS = {"ana": A("a"), "ben": A("b")}
TRAINER, BIDDER2, CONSUMER, CHECKER_AUTHOR, VALIDATOR = A("c"), A("d"), A("e"), A("f"), A("5")
KIM, RAJ, LEE, AUTO = A("7"), A("8"), A("6"), A("4")
NAMES = {**{v: k for k, v in PRODUCERS.items()}, TRAINER: "trainer", BIDDER2: "bidder-2", CONSUMER: "consumer",
         CHECKER_AUTHOR: "checker author", VALIDATOR: "validator", KIM: "kim (coin)", RAJ: "raj (coin)",
         LEE: "lee (coin)"}
usd = lambda m: f"${m / 1e6:,.4f}"


def main(port=8799, live=None):
    model = Model(live=live)
    ex, srv = serve(port, ":memory:", k=2, reserve_micros=50_000, validators=(VALIDATOR,))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    node = Client(url)
    node._call("POST", "/v0/checkers", {"id": "flight-rules", "author": CHECKER_AUTHOR})

    def agent(addr):
        return AdaptiveAgent(model, flight.check, fields=flight.FIELDS, task=flight.TASK, base_model=Model.name,
                             checker=flight.CHECKER, producer=addr, clean=flight.clean, narrow=flight.context_for,
                             evidence=flight.evidence)

    print("0. An agent with a problem posts a bounty")
    from traceex.adapt import attest as _attest
    eval_hash = _attest(VALIDATOR, flight.EVAL, "", 0, 0)["eval_set"]     # the bounty names its hidden eval by hash
    bounty = Client(url, CONSUMER).post_bounty(title="Flight extraction: 70% first-pass on unseen airlines",
                                               path="extract/travel/flight", eval_set=eval_hash, target=0.70,
                                               base_model=Model.name)
    print(f"   bounty #{bounty['id']} posted free: extract/travel/flight on needle3, reach 70% first-pass on a hidden "
          f"eval. Its coin starts at {usd(bounty['price_micros'])}")
    k = Client(url, KIM).buy_coins(bounty["id"], 2_000_000)
    r = Client(url, RAJ).buy_coins(bounty["id"], 3_000_000)
    print(f"   kim backs it early: $2.00 buys {k['coins']:,.0f} coins (avg {usd(k['avg_price_micros'])})")
    print(f"   raj backs it later: $3.00 buys {r['coins']:,.0f} coins (avg {usd(r['avg_price_micros'])}); "
          f"pool {usd(r['pool_micros'])}, next coin {usd(r['next_price_micros'])}")
    Client(url, KIM).transfer_coins(bounty["id"], LEE, k["coins"] / 2)
    print(f"   kim sells half her coins to lee off-exchange (transfer {k['coins'] / 2:,.0f} coins)")

    print("\n1. Agents fix their own mistakes; fixes become classified skeleton traces")
    owned = {"ana": ["southwest", "united"], "ben": ["delta"]}
    traces = []
    for who, emails in owned.items():
        ag = agent(PRODUCERS[who])
        for name in emails:
            out = ag.run(flight.TRAIN[name], created="2026-10-03T00:00:00Z")
            t = out.get("trace")
            print(f"   {who:4} {name:9} first pass {10 - len(flight.check(flight.TRAIN[name], out['first']))}/10 -> "
                  f"verified 10/10; fixed {', '.join(t['fixed_fields'])}")
            r = Client(url, PRODUCERS[who]).submit(t)
            c = r["classified"]
            print(f"        filed under {c['path_str']} ({c['engine']}); failure {c['signature']}"
                  + (f"; feeds bounty #{', #'.join(map(str, r['bounties']))}" if r["bounties"] else ""))
            traces.append(t)
    sample = traces[0]
    print("   what actually leaves the device (first line of a trace):")
    print("     " + sample["input"].splitlines()[2][:110])
    lot = node.lots()["lots"][0]
    print(f"   lot {lot['lot']}: {lot['traces']} traces from {lot['producers']} producers")
    s = node.search(path="extract/travel", failure="role_swap")
    print(f"   search extract/travel + role_swap: {s['count']} traces, {len(s['bounties'])} open bounty")

    print("\n2. Trainers bid; the batch auction clears")
    Client(url, TRAINER).bid(lot["lot"], 900_000)
    Client(url, BIDDER2).bid(lot["lot"], 600_000)
    Client(url, A("9")).bid(lot["lot"], 250_000)
    c = node.clear()["cleared"][0]
    print(f"   top {len(c['winners'])} win, each pays the next bid down: {usd(c['price_micros'])} "
          f"(bids were $0.90, $0.60, $0.25)")

    print("\n3. A learning is built from the traces and must prove itself on held-out emails")
    artifact, parents = routing_from_traces(traces)
    print(f"   learning: route {', '.join(artifact['body']['focus'])} to focused extraction")
    before, _ = first_pass_score(flight.EVAL, model, flight.check, fields=flight.FIELDS, clean=flight.clean)
    after, _ = first_pass_score(flight.EVAL, apply_routing(model, artifact, flight.context_for), flight.check,
                                fields=flight.FIELDS, clean=flight.clean)
    att = attest(VALIDATOR, flight.EVAL, "first_pass_field_accuracy", before, after)
    L = Learning.build(kind="routing", task=flight.TASK, base_model=Model.name, artifact=artifact,
                       parents=parents,
                       trainer=TRAINER, attestation=att, per_call_micros=200)
    lid = Client(url, TRAINER).register_learning(L)["id"]
    print(f"   validator on 3 unseen emails: first-pass accuracy {before:.1%} -> {after:.1%}  (accepted {lid[:19]}...)")
    won = Client(url, TRAINER).claim_bounty(bounty["id"], lid)
    print(f"   {after:.1%} beats the bounty's 70% target: bounty #{won['bounty']} solved, its {usd(won['pool_micros'])} "
          "pool pays " + ", ".join(f"{NAMES[a]} {usd(m)}" for a, m in sorted(won["payout"].items(), key=lambda kv: -kv[1])))
    print(f"   from now on the bounty's coin holders earn {won['holders_now_earn']}")

    print("\n4. A new agent adopts it and gets better without ever seeing the training emails")
    newbie = agent(CONSUMER)
    newbie.adopt(L)
    calls = 0
    for name, email in flight.EVAL.items():
        out = newbie.run(email)
        calls += 1
        handed = f"; {len(out['failing'])} handed to a bigger model" if out["failing"] else ""
        print(f"   {name:9} first pass {10 - len(flight.check(email, out['first']))}/10, "
              f"after checks {10 - len(out['failing'])}/10{handed}")
    Client(url, CONSUMER).report_usage(lid, 5_000)
    print(f"   metered: 5,000 calls x {usd(L['royalty']['per_call_micros'])} = {usd(5_000 * 200)}")

    print("\n5. Settlement: royalties flow down the family tree; one Merkle root goes on-chain")
    s = node.settle()
    for acct, claim in sorted(s["claims"].items(), key=lambda kv: -kv[1]["amount_micros"]):
        ok = merkle.verify(bytes.fromhex(s["root"][2:]), merkle.leaf(s["epoch"], acct, claim["amount_micros"]),
                           [bytes.fromhex(h[2:]) for h in claim["proof"]])
        print(f"   {NAMES.get(acct, acct[:10]):15} {usd(claim['amount_micros']):>10}   proof {'valid' if ok else 'INVALID'}")
    print(f"   epoch {s['epoch']} payout root {s['root'][:18]}...  total {usd(s['total_micros'])}")
    prov = node.provenance(lid)
    print("   provenance: learning <- " + ", ".join(f"{NAMES[p['producer']]}'s trace (w {p['weight']:.2f})"
                                                 for p in prov["parents"]))

    print("\n6. A new agent on autopilot: nobody tells it about the exchange")
    pilot = Autopilot(Client(url, AUTO), task=flight.TASK, base_model=Model.name, checker=flight.CHECKER,
                      policy=Policy(bounty_after=2, back_micros=1_000_000, budget_micros=1_000_000))
    auto = agent(AUTO)
    auto.autopilot = pilot
    acts = []
    for name, email in flight.LIVE.items():
        out = auto.run(email)
        for a in out["autopilot"]:
            acts.append(a)
            if a["action"] == "adopt":
                print(f"   {name:10} fails {len(out['failing'])} fields -> finds a learning on the exchange "
                      f"(attested +{a['gain'] * 100:.0f} points on held-out data) and adopts it" + (" automatically" if a.get("adopted") else ""))
            elif a["action"] == "noted":
                print(f"   {name:10} still fails {len(out['failing'])} fields; no learning fixes them yet (seen {a['seen']}x)")
            elif a["action"] in ("posted_bounty", "backed_existing_bounty"):
                print(f"   {name:10} same failure again -> posts bounty #{a['bounty']} for it, free: "
                      f"{a['failure'].split(':', 1)[1].replace(',', ', ')}")
                print(f"              its {a['cases']} failing emails stay on this device as the hidden eval "
                      f"({a.get('eval_set', '')[:19]}…); target {a.get('target', 0):.0%}; "
                      f"backed with {usd(a.get('backed_micros', 0))} of its {usd(pilot.policy.budget_micros)} budget")
            elif a["action"] == "already_posted":
                print(f"   {name:10} same failure: bounty #{a['bounty']} already stands, nothing new to post")
    srv.shutdown()
    srv.server_close()
    return {"before": before, "after": after, "settle": s, "autopilot": acts}


if __name__ == "__main__":
    main()
