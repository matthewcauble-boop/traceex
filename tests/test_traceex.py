"""python -m unittest discover tests   (standard library only)"""
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node"), os.path.join(ROOT, "examples", "flight_emails")]

from traceex import (Bid, Trace, Learning, clear_shared, clear_exclusive, find_pii, skeletonize, split_trace_sale,
                     split_usage, object_id, routing_from_traces, attest)
from traceex import merkle
from exchange import Exchange

A = lambda c: "0x" + c * 40
EMAIL = """Confirmation # QX7PLM
Thursday, October 15, 2026  Flight 1123  Departs Austin (AUS) 8:05 AM  Arrives Denver (DEN) 9:35 AM
Passengers: Jordan Parker, Casey Parker. Questions? jordan@example.com or (512) 555-0142
Total paid: $1,284.40"""
PRED = {"confirmation_number": "QX7PLM", "outbound_flight_number": "AUS", "origin_airport_code": "AUS",
        "destination_airport_code": "AUS", "total_paid": 1284.4}
FIXED = dict(PRED, outbound_flight_number="1123", destination_airport_code="DEN")


def make_trace(producer=A("a"), email=EMAIL):
    return Trace.from_fix(task="extract.flight", base_model="needle3", input=email, model_output=PRED,
                          verified_output=FIXED, checker="flight-rules@1", producer=producer,
                          created="2026-10-03T00:00:00Z", fixed_by={"outbound_flight_number": "model"})


class Skeleton(unittest.TestCase):
    def test_no_personal_values_survive(self):
        t = make_trace()
        for raw in ("QX7PLM", "1123", "Jordan", "Parker", "jordan@example.com", "555-0142", "1,284.40", "8:05", "October 15"):
            self.assertNotIn(raw, t["input"], raw)
            self.assertNotIn(raw, str(t["verified_output"]), raw)
        self.assertEqual(find_pii(t["input"]), [])

    def test_structure_survives(self):
        t = make_trace()
        self.assertIn("Departs {", t["input"])
        self.assertIn("Flight {", t["input"])
        self.assertEqual(t["input"].count("\n"), EMAIL.count("\n"))
        self.assertIn("\nTotal paid:", t["input"])

    def test_placeholders_link_input_and_output(self):
        t = make_trace()
        ph = t["verified_output"]["outbound_flight_number"]
        self.assertRegex(ph, r"^\{[A-Z]+_\d+\}$")
        self.assertIn(ph, t["input"])
        self.assertEqual(t["fixed_fields"], ["destination_airport_code", "outbound_flight_number"])
        self.assertEqual(t["fixed_by"], {"destination_airport_code": "unknown", "outbound_flight_number": "model"})

    def test_same_fix_same_id(self):
        self.assertEqual(make_trace().id, make_trace().id)
        self.assertNotEqual(make_trace().id, make_trace(producer=A("b")).id)

    def test_id_stable_across_processes(self):
        # placeholder numbering must not depend on Python's per-process hash seed
        import subprocess
        code = ("import sys; sys.path[:0]=[%r, %r]; import test_traceex as t; print(t.make_trace().id)"
                % (os.path.join(ROOT, "sdk", "python"), os.path.dirname(os.path.abspath(__file__))))
        ids = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env={**os.environ, "PYTHONHASHSEED": str(s)}).stdout.strip() for s in (1, 2, 3, 4)}
        self.assertEqual(len(ids), 1, ids)


class Auctions(unittest.TestCase):
    def test_uniform_price_vickrey(self):
        bids = [Bid("x", 900), Bid("y", 600), Bid("z", 250), Bid("w", 100)]
        self.assertEqual(clear_shared(bids, 2, 50), (["x", "y"], 250))

    def test_reserve_when_undersubscribed(self):
        self.assertEqual(clear_shared([Bid("x", 900)], 3, 50), (["x"], 50))
        self.assertEqual(clear_shared([Bid("x", 40)], 3, 50), ([], None))

    def test_one_bid_per_bidder(self):
        winners, price = clear_shared([Bid("x", 900), Bid("x", 800), Bid("y", 300)], 1)
        self.assertEqual((winners, price), (["x"], 300))

    def test_exclusive_second_price(self):
        self.assertEqual(clear_exclusive([Bid("x", 9), Bid("y", 7), Bid("z", 3)]), ("x", 7))

    def test_truthful_bidding_never_loses_to_shading(self):
        others = [Bid("o1", 500), Bid("o2", 300), Bid("o3", 200)]
        value = 400
        for shade in range(0, 1000, 25):
            w, p = clear_shared(others + [Bid("me", shade)], 2)
            w2, p2 = clear_shared(others + [Bid("me", value)], 2)
            u = (value - p) if "me" in w else 0
            u_true = (value - p2) if "me" in w2 else 0
            self.assertGreaterEqual(u_true, u)


class Royalties(unittest.TestCase):
    def test_trace_sale_split_is_exact(self):
        out = split_trace_sale(1_000_001, A("a"), A("c"), [A("5"), A("6")])
        self.assertEqual(sum(out.values()), 1_000_001)
        self.assertAlmostEqual(out[A("a")] / 1_000_001, 0.85, places=4)

    def test_usage_follows_family_tree(self):
        t1, t2 = "sha256:t1", "sha256:t2"
        info = {t1: {"producer": A("a"), "checker_author": A("c")}, t2: {"producer": A("b"), "checker_author": A("c")}}
        base = {"trainer": A("e"), "parents": [{"trace": t1, "weight": 0.75}, {"trace": t2, "weight": 0.25}],
                "royalty": {"per_call_micros": 10, "split": Learning.DEFAULT_SPLIT}}
        child = {"trainer": A("f"), "parents": [{"trace": "sha256:base", "weight": 1.0}],
                 "royalty": {"per_call_micros": 10, "split": Learning.DEFAULT_SPLIT}}
        out = split_usage(1_000_000, child, info, [A("5")], {"sha256:base": base})
        self.assertEqual(sum(out.values()), 1_000_000)
        self.assertGreater(out[A("a")], out[A("b")])      # bigger contribution, bigger share
        self.assertIn(A("e"), out)                          # the parent learning's trainer is paid too


class Merkle(unittest.TestCase):
    def test_every_proof_verifies_and_tampering_fails(self):
        pays = {A(c): (i + 1) * 1000 for i, c in enumerate("abcdef5")}
        leaves = {a: merkle.leaf(3, a, m) for a, m in pays.items()}
        levels = merkle.build_tree(list(leaves.values()))
        root = levels[-1][0]
        for a, m in pays.items():
            p = merkle.proof(levels, leaves[a])
            self.assertTrue(merkle.verify(root, leaves[a], p))
            self.assertFalse(merkle.verify(root, merkle.leaf(3, a, m + 1), p))

    def test_leaf_layout_matches_solidity_encode_packed(self):
        # abi.encodePacked(uint64, address, uint256) = 8 + 20 + 32 bytes
        import hashlib
        raw = (7).to_bytes(8, "big") + bytes.fromhex("ab" * 20) + (5).to_bytes(32, "big")
        self.assertEqual(merkle.leaf(7, "0x" + "ab" * 20, 5), hashlib.sha256(raw).digest())


class Node(unittest.TestCase):
    def setUp(self):
        self.ex = Exchange(":memory:", k=2, reserve_micros=100, validators=(A("5"),))
        self.ex.register_checker("flight-rules", A("c"))

    def test_rejects_leaky_trace(self):
        t = dict(make_trace())
        t["input"] += "\nCall Jordan Parker at (512) 555-0142"
        with self.assertRaises(ValueError):
            self.ex.submit_trace(t)

    def test_duplicate_is_idempotent(self):
        t = make_trace()
        self.ex.submit_trace(dict(t))
        self.assertTrue(self.ex.submit_trace(dict(t))["duplicate"])

    def test_learning_needs_improvement(self):
        tid = self.ex.submit_trace(dict(make_trace()))["id"]
        bad = Learning.build(kind="routing", task="extract.flight", base_model="needle3", artifact={"uri": "inline"},
                             parents=[(tid, 1)], trainer=A("e"), per_call_micros=10,
                             attestation=attest(A("5"), {"x": "y"}, "acc", 0.7, 0.6))
        with self.assertRaises(ValueError):
            self.ex.register_learning(bad)

    def test_full_epoch_balances(self):
        tid = self.ex.submit_trace(dict(make_trace()))["id"]
        lot = self.ex.lots()["lots"][0]["lot"]
        for who, p in ((A("e"), 900), (A("d"), 600), (A("9"), 300)):
            self.ex.bid({"lot": lot, "bidder": who, "price_micros": p})
        c = self.ex.clear()["cleared"][0]
        self.assertEqual((c["winners"], c["price_micros"]), ([A("e"), A("d")], 300))
        L = Learning.build(kind="routing", task="extract.flight", base_model="needle3", artifact={"uri": "inline"},
                           parents=[(tid, 1)], trainer=A("e"), per_call_micros=10,
                           attestation=attest(A("5"), {"x": "y"}, "acc", 0.6, 0.7))
        lid = self.ex.register_learning(L)["id"]
        self.ex.usage({"learning": lid, "consumer": A("8"), "calls": 1000})
        s = self.ex.settle()
        self.assertEqual(s["total_micros"], 2 * 300 + 1000 * 10)    # everything paid in is paid out
        self.assertEqual(self.ex.epoch, 2)
        root = bytes.fromhex(s["root"][2:])
        for a, c in s["claims"].items():
            self.assertTrue(merkle.verify(root, merkle.leaf(1, a, c["amount_micros"]), [bytes.fromhex(h[2:]) for h in c["proof"]]))
        self.assertEqual(self.ex.provenance(lid)["parents"][0]["producer"], A("a"))


class Classifier(unittest.TestCase):
    def test_failure_modes_from_placeholders(self):
        from traceex.classify import failure_modes
        t = {"input": "Flight {NUM_1} departs {CODE_1} arrives {CODE_2}", "fixed_fields": ["a", "b", "c", "d"],
             "model_output": {"a": "{CODE_1}", "b": "{CODE_1}", "c": "{NUM_9}", "d": ""},
             "verified_output": {"a": "{NUM_1}", "b": "{CODE_2}", "c": "{NUM_1}", "d": "{CODE_2}", "o": "{CODE_1}"}}
        self.assertEqual(failure_modes(t), {"a": "type_mismatch", "b": "role_swap", "c": "invented", "d": "omission"})

    def test_rules_engine_paths(self):
        from traceex.classify import RulesEngine, classify
        E = RulesEngine()
        self.assertEqual(classify(make_trace(), E)["path_str"], "extract/travel/flight")
        self.assertEqual(E.classify("Invoice {NUM_1}. Amount due {MONEY_1}, due date {DATE_1}, Net 30")["path"],
                         ["extract", "commerce", "invoice"])
        self.assertEqual(E.classify("Traceback: exception in test, failing")["path"][:1], ["code"])

    def test_jev_beam_walks_the_tree(self):
        from traceex.classify import JevEngine
        asked = []

        def fake(req):
            asked.append(req["questions"]["category"]["instructions"])
            kids = list(req["questions"]["category"]["criteria"])
            pick = {"extract", "travel", "flight"} & set(kids)
            return {k: (.9 if k in pick else .1 / len(kids)) for k in kids}
        out = JevEngine("k", beam=1).classify("x", ask=fake)
        self.assertEqual(out["path"], ["extract", "travel", "flight"])
        self.assertEqual(len(asked), 3)

    def test_hosted_engine_failure_falls_back(self):
        from traceex.classify import JevEngine, classify
        broken = JevEngine("k", base_url="http://127.0.0.1:9", timeout=.5)
        c = classify(make_trace(), broken)
        self.assertEqual(c["engine"], "rules")
        self.assertIn("fallback", c)


class Bounties(unittest.TestCase):
    def setUp(self):
        self.ex = Exchange(":memory:", k=2, reserve_micros=100, validators=(A("5"),))
        self.ex.register_checker("flight-rules", A("c"))
        self.tid = self.ex.submit_trace(dict(make_trace()))["id"]

    def learning(self, eval_set, after):
        L = Learning.build(kind="routing", task="extract.flight", base_model="needle3", artifact={"uri": "inline"},
                           parents=[(self.tid, 1)], trainer=A("e"), per_call_micros=10,
                           attestation={"validator": A("5"), "eval_set": eval_set, "metric": "acc", "before": .5,
                                        "after": after, "sig": "x"})
        return self.ex.register_learning(L)["id"]

    def test_search_and_matching(self):
        b = self.ex.post_bounty({"poster": A("8"), "path": "extract/travel", "eval_set": "sha256:E", "target": .7,
                                 "reward_micros": 1000})
        r = self.ex.submit_trace(dict(make_trace(producer=A("b"))))
        self.assertEqual(r["bounties"], [b["id"]])
        s = self.ex.search(q="Departs", path="extract", failure="role_swap")
        self.assertEqual(s["count"], 2)
        self.assertEqual(len(s["bounties"]), 1)
        self.assertEqual(self.ex.search(path="code")["count"], 0)
        tax = {n["path"]: n["traces"] for n in self.ex.taxonomy()["nodes"]}
        self.assertEqual(tax["extract"], 2)

    def test_claim_rules_and_payout(self):
        b = self.ex.post_bounty({"poster": A("8"), "path": "extract/travel/flight", "eval_set": "sha256:E",
                                 "target": .7, "reward_micros": 10_000})["id"]
        with self.assertRaises(ValueError):
            self.ex.claim_bounty(b, self.learning("sha256:OTHER", .9))      # wrong eval set
        with self.assertRaises(ValueError):
            self.ex.claim_bounty(b, self.learning("sha256:E", .65))         # below target
        won = self.ex.claim_bounty(b, self.learning("sha256:E", .75))
        self.assertEqual(sum(won["payout"].values()), 10_000)
        self.assertEqual(won["payout"][A("e")], 7_000)                       # solver 70%
        self.assertEqual(won["payout"][A("a")], 2_000)                       # the trace still earns 20%
        with self.assertRaises(ValueError):
            self.ex.claim_bounty(b, self.learning("sha256:E", .8))           # already paid

    def test_unclaimed_bounty_refunds(self):
        self.ex.post_bounty({"poster": A("8"), "path": "extract", "eval_set": "sha256:E", "target": .9,
                             "reward_micros": 5_000, "epochs": 0})
        self.ex.settle()                                   # epoch 1 -> 2; deadline was epoch 1
        s = self.ex.settle()
        self.assertEqual(s["claims"][A("8")]["amount_micros"], 5_000)
        self.assertEqual(self.ex.bounties()["bounties"][0]["status"], "expired")


class BountyCoins(unittest.TestCase):
    def test_curve_maths(self):
        from traceex import bountycoin as c
        n = c.coins_for(0, 2_000_000)
        self.assertAlmostEqual(c.cost(0, n), 2_000_000, delta=1)
        self.assertAlmostEqual(c.sell_value(n, n), 2_000_000, delta=1)         # selling everything empties the pool
        self.assertGreater(c.coins_for(0, 1_000_000), c.coins_for(500, 1_000_000))   # early backers get more coins

    def setUp(self):
        self.ex = Exchange(":memory:", k=2, reserve_micros=100, validators=(A("5"),))
        self.ex.register_checker("flight-rules", A("c"))
        self.tid = self.ex.submit_trace(dict(make_trace()))["id"]
        self.b = self.ex.post_bounty({"poster": A("8"), "path": "extract/travel/flight", "eval_set": "sha256:E",
                                      "target": .7})["id"]

    def test_free_post_buy_sell_transfer(self):
        self.assertEqual(self.ex.holders(self.b)["pool_micros"], 0)              # posting is free
        k = self.ex.buy_coins(self.b, A("1"), 2_000_000)
        r = self.ex.buy_coins(self.b, A("2"), 2_000_000)
        self.assertGreater(k["coins"], r["coins"])
        s = self.ex.sell_coins(self.b, A("2"), r["coins"])                         # raj exits at the curve
        self.assertAlmostEqual(s["paid_micros"], 2_000_000, delta=2)
        self.ex.transfer_coins(self.b, A("1"), A("3"), k["coins"] / 4)
        h = self.ex.holders(self.b)["holders"]
        self.assertAlmostEqual(h[A("3")], k["coins"] / 4, places=4)
        with self.assertRaises(ValueError):
            self.ex.transfer_coins(self.b, A("3"), A("4"), k["coins"])            # more than held

    def test_holders_earn_from_the_solution(self):
        k = self.ex.buy_coins(self.b, A("1"), 1_000_000)
        self.ex.buy_coins(self.b, A("2"), 1_000_000)
        self.ex.transfer_coins(self.b, A("1"), A("3"), k["coins"] / 2)
        L = Learning.build(kind="routing", task="extract.flight", base_model="needle3", artifact={"uri": "inline"},
                           parents=[(self.tid, 1)], trainer=A("e"), per_call_micros=10,
                           attestation={"validator": A("5"), "eval_set": "sha256:E", "metric": "acc", "before": .5,
                                        "after": .8, "sig": "x"})
        lid = self.ex.register_learning(L)["id"]
        won = self.ex.claim_bounty(self.b, lid)
        self.assertEqual(sum(won["payout"].values()), 2_000_000)
        with self.assertRaises(ValueError):
            self.ex.buy_coins(self.b, A("4"), 1000)                                # curve closes once solved
        self.ex.usage({"learning": lid, "consumer": A("9"), "calls": 100_000})    # $1.00 of usage
        s = self.ex.settle()
        cut = {a: s["claims"][a]["amount_micros"] for a in (A("1"), A("2"), A("3"))}
        self.assertEqual(sum(cut.values()), 200_000)                               # holders' 20%
        self.assertAlmostEqual(cut[A("1")], cut[A("3")], delta=1)                  # split by coins held now
        self.assertGreater(cut[A("1")] + cut[A("3")], cut[A("2")])                 # early backer earns more per dollar
        self.assertEqual(s["total_micros"], 2_000_000 + 1_000_000)


class Adaptation(unittest.TestCase):
    def test_routing_uses_model_fixes_only(self):
        t1 = make_trace()
        t2 = make_trace(producer=A("b"))
        art, parents = routing_from_traces([t1, t2])
        self.assertEqual(art["body"]["focus"], ["destination_airport_code", "outbound_flight_number"])
        t1["fixed_by"]["destination_airport_code"] = t2["fixed_by"]["destination_airport_code"] = "rule"
        art, _ = routing_from_traces([t1, t2])
        self.assertEqual(art["body"]["focus"], ["outbound_flight_number"])

    def test_demo_learning_improves_held_out(self):
        import demo
        r = demo.main(port=8798, live=False)
        self.assertGreater(r["after"], r["before"])


if __name__ == "__main__":
    unittest.main()
