"""python -m unittest discover tests   (standard library only)"""
import json
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node"), os.path.join(ROOT, "examples", "flight_emails")]

from traceex import (Client, Bid, Trace, Learning, clear_shared, clear_exclusive, find_pii, skeletonize, split_trace_sale,
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

    def test_lone_names_and_places(self):
        sk, _, _ = skeletonize("Flight 1123 departs Austin (AUS) 8:05 AM. Passenger Jordan Parker, call Riley at noon.\n"
                               "Thanks for flying with us.")
        for raw in ("Austin", "Jordan", "Parker", "Riley"):
            self.assertNotIn(raw, sk)
        self.assertIn("Thanks for flying", sk)              # sentence starts are left alone
        self.assertEqual(find_pii(sk), [])
        self.assertTrue(find_pii("we land in Denver at noon"))
        sk, _, _ = skeletonize("Passenger: Riley. Meet at the (Hilton) - Boston, MA")
        for raw in ("Riley", "Hilton", "Boston"):
            self.assertNotIn(raw, sk)

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

        def fake(req):                     # one request per level, one question per branch on the beam
            asked.append(sorted(req["questions"]))
            out = {}
            for name, qn in req["questions"].items():
                kids = list(qn["criteria"])
                pick = {"extract", "travel", "flight"} & set(kids)
                out[name] = {k: (.9 if k in pick else .1 / len(kids)) for k in kids}
            return out
        out = JevEngine("k", beam=2).classify("x", ask=fake)
        self.assertEqual(out["path"], ["extract", "travel", "flight"])
        self.assertEqual(len(asked), 3)                       # three levels, three requests
        self.assertEqual(asked[0], ["p0"])                    # the top level is one question

    def test_jev_daily_budget_falls_back_to_rules(self):
        import datetime as dt
        from traceex.classify import JevEngine, JevBudget, classify
        eng = JevEngine("k", daily_requests=2)
        eng._spend(); eng._spend()
        with self.assertRaises(JevBudget):
            eng._spend()
        c = classify(make_trace(), eng)
        self.assertEqual((c["engine"], c["path_str"]), ("rules", "extract/travel/flight"))
        self.assertIn("JevBudget", c["fallback"])
        eng.usage["day"] = (dt.date.today() - dt.timedelta(days=1)).isoformat()   # a new day resets the count
        eng._spend()

    def test_hosted_engine_failure_falls_back(self):
        from traceex.classify import JevEngine, classify
        broken = JevEngine("k", base_url="http://127.0.0.1:9", timeout=.5)
        c = classify(make_trace(), broken)
        self.assertEqual(c["engine"], "rules")
        self.assertIn("fallback", c)


class FakeJev:
    """Stands in for the hosted engine: files everything under one branch, and checks it is never called while the
    node holds its write lock."""
    name = "jev"

    def __init__(self, path, ex=None):
        self.path, self.ex, self.calls = path, ex, 0

    def classify(self, text):
        self.calls += 1
        if self.ex is not None:                       # another thread can take the lock: nobody holds it now
            import threading
            got = []
            t = threading.Thread(target=lambda: got.append(self.ex.lock.acquire(blocking=False) and (self.ex.lock.release() or True)))
            t.start()
            t.join()
            assert got[0], "classified while holding the write lock"
        return {"path": self.path.split("/"), "confidence": .9, "engine": self.name}


CODE = "Write a python function to find the shared elements from the given two lists.\nassert similar((3, 4), (4, 5)) == (4,)"


def search_trace(producer=A("b"), text=CODE, mode="wrong_answer"):
    return Trace.from_fix(task="code.python", base_model="Qwen/Qwen2.5-0.5B-Instruct", input=text,
                          model_output={"code": "def f(): pass"}, verified_output={"code": "def f(): return 1"},
                          checker="mbpp-tests@1", producer=producer, created="2026-10-03T00:00:00Z", privacy="open",
                          failure_modes={"code": mode})


class Search(unittest.TestCase):
    def setUp(self):
        self.ex = Exchange(":memory:", validators=(A("5"),))
        self.flight = self.ex.submit_trace(dict(make_trace()))["id"]
        self.code = [self.ex.submit_trace(dict(search_trace(text=CODE + f"\n# case {i}", mode=m)))["id"]
                     for i, m in enumerate(["wrong_answer", "runtime_error", "wrong_answer"])]

    def test_relevance_new_and_bounty_sorts(self):
        r = self.ex.search("flight")
        self.assertEqual(r["sort"], "relevant")
        self.assertEqual(r["results"][0]["id"], self.flight)
        self.assertEqual([h["id"] for h in self.ex.search(limit=10)["results"]][0], self.code[-1])   # newest first
        b = self.ex.post_bounty({"poster": A("8"), "path": "code", "eval_set": "sha256:E", "target": .5, "failure": "runtime_error"})
        self.ex.buy_coins(b["id"], A("8"), 2_000_000)
        top = self.ex.search(sort="bounty")["results"][0]
        self.assertEqual((top["id"], top["bounty_pool_micros"]), (self.code[1], 2_000_000))
        with self.assertRaises(ValueError):
            self.ex.search(sort="sideways")

    def test_filters_paging_and_facets(self):
        r = self.ex.search(path="code", failure="wrong_answer", limit=1, facets=True)
        self.assertEqual((r["total"], r["count"]), (2, 1))
        self.assertEqual(self.ex.search(path="code", failure="wrong_answer", limit=1, offset=1)["count"], 1)
        self.assertEqual(r["facets"]["modes"], {"wrong_answer": 2, "runtime_error": 1})   # modes within the branch
        self.assertEqual(sum(r["facets"]["paths"].values()), 4)                            # paths across all branches
        self.assertEqual(self.ex.search(failure="swap")["total"], 0)                      # whole labels, not substrings

    def test_any_word_when_all_words_find_nothing(self):
        self.assertEqual(self.ex.search("flight zebra")["results"][0]["id"], self.flight)
        for odd in ('"', "--", "*", "AND OR NOT", "(("):
            self.assertIsInstance(self.ex.search(odd)["total"], int)

    def test_reclassify_moves_keyword_labels(self):
        self.assertEqual(self.ex.reclassify()["skipped"][:3], "the")      # nothing to gain without a hosted engine
        jev = FakeJev("code/test")                 # several workers: another one may hold the lock, that's fine
        st = self.ex.reclassify(engine=jev)
        self.assertEqual((st["done"], st["total"], st["running"]), (4, 4, False))
        self.assertEqual(self.ex.search(path="code/test")["total"], 4)
        self.assertEqual(self.ex.search("test")["total"], 4)             # the full-text row moved too
        self.assertEqual(self.ex.stats()["classifier"]["filed_by"], {"jev": 4})
        self.assertEqual(self.ex.reclassify(engine=jev)["total"], 0)     # only keyword labels are re-filed

    def test_hosted_engine_runs_outside_the_lock(self):
        ex = Exchange(":memory:", validators=(A("5"),))
        ex.engine = FakeJev("code/repair", ex)
        r = ex.submit_trace(dict(search_trace()))
        self.assertEqual(r["classified"]["path_str"], "code/repair")
        self.assertTrue(ex.submit_trace(dict(search_trace()))["duplicate"])
        self.assertEqual(ex.engine.calls, 1)                              # a duplicate is never paid for twice


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


CODE_PROMPT = "Write a function to add two numbers.\nassert add(1, 2) == 3"


def code_trace(producer=A("a"), bad="def add(a, b):\n    return a - b", good="def add(a, b):\n    return a + b"):
    return Trace.from_fix(task="code.python", base_model="Qwen/Qwen2.5-0.5B-Instruct", input=CODE_PROMPT,
                          model_output={"code": bad}, verified_output={"code": good}, checker="mbpp-tests@1",
                          producer=producer, created="2026-10-03T00:00:00Z", privacy="open",
                          feedback=["Failed: assert add(1, 2) == 3\n  your function returned -1, expected 3"],
                          failure_modes={"code": "wrong_answer"}, fixed_by={"code": "model_repair"})


class OpenTraces(unittest.TestCase):
    def test_open_keeps_the_text(self):
        t = code_trace()
        self.assertEqual(t["input"], CODE_PROMPT)
        self.assertIn("return a + b", t["verified_output"]["code"])
        self.assertEqual(t["privacy"], "open")
        self.assertEqual(t["feedback"][0].splitlines()[0], "Failed: assert add(1, 2) == 3")

    def test_secrets_are_refused_everywhere(self):
        from traceex.client import privacy_leaks
        from traceex import find_secrets
        self.assertEqual(privacy_leaks(code_trace()), [])
        leaky = code_trace(good='API_KEY = "sk-live-' + "a1B2c3D4e5F6g7H8i9J0k1L2" + '"\ndef add(a, b):\n    return a + b')
        self.assertTrue(privacy_leaks(leaky))
        self.assertTrue(find_secrets("token: ghp_" + "x" * 36))
        self.assertTrue(privacy_leaks(code_trace(good="# mail jo@example.com\ndef add(a, b):\n    return a + b")))
        ex = Exchange(":memory:", validators=(A("5"),))
        with self.assertRaises(ValueError):
            ex.submit_trace(dict(leaky))
        r = ex.submit_trace(dict(code_trace()))
        self.assertEqual(r["classified"]["failure_modes"], {"code": "wrong_answer"})

    def test_skeleton_feedback_is_skeletonised_too(self):
        t = Trace.from_fix(task="extract.flight", base_model="needle3", input=EMAIL, model_output=PRED,
                           verified_output=FIXED, checker="flight-rules@1", producer=A("a"),
                           created="2026-10-03T00:00:00Z", feedback=["the 'arriv' airport on the outbound line is DEN, not AUS"])
        self.assertNotIn("DEN", t["feedback"][0])
        self.assertIn("{CODE_", t["feedback"][0])
        self.assertEqual(find_pii("\n".join(t["feedback"])), [])

    def test_learning_kinds_and_release(self):
        with self.assertRaises(ValueError):
            Learning.build(kind="vibes", task="t", base_model="m", artifact={}, parents=[("x", 1)], trainer=A("e"),
                           attestation={}, per_call_micros=1)
        L = Learning.build(kind="lora", task="code.python", base_model="m", artifact={"uri": "hf://x"},
                           parents=[("x", 1)], trainer=A("e"), attestation={}, per_call_micros=1, release="open")
        self.assertEqual(L["release"], "open")


class Export(unittest.TestCase):
    def test_sft_dpo_repair_shapes(self):
        from traceex import export
        ts = [code_trace(), code_trace(producer=A("b"))]
        sft, dpo, rep = export.to_sft(ts), export.to_dpo(ts), export.to_repair(ts)
        self.assertEqual(sft[0]["messages"][1]["content"], "```python\ndef add(a, b):\n    return a + b\n```")
        self.assertIn("a - b", dpo[0]["rejected"])
        self.assertIn("a + b", dpo[0]["chosen"])
        self.assertIn("your function returned -1", rep[0]["messages"][0]["content"])
        self.assertEqual(sft[0]["trace"], ts[0].id)

    def test_refill_is_synthetic_and_consistent(self):
        from traceex import export
        t = make_trace()
        a, b = export.refill(t, 0), export.refill(t, 1)
        self.assertNotIn("{", a["input"].replace("{\"", ""))                      # every placeholder filled
        for raw in ("QX7PLM", "Jordan", "1123", "DEN"):
            self.assertNotIn(raw, a["input"])                                     # never the original values
        self.assertNotEqual(a["input"], b["input"])                               # different seed, different values
        code = a["verified_output"]["outbound_flight_number"]
        self.assertIn(code, a["input"])                                           # same placeholder, same value
        self.assertEqual(export.refill(t, 0), a)                                  # deterministic
        rows = export.to_sft([t], refills=3)
        self.assertEqual(len(rows), 3)

    def test_cards_credit_producers(self):
        from traceex import export
        ts = [code_trace(), code_trace(producer=A("b"))]
        card = export.dataset_card(ts, name="mbpp fixes")
        self.assertIn(A("a"), card)
        self.assertIn("wrong_answer 2", card)
        L = Learning.build(kind="lora", task="code.python", base_model="Qwen/Qwen2.5-0.5B-Instruct",
                           artifact={"uri": "x", "hash": "sha256:ab"}, parents=[(t.id, 1) for t in ts], trainer=A("e"),
                           attestation={"metric": "pass@1", "before": .3, "after": .35}, per_call_micros=1)
        mc = export.model_card(L, title="t", base_model_license="apache-2.0", eval_name="e",
                               producers={A("a"): 1, A("b"): 1})
        self.assertIn("30.0% -> 35.0%", mc)
        self.assertIn(A("b"), mc)


class AgentsUseTheExchange(unittest.TestCase):
    """MCP server (stdio + the node's /mcp) and the autopilot: agents acting on the exchange by themselves."""

    def setUp(self):
        import threading
        from exchange import serve
        self.ex, self.srv = serve(0, ":memory:", k=2, reserve_micros=100, validators=(A("5"),))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.ex.register_checker("flight-rules", A("c"))

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def call(self, backend, name, args, mid=1):
        from traceex.mcp import handle
        return handle({"jsonrpc": "2.0", "id": mid, "method": "tools/call", "params": {"name": name, "arguments": args}},
                      backend)["result"]

    def test_handshake_and_tool_list(self):
        from traceex.mcp import handle, NodeBackend, TOOLS
        b = NodeBackend(self.ex)
        init = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}}}, b)
        self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("proactively", init["result"]["instructions"])
        self.assertIsNone(handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, b))
        tools = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, b)["result"]["tools"]
        self.assertEqual(len(tools), len(TOOLS))
        self.assertIn("address", next(t for t in tools if t["name"] == "traceex_post_bounty")["inputSchema"]["properties"])
        self.assertEqual(handle({"jsonrpc": "2.0", "id": 3, "method": "nope"}, b)["error"]["code"], -32601)

    def test_stdio_server_round_trip(self):
        import io
        from traceex.mcp import serve_stdio, ClientBackend
        inp = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}\n'
                          '{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
                          '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"traceex_taxonomy","arguments":{}}}\n')
        out = io.StringIO()
        serve_stdio(ClientBackend(Client(self.url, A("a"))), inp, out)
        lines = [json.loads(x) for x in out.getvalue().splitlines()]
        self.assertEqual([m["id"] for m in lines], [1, 2])
        self.assertEqual(lines[1]["result"]["structuredContent"]["version"], "taxonomy/0.1")

    def test_budget_and_privacy_guards(self):
        from traceex.mcp import ClientBackend, NodeBackend
        local = ClientBackend(Client(self.url, A("a")), max_spend_micros=1000)
        b = self.call(local, "traceex_post_bounty", {"title": "t", "path": "extract", "eval_set": "sha256:e", "target": .5})
        bid = b["structuredContent"]["id"]
        self.assertFalse(self.call(local, "traceex_back_bounty", {"bounty_id": bid, "micros": 800}).get("isError"))
        over = self.call(local, "traceex_back_bounty", {"bounty_id": bid, "micros": 800})
        self.assertTrue(over["isError"])
        self.assertIn("over budget", over["content"][0]["text"])
        fix = {"task": "extract.flight", "base_model": "needle3", "input": EMAIL, "model_output": PRED,
               "verified_output": FIXED, "checker": "flight-rules@1"}
        sent = self.call(local, "traceex_submit_fix", fix)                       # local: skeletonised here, accepted
        self.assertFalse(sent.get("isError"), sent)
        self.assertNotIn("Jordan", json.dumps(self.ex.search(q="Departs")))
        remote = self.call(NodeBackend(self.ex), "traceex_submit_fix", dict(fix, address=A("b")))
        self.assertTrue(remote["isError"])                                       # remote: raw personal text refused
        self.assertIn("own machine", remote["content"][0]["text"])

    def test_node_speaks_mcp_over_http_and_describes_itself(self):
        import urllib.request
        req = urllib.request.Request(self.url + "/mcp", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}).encode())
        with urllib.request.urlopen(req) as r:
            self.assertEqual(json.loads(r.read())["id"], 7)
        note = urllib.request.Request(self.url + "/mcp", method="POST", headers={"Content-Type": "application/json"},
                                      data=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode())
        with urllib.request.urlopen(note) as r:
            self.assertEqual(r.status, 202)
        d = Client(self.url).describe()
        self.assertEqual(d["mcp"], "/mcp")

    def test_autopilot_posts_a_bounty_then_adopts_a_learning(self):
        from traceex.autopilot import Autopilot, Policy
        from traceex.classify import RulesEngine
        pilot = Autopilot(Client(self.url, A("7")), task="extract.flight", base_model="needle3", checker="flight-rules@1",
                          policy=Policy(bounty_after=2, back_micros=500, budget_micros=600), engine=RulesEngine())
        fail = {"result": {"outbound_date": "2026-03-14"}, "failing": ["outbound_date", "return_date"]}
        email2 = EMAIL.replace("October 15", "3/14/2027")
        a1 = pilot.on_result(email2, fail)[0]
        self.assertEqual((a1["action"], a1["path"], a1["seen"]), ("noted", "extract/travel/flight", 1))
        a2 = pilot.on_result(email2.replace("QX7PLM", "TRVQ8B"), fail)[0]
        self.assertEqual(a2["action"], "posted_bounty")
        self.assertEqual(a2["backed_micros"], 500)
        b = self.ex.bounties(status="open")["bounties"][0]
        self.assertEqual((b["path"], b["failure"]), ("extract/travel/flight", "unresolved:outbound_date,return_date"))
        self.assertEqual(b["eval_set"], a2["eval_set"])                         # only the hash is public
        self.assertNotIn("3/14/2027", json.dumps(self.ex.bounties()))
        self.assertEqual(pilot.on_result(email2, fail)[0]["action"], "already_posted")
        self.assertEqual(pilot.spent, 500)                                       # never past the budget
        # a learning for that branch arrives: the autopilot now adopts instead
        tid = self.ex.submit_trace(dict(make_trace()))["id"]
        L = Learning.build(kind="routing", task="extract.flight", base_model="needle3", artifact={"uri": "inline"},
                           parents=[(tid, 1)], trainer=A("e"), per_call_micros=10,
                           attestation={"validator": A("5"), "eval_set": "sha256:x", "metric": "acc", "before": .6, "after": .7})
        lid = self.ex.register_learning(L)["id"]
        found = self.ex.find_learnings(path="extract/travel", model="needle3")["learnings"]
        self.assertEqual((found[0]["id"], found[0]["path"], found[0]["gain"]), (lid, "extract/travel/flight", 0.1))
        self.assertEqual(pilot.tick()[0]["learning"]["id"], lid)                  # tick spots it first
        act = pilot.on_result(email2, fail)[0]
        self.assertEqual((act["action"], act["learning"]["id"]), ("adopt", lid))
        self.assertEqual(pilot.tick(), [])                                       # tried: not suggested again
        self.assertEqual(pilot.on_result(email2, fail)[0]["action"], "already_posted")   # still failing: the bounty stands
        self.assertEqual(len(pilot.hidden_eval(b["id"])), 2)                     # two distinct failing cases, kept locally


class HostedNode(unittest.TestCase):
    """The public deployment: operator-only calls, testnet wallets, limits, the website and the first-boot seed."""

    def setUp(self):
        import threading
        from exchange import serve
        self.ex, self.srv = serve(0, ":memory:", public=True, admin_token="op-secret", test_credits=5_000_000,
                                  validators=(A("5"),))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def raw(self, method, path, body=None, headers=None):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.srv.server_address[1], timeout=10)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        out = (r.status, r.getheader("Content-Type"), r.read())
        c.close()
        return out

    def test_operator_calls_need_the_token(self):
        for path in ("/v0/epochs/settle", "/v0/epochs/clear", "/v0/learnings", "/v0/checkers"):
            with self.assertRaisesRegex(RuntimeError, "^403"):
                Client(self.url)._call("POST", path, {})
        self.assertEqual(Client(self.url, token="op-secret").settle()["epoch"], 1)
        with self.assertRaisesRegex(RuntimeError, "^403"):
            Client(self.url, token="wrong").clear()
        Client(self.url, A("8")).faucet()
        b = Client(self.url, A("8")).post_bounty(title="t", path="extract", eval_set="sha256:E", target=.5)
        Client(self.url, A("8")).buy_coins(b["id"], 1_000_000)
        with self.assertRaisesRegex(RuntimeError, "^403"):          # transfers wait for signed wallets
            Client(self.url, A("8")).transfer_coins(b["id"], A("9"), 1)

    def test_testnet_wallets_must_cover_every_spend(self):
        me = Client(self.url, A("8"))
        with self.assertRaisesRegex(RuntimeError, "not enough test credits"):
            me.post_bounty(title="t", path="extract", eval_set="sha256:E", target=.5, seed_micros=1_000_000)
        w = me.faucet()
        self.assertEqual((w["grant_micros"], w["balance_micros"]), (5_000_000, 5_000_000))
        self.assertTrue(me.faucet()["already"])                     # once per address
        b = me.post_bounty(title="t", path="extract", eval_set="sha256:E", target=.5)
        with self.assertRaisesRegex(RuntimeError, "not enough test credits"):
            me.buy_coins(b["id"], 6_000_000)
        me.buy_coins(b["id"], 2_000_000)
        w = me.wallet()
        self.assertEqual(w["balance_micros"], 3_000_000)
        self.assertGreater(w["coins"][str(b["id"])], 0)
        me.sell_coins(b["id"], w["coins"][str(b["id"])])
        self.assertEqual(me.wallet()["balance_micros"], 5_000_000)  # the curve buys back at cost while it's open
        with self.assertRaisesRegex(RuntimeError, "(?s)^400.*40 hex"):
            Client(self.url, "0xnope").faucet()
        for i in range(2):                                           # three new wallets per source per day
            Client(self.url, A(str(i))).faucet()
        with self.assertRaisesRegex(RuntimeError, "^403"):
            Client(self.url, A("3")).faucet()
        texts = [e["text"] for e in me.events()["events"]]
        self.assertTrue(any(t.startswith("bounty #1 posted free") for t in texts), texts)
        self.assertEqual(me.stats()["bounties_open"], 1)

    def test_bounty_inputs_are_bounded(self):
        me = Client(self.url, A("8"))
        for bad in ({"path": "Extract/../x"}, {"title": "x" * 500}, {"target": 3}):
            args = dict(title="t", path="extract", eval_set="sha256:E", target=.5)
            args.update(bad)
            with self.assertRaisesRegex(RuntimeError, "^400"):
                me.post_bounty(**args)
        status, _, body = self.raw("POST", "/v0/bounties", b"{" + b" " * 70_000 + b"}",
                                   {"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertIn(b"over 64 KB", body)

    def test_serves_the_site_and_nothing_else(self):
        status, ctype, body = self.raw("GET", "/")
        self.assertEqual((status, ctype), (200, "text/html; charset=utf-8"))
        self.assertIn(b"traceX", body)
        self.assertEqual(self.raw("GET", "/healthz")[0], 200)
        for path in ("/site/../node/exchange.py", "/../README.md", "/node/exchange.py", "/site/%2e%2e/LICENSE"):
            self.assertEqual(self.raw("GET", path)[0], 404, path)

    def test_bad_input_cannot_stall_settlement_or_inject_markup(self):
        ex = Exchange(":memory:", validators=(A("5"),))
        t = make_trace()
        for bad, why in ((dict(t, producer="bob"), "producer"),
                         (dict(t, failure_modes={"outbound_flight_number": "<img src=x onerror=alert(1)>"}), "failure modes"),
                         (dict(t, task="<b>x</b>"), "task"),
                         (dict(t, input=t["input"] + " pad" * 9000), "KB")):
            with self.assertRaisesRegex(ValueError, why):
                ex.submit_trace(bad)
        for call in (lambda: ex.buy_coins(1, "bob", 10), lambda: ex.bid({"lot": "x", "bidder": "bob", "price_micros": 5}),
                     lambda: ex.post_bounty({"poster": "bob", "path": "extract", "eval_set": "sha256:E", "target": .5})):
            with self.assertRaisesRegex(ValueError, "0x address"):
                call()
        ex._credit("not-an-address", 5, "legacy row")              # even a bad row already in the ledger
        ex._credit(A("a"), 7, "ok")
        s = ex.settle()
        self.assertEqual(list(s["claims"]), [A("a")])

    def test_operator_takedown(self):
        Client(self.url, A("8")).faucet()
        me = Client(self.url, A("8"))
        b = me.post_bounty(title="spam", path="extract", eval_set="sha256:E", target=.5)
        me.buy_coins(b["id"], 1_000_000)
        r = me.submit(make_trace(producer=A("8")))
        with self.assertRaisesRegex(RuntimeError, "^403"):
            me._call("POST", "/v0/admin/remove", {"kind": "bounty", "id": b["id"]})
        op = Client(self.url, token="op-secret")
        op._call("POST", "/v0/admin/remove", {"kind": "bounty", "id": b["id"]})
        op._call("POST", "/v0/admin/remove", {"kind": "trace", "id": r["id"]})
        self.assertEqual(me.bounties(status="")["bounties"], [])
        self.assertEqual(me.search()["count"], 0)
        self.assertFalse(any("spam" in e["text"] for e in me.events()["events"]))
        self.assertEqual(me.wallet()["balance_micros"], 5_000_000)  # the pool went back to its backer

    def test_full_disk_stops_new_writes_only(self):
        import tempfile
        from exchange import Full
        with tempfile.TemporaryDirectory() as d:
            ex = Exchange(os.path.join(d, "x.db"), max_db_bytes=1)
            with self.assertRaises(Full):
                ex.submit_trace(make_trace())
            self.assertEqual(ex.stats()["traces"], 0)
            ex.db.close()

    def test_rate_limit(self):
        from exchange import RateLimit
        r = RateLimit(reads=2, writes=1)
        self.assertEqual([r.allow("x", "read") for _ in range(3)], [True, True, False])
        self.assertEqual([r.allow("x", "write"), r.allow("y", "write")], [True, True])

    def test_seed_once(self):
        from seed import seed_if_empty
        ex = Exchange(":memory:", test_credits=25_000_000, validators=(A("5"),))
        seed_if_empty(ex)
        s = ex.stats()
        self.assertEqual((s["traces"], s["learnings"], s["bounties_open"], s["bounties_solved"], s["epoch"]),
                         (247, 2, 2, 1, 2))
        self.assertIn("left alone", seed_if_empty(ex))
        lora = [L for L in ex.find_learnings(path="code/generate")["learnings"] if L["kind"] == "lora"][0]
        self.assertEqual((lora["before"], lora["after"]), (0.294, 0.372))
        self.assertLess(lora["p_value"], 0.001)
        self.assertEqual(lora["first_try"]["after"], 0.288)


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
