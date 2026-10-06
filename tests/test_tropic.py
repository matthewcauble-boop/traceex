"""traceX v0.8: step graphs, tropical composition across contributors (after TROPIC, arXiv 2610.02478), credit for the
step traces a verified path is built from, and the node's replay-on-arrival.   python -m pytest -q tests/test_tropic.py"""
import hashlib
import json
import math
import os
import sys
import threading
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node"),
                os.path.join(ROOT, "examples", "tropical_composition")]

from traceex import Trace, Learning, attest, Client  # noqa: E402
from traceex import tropic as T  # noqa: E402
from traceex.royalty import split_trace_sale, split_usage  # noqa: E402
from sats import SatsExchange, Params, attestation_digest  # noqa: E402
from exchange import Exchange, serve  # noqa: E402
import countdown as cd  # noqa: E402

A = lambda c: "0x" + c * 40
V = lambda i: "0x" + f"{0xa0 + i:02x}" * 20
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
PROBLEM = {"target": 24, "nums": [2, 3, 4]}
ROOT_STATE = cd.root(PROBLEM)


def walk(state, actions):
    steps = []
    for a in actions:
        state = cd.step(state, a)
        steps.append({"action": a, "state": state, "success": cd.check(state)})
    return steps


def verify(g, problem=PROBLEM):
    return lambda path: (T.replay(g, path, cd.step, cd.check)
                         and cd.expression_check(problem, [g.edges[k]["action"] for k in path]))


class Core(unittest.TestCase):
    def test_values_are_max_plus_and_paths_explicit(self):
        g = T.StepGraph(ROOT_STATE, "p")
        g.add_fragment("f1", A("1"), walk(ROOT_STATE, ["2 * 3", "6 * 4"]))      # 24: passes
        g.add_fragment("f2", A("2"), walk(ROOT_STATE, ["3 * 4", "12 * 2"]))     # 24: passes, another way
        scores = {k: (-0.1 if e["action"] in ("2 * 3", "6 * 4") else -2.0) for k, e in g.edges.items()}
        g.set_scores(scores)
        self.assertEqual(len(T.admit(g, verify(g))), 2)
        vals = g.values()
        self.assertAlmostEqual(vals[g.root]["suffix"], -0.2)                     # the max over paths, not the sum
        self.assertAlmostEqual(vals[g.root]["prefix"], 0.0)
        best = g.best_path()
        self.assertEqual([g.edges[k]["action"] for k in best], ["2 * 3", "6 * 4"])
        self.assertAlmostEqual(g.score(best), -0.2)

    def test_a_failed_prefix_and_another_agents_suffix_join_into_a_verified_path(self):
        g = T.StepGraph(ROOT_STATE, "p")
        g.add_fragment("a-fail", A("a"), walk(ROOT_STATE, ["2 * 3"]))           # A gets to [4, 6], then stops
        mid = cd.step(ROOT_STATE, "2 * 3")
        self.assertEqual([f["state"] for f in g.frontier_states(5, alive=cd.alive)], [mid])
        prefix = g.choose_frontier(5, alive=cd.alive)
        state, depth = g.end_state(prefix)
        g.add_fragment("b-restart", A("b"), walk(state, ["4 * 6"]), start_state=state, start_depth=depth, prefix=prefix)
        new = T.admit(g, verify(g))
        self.assertEqual(len(new), 1)
        path, origin = new[0]
        self.assertEqual(origin, "restart")
        self.assertEqual(g.producers(path), [A("a"), A("b")])                   # provenance: both agents' steps
        self.assertEqual(T.step_sources(g, path), ["a-fail", "b-restart"])
        self.assertEqual(g.fragments["a-fail"]["outcome"], "fail")
        self.assertEqual(T.credit(g, path), {"b-restart": 1.0})                  # payment: the passing trace only
        g.add_fragment("c-copy", A("c"), walk(ROOT_STATE, ["3 * 2", "4 * 6"]))  # wraps A's step in a passing attempt
        T.admit(g, verify(g))
        for p in g.solutions:
            self.assertNotIn("c-copy", T.credit(g, p))                          # re-filed steps aren't new work

    def test_composition_joins_prefixes_and_suffixes_from_different_rollouts(self):
        g = T.StepGraph({"target": 10, "nums": ["1", "2", "3", "4"]}, "p")
        root = g.nodes[g.root]["state"]
        g.add_fragment("x", "X", walk(root, ["1 + 2", "3 * 3", "9 + 4"]))       # 13: a failed complete attempt
        g.add_fragment("y", "Y", walk(root, ["2 + 1"]))                         # also reaches [3, 3, 4]
        g.add_fragment("z", "Z", walk(root, ["4 + 3", "7 * 1", "7 + 2"]))       # 9: fails
        g.add_fragment("w", "W", walk(root, ["1 + 2", "3 + 3", "6 + 4"]))       # 10: passes
        problem = {"target": 10, "nums": [1, 2, 3, 4]}
        self.assertEqual(len(T.admit(g, verify(g, problem))), 1)
        joined = T.compose(g, verify(g, problem))                              # "2 + 1" then W's suffix
        self.assertTrue(any(g.edges[p[0]]["action"] == "2 + 1" for p, _ in joined))
        for p, _ in joined:
            self.assertTrue(T.replay(g, p, cd.step, cd.check))

    def test_a_join_that_does_not_replay_is_refused_and_remembered(self):
        g = T.StepGraph(ROOT_STATE, "p")
        g.add_fragment("liar", A("e"), [{"action": "2 * 3", "state": {"target": 24, "nums": ["24"]}, "success": True}])
        self.assertEqual(T.admit(g, verify(g)), [])                             # its claimed step never happens
        path = tuple(g.fragments["liar"]["edges"])
        self.assertFalse(T.replay(g, path, cd.step, cd.check))
        g.reject_join(path)
        self.assertEqual(g.candidates(), [])
        self.assertEqual(g.solutions, {})

    def test_credit_is_per_passing_trace_and_only_for_transitions_it_filed_first(self):
        g = T.StepGraph(ROOT_STATE, "p")
        g.add_fragment("honest", A("1"), walk(ROOT_STATE, ["2 * 3", "6 * 4"]))
        g.add_fragment("reword", A("2"), walk(ROOT_STATE, ["3 * 2", "4 * 6"]))  # the same transitions, other words
        g.set_scores({k: (-0.01 if e["action"] in ("3 * 2", "4 * 6") else -1.0) for k, e in g.edges.items()})
        T.admit(g, verify(g))
        best = g.best_path()
        self.assertEqual([g.edges[k]["action"] for k in best], ["3 * 2", "4 * 6"])   # the likelier wording is used
        self.assertEqual(T.producer_credit(g, best), {A("1"): 1.0})                   # the credit stays with the first
        g2 = T.StepGraph(ROOT_STATE, "p")                                        # the same work, split in two fragments
        g2.add_fragment("part1", A("1"), walk(ROOT_STATE, ["2 * 3"]))
        mid = cd.step(ROOT_STATE, "2 * 3")
        g2.add_fragment("part2", A("1"), walk(mid, ["6 * 4"]), start_state=mid, start_depth=1,
                        prefix=g2.best_prefix(mid, 1))
        T.admit(g2, verify(g2))
        self.assertEqual(T.producer_credit(g2, g2.best_path()), {A("1"): 1.0})

    def test_two_passing_traces_split_equally_and_a_path_with_none_pays_nobody(self):
        root = {"target": 10, "nums": ["1", "2", "3", "4"]}
        problem = {"target": 10, "nums": [1, 2, 3, 4]}
        g = T.StepGraph(root, "p")
        g.add_fragment("p1", "P1", walk(root, ["1 + 2", "3 + 3", "6 + 4"]))      # passes from the root
        mid = cd.step(root, "1 + 2")
        g.add_fragment("p2", "P2", walk(mid, ["4 + 3", "7 + 3"]), start_state=mid, start_depth=1,
                       prefix=g.best_prefix(mid, 1))                            # passes after P1's first step
        g.add_fragment("f", "F", walk(mid, ["4 * 3"]), start_state=mid, start_depth=1)   # fails
        T.admit(g, verify(g, problem))
        restart = next(p for p, o in g.solutions.items() if o == "restart")
        self.assertEqual(T.credit(g, restart), {"p1": 0.5, "p2": 0.5})          # per whole trace, not per step
        g.fragments["p1"]["producer"] = g.fragments["p2"]["producer"] = "node:x"  # no passing contributor left
        self.assertEqual(T.credit(g, restart), {})
        self.assertEqual(T.producer_credit(g, restart), {})

    def test_restarts_must_start_inside_the_graph(self):
        g = T.StepGraph(ROOT_STATE, "p")
        with self.assertRaises(ValueError):
            g.add_fragment("x", "X", walk({"target": 24, "nums": ["4", "6"]}, ["4 * 6"]),
                           start_state={"target": 24, "nums": ["4", "6"]}, start_depth=1)

    def test_producer_quota_caps_what_one_producer_can_fill(self):
        g = T.StepGraph({"target": 99, "nums": [str(i) for i in range(1, 7)]}, "p", max_edges_per_producer=3)
        root = g.nodes[g.root]["state"]
        for i, a in enumerate(["1 + 2", "1 + 3", "1 + 4", "1 + 5", "1 + 6"]):
            g.add_fragment(f"s{i}", "spammer", walk(root, [a]))
        self.assertEqual(len(g.edges), 3)
        g.add_fragment("h", "honest", walk(root, ["2 + 3"]))
        self.assertEqual(len(g.edges), 4)

    def test_loops_are_cut_out_and_kept_out_of_the_basis(self):
        step = lambda st, a: ({"pos": st["pos"] + {"+1": 1, "-1": -1, "+2": 2}[a], "goal": st["goal"]}
                              if a in ("+1", "-1", "+2") else None)
        check = lambda st: st["pos"] == st["goal"]
        root = {"pos": 0, "goal": 3}
        g = T.StepGraph(root, "line")
        g.add_fragment("h", "H", [{"action": "+1", "state": {"pos": 1, "goal": 3}, "success": False}])
        one = {"pos": 1, "goal": 3}
        st, steps = one, []
        for a in ["+1", "-1", "+1", "-1", "+2"]:
            st = step(st, a)
            steps.append({"action": a, "state": st, "success": check(st)})
        g.add_fragment("pad", "P", steps, start_state=one, start_depth=1, prefix=g.best_prefix(one, 1))
        (padded, _), = T.admit(g, lambda p: T.replay(g, p, step, check))
        self.assertTrue(g.loops(padded))
        self.assertEqual(T.producer_credit(g, padded), {"P": 1.0})               # one passing trace, however long
        short = T.shortcut(g, padded, step, check)
        self.assertEqual([g.edges[k]["action"] for k in short], ["+1", "+2"])
        self.assertEqual(T.producer_credit(g, short), {"P": 1.0})                # H's failed attempt: unpaid
        self.assertEqual(g.select_basis(2), [short])                                # and stay out of training

    def test_round_trip_and_tropic_exports(self):
        g = T.StepGraph(ROOT_STATE, "p")
        g.add_fragment("f1", "A", walk(ROOT_STATE, ["2 * 3", "6 * 4"]))
        g.add_fragment("f2", "B", walk(ROOT_STATE, ["3 * 4", "12 * 2"]))
        T.admit(g, verify(g))
        h = T.StepGraph.from_dict(json.loads(json.dumps(g.to_dict())))
        self.assertEqual(h.solutions, g.solutions)
        self.assertEqual(h.values(), g.values())
        g3 = T.StepGraph({"target": 5, "nums": ["2", "3"]}, "q")
        g3.add_fragment("f3", "C", walk(g3.nodes[g3.root]["state"], ["2 + 3"]))
        T.admit(g3, lambda p: T.replay(g3, p, cd.step, cd.check))
        rows = T.export_tropic([g, g3], basis_size=2)
        self.assertAlmostEqual(sum(r["weight"] for r in rows), 1.0)               # each problem counts once
        self.assertAlmostEqual(sum(r["weight"] for r in rows if r["problem"] == "q"), 0.5)
        ck = T.to_tropic_checkpoint([g], lambda s: [1, 2, len(s["nums"])], lambda a: [7, len(a)])
        graph = ck["graphs"][0]
        keys = set()
        for e in graph["edges"]:                                                 # TROPIC's Edge.key, recomputed
            fields = [e["source"], e["target"], e["action"], e["prompt_ids"], e["completion_ids"]]
            keys.add(hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest())
        self.assertTrue(all(k in keys for path, _ in graph["solutions"] for k in path))
        self.assertEqual(len(graph["solutions"]), 2)

    def test_fragment_builder_and_scan(self):
        f = T.fragment(failure_id="TXF-2026-000001", task="math.countdown", base_model=QWEN, checker="countdown@1",
                       root=ROOT_STATE, steps=walk(ROOT_STATE, ["2 * 3"]), producer=A("a"))
        self.assertEqual(f["v"], "steps/0.1")
        self.assertEqual(T.fragment_leaks(f), [])
        f["steps"][0]["action"] = "call me on +1 415 555 0100"
        self.assertTrue(T.fragment_leaks(f))
        self.assertIn("tropic", Learning.KINDS)


def failure_trace(producer=A("b")):
    return Trace.from_fix(task="math.countdown", base_model=QWEN, input="Target 24 from the numbers 2 3 4.",
                          model_output={"answer": "2 + 3 + 4"}, verified_output={"answer": "2 * 3 * 4"},
                          checker="countdown@1", producer=producer, created="2026-10-06T00:00:00Z", privacy="open",
                          failure_modes={"answer": "wrong_answer"})


def countdown_node(sats=True):
    if sats:
        ex = SatsExchange(":memory:", test_credits=30_000_000, beacon_delay=0, params=Params(quorum=3), fee_to=A("f"))
        for i in range(5):
            ex.faucet(V(i))
            ex.register_validator(V(i), 15_000_000)
    else:
        ex = Exchange(":memory:")
    for w in "abcde12":
        if sats:
            ex.faucet(A(w))
    ex.register_checker("countdown", A("d"))
    ex.register_env("countdown@1", cd.step, cd.check, cd.alive)
    fid = ex.submit_trace(dict(failure_trace()))["failure_id"]
    return ex, fid


def frag(fid, producer, actions, start=None, root=ROOT_STATE):
    begin = start["state"] if start else root
    return T.fragment(failure_id=fid, task="math.countdown", base_model=QWEN, checker="countdown@1", root=root,
                      steps=walk(begin, actions), producer=producer, start=start, created="2026-10-06T00:00:00Z")


def accept(ex, lid):
    att = lambda v: {"validator": v, "eval_set": "sha256:p" + v[-2:], "metric": "pass@1", "before": .6, "after": .7,
                     "n": 300, "audit": {"checked": 10, "bad": 0}}
    for v in ex.verdict(lid)["assigned"]:
        ex.commit(lid, v, attestation_digest(att(v), "s"))
    for v in ex.verdict(lid)["assigned"]:
        ex.reveal(lid, v, att(v), "s")
    return ex.verdict(lid)["status"]


class Node(unittest.TestCase):
    def test_two_producers_steps_compose_on_the_node_and_both_are_paid(self):
        ex, fid = countdown_node()
        r1 = ex.submit_fragment(fid, frag(fid, A("a"), ["2 * 3"]))               # A: a failed attempt, one valid step
        self.assertTrue(r1["accepted"])
        self.assertEqual((r1["outcome"], r1["joins"]), ("fail", []))
        front = ex.frontier(fid)["graphs"][0]["frontier"]
        self.assertEqual(front[0]["state"], cd.step(ROOT_STATE, "2 * 3"))
        start = {"state": front[0]["state"], "depth": front[0]["depth"]}
        r2 = ex.submit_fragment(fid, frag(fid, A("e"), ["4 * 6"], start=start))   # B restarts there and finishes
        self.assertEqual(len(r2["joins"]), 1)
        join = r2["joins"][0]
        self.assertEqual(join["actions"], ["2 * 3", "4 * 6"])
        self.assertEqual(join["producers"], {A("e"): 1.0})                       # the failed attempt is unpaid
        t = ex.get_trace(join["trace"])
        self.assertEqual((t["failure_id"], t["composed"]["origin"]), (fid, "restart"))
        self.assertEqual(ex.provenance(join["trace"])["fragments"], join["fragments"])
        self.assertEqual(ex.step_graphs(fid)["graphs"][0]["verified_paths"], 1)
        f = ex.get_failure(fid)                                                  # but it is a reported case
        self.assertEqual(f["occurrences"], 2)
        self.assertEqual([c["step_trace"] for c in f["step_trace_cases"]], [r1["id"]])
        before = {a: ex.wallet(a)["balance_msats"] for a in (A("a"), A("e"))}
        L = Learning.build(kind="tropic", task="math.countdown", base_model=QWEN, artifact={"uri": "w", "hash": None},
                           parents=[(join["trace"], 1)], trainer=A("c"), per_call_msats=1_000,
                           attestation=attest(V(0), "sha256:c", "pass@1", .6, .7))
        lid = ex.register_learning(L)["id"]
        self.assertEqual(accept(ex, lid), "accepted")
        ex.usage({"learning": lid, "consumer": A("2"), "calls": 10_000})        # 10,000 sats of paid use
        for _ in range(ex.p.vest_epochs + 2):
            ex.settle()
        got = {a: ex.wallet(a)["balance_msats"] - before[a] for a in before}
        self.assertEqual(got[A("a")], 0)                                        # nothing pays a failed attempt
        self.assertEqual(got[A("e")], 10_000_000 * 60 // 100)                   # the whole traces share
        self.assertTrue(ex.audit()["balanced"])

    def test_a_path_with_no_passing_trace_returns_its_share_to_the_payer(self):
        ex, fid = countdown_node()
        ex.submit_fragment(fid, frag(fid, A("a"), ["2 * 3", "6 * 4"]))
        (tid,) = [j["trace"] for j in ex.joins(fid)["joins"]]
        ex.db.execute("UPDATE joins SET producers='{}' WHERE trace=?", (tid,))  # as if no passing trace were used
        L = Learning.build(kind="tropic", task="math.countdown", base_model=QWEN, artifact={"uri": "v", "hash": None},
                           parents=[(tid, 1)], trainer=A("c"), per_call_msats=1_000,
                           attestation=attest(V(0), "sha256:c", "pass@1", .6, .7))
        lid = ex.register_learning(L)["id"]
        self.assertEqual(accept(ex, lid), "accepted")
        payer = ex.wallet(A("2"))["balance_msats"]
        ex.usage({"learning": lid, "consumer": A("2"), "calls": 10_000})
        for _ in range(ex.p.vest_epochs + 2):
            ex.settle()
        spent = payer - ex.wallet(A("2"))["balance_msats"]
        self.assertEqual(spent, 10_000_000 - 6_000_000 + ex.tx_fee_msats)       # the traces share came back
        self.assertEqual(ex.wallet(A("a"))["balance_msats"] - 30_000_000 + ex.tx_fee_msats, 0)
        self.assertTrue(ex.audit()["balanced"])

    def test_steps_that_do_not_replay_are_refused_and_pay_their_fee(self):
        ex, fid = countdown_node()
        lie = frag(fid, A("e"), ["2 * 3"])
        lie["steps"][0]["state"] = {"target": 24, "nums": ["24"]}               # claims a state the step never reaches
        lie["steps"][0]["success"] = True
        before = ex.wallet(A("e"))["balance_msats"]
        r = ex.submit_fragment(fid, lie)
        self.assertFalse(r["accepted"])
        self.assertEqual(before - ex.wallet(A("e"))["balance_msats"], ex.tx_fee_msats)
        self.assertEqual(ex.step_graphs(fid)["graphs"], [])
        self.assertEqual(ex.step_graphs(fid)["refused_fragments"], 1)
        bad = frag(fid, A("e"), ["2 * 3"])
        bad["steps"][0]["action"] = "2 * 99"
        self.assertFalse(ex.submit_fragment(fid, bad)["accepted"])
        nowhere = frag(fid, A("e"), ["4 * 6"], start={"state": {"target": 24, "nums": ["4", "6"]}, "depth": 1})
        self.assertIn("frontier", ex.submit_fragment(fid, nowhere)["reason"])
        other = dict(frag(fid, A("e"), ["2 * 3"]), checker={"id": "my-own-checker", "version": "1", "hash": None})
        with self.assertRaises(ValueError):                                     # only the operator's environments replay
            ex.submit_fragment(fid, other)
        with self.assertRaises(KeyError):
            ex.submit_fragment("TXF-2026-999999", dict(frag(fid, A("e"), ["2 * 3"]), failure_id="TXF-2026-999999"))
        self.assertTrue(ex.audit()["balanced"])

    def test_a_licence_sale_of_a_path_trace_pays_its_contributors_exactly(self):
        out = split_trace_sale(1_000_001, {A("a"): 0.5, A("e"): 0.5}, A("d"), [V(0)])
        self.assertEqual(sum(out.values()), 1_000_001)
        self.assertEqual(abs(out[A("a")] - out[A("e")]) <= 1, True)
        L = {"trainer": A("c"), "parents": [{"trace": "j", "weight": 1}],
             "royalty": {"split": {"traces": .6, "trainer": .25, "checkers": .1, "validators": .05}}}
        pay = split_usage(999_999, L, {"j": {"producer": {A("a"): 2, A("e"): 1}, "checker_author": A("d")}}, [V(0)])
        self.assertEqual(sum(pay.values()), 999_999)
        self.assertGreater(pay[A("a")], pay[A("e")])

    def test_dollar_node_composes_too(self):
        ex, fid = countdown_node(sats=False)
        ex.submit_fragment(fid, frag(fid, A("a"), ["3 * 4", "12 * 2"]))
        self.assertEqual(len(ex.joins(fid)["joins"]), 1)

    def test_http_routes(self):
        ex, srv = serve(0, ":memory:", economy="sats", public=True, admin_token="op", test_credits=30_000_000)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            ex.register_env("countdown@1", cd.step, cd.check, cd.alive)
            for w in "ab":
                Client(url).faucet(A(w)) if hasattr(Client, "faucet") else ex.faucet(A(w))
            fid = Client(url, A("b")).submit(dict(failure_trace()))["failure_id"]
            c = Client(url, A("a"))
            self.assertTrue(c.submit_fragment(frag(fid, A("a"), ["2 * 3"]))["accepted"])
            front = c.frontier(fid)["graphs"][0]["frontier"][0]
            r = c.submit_fragment(frag(fid, A("b"), ["4 * 6"], start={"state": front["state"], "depth": front["depth"]}))
            self.assertEqual(len(r["joins"]), 1)
            self.assertEqual(c.joins(fid)["joins"][0]["producers"], {A("b"): 1.0})
            self.assertEqual(c.step_graphs(fid)["graphs"][0]["verified_paths"], 1)
            with self.assertRaises(Exception):
                bad = frag(fid, A("a"), ["2 * 3"])
                bad["steps"][0]["state"] = "ring +1 415 555 0100"
                c.submit_fragment(bad)
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
