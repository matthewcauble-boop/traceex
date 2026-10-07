"""Challenge bounties (SPEC 4k, node/challenges.py): python -m pytest -q tests/test_challenges.py

Big open problems with a deterministic verifier, a direction and a public leaderboard, paid per verified improvement
out of refundable pledges. Every test ends with the books balanced (audit()): every payout is a split of a real payment."""
import hashlib
import json
import os
import sys
import threading
import unittest
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node")]

from traceex import Trace  # noqa: E402
from traceex import challenges as C  # noqa: E402
from traceex.verifiers import (circle_packing_sum_radii, tammes_min_distance, lean_gate, Invalid,  # noqa: E402
                               similarity)
from sats import SatsExchange, Params  # noqa: E402
from registry import measurement_digest  # noqa: E402

A = lambda c: "0x" + c * 40
V = lambda i: "0x" + f"{0xa0 + i:02x}" * 20
POSTER, SOLVER, BACKER, AUTHOR, OTHER, OPERATOR = A("b"), A("c"), A("d"), A("e"), A("7"), A("f")
FEE = 58
SAMPLE = os.path.join(ROOT, "examples", "challenges", "sample")


def toy(sol, inst):
    x = sol.get("x") if isinstance(sol, dict) else None
    return float(x) if isinstance(x, (int, float)) and 0 <= x <= 100 else None


def make_ex(validators=3, **params):
    ex = SatsExchange(":memory:", test_credits=30_000_000, beacon_delay=0, params=Params(quorum=3, **params),
                      fee_to=OPERATOR)
    for i in range(validators):
        ex.faucet(V(i))
        ex.register_validator(V(i), 15_000_000)
    for w in (POSTER, SOLVER, BACKER, AUTHOR, OTHER):
        ex.faucet(w)
    ex.register_verifier("toy-max@1", toy, author=AUTHOR)
    return ex


def toy_challenge(key="toy:1", target=20.0, baseline=10.0, **metric):
    return {"v": "challenge/0.1", "title": "Toy: make x large", "statement": "Maximize x in [0, 100].",
            "path": "math/toy", "key": key, "metric": dict({"direction": "maximize", "baseline": baseline,
                                                            "target": target}, **metric),
            "verifier": {"id": "toy-max@1", "kind": "python"}}


def held(ex, who):
    w = ex.wallet(who)
    return w["balance_msats"] + w["vesting_msats"]


def balanced(test, ex):
    a = ex.audit()
    test.assertTrue(a["balanced"], a)


class Curve(unittest.TestCase):
    """The payout curve depends on the best score alone, so payouts telescope."""

    def test_many_small_steps_pay_exactly_what_one_step_pays(self):
        for target in (20.0, None):
            m = C.normalize(toy_challenge(target=target, scale=None if target else 2.0))["metric"]
            amount, total, best = 7_777_777, 0, 10.0
            for k in range(1, 41):
                new = 10.0 + 0.2 * k
                total += C.owed(amount, m, 10.0, new) - C.owed(amount, m, 10.0, best)
                best = new
            self.assertEqual(total, C.owed(amount, m, 10.0, 18.0))
            self.assertLessEqual(C.owed(amount, m, 10.0, 1e9), amount)

    def test_with_a_target_half_streams_and_half_waits_for_the_target(self):
        m = C.normalize(toy_challenge())["metric"]
        self.assertEqual(C.owed(1_000_000, m, 10, 15), 250_000)      # halfway: half of the streaming half
        self.assertAlmostEqual(C.owed(1_000_000, m, 10, 19.99), 499_500, delta=1)   # floor of floats
        self.assertEqual(C.owed(1_000_000, m, 10, 20), 1_000_000)    # the target releases everything
        m = C.normalize(toy_challenge(target=None, scale=1.0))["metric"]
        self.assertEqual(C.owed(1_000_000, m, 10, 11), 500_000)      # no target: each `scale` releases half of the rest
        self.assertEqual(C.owed(1_000_000, m, 10, 12), 750_000)
        self.assertEqual(C.owed(1_000_000, m, 11, 12), 500_000)      # a later pledge pays only for progress after it

    def test_minimize_and_the_minimum_step(self):
        m = C.normalize(dict(toy_challenge(target=None), metric={"direction": "-", "baseline": 5.0, "min_step": 0.1,
                                                                 "min_step_rel": 0.05}))["metric"]
        self.assertEqual(m["direction"], "minimize")
        self.assertFalse(C.improves(m, 5.0, 4.8))                     # 0.2 < 5% of 5
        self.assertTrue(C.improves(m, 5.0, 4.75))
        self.assertFalse(C.improves(m, 5.0, 4.7, se=0.05))            # beyond twice the validators' error


class Paying(unittest.TestCase):
    def setUp(self):
        self.ex = make_ex()
        self.cid = self.ex.post_challenge(dict(toy_challenge(), poster=POSTER))["id"]
        self.ex.pledge_challenge(self.cid, BACKER, 10_000_000)

    def test_posting_is_free_and_a_verified_improvement_pays_70_20_5_5_vesting(self):
        ex = self.ex
        self.assertEqual(ex.wallet(POSTER)["balance_msats"], 30_000_000 - FEE)          # the fee only
        s0, a0 = held(ex, SOLVER), held(ex, AUTHOR)
        r = ex.submit_solution(self.cid, {"submitter": SOLVER, "solution": {"x": 12}})
        self.assertEqual((r["status"], r["score"], r["bond_msats"]), ("best", 12.0, 1_000_000))   # held: prior art
        # 12 is 20% of the way to 20: 10% of the pledge streams out (half the pledge streams, half waits for the target)
        self.assertEqual(ex.challenge_backers(self.cid)["pledges"][0]["released_msats"], 1_000_000)
        self.assertEqual(ex.wallet(SOLVER)["vesting_msats"], 900_000)                   # solver 70 + its trace 20
        self.assertEqual(held(ex, AUTHOR) - a0, 50_000)                                # the verifier's author: 5%
        self.assertEqual(held(ex, SOLVER) - s0, 900_000 - FEE - 1_000_000)             # the bond waits too
        self.assertEqual(ex.wallet(SOLVER)["balance_msats"], 29_000_000 - FEE)         # all of it vests first
        for _ in range(ex.p.vest_epochs + 1):
            ex.settle()
        self.assertEqual(ex.wallet(SOLVER)["balance_msats"], 30_900_000 - FEE)         # tranche and bond, home
        L = ex.get_learning(r["learning"])
        self.assertEqual((L["kind"], L["verdict"]["status"]), ("challenge_solution", "accepted"))
        self.assertEqual(ex.get_trace(r["trace"])["trace"]["checker"]["id"], "toy-max")
        balanced(self, ex)

    def test_below_the_minimum_step_duplicates_and_invalid_submissions_pay_nothing(self):
        ex = self.ex
        ex.submit_solution(self.cid, {"submitter": SOLVER, "solution": {"x": 12}})
        before = held(ex, OTHER)
        r = ex.submit_solution(self.cid, {"submitter": OTHER, "solution": {"x": 12.05}})   # min step 0.1
        self.assertEqual((r["status"], r["paid_msats"]), ("scored", 0))
        with self.assertRaisesRegex(ValueError, "already submitted"):
            ex.submit_solution(self.cid, {"submitter": OTHER, "solution": {"x": 12}})
        r = ex.submit_solution(self.cid, {"submitter": OTHER, "solution": {"x": 1000}})   # fails the gate
        self.assertEqual(r["status"], "invalid")
        self.assertEqual(held(ex, OTHER) - before, -2 * FEE)
        top = ex.leaderboard(self.cid)["leaderboard"]                                   # ranked by score...
        self.assertEqual([(s["submitter"], s["status"]) for s in top], [(OTHER, "scored"), (SOLVER, "best")])
        self.assertEqual(ex.get_challenge(self.cid)["best"], 12.0)                      # ...but only a step moves it
        balanced(self, ex)

    def test_reaching_the_target_releases_everything_and_solves_it(self):
        ex = self.ex
        ex.submit_solution(self.cid, {"submitter": SOLVER, "solution": {"x": 12}})
        r = ex.submit_solution(self.cid, {"submitter": OTHER, "solution": {"x": 21}})
        self.assertEqual(ex.get_challenge(self.cid)["status"], "solved")
        self.assertEqual(ex.challenge_backers(self.cid)["pledges"][0]["held_msats"], 0)
        self.assertEqual(r["paid_msats"], 9_000_000 * 90 // 100)
        with self.assertRaisesRegex(ValueError, "solved"):
            ex.pledge_challenge(self.cid, BACKER, 1_000)
        balanced(self, ex)

    def test_unreleased_pledges_come_back_at_expiry_and_unbacked_challenges_expire(self):
        ex = self.ex
        ex.submit_solution(self.cid, {"submitter": SOLVER, "solution": {"x": 12}})
        lonely = ex.post_challenge(dict(toy_challenge(key="toy:lonely"), poster=POSTER))["id"]
        b0 = ex.wallet(BACKER)["balance_msats"]
        for _ in range(30):
            ex.settle()
        st = {c["id"]: c["status"] for c in ex.challenges(status="")["challenges"]}
        self.assertEqual((st[self.cid], st[lonely]), ("expired", "expired"))
        self.assertEqual(ex.wallet(BACKER)["balance_msats"] - b0, 9_000_000)            # exactly what was left
        self.assertEqual(ex._row(lonely)["note"], "unbacked")
        balanced(self, ex)

    def test_a_pledge_pays_only_for_progress_beyond_its_from_score(self):
        ex = self.ex
        ex.pledge_challenge(self.cid, OTHER, 4_000_000, from_score=15)
        ex.submit_solution(self.cid, {"submitter": SOLVER, "solution": {"x": 14}})
        p = {x["backer"]: x for x in ex.challenge_backers(self.cid)["pledges"]}
        self.assertEqual(p[OTHER]["released_msats"], 0)
        ex.submit_solution(self.cid, {"submitter": SOLVER, "solution": {"x": 17.5}})
        p = {x["backer"]: x for x in ex.challenge_backers(self.cid)["pledges"]}
        self.assertEqual(p[OTHER]["released_msats"], 4_000_000 // 6)                   # (0.375 - 0.25) / (1 - 0.25)
        balanced(self, ex)

    def test_a_copy_of_the_best_cites_it_and_pays_its_producer(self):
        ex = make_ex()
        cid = ex.post_challenge(dict(toy_challenge(key="vec", target=None, baseline=4.0, scale=1.0, min_step=0.01), poster=POSTER,
                                     verifier={"id": "toy-sum@1", "kind": "python"}))["id"]
        ex.register_verifier("toy-sum@1", lambda s, i: float(sum(s["v"])) if all(0 <= x <= 1 for x in s["v"]) else None,
                             author=AUTHOR)
        ex.pledge_challenge(cid, BACKER, 10_000_000)
        first = ex.submit_solution(cid, {"submitter": SOLVER, "solution": {"v": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}})
        s0 = held(ex, SOLVER)
        copy = ex.submit_solution(cid, {"submitter": OTHER, "solution": {"v": [0.5, 0.6, 0.7, 0.8, 0.95, 1.0]}})
        L = ex.get_learning(copy["learning"])
        self.assertIn(first["trace"], [p["trace"] for p in L["parents"]])             # lineage, found by similarity
        self.assertGreater(held(ex, SOLVER) - s0, 0)                                   # the original's producer earns
        balanced(self, ex)

    def test_a_nondeterministic_verifier_is_refused(self):
        ex = make_ex()
        import itertools
        n = itertools.count()
        ex.register_verifier("flaky@1", lambda s, i: float(next(n)), author=AUTHOR)
        cid = ex.post_challenge(dict(toy_challenge(key="flaky"), verifier={"id": "flaky@1", "kind": "python"},
                                     poster=POSTER))["id"]
        r = ex.submit_solution(cid, {"submitter": SOLVER, "solution": {"x": 1}})
        self.assertEqual(r["status"], "invalid")

    def test_a_solution_is_reused_like_any_learning(self):
        ex = self.ex
        r = ex.submit_solution(self.cid, {"submitter": SOLVER, "solution": {"x": 12}, "per_call_msats": 100})
        s0 = held(ex, SOLVER)
        ex.usage({"learning": r["learning"], "consumer": OTHER, "calls": 1_000})       # 100,000 msats of use
        ex.settle()
        self.assertEqual(held(ex, SOLVER) - s0, 85_000)                               # trainer 25 + its trace 60
        with self.assertRaisesRegex(ValueError, "not re-measured"):
            ex.challenge(r["learning"], OTHER)
        balanced(self, ex)


class References(unittest.TestCase):
    """A baseline is only as good as its reference; nobody can squat a problem's key with a different test."""

    def test_a_verified_reference_sets_the_baseline_raises_the_best_and_rebases_pledges(self):
        ex = make_ex()
        with self.assertRaisesRegex(ValueError, "fails the verifier"):
            ex.post_challenge(dict(toy_challenge(key="ref"), poster=POSTER, reference={"x": 500}))
        cid = ex.post_challenge(dict(toy_challenge(key="ref"), poster=POSTER, reference={"x": 11}))["id"]
        self.assertEqual(ex.get_challenge(cid)["baseline"], 11.0)                     # the node scored it
        ex.pledge_challenge(cid, BACKER, 10_000_000)                                  # trusts 11; the record is 15
        r = ex.post_challenge(dict(toy_challenge(key="ref"), poster=OTHER, reference={"x": 15}))
        self.assertEqual((r["id"], r.get("merged")), (cid, True))
        self.assertEqual(ex.get_challenge(cid)["best"], 15.0)
        board = ex.leaderboard(cid)["leaderboard"]
        self.assertEqual([s["status"] for s in board], ["reference", "reference"])     # on the board, never paid
        self.assertEqual(ex.challenge_backers(cid)["pledges"][0]["from_score"], 15.0)  # rebased
        self.assertEqual(ex.submit_solution(cid, {"submitter": SOLVER, "solution": {"x": 15.05}})["status"], "scored")
        ex.submit_solution(cid, {"submitter": SOLVER, "solution": {"x": 16.8}})       # beyond the record: paid
        m = ex._body(cid)["metric"]
        self.assertEqual(ex.challenge_backers(cid)["pledges"][0]["released_msats"], C.owed(10_000_000, m, 15, 16.8))
        balanced(self, ex)

    def test_a_key_borrowed_with_another_verifier_opens_its_own_challenge(self):
        ex = make_ex()
        imp = ex.post_challenge(dict(toy_challenge(key="the-problem"), poster=POSTER))["id"]
        ex.register_verifier("rigged@1", lambda s, i: 99.0, author=OTHER)
        squat = ex.post_challenge(dict(toy_challenge(key="the-problem"), poster=OTHER,
                                       verifier={"id": "rigged@1", "kind": "python"}))
        self.assertNotEqual(squat["id"], imp)
        self.assertFalse(squat.get("merged"))
        self.assertEqual(ex.get_challenge(imp)["verifier"], "toy-max@1")


class PriorArt(unittest.TestCase):
    """A result that was already known pays nobody: prior-art claims during the vesting window."""

    def lure(self):
        ex = make_ex(validators=5)
        old = ex.post_challenge(dict(toy_challenge(key="elsewhere"), poster=OTHER))["id"]   # the record, on traceX before
        ex.pledge_challenge(old, OTHER, 1_000)
        ex.submit_solution(old, {"submitter": OTHER, "solution": {"x": 18}})
        cid = ex.post_challenge(dict(toy_challenge(key="lure"), poster=SOLVER))["id"]     # baseline 10 understated
        ex.pledge_challenge(cid, BACKER, 10_000_000)
        sub = ex.submit_solution(cid, {"submitter": SOLVER, "solution": {"x": 18}})
        self.assertEqual(ex.wallet(SOLVER)["vesting_msats"], 10_000_000 * 4 // 10 * 9 // 10)
        return ex, cid, sub

    def test_a_tracex_record_claws_back_the_lure_rebases_and_destroys_the_bond(self):
        ex, cid, sub = self.lure()
        w0, b0 = held(ex, POSTER), ex.wallet(SOLVER)["balance_msats"]
        c = ex.file_prior_art(cid, {"challenger": POSTER, "reference": {"x": 18}, "provenance": {"kind": "tracex"}})
        self.assertEqual(c["status"], "upheld")
        self.assertEqual(ex.wallet(SOLVER)["vesting_msats"], 0)                        # every tranche clawed back
        self.assertEqual(ex.submission(sub["id"])["status"], "prior_art")
        self.assertEqual(held(ex, POSTER) - w0, 500_000 - FEE)                         # reward, from the bond only
        g = ex.get_challenge(cid)
        self.assertEqual((g["best"], g["escrow_msats"]), (18.0, 10_000_000))           # all of it back in escrow
        self.assertEqual(ex.challenge_backers(cid)["pledges"][0]["from_score"], 18.0)
        for _ in range(30):
            ex.settle()
        self.assertEqual(ex.wallet(SOLVER)["balance_msats"], b0)                       # nothing came home: no bond
        self.assertEqual(ex.wallet(BACKER)["balance_msats"], 30_000_000 - FEE)         # the backer: whole again
        balanced(self, ex)

    def test_beyond_the_known_result_is_paid_again_for_the_new_part_only(self):
        ex, cid, sub = self.lure()
        better = ex.submit_solution(cid, {"submitter": AUTHOR, "solution": {"x": 19}})
        ex.file_prior_art(cid, {"challenger": POSTER, "reference": {"x": 18}, "provenance": {"kind": "tracex"}})
        m = ex._body(cid)["metric"]
        self.assertEqual(ex.submission(better["id"])["status"], "best")
        self.assertEqual(ex.challenge_backers(cid)["pledges"][0]["released_msats"], C.owed(10_000_000, m, 18, 19))
        self.assertEqual(ex.get_challenge(cid)["best"], 19.0)
        balanced(self, ex)

    def test_an_external_record_is_checked_by_validators_and_a_false_claim_loses_its_stake(self):
        ex, cid, sub = self.lure()
        before = held(ex, POSTER)
        c = ex.file_prior_art(cid, {"challenger": POSTER, "reference": {"x": 17},
                                    "provenance": {"kind": "external", "url": "https://example.org/x", "date": "2031-01-01"}})
        self.assertEqual(c["status"], "pending")
        ex.settle()                                                                   # paused: nothing vests meanwhile
        for v in c["validators"]:
            ex.commit_prior(c["id"], v, measurement_digest({"prior": False}, "s"))
        for v in c["validators"]:
            r = ex.reveal_prior(c["id"], v, {"prior": False}, "s")
        self.assertEqual(r["status"], "rejected")
        self.assertEqual(held(ex, POSTER) - before, -2_000_000 - FEE)
        self.assertGreater(ex.wallet(SOLVER)["vesting_msats"], 0)                     # the payout stands
        with self.assertRaisesRegex(ValueError, "earlier record"):
            ex.file_prior_art(cid, {"challenger": POSTER, "reference": {"x": 17.5}, "provenance": {"kind": "tracex"}})
        with self.assertRaisesRegex(ValueError, "starts below"):
            ex.file_prior_art(cid, {"challenger": POSTER, "reference": {"x": 9}, "provenance": {"kind": "tracex"}})
        balanced(self, ex)

    def test_once_the_window_closes_the_bond_and_tranche_go_home(self):
        ex, cid, sub = self.lure()
        for _ in range(ex.p.vest_epochs + 1):
            ex.settle()
        with self.assertRaisesRegex(ValueError, "still vesting"):
            ex.file_prior_art(cid, {"challenger": POSTER, "reference": {"x": 18}, "provenance": {"kind": "tracex"}})
        balanced(self, ex)


class Validators(unittest.TestCase):
    """Hidden instances: drawn validators run the verifier on their own instances; overfitting loses the bond."""

    def setUp(self):
        self.ex = make_ex(validators=5)
        self.ex.register_verifier("toy-public@1", toy, author=AUTHOR)
        self.cid = self.ex.post_challenge(dict(toy_challenge(key="hidden"), poster=POSTER,
                                               verifier={"id": "toy-public@1", "kind": "python"},
                                               instances={"hidden": {"digest": "sha256:h", "count": 50}}))["id"]
        self.ex.pledge_challenge(self.cid, BACKER, 10_000_000)

    def measure(self, sid, scores):
        ex = self.ex
        drawn = ex.submission(sid)["validators"]
        self.assertNotIn(ex.submission(sid)["submitter"], drawn)
        for v in drawn:
            ex.commit_solution(sid, v, measurement_digest({"score": scores, "se": 0.1}, "s" + v))
        for v in drawn:
            r = ex.reveal_solution(sid, v, {"score": scores, "se": 0.1}, "s" + v)
        return r

    def test_the_validators_median_sets_the_board_and_the_backers_judge_releases_the_money(self):
        ex = self.ex
        b0 = ex.wallet(SOLVER)["balance_msats"]
        r = ex.submit_solution(self.cid, {"submitter": SOLVER, "solution": {"x": 13}})
        self.assertEqual((r["status"], r["bond_msats"], len(r["validators"])), ("pending", 1_000_000, 3))
        r = self.measure(r["id"], 12.9)
        self.assertEqual((r["status"], r["score"]), ("best", 12.9))
        self.assertEqual(ex.wallet(SOLVER)["balance_msats"] - b0, -FEE - 1_000_000)   # the bond waits out the window
        self.assertEqual(ex.wallet(SOLVER)["vesting_msats"], 0)                        # ...and no verdict paid anyone
        with self.assertRaises(PermissionError):
            ex.confirm_solution(r["id"], OTHER, {"score": 13})                         # only a pledge's own judge
        c = ex.confirm_solution(r["id"], BACKER, {"score": 12.5})                      # the backer measures 12.5
        self.assertEqual(c["confirmed_level"], 12.5)                                   # the lower of the two counts
        self.assertEqual(ex.wallet(SOLVER)["vesting_msats"], 10_000_000 * 125 // 1000 * 90 // 100)
        balanced(self, ex)

    def test_a_captured_validator_majority_moves_no_backers_money(self):
        ex = self.ex
        r = ex.submit_solution(self.cid, {"submitter": SOLVER, "artifact": {"uri": "junk", "hash": "sha256:j"}})
        r = self.measure(r["id"], 20.0)                                                # every drawn validator lies
        self.assertEqual(ex.get_challenge(self.cid)["best"], 20.0)                     # the board says solved...
        honest = ex.submit_solution(self.cid, {"submitter": OTHER, "solution": {"x": 14}})
        honest = self.measure(honest["id"], 14.0)
        self.assertEqual(honest["status"], "scored")                                   # not "better" than the fake...
        ex.confirm_solution(honest["id"], BACKER, {"score": 14.0})                     # ...but the backer pays for it
        self.assertEqual(ex.wallet(OTHER)["vesting_msats"], 10_000_000 * 2 // 10 * 90 // 100)
        b0 = ex.wallet(BACKER)["balance_msats"]
        for _ in range(30):
            ex.settle()
        self.assertEqual(ex.wallet(BACKER)["balance_msats"] - b0, 8_000_000)           # the fake was never confirmed
        self.assertEqual(ex.wallet(SOLVER)["vesting_msats"], 0)
        balanced(self, ex)

    def test_overfitting_the_public_instances_destroys_the_bond_and_pays_nothing(self):
        ex = self.ex
        b0 = held(ex, SOLVER)
        r = ex.submit_solution(self.cid, {"submitter": SOLVER, "solution": {"x": 19}})  # public 19, hidden 10.5
        r = self.measure(r["id"], 10.5)
        self.assertEqual((r["status"], r["paid_msats"]), ("overfit", 0))
        self.assertEqual(held(ex, SOLVER) - b0, -1_000_000 - FEE)
        self.assertEqual(ex.get_challenge(self.cid)["best"], 10.0)
        balanced(self, ex)

    def test_an_invalid_proof_loses_its_bond(self):
        ex = self.ex
        r = ex.submit_solution(self.cid, {"submitter": SOLVER, "artifact": {"uri": "x", "hash": "sha256:x"}})
        r = self.measure(r["id"], None)
        self.assertEqual(r["status"], "invalid")
        self.assertEqual(ex.economy_stats()["forfeited_msats"], 1_000_000)
        balanced(self, ex)


class Posting(unittest.TestCase):
    def test_the_same_problem_from_two_sources_merges_and_the_verifier_upgrades(self):
        ex = make_ex()
        yaml = open(os.path.join(SAMPLE, "erdosproblems", "problems_sample.yaml"), encoding="utf-8").read()
        listed = C.from_erdosproblems_yaml(yaml)
        self.assertEqual([c["key"] for c in listed], ["erdos:3", "erdos:28"])          # the solved one is skipped
        fees = ex.fees()["paid_msats"]
        ids = [ex.post_challenge(c, origin="import")["id"] for c in listed]
        self.assertEqual(ex.fees()["paid_msats"], fees)                                # operator imports pay no fee
        with self.assertRaisesRegex(ValueError, "no verifier"):
            ex.submit_solution(ids[1], {"submitter": SOLVER, "solution": {"proof": "theorem x : True := trivial"}})
        lean = os.path.join(SAMPLE, "formal_conjectures", "FormalConjectures", "ErdosProblems", "28.lean")
        (fc,) = C.load(lean)
        r = ex.post_challenge(dict(fc, poster=POSTER))
        self.assertEqual((r["id"], r["merged"]), (ids[1], True))
        c = ex.get_challenge(ids[1])
        self.assertEqual(c["verifier_kind"], "lean4")
        self.assertEqual(len(c["sources"]), 2)
        self.assertIn(fc["key"], c["aliases"])
        self.assertEqual(ex.post_challenge(dict(toy_challenge(key="other"), aliases=["erdos:3"], poster=POSTER))["id"],
                         ids[0])                                                       # an alias merges too
        balanced(self, ex)

    def test_a_yukon_benchmark_posts_and_exports(self):
        ex = make_ex()
        bench = json.load(open(os.path.join(SAMPLE, "yukon_export", "benchmark.json"), encoding="utf-8"))
        r = ex.post_challenge({"format": "yukon", "benchmark": bench, "baseline": 2.6, "poster": POSTER})
        c = ex.get_challenge(r["id"])
        self.assertEqual((c["verifier_kind"], c["direction"]), ("command", "maximize"))
        self.assertEqual(c["challenge"]["metric"]["min_step_rel"], 0.0001)
        sub = ex.submit_solution(r["id"], {"submitter": SOLVER, "artifact": {"uri": "git:abc", "hash": "sha256:a"}})
        self.assertEqual(sub["status"], "pending")                                     # validators run commands
        out = ex.export_challenge(r["id"])["files"]["benchmark.json"]
        for k in C.YUKON_REQUIRED:
            self.assertIn(k, out)
        v2 = {"schemaVersion": 2, "name": "two", "description": "d", "category": "math", "tracks": [
            dict(bench, trackName="a"), dict(bench, trackName="b", direction="-")]}
        self.assertEqual([c["metric"]["direction"] for c in C.from_yukon(v2, baseline=1.0)], ["maximize", "minimize"])

    def test_escalation_a_growing_open_failure_becomes_an_unfunded_challenge(self):
        ex = make_ex(escalate_after=2, escalate_streak=2)
        ex.post_reporter_bond(SOLVER)
        fid = None
        for i in range(6):
            t = Trace.from_fix(task="code.python", base_model="qwen", input=f"Write function {i} to add.\nassert f(1,2)==3",
                               model_output={"code": "x"}, verified_output={"code": f"return a+b+{i}"},
                               checker="unit-tests@1", producer=SOLVER, privacy="open",
                               failure_modes={"code": "wrong_answer"}, created="2026-10-06T00:00:00Z")
            fid = ex.submit_trace(dict(t))["failure_id"]
            ex.settle()
        cs = ex.challenges(origin="escalation")["challenges"]
        self.assertEqual(len(cs), 1)
        self.assertEqual((cs[0]["failure_id"], cs[0]["key"], cs[0]["escrow_msats"]), (fid, f"failure|{fid}".lower(), 0))
        for _ in range(ex.unbacked_epochs + 1):
            ex.settle()
        self.assertEqual(ex.challenges(status="")["challenges"][0]["status"], "expired")
        balanced(self, ex)


class Importers(unittest.TestCase):
    def test_the_alphaevolve_sample_verifies_its_reference_construction(self):
        sheet = C.load(os.path.join(SAMPLE, "alphaevolve_sample.json"))
        c26 = next(c for c in sheet if c["key"].endswith("n=26"))
        ref = json.load(open(os.path.join(SAMPLE, "reference", "circle_packing_26.json")))
        self.assertEqual(circle_packing_sum_radii(ref, {"n": 26}), c26["metric"]["baseline"])
        self.assertAlmostEqual(c26["metric"]["baseline"], 2.635983, places=6)
        bad = dict(ref, radii=[r * 1.01 for r in ref["radii"]])
        self.assertIsNone(circle_packing_sum_radii(bad, {"n": 26}))
        self.assertIsNone(circle_packing_sum_radii({"centers": [[0.5, 0.5]], "radii": [float("nan")]}, {"n": 1}))
        self.assertAlmostEqual(tammes_min_distance({"points": [[0, 0, 1], [0, 0, -2]]}, {"n": 2}), 2.0)

    def test_an_alphaevolve_notebook_is_read_without_running_it(self):
        nb = {"cells": [{"cell_type": "code", "source": [
            "centers_2 = np.array([[0.25, 0.5], [0.75, 0.5]])\n", "radii_2 = np.array([0.25, 0.25])\n",
            "centers_2 = np.array([[0.3, 0.5], [0.7, 0.5]])\n", "radii_2 = np.array([0.2, 0.2])\n",
            "radii_3 = np.array([__import__('os').system('echo no')])\n"]}]}
        (x,) = C.from_alphaevolve_notebook(nb, "packing_circles_max_sum_of_radii")
        self.assertEqual(x["challenge"]["metric"]["baseline"], 0.5)                     # the best valid construction
        self.assertEqual(x["challenge"]["verifier"]["instance"], {"n": 2})

    def test_formal_conjectures_open_statements_and_the_lean_gate(self):
        f = os.path.join(SAMPLE, "formal_conjectures", "FormalConjectures", "ErdosProblems", "3.lean")
        cs = C.load(f)
        self.assertEqual([c["aliases"] for c in cs], [["erdos:3"]])                    # solved variants are skipped
        inst = cs[0]["verifier"]["instance"]
        ok = f"theorem erdos_3 {inst['statement'].replace('answer(sorry)', 'answer(True)')} := by\n  exact foo"
        self.assertEqual(lean_gate(inst["statement"], "erdos_3", ok), "True")
        for bad in (ok.replace("exact foo", "sorry"), ok.replace("Set ℕ", "Set ℤ"), ok.replace("erdos_3", "erdos_4", 1),
                    ok.replace("exact foo", "native_decide"), ok + "\naxiom cheat : False"):
            with self.assertRaises(Invalid):
                lean_gate(inst["statement"], "erdos_3", bad)

    def test_the_exported_verify_script_keeps_the_score_file_contract(self):
        import subprocess
        import tempfile
        d = tempfile.mkdtemp()
        script = os.path.join(SAMPLE, "yukon_export", "verify.py")
        ref = os.path.join(SAMPLE, "reference", "circle_packing_26.json")
        out = os.path.join(d, "score.json")
        r = subprocess.run([sys.executable, script, ref, out], capture_output=True)
        self.assertEqual(r.returncode, 0)
        self.assertAlmostEqual(json.load(open(out))["score"], 2.6359830849176067)
        bad = os.path.join(d, "bad.json")
        json.dump({"centers": [[0.5, 0.5]] * 26, "radii": [0.4] * 26}, open(bad, "w"))
        r = subprocess.run([sys.executable, script, bad, out], capture_output=True)
        self.assertEqual(r.returncode, 1)
        self.assertFalse(os.path.exists(out))                                          # no stale score survives
        self.assertEqual(similarity({"v": [1.0, 2.0, 3.0, 4.0]}, {"v": [1.0, 2.0, 3.0, 4.0]}), 1.0)


class Interfaces(unittest.TestCase):
    """HTTP, MCP and the SDK client, against a live sats node."""

    def setUp(self):
        from exchange import make_handler
        from http.server import ThreadingHTTPServer
        self.ex = make_ex()
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.ex))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()

    def test_http_client_and_mcp(self):
        from traceex import Client
        from traceex.mcp import handle, NodeBackend, ClientBackend, TOOLS
        poster, solver = Client(self.url, POSTER), Client(self.url, SOLVER)
        cid = poster.post_challenge(toy_challenge(), seed_msats=2_000_000)["id"]
        self.assertEqual(Client(self.url, BACKER).pledge_challenge(cid, 3_000_000)["pledged_msats"], 3_000_000)
        r = solver.submit_solution(cid, {"x": 15})
        self.assertEqual(r["status"], "best")
        self.assertEqual(solver.leaderboard(cid)["leaderboard"][0]["score"], 15.0)
        self.assertEqual(poster.challenges()["challenges"][0]["id"], cid)
        self.assertEqual(poster.get_challenge(cid)["best"], 15.0)
        self.assertIn("benchmark.json", poster.challenge_export(cid)["files"])
        self.assertEqual(solver.submission(r["id"])["score"], 15.0)
        with urllib.request.urlopen(self.url + "/.well-known/trace-exchange.json") as f:
            self.assertIn("challenges", json.load(f))
        call = lambda b, name, args: handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                             "params": {"name": name, "arguments": args}}, b)["result"]
        nb = NodeBackend(self.ex)
        self.assertEqual(call(nb, "traceex_challenges", {})["structuredContent"]["challenges"][0]["id"], cid)
        got = call(nb, "traceex_submit_challenge", {"id": cid, "solution": {"x": 17}, "address": OTHER})
        self.assertEqual(got["structuredContent"]["status"], "best")
        cb = ClientBackend(Client(self.url, BACKER), max_spend_msats=1_000)
        self.assertTrue(call(cb, "traceex_back_challenge", {"id": cid, "msats": 5_000}).get("isError"))  # over budget
        names = {t["name"] for t in TOOLS}
        self.assertTrue({"traceex_challenges", "traceex_post_challenge", "traceex_submit_challenge",
                         "traceex_back_challenge"} <= names)
        balanced(self, self.ex)

    def test_autopilot_posts_a_challenge_when_a_bounty_did_not_fix_it(self):
        from traceex import Client
        from traceex.autopilot import Autopilot, Policy
        from traceex.classify import RulesEngine
        pilot = Autopilot(Client(self.url, OTHER), task="code.python", base_model="qwen", checker="unit-tests@1",
                          engine=RulesEngine(), policy=Policy(privacy="open", bounty_after=2, challenge_after=4,
                                                              back_msats=500_000, budget_msats=1_000_000))
        acts = []
        for i in range(5):
            acts += pilot.on_result(f"Write function number {i} that adds.", {"failing": ["code"], "result": {}},
                                    failure="wrong_answer")
        kinds = [a["action"] for a in acts]
        self.assertIn("posted_bounty", kinds)
        self.assertIn("posted_challenge", kinds)
        posted = next(a for a in acts if a["action"] == "posted_challenge")
        c = self.ex.get_challenge(posted["challenge"])
        self.assertEqual((c["verifier_kind"], c["escrow_msats"]), ("registry", 500_000))
        self.assertEqual(pilot.spent, 1_000_000)                                       # bounty + challenge, in budget
        other = Autopilot(Client(self.url, POSTER), task="code.python", base_model="qwen", checker="unit-tests@1",
                          engine=RulesEngine(), policy=Policy(privacy="open", bounty_after=1, challenge_after=1))
        acts = []
        for i in range(2):
            acts += other.on_result(f"Write function number {i} that adds.", {"failing": ["code"], "result": {}},
                                    failure="wrong_answer")
        self.assertIn("backed_existing_challenge", [a["action"] for a in acts])        # the same problem merges
        balanced(self, self.ex)


if __name__ == "__main__":
    unittest.main()
