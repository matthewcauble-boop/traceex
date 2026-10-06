"""traceX v0.7: the failure registry, fix tracking, the Leviathan-style search, and the two ingestion paths (the pytest
plugin and the OpenTelemetry exporter).   python -m pytest -q tests/test_registry.py"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SDK = os.path.join(ROOT, "sdk", "python")
sys.path[:0] = [SDK, os.path.join(ROOT, "node")]

from traceex import Trace, Learning, attest, Client, find_secrets  # noqa: E402
from traceex.client import privacy_leaks  # noqa: E402
from exchange import Exchange, serve  # noqa: E402
from sats import SatsExchange, Params, attestation_digest  # noqa: E402
from registry import measurement_digest, model_family, failure_signature, FAILURE_ID  # noqa: E402

A = lambda c: "0x" + c * 40
V = lambda i: "0x" + f"{0xa0 + i:02x}" * 20
VAL = A("5")
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"


def case(i, producer=A("b"), model=QWEN, exc="TypeError", mode="runtime_error", text=None):
    return Trace.from_fix(task="code.python", base_model=model,
                          input=text or f"Write a function number {i} that merges two tuples into a list.",
                          model_output={"code": "return a + list(b)"}, verified_output={"code": f"return list(a) + list(b)  # {i}"},
                          checker="unit-tests@1", producer=producer, created="2026-10-04T00:00:00Z", privacy="open",
                          failure_modes={"code": mode}, feedback=[f"{exc}: can only concatenate tuple (case {i})"])


def sats_node(validators=5, quorum=3, **kw):
    ex = SatsExchange(":memory:", test_credits=30_000_000, beacon_delay=0, params=Params(quorum=quorum), fee_to=A("f"), **kw)
    vals = []
    for i in range(validators):
        ex.faucet(V(i))
        ex.register_validator(V(i), 15_000_000)
        vals.append(V(i))
    for w in "bcde8796":
        ex.faucet(A(w))
    ex.register_checker("unit-tests", A("b"))
    return ex, vals


def measure(ex, fix_id, results, salt="s"):
    """Every drawn validator commits, then reveals, the same measurement: {failure: (passed, n)}."""
    m = {"results": {f: {"passed": p, "n": n} for f, (p, n) in results.items()}}
    drawn = ex.get_fix(fix_id)["assigned"]
    for v in drawn:
        ex.commit_fix(fix_id, v, measurement_digest(m, salt + v))
    for v in drawn:
        ex.reveal_fix(fix_id, v, m, salt + v)
    return ex.get_fix(fix_id)


def accept(ex, lid):
    for v in ex.verdict(lid)["assigned"]:
        ex.commit(lid, v, attestation_digest(_att(v), "s"))
    for v in ex.verdict(lid)["assigned"]:
        ex.reveal(lid, v, _att(v), "s")
    return ex.verdict(lid)["status"]


def _att(v):
    return {"validator": v, "eval_set": "sha256:p" + v[-2:], "metric": "pass@1", "before": .6, "after": .7, "n": 300,
            "audit": {"checked": 10, "bad": 0}}


def learning(ex, parents, trainer=A("c")):
    L = Learning.build(kind="lora", task="code.python", base_model=QWEN, artifact={"uri": "x", "hash": None},
                       parents=[(p, 1) for p in parents], trainer=trainer, per_call_msats=200,
                       attestation=attest(V(0), "sha256:c", "pass@1", .6, .7))
    return ex.register_learning(L)["id"]


class Dedup(unittest.TestCase):
    """Every trace lands in one canonical failure: branch + failure signature + model family."""

    def test_same_failure_same_id_and_ids_are_stable(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "x.db")
            ex = Exchange(db, validators=(VAL,))
            a = ex.submit_trace(dict(case(1)))["failure_id"]
            self.assertRegex(a, r"^TXF-\d{4}-000001$")
            self.assertEqual(ex.submit_trace(dict(case(2, producer=A("e"))))["failure_id"], a)       # another case
            self.assertEqual(ex.submit_trace(dict(case(3, model="Qwen/Qwen2.5-7B-Instruct")))["failure_id"], a)
            self.assertEqual(ex.submit_trace(dict(case(1)))["failure_id"], a)                        # a duplicate
            other = {ex.submit_trace(dict(case(4, exc="IndexError")))["failure_id"],                # another exception
                     ex.submit_trace(dict(case(5, model="meta-llama/Llama-3.1-8B")))["failure_id"],  # another family
                     ex.submit_trace(dict(case(6, mode="wrong_answer", exc="AssertionError")))["failure_id"]}
            self.assertEqual(len(other | {a}), 4)
            f = ex.get_failure(a)
            self.assertEqual((f["occurrences"], f["family"], f["signature"]), (3, "qwen2.5", "code:runtime_error/TypeError"))
            self.assertEqual(f["models"], {QWEN: 2, "Qwen/Qwen2.5-7B-Instruct": 1})
            ids = [x["id"] for x in ex.failures(sort="new", limit=50)["failures"]]
            ex.db.close()
            again = Exchange(db, validators=(VAL,))                                                   # reopened
            self.assertEqual(sorted(x["id"] for x in again.failures(limit=50)["failures"]), sorted(ids))
            self.assertEqual(again.submit_trace(dict(case(7)))["failure_id"], a)
            self.assertTrue(again.submit_trace(dict(case(8, exc="KeyError")))["failure_id"].endswith("000005"))
            again.db.close()

    def test_family_and_signature(self):
        self.assertEqual([model_family(m) for m in (QWEN, "llama-3.1-8b", "needle3", "gpt-4o-mini", {"name": "x",
                          "family": "Mine"})], ["qwen2.5", "llama3.1", "needle3", "gpt4", "mine"])
        t = case(1)
        self.assertEqual(failure_signature(t, {"code": "runtime_error"}), "code:runtime_error/TypeError")
        self.assertEqual(failure_signature(t, {"b": "role_swap", "a": "omission"}), "a:omission b:role_swap")

    def test_reclassify_moves_the_case_and_keeps_both_ids(self):
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        from test_traceex import FakeJev
        ex = Exchange(":memory:", validators=(VAL,))
        old = ex.submit_trace(dict(case(1)))["failure_id"]
        ex.reclassify(engine=FakeJev("code/repair"))
        new = ex.get_trace(ex.search()["results"][0]["id"])["failure_id"]
        self.assertNotEqual(new, old)
        self.assertEqual(ex.get_failure(new)["path"], "code/repair")
        self.assertEqual(ex.get_failure(old)["occurrences"], 0)                 # the id stays; nothing filed under it
        self.assertNotIn(old, [f["id"] for f in ex.failures()["failures"]])

    def test_takedown_leaves_the_counters(self):
        ex = Exchange(":memory:", validators=(VAL,))
        r = [ex.submit_trace(dict(case(i))) for i in range(2)]
        ex.remove("trace", r[0]["id"])
        self.assertEqual(ex.get_failure(r[0]["failure_id"])["occurrences"], 1)
        self.assertEqual(ex.search(kind="trace")["total"], 1)


class Counters(unittest.TestCase):
    def test_distinct_verified_reporters_copies_and_growth(self):
        ex, vals = sats_node()
        fid = ex.submit_trace(dict(case(1, producer=A("b"))))["failure_id"]
        ex.submit_trace(dict(case(2, producer=A("e"))))
        f = ex.get_failure(fid)
        self.assertEqual((f["reporters"], f["reporters_unverified"], f["occurrences"]), (0, 2, 2))   # no bonds yet
        for w in "be":
            ex.post_reporter_bond(A(w))
        self.assertEqual(ex.get_failure(fid)["reporters"], 2)
        copy = case(1, producer=A("d"), text="WRITE A FUNCTION NUMBER 1 THAT MERGES TWO TUPLES INTO A LIST.")
        r = ex.submit_trace(dict(copy))
        self.assertEqual(r["near_duplicate_of"], ex.search(limit=10)["results"][-1]["id"])
        ex.post_reporter_bond(A("d"))
        f = ex.get_failure(fid)
        self.assertEqual((f["reporters"], f["occurrences"], f["reports"]), (2, 2, 3))   # a copy adds neither
        self.assertEqual((f["first_epoch"], f["last_epoch"], f["growth"]), (1, 1, 2))
        for _ in range(4):
            ex.settle()
        ex.submit_trace(dict(case(9, producer=A("b"))))
        f = ex.get_failure(fid)
        self.assertEqual((f["last_epoch"], f["recent"], f["growth"]), (5, 1, -1))   # 1 new case, 2 in the 3 epochs before
        ex.withdraw_reporter(A("e"))
        self.assertEqual(ex.get_failure(fid)["reporters"], 1)                  # leaving: no longer counted
        self.assertTrue(ex.audit()["balanced"])

    def test_refuted_cases_leave_the_counters_and_cost_the_bond(self):
        ex, vals = sats_node()
        ex.post_reporter_bond(A("b"))
        fid = ex.submit_trace(dict(case(1, producer=A("b"))))["failure_id"]
        sybil = A("9")
        ex.faucet(sybil)
        ex.post_reporter_bond(sybil)
        fake = ex.submit_trace(dict(case(77, producer=sybil)))["id"]
        self.assertEqual(ex.get_failure(fid)["reporters"], 2)
        with self.assertRaises(PermissionError):
            ex.repro_check(fid, A("c"), {fake: False})                           # only validators re-check
        ex.repro_check(fid, vals[0], {fake: False})
        self.assertEqual(ex.get_failure(fid)["reporters"], 2)                  # one vote is not a majority of 2
        r = ex.repro_check(fid, vals[1], {fake: False})
        self.assertEqual(r["refuted"], [fake])
        self.assertEqual((ex.get_failure(fid)["reporters"], ex.get_failure(fid)["occurrences"]), (1, 1))
        self.assertEqual(ex.reporter(sybil)["bond_msats"], 0)
        self.assertGreaterEqual(ex.economy_stats()["forfeited_msats"], 1_000_000)
        self.assertTrue(ex.audit()["balanced"])


class FixTracking(unittest.TestCase):
    """open -> partly_fixed -> fixed, a model version that regresses it, and one that fixes it again."""

    def setUp(self):
        self.ex = Exchange(":memory:", validators=(VAL,))
        self.fid = self.ex.submit_trace(dict(case(1)))["failure_id"]

    def fix(self, passed, n=20, kind="prompt_patch", model=QWEN):
        f = self.ex.claim_fix({"claimant": A("c"), "kind": kind, "claims": [self.fid], "model": model})
        self.assertEqual(f["status"], "pending")
        return self.ex.reveal_fix(f["id"], VAL, {"results": {self.fid: {"passed": passed, "n": n}}})

    def test_status_transitions_and_history(self):
        ex = self.ex
        f = self.fix(3)
        self.assertEqual((f["status"], f["claims"][0]["status"], f["claims"][0]["pass_rate"]), ("validated", "partly_fixed", .15))
        self.assertEqual(ex.get_failure(self.fid)["status"], "partly_fixed")
        self.fix(19)
        self.assertEqual((ex.get_failure(self.fid)["status"], ex.get_failure(self.fid)["pass_rate"]), ("fixed", .95))
        weak = self.fix(1)                                                       # a weaker fix never hides a stronger one
        self.assertEqual((weak["claims"][0]["status"], ex.get_failure(self.fid)["status"]), ("open", "fixed"))
        self.assertEqual(weak["status"], "rejected")
        thin = self.fix(5, n=5)                                                  # too few cases: recorded, not counted
        self.assertEqual((thin["status"], thin["claims"][0]["validated"]), ("inconclusive", False))
        rep = ex.register_model({"version": "Qwen/Qwen2.5-1.5B-Instruct"})
        self.assertEqual([x["failure"] for x in rep["pending"]], [self.fid])
        ex.reveal_fix(rep["recheck"], VAL, {"results": {self.fid: {"passed": 1, "n": 20}}})
        rep = ex.model_report("Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(rep["summary"]["regressed"], 1)
        self.assertEqual(rep["regressed"][0]["before"]["status"], "fixed")
        self.assertEqual(ex.get_failure(self.fid)["status"], "regressed")
        rep = ex.register_model({"version": "Qwen/Qwen2.5-3B-Instruct"})
        ex.reveal_fix(rep["recheck"], VAL, {"results": {self.fid: {"passed": 20, "n": 20}}})
        self.assertEqual(ex.model_report("Qwen/Qwen2.5-3B-Instruct")["summary"]["fixed"], 1)
        h = [(x["status"], x["kind"], x["validated"]) for x in ex.failure_history(self.fid)["history"]]
        self.assertEqual(h, [("partly_fixed", "prompt_patch", True), ("fixed", "prompt_patch", True),
                             ("open", "prompt_patch", True), ("open", "prompt_patch", False),
                             ("regressed", "model", True), ("fixed", "model", True)])
        by = {(r["model"], r["kind"]): r["status"] for r in ex.get_failure(self.fid)["by_model"]}
        self.assertEqual(by[("Qwen/Qwen2.5-1.5B-Instruct", "model")], "regressed")
        with self.assertRaisesRegex(ValueError, "already registered"):
            ex.register_model({"version": "Qwen/Qwen2.5-3B-Instruct"})

    def test_claims_are_checked(self):
        ex = self.ex
        for bad, msg in (({"claims": ["TXF-2026-999999"]}, "unknown failure"), ({"claims": []}, "at least one"),
                         ({"kind": "model"}, "POST /v0/models"), ({"kind": "magic"}, "kind is one of"),
                         ({"model": ""}, "model")):
            body = dict({"claimant": A("c"), "kind": "tool", "claims": [self.fid], "model": QWEN}, **bad)
            with self.assertRaisesRegex(ValueError, msg):
                ex.claim_fix(body)
        tid = ex.search()["results"][0]["id"]
        L = Learning.build(kind="lora", task="code.python", base_model=QWEN, artifact={"uri": "x"}, parents=[(tid, 1)],
                           trainer=A("e"), per_call_micros=1, attestation=attest(VAL, "sha256:x", "acc", .5, .6))
        lid = ex.register_learning(L)["id"]
        with self.assertRaisesRegex(ValueError, "only the learning's trainer"):
            ex.claim_fix({"claimant": A("c"), "claims": [self.fid], "model": QWEN, "learning": lid})
        with self.assertRaises(PermissionError):                                 # only a drawn validator measures
            f = ex.claim_fix({"claimant": A("c"), "claims": [self.fid], "model": QWEN})
            ex.reveal_fix(f["id"], A("9"), {"results": {self.fid: {"passed": 1, "n": 20}}})


class SatsFixes(unittest.TestCase):
    """On a sats node: bonds, random draws, commit-reveal, and the bounty that pays itself when its failure is fixed."""

    def setUp(self):
        self.ex, self.vals = sats_node()
        ex = self.ex
        self.tids = [ex.submit_trace(dict(case(i)))["id"] for i in range(4)]
        self.fid = ex._failure_of(self.tids[0])
        self.lid = learning(ex, self.tids)
        self.assertEqual(accept(ex, self.lid), "accepted")
        self.b = ex.post_bounty({"poster": A("d"), "failure_id": self.fid, "eval_set": "sha256:hidden", "target": .8,
                                 "title": "stop the TypeError"})
        ex.pledge(self.b["id"], A("8"), 5_000_000)

    def test_a_bounty_on_a_failure_merges_and_pays_when_it_is_fixed(self):
        ex = self.ex
        again = ex.post_bounty({"poster": A("e"), "failure_id": self.fid, "eval_set": "sha256:other", "target": .5,
                                "seed_sats": 1_000})
        self.assertEqual((again["id"], again["merged"]), (self.b["id"], True))
        self.assertEqual(ex.failures(sort="bounty")["failures"][0]["open_bounty_msats"], 6_000_000)
        before = ex.wallet(A("c"))
        f = ex.claim_fix({"claimant": A("c"), "claims": [self.fid], "model": QWEN, "learning": self.lid})
        self.assertEqual(ex.wallet(A("c"))["balance_msats"], before["balance_msats"] - 2_000_000 - 58)   # bond + fee
        self.assertNotIn(A("c"), f["assigned"])
        with self.assertRaisesRegex(ValueError, "commit first"):
            ex.reveal_fix(f["id"], f["assigned"][0], {"results": {self.fid: {"passed": 19, "n": 20}}})
        f = measure(ex, f["id"], {self.fid: (19, 20)})
        self.assertEqual((f["status"], ex.get_failure(self.fid)["status"]), ("validated", "fixed"))
        self.assertEqual(ex.bounties()["bounties"][0]["status"], "open")        # fixed, but the poster hasn't measured
        low = ex.poster_measure(self.b["id"], f["id"], attest(A("d"), "sha256:hidden", "pass@1", .1, .5))
        self.assertEqual(low["paid"], [])                                        # under the poster's target
        with self.assertRaisesRegex(ValueError, "poster's own measurement"):
            ex.poster_measure(self.b["id"], f["id"], attest(A("c"), "sha256:hidden", "pass@1", .1, .95))
        r = ex.poster_measure(self.b["id"], f["id"], attest(A("d"), "sha256:hidden", "pass@1", .1, .9))
        self.assertEqual(r["paid"], [self.b["id"]])
        self.assertEqual(ex.bounties(status="")["bounties"][0]["status"], "solved")
        vest = {a: m for a, m in ex.db.execute("SELECT account, SUM(msats) FROM vesting GROUP BY account").fetchall()}
        self.assertEqual(vest[A("c")], 4_200_000)                                # 70% of 6,000 sats, vesting
        self.assertEqual(sum(vest.values()), 6_000_000)                          # the pledges, exactly
        f2 = ex.claim_fix({"claimant": A("c"), "claims": [self.fid], "model": QWEN, "learning": self.lid})
        b2 = ex.post_bounty({"poster": A("e"), "failure_id": self.fid, "eval_set": "sha256:e2",
                             "target": .5, "seed_sats": 500})
        measure(ex, f2["id"], {self.fid: (19, 20)}, salt="t")
        r2 = ex.poster_measure(b2["id"], f2["id"], attest(A("e"), "sha256:e2", "pass@1", .1, .9))
        self.assertEqual(r2["paid"], [])                                         # one payout per failure and learning
        for _ in range(8):
            ex.settle()
        self.assertTrue(ex.audit()["balanced"])

    def test_copied_public_answers_never_fix_a_failure(self):
        ex = self.ex
        outputs = {t: json.loads(ex.db.execute("SELECT body FROM traces WHERE id=?", (t,)).fetchone()[0])["verified_output"]
                   for t in self.tids}
        burned = ex.economy_stats()["forfeited_msats"]
        f = ex.claim_fix({"claimant": A("e"), "kind": "prompt_patch", "claims": [self.fid], "model": QWEN, "outputs": outputs})
        self.assertEqual(f["claims"][0]["public_repro"], {"passed": 4, "n": 4})
        f = measure(ex, f["id"], {self.fid: (0, 20)})
        self.assertEqual((f["status"], f["claims"][0]["status"]), ("rejected", "open"))
        self.assertEqual(ex.get_failure(self.fid)["status"], "open")
        self.assertEqual(ex.economy_stats()["forfeited_msats"] - burned, 2_000_000)   # its bond is destroyed
        self.assertTrue(ex.audit()["balanced"])

    def test_a_fix_that_fails_its_own_public_repro_is_not_fixed(self):
        ex = self.ex
        f = ex.claim_fix({"claimant": A("e"), "kind": "tool", "claims": [self.fid], "model": QWEN,
                          "outputs": {self.tids[0]: {"code": "wrong"}}})
        f = measure(ex, f["id"], {self.fid: (20, 20)})
        self.assertEqual(f["claims"][0]["status"], "partly_fixed")

    def test_regressions_come_only_from_model_rechecks(self):
        ex = self.ex
        f = ex.claim_fix({"claimant": A("c"), "claims": [self.fid], "model": QWEN, "learning": self.lid})
        measure(ex, f["id"], {self.fid: (20, 20)})
        bad = ex.claim_fix({"claimant": A("e"), "kind": "prompt_patch", "claims": [self.fid], "model": QWEN,
                            "outputs": {t: {"code": "broken"} for t in self.tids}})
        measure(ex, bad["id"], {self.fid: (0, 20)}, salt="b")
        self.assertEqual(ex.get_failure(self.fid)["status"], "fixed")           # a broken claim can't reopen it
        rep = ex.register_model({"version": "Qwen/Qwen2.5-1.5B-Instruct"})
        measure(ex, rep["recheck"], {self.fid: (2, 20)}, salt="m")
        self.assertEqual(ex.model_report("Qwen/Qwen2.5-1.5B-Instruct")["regressed"][0]["failure"], self.fid)
        self.assertEqual(ex.get_failure(self.fid)["status"], "regressed")
        self.assertTrue(ex.audit()["balanced"])


class Migration(unittest.TestCase):
    def test_a_v06_database_gains_the_registry(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "v06.db")
            ex = SatsExchange(db, test_credits=30_000_000)
            ex.faucet(A("b"))
            ids = [ex.submit_trace(dict(case(i)))["failure_id"] for i in range(3)] + \
                  [ex.submit_trace(dict(case(9, exc="IndexError")))["failure_id"]]
            ex.db.close()
            con = sqlite3.connect(db)                                            # make it look like v0.6 left it
            for t in ("failures", "occurrences", "fixes", "fstatus", "fhistory", "lx_records", "lx_fts", "lx_branches",
                      "lx_lines", "models", "reporters"):
                con.execute(f"DROP TABLE {t}")
            con.execute("ALTER TABLE bounties DROP COLUMN failure_id")
            con.execute("CREATE VIRTUAL TABLE trace_fts USING fts5(id UNINDEXED, path, signature, model, task, body)")
            con.execute("UPDATE meta SET v='0.6' WHERE k='sats_version'")
            con.commit()
            con.close()
            up = SatsExchange(db, test_credits=30_000_000)
            self.assertEqual(up._meta("sats_version"), "0.7")
            self.assertEqual(sorted({f["id"] for f in up.failures()["failures"]}), sorted(set(ids)))
            self.assertEqual(up.search("tuples")["total"], 4)
            self.assertIsNone(up.db.execute("SELECT name FROM sqlite_master WHERE name='trace_fts'").fetchone())
            self.assertIn("upgraded from v0.6", up.events()["events"][0]["text"])
            up.db.close()


class SearchIndex(unittest.TestCase):
    """The Leviathan-style search on the seeded testnet (247 real traces)."""

    @classmethod
    def setUpClass(cls):
        from seed import seed_if_empty
        os.environ.pop("TYPESAFE_API_KEY", None)
        cls.ex = Exchange(":memory:", test_credits=25_000_000)
        seed_if_empty(cls.ex)
        cls.meta = {}
        for tid, body in cls.ex.db.execute("SELECT id, body FROM traces").fetchall():
            t = json.loads(body)
            cls.meta[tid] = {"task": (t.get("source") or {}).get("task_id"), "modes": set((t.get("failure_modes") or {}).values()),
                             "model": t["base_model"]["name"], "input": t["input"].lower()}
        for tid, modes in cls.ex.db.execute("SELECT id, modes FROM labels").fetchall():
            cls.meta[tid]["modes"] |= set(json.loads(modes).values())

    def test_quality_on_frozen_queries(self):
        """A subset of the frozen leviathan-eval queries, relevance from metadata only (MBPP task ids, failure modes,
        model). The evaluation that chose this design measured 87% top-1 / 94% top-5 (old search: 70% / 77%)."""
        task = lambda *ids: lambda m: m["task"] in ids
        fail = lambda *ms: lambda m: bool(m["modes"] & set(ms))
        Q = [("wrong_name", fail("wrong_name")), ("syntax_error", fail("syntax_error")), ("role_swap", fail("role_swap")),
             ("runtime_error tuple", lambda m: "runtime_error" in m["modes"] and "tuple" in m["input"]),
             ("find the previous palindrome of a specified number", task(909)),
             ("convert an integer into a roman numeral", task(958)),
             ("check whether the given ip address is valid using regex", task(669)),
             ("lateral surface area of a cone", task(731)), ("count coin change", task(918)),
             ("turn a decimal number into Roman numerals", task(958)),
             ("least common multiple of two integers", task(876)),
             ("binomial coefficient modulo a prime", task(952)), ("palindrom", task(909, 864)),
             ("rhombus perimeter", task(716)), ("python functions with wrong_name or syntax_error",
                                                fail("wrong_name", "syntax_error")),
             ("needle3 failures", lambda m: m["model"] == "needle3")]
        top1 = top5 = 0
        for q, rel in Q:
            ids = [h["id"] for h in self.ex.search(q, limit=5)["results"]]
            hits = [rel(self.meta[i]) for i in ids]
            top1 += bool(hits[:1] and hits[0])
            top5 += any(hits)
        self.assertGreaterEqual(top5 / len(Q), 0.9, (top1, top5))
        self.assertGreaterEqual(top1 / len(Q), 0.8, (top1, top5))

    def test_branches_resolve_in_tiers_and_never_guess(self):
        ex = self.ex
        r = ex.search(path="code/gen", limit=3)
        self.assertEqual((r["status"], r["branch"]["key"], r["branch"]["match_type"]), ("ok", "code/generate", "contains"))
        r = ex.search(path="travel")
        self.assertEqual((r["branch"]["key"], r["branch"]["match_type"]), ("extract/travel", "contains+subtree"))
        amb = ex.search("tuple", path="re")
        self.assertEqual((amb["status"], amb["results"]), ("ambiguous_branch", []))
        self.assertGreater(len(amb["candidates"]), 1)
        self.assertEqual(ex.search(path="zzzz")["status"], "unknown_branch")

    def test_labelled_fallback_and_compact_cards(self):
        ex = self.ex
        r = ex.search("tuple", path="extract/travel/flight", limit=3)
        self.assertEqual(r["results"], [])
        self.assertTrue(r["other_branches"] and all(c["other_branch"] for c in r["other_branches"]))
        self.assertIn("OTHER", r["notes"][0])
        txt = ex.search("runtime error on tuples", limit=5, fmt="text")["text"]
        self.assertRegex(txt, r"shown 5 of \d+")
        self.assertIn("match: TypeError", txt)
        self.assertLess(len(txt), 2400)                                          # ~600 tokens for five cards
        self.assertIn("next: GET /v0/traces/<id>", txt)
        f = ex.search("type error", kind="failure", limit=3)["results"]
        self.assertEqual(f[0]["kind"], "failure")
        self.assertIn("TypeError", f[0]["title"])
        both = ex.search("type error", kind="all", limit=20)["results"]
        self.assertEqual({h["kind"] for h in both}, {"trace", "failure"})

    def test_index_is_written_in_the_same_transaction(self):
        ex = Exchange(":memory:", validators=(VAL,))
        boom = RuntimeError("crash after indexing, before commit")

        def crash(*a, **k):
            raise boom
        ex._event = crash
        with self.assertRaises(RuntimeError):
            ex.submit_trace(dict(case(1)))
        ex.db.conn.rollback()
        self.assertEqual(ex.db.execute("SELECT COUNT(*) FROM traces").fetchone()[0], 0)
        self.assertEqual(ex.index.count(), 0)                                     # the index row went with it
        del ex._event
        ex.submit_trace(dict(case(1)))
        self.assertFalse(ex.db.conn.in_transaction)
        self.assertEqual((ex.index.count("trace"), ex.index.count("failure")), (1, 1))


class Http(unittest.TestCase):
    def setUp(self):
        self.ex, self.srv = serve(0, ":memory:", economy="sats", public=True, admin_token="op", test_credits=30_000_000,
                                  beacon_delay=0, params=Params(quorum=1))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.op = Client(self.url, token="op")
        for v in (V(0), V(1)):
            self.ex.faucet(v)
            self.ex.register_validator(v, 15_000_000)

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_registry_endpoints(self):
        me = Client(self.url, A("b"))
        me.faucet()
        me.reporter_bond()
        r = me.submit(case(1))
        fid = r["failure_id"]
        self.assertTrue(FAILURE_ID.fullmatch(fid))
        self.assertEqual(me.failures()["failures"][0]["reporters"], 1)
        self.assertEqual(me.failure(fid)["repro"]["public"][0]["trace"], r["id"])
        self.assertEqual(me.trace(r["id"][:20])["failure_id"], fid)
        txt = me.search("tuples", fmt="text", kind="all")["text"]
        self.assertTrue(txt.startswith("tracex search") and fid in txt)
        f = me.claim_fix([fid], QWEN, kind="tool")                               # public: it pays its bond
        for path, body in ((f"/v0/fixes/{f['id']}/commits", {}), (f"/v0/fixes/{f['id']}/reveals", {}),
                           ("/v0/models", {"version": "x"}), (f"/v0/failures/{fid}/repro", {}),
                           ("/v0/bounties/1/measurements", {})):
            with self.assertRaisesRegex(RuntimeError, "^403"):
                me._call("POST", path, body)
        v = Client(self.url, f["assigned"][0], token="op")
        m = {"results": {fid: {"passed": 19, "n": 20}}}
        v.commit_fix(f["id"], measurement_digest(m, "x"))
        self.assertEqual(v.reveal_fix(f["id"], m, "x")["claims"][0]["status"], "fixed")
        self.assertEqual(me.failure_history(fid)["history"][0]["status"], "fixed")
        rep = self.op.register_model("Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(rep["summary"]["pending"], 1)
        self.assertEqual(me.model_report("Qwen/Qwen2.5-1.5B-Instruct")["version"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(me.fixes(failure=fid)["count"], 2)
        self.assertEqual(me.reporter()["verified"], True)
        d = me.describe()
        self.assertIn("/v0/failures", d["start_here"][0])

    def test_mcp_tools(self):
        from traceex.mcp import handle, NodeBackend
        self.ex.faucet(A("b"))
        fid = self.ex.submit_trace(dict(case(1)))["failure_id"]
        call = lambda name, args: handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                          "params": {"name": name, "arguments": args}}, NodeBackend(self.ex))["result"]
        s = call("traceex_search", {"q": "type error"})
        self.assertTrue(s["content"][0]["text"].startswith("tracex search"))   # cards by default, not JSON
        self.assertIn(fid, s["content"][0]["text"])
        self.assertEqual(call("traceex_failures", {})["structuredContent"]["failures"][0]["id"], fid)
        self.assertEqual(call("traceex_failures", {"id": fid})["structuredContent"]["id"], fid)
        c = call("traceex_claim_fix", {"claims": [fid], "model": QWEN, "kind": "tool", "address": A("b")})
        self.assertFalse(c.get("isError"), c)


class PytestPlugin(unittest.TestCase):
    """Ingestion path A: a test that fails, then passes after a code change, becomes a skeleton trace."""

    SRC = ('API_KEY = "AKIAIOSFODNN7EXAMPLE"\nOWNER = "jordan.parker@example.com"\n\n'
           'def total_due(lines, discount_code="SPRING-2026"):\n'
           '    # the invoice total for Jordan Parker\'s account\n'
           '    subtotal = sum(price * qty for price, qty in lines[1:])\n'
           '    if discount_code == "SPRING-2026":\n        subtotal = subtotal * 0.9\n    return round(subtotal, 2)\n')
    TEST = ('from billing.invoice import total_due\n\n\ndef test_total_due_counts_every_line():\n'
            '    assert total_due([(1999.0, 2), (450.5, 1)], discount_code="NONE") == 4448.5\n')

    def setUp(self):
        try:
            import pytest  # noqa: F401
        except ImportError:
            self.skipTest("the plugin tests run pytest in a subprocess")
        self.dir = tempfile.mkdtemp(prefix="traceex-plugin-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        os.makedirs(os.path.join(self.dir, "billing"))
        os.makedirs(os.path.join(self.dir, "tests"))
        self.write("billing/__init__.py", "")
        self.write("billing/invoice.py", self.SRC)
        self.write("tests/test_invoice.py", self.TEST)

    def write(self, rel, text):
        with open(os.path.join(self.dir, rel), "w", encoding="utf-8") as f:
            f.write(text)

    def pytest(self, *args):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([SDK, self.dir]), PYTHONIOENCODING="utf-8")
        env.pop("TRACEX_ADDRESS", None)
        return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "traceex.pytest_plugin", "-p",
                               "no:cacheprovider", *args], cwd=self.dir, env=env, capture_output=True, text=True,
                              timeout=120)

    def outbox(self):
        box = os.path.join(self.dir, ".traceex", "outbox")
        return [json.load(open(os.path.join(box, f), encoding="utf-8")) for f in sorted(os.listdir(box))
                if f.endswith(".json")] if os.path.isdir(box) else []

    def test_a_fixed_test_becomes_a_skeleton_trace(self):
        self.assertEqual(self.pytest("--traceex").returncode, 1)
        self.assertEqual(self.outbox(), [])
        self.write("billing/invoice.py", self.SRC.replace("in lines[1:])", "in lines)"))
        out = self.pytest("--traceex")
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertIn("1 trace written to", out.stdout)
        items = self.outbox()
        self.assertEqual(len(items), 1)
        t = items[0]["trace"]
        sent = json.dumps(t)
        for secret in ("AKIA", "jordan", "Jordan", "SPRING", "NONE", "billing", "invoice", "total_due", "discount",
                       "subtotal", "1999", "4448", "450.5", "traceex-plugin", os.path.basename(self.dir), "Users"):
            self.assertNotIn(secret, sent)
        self.assertEqual((t["task"], t["privacy"], t["checker"]["id"]), ("code.repair", "open", "pytest"))
        self.assertEqual(t["failure_modes"], {"code": "wrong_answer"})
        self.assertRegex(t["model_output"]["code"], r"for \{ID_\d+\}, \{ID_\d+\} in lines\[1:\]\)$")
        self.assertRegex(t["verified_output"]["code"], r"for \{ID_\d+\}, \{ID_\d+\} in lines\)$")
        self.assertIn("test_total_", t["input"])
        self.assertIn("failed with AssertionError", t["input"])
        self.assertEqual((privacy_leaks(t), find_secrets(sent)), ([], []))
        self.assertTrue(open(os.path.join(self.dir, ".traceex", ".gitignore")).read().strip().endswith("*"))

    def test_flaky_tests_and_runtime_errors(self):
        self.write("tests/test_flaky.py", "import os\n\ndef test_flag():\n    assert os.path.exists('flag')\n")
        self.pytest("--traceex")
        self.write("flag", "")
        out = self.pytest("--traceex", "tests/test_flaky.py")
        self.assertIn("passed again with no change: flaky", out.stdout)
        self.assertEqual(self.outbox(), [])
        self.write("billing/invoice.py", self.SRC.replace("price * qty", "price * str(qty)"))
        self.pytest("--traceex", "tests/test_invoice.py")
        self.write("billing/invoice.py", self.SRC.replace("in lines[1:])", "in lines)"))
        self.pytest("--traceex", "tests/test_invoice.py")
        t = self.outbox()[0]["trace"]
        self.assertEqual(t["failure_modes"], {"code": "runtime_error"})
        self.assertTrue(t["feedback"][0].startswith("TypeError: "))

    def test_off_unless_asked_and_submit_to_a_node(self):
        out = self.pytest()
        self.assertFalse(os.path.exists(os.path.join(self.dir, ".traceex")))
        self.assertNotIn("traceX", out.stdout)
        ex, srv = serve(0, ":memory:")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            self.pytest("--traceex")
            self.write("billing/invoice.py", self.SRC.replace("in lines[1:])", "in lines)"))
            out = self.pytest("--traceex", "--traceex-submit", "--traceex-node", url, "--traceex-address", A("b"),
                              "--traceex-model", QWEN)
            self.assertIn("sent to", out.stdout)
            self.assertEqual(self.outbox(), [])
            hit = ex.search(kind="trace")["results"][0]
            self.assertEqual((hit["producer"], hit["model"]), (A("b"), QWEN))
            self.assertTrue(hit["failure_id"].startswith("TXF-"))
        finally:
            srv.shutdown()
            srv.server_close()


class FakeSpan:
    """Just enough of an OpenTelemetry ReadableSpan."""
    class _Ctx:
        def __init__(self, sid, tid):
            self.span_id, self.trace_id = sid, tid

    class _Status:
        def __init__(self, code, desc=None):
            self.status_code = type("C", (), {"name": code})()
            self.description = desc

    class _Event:
        def __init__(self, name, attrs):
            self.name, self.attributes = name, attrs

    def __init__(self, name, sid, parent, start, attrs=None, events=(), status="UNSET", desc=None, trace=1):
        self.name, self.attributes, self.start_time, self.end_time = name, attrs or {}, start, start + 1
        self.context = self._Ctx(sid, trace)
        self.parent = self._Ctx(parent, trace) if parent else None
        self.events = [self._Event(n, a) for n, a in events]
        self.status = self._Status(status, desc)


EMAIL = "Hi Jordan Parker, your flight departs Austin on Thursday, October 15, 2026 at 8:05 AM. Call (512) 555-0142."


def llm(sid, start, answer, legacy=False, model="qwen2.5-0.5b"):
    if legacy:
        attrs = {"gen_ai.request.model": model, "gen_ai.prompt.0.role": "user", "gen_ai.prompt.0.content": EMAIL,
                 "gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": answer}
    else:
        attrs = {"gen_ai.operation.name": "chat", "gen_ai.request.model": model,
                 "gen_ai.input.messages": json.dumps([{"role": "system", "parts": [{"type": "text", "content": "SYSTEM SECRET"}]},
                                                      {"role": "user", "parts": [{"type": "text", "content": EMAIL}]}]),
                 "gen_ai.output.messages": json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": answer}]}])}
    return FakeSpan("chat", sid, 1, start, attrs)


class OtelExporter(unittest.TestCase):
    """Ingestion path B: model answer, a checker fails it, a retry the checker passes -> a skeleton trace."""

    def setUp(self):
        self.box = tempfile.mkdtemp(prefix="traceex-otel-")
        self.addCleanup(shutil.rmtree, self.box, True)

    def spans(self, second_ok=True, legacy=False):
        root = FakeSpan("invoke_agent", 1, None, 0, {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "Trips"})
        fail = FakeSpan("execute_tool date_rules", 3, 1, 3, {"gen_ai.operation.name": "execute_tool",
                                                              "gen_ai.tool.name": "date_rules"},
                        events=[("exception", {"exception.type": "ValueError",
                                               "exception.message": "'3/02/2026' is not the departure date"})],
                        status="ERROR")
        ok = FakeSpan("execute_tool date_rules", 5, 1, 7, {"gen_ai.operation.name": "execute_tool",
                                                            "gen_ai.tool.name": "date_rules"},
                      status="OK" if second_ok else "ERROR")
        return [llm(2, 1, "3/02/2026", legacy), fail, llm(4, 5, "2026-10-15", legacy), ok, root]

    def test_the_pattern_becomes_a_skeleton_trace_in_the_outbox(self):
        from traceex.otel import TraceexSpanExporter
        ex = TraceexSpanExporter(outbox=self.box)
        spans = self.spans()
        ex.export(spans[:-1])                                     # held until the root span ends
        self.assertEqual(os.listdir(self.box), [])
        ex.export(spans[-1:])
        files = os.listdir(self.box)
        self.assertEqual(len(files), 1)
        t = json.load(open(os.path.join(self.box, files[0]), encoding="utf-8"))["trace"]
        sent = json.dumps(t)
        for v in ("Jordan", "Parker", "Austin", "555-0142", "October 15", "3/02/2026", "2026-10-15", "SYSTEM SECRET"):
            self.assertNotIn(v, sent)
        self.assertEqual((t["task"], t["base_model"]["name"], t["checker"]["id"], t["privacy"]),
                         ("agent.trips", "qwen2.5-0.5b", "date_rules", "skeleton"))
        self.assertEqual(t["failure_modes"], {"output": "wrong_answer"})
        self.assertEqual(t["model_output"]["output"], "{DATE_1}")
        self.assertIn("is not the departure date", t["feedback"][0])
        self.assertEqual(privacy_leaks(t), [])

    def test_semconv_variants_and_non_patterns(self):
        from traceex.otel import TraceexSpanExporter, find_fixes
        self.assertEqual(len(find_fixes(self.spans(legacy=True))), 1)            # gen_ai.prompt.N / completion.N
        self.assertEqual(find_fixes(self.spans(second_ok=False)), [])            # the retry failed too
        ev = lambda label: [("gen_ai.evaluation.result", {"gen_ai.evaluation.name": "judge",
                                                          "gen_ai.evaluation.score.label": label,
                                                          "gen_ai.evaluation.explanation": "wrong date"})]
        spans = [llm(2, 1, "3/02/2026"), FakeSpan("score", 3, 2, 2, events=ev("fail")),
                 llm(4, 5, "2026-10-15"), FakeSpan("score", 5, 4, 6, events=ev("pass"))]
        fx = find_fixes(spans)
        self.assertEqual(len(fx), 1)
        self.assertEqual(fx[0][4], "wrong date")
        ex = TraceexSpanExporter(outbox=self.box)
        ex.export(spans)                                                         # no root: flushed on shutdown
        ex.shutdown()
        self.assertEqual(len(os.listdir(self.box)), 1)
        leaky = [llm(2, 1, "3/02/2026"), FakeSpan("check", 3, 2, 2, {"traceex.check.passed": False}),
                 FakeSpan("chat", 4, 1, 5, {"gen_ai.operation.name": "chat", "gen_ai.request.model": "m",
                                            "traceex.output": "use key AKIAIOSFODNN7EXAMPLE"}),
                 FakeSpan("check", 5, 4, 6, {"traceex.check.passed": True})]
        ex2 = TraceexSpanExporter(outbox=os.path.join(self.box, "two"), privacy="open")
        ex2.export(leaky)
        ex2.force_flush()
        self.assertEqual((ex2.refused, ex2.written), (1, []))                    # a secret: dropped, never written

    def test_submit_to_a_node_and_the_real_sdk_demo(self):
        from traceex.otel import TraceexSpanExporter
        node, srv = serve(0, ":memory:")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            ex = TraceexSpanExporter(submit=True, client=Client(url, A("b")), outbox=self.box)
            ex.export(self.spans())
            self.assertEqual(len(ex.sent), 1)
            self.assertTrue(ex.sent[0]["failure_id"].startswith("TXF-"))
            self.assertEqual(os.listdir(self.box), [])
        finally:
            srv.shutdown()
            srv.server_close()
        try:
            import opentelemetry.sdk  # noqa: F401
        except ImportError:
            self.skipTest("opentelemetry-sdk not installed")
        import contextlib
        import importlib.util
        import io
        spec = importlib.util.spec_from_file_location("otel_agent_demo", os.path.join(ROOT, "examples", "otel_agent",
                                                                                       "demo.py"))
        otel_demo = importlib.util.module_from_spec(spec)                  # its own name: never shadows another demo
        spec.loader.exec_module(otel_demo)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(otel_demo.main(), 0)
        self.assertIn("personal values in what would be sent: none", buf.getvalue())


class Preview(unittest.TestCase):
    def test_the_seeded_preview_shows_a_realistic_registry(self):
        from seed import seed_if_empty
        ex = SatsExchange(":memory:", test_credits=30_000_000, reserve_msats=50_000)
        seed_if_empty(ex)
        f = ex.failures()
        self.assertGreaterEqual(f["total"], 10)
        top = f["failures"][0]
        self.assertEqual((top["family"], top["reporters"]), ("qwen2.5", 3))
        self.assertGreater(f["statuses"]["partly_fixed"], 3)
        self.assertTrue(all(0.1 <= x["pass_rate"] < 0.9 for x in f["failures"] if x["status"] == "partly_fixed"))
        rep = ex.model_report(QWEN + "+lora-v1")
        self.assertEqual(rep["summary"]["pending"], 0)
        typeerr = [x for x in f["failures"] if x["signature"] == "code:runtime_error/TypeError"][0]
        self.assertEqual(typeerr["open_bounty_msats"], 2_000_000)
        self.assertEqual(ex.failures(sort="bounty")["failures"][0]["id"], typeerr["id"])
        self.assertTrue(ex.audit()["balanced"])


if __name__ == "__main__":
    unittest.main()
