"""Agent Beacon -> traceX exporter (adapters/beacon). Synthetic logs only (tests/beacon_fixtures.py,
tests/fixtures/beacon/runtime.jsonl): never a real session log.   python -m unittest tests.test_beacon_export"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path[:0] = [HERE, os.path.join(ROOT, "sdk", "python"), ROOT]

import beacon_fixtures as F                                        # noqa: E402
import adapters.beacon.export as X                                 # noqa: E402
from adapters.beacon import episodes as E, events as V, scrub as S  # noqa: E402
from traceex import outbox as tx_outbox                            # noqa: E402
from traceex.client import privacy_leaks                           # noqa: E402

STATIC = os.path.join(HERE, "fixtures", "beacon", "runtime.jsonl")


def episodes_of(evs):
    evs, stats = V.read([json.dumps(e) for e in evs])
    out, agg = [], {"flaky": 0}
    for key, sess in V.sessions(evs).items():
        eps, st = E.find(sess)
        agg["flaky"] += st["flaky"]
        for ep in eps:
            ep["session"] = key
        out += eps
    return out, agg, stats


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def all_text(folder):
    """Everything the exporter wrote under folder, as one string."""
    parts = []
    for d, _, files in os.walk(folder):
        for f in files:
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                parts.append(f + "\n" + fh.read())
    return "\n".join(parts)


class Scrub(unittest.TestCase):
    def s(self, text, **kw):
        return S.Scrubber(**kw).scrub(text)

    def test_provider_keys_and_tokens(self):
        p = F.planted()
        for name in ("openai", "anthropic", "github", "github_pat", "aws_id", "jwt", "fal", "hf", "slack", "google",
                     "stripe"):
            out = self.s(f"value {p[name]} end")
            self.assertNotIn(p[name], out, name)
            self.assertIn("[REDACTED_", out, name)
            self.assertTrue(out.startswith("value ") and out.endswith(" end"), (name, out))

    def test_headers_assignments_urls(self):
        p = F.planted()
        cases = [f"Authorization: Bearer {p['bearer']}", f"-H 'Authorization: Key {p['fal']}'",
                 f"x-api-key: {p['cloudflare']}", f"CLOUDFLARE_API_TOKEN={p['cloudflare']}",
                 f"export FAL_KEY={p['bearer']}", f'"password": "{p["db_password"]}"',
                 f"db_password = '{p['db_password']}'", f"--token {p['bearer']}",
                 f"AWS_SECRET_ACCESS_KEY={p['aws_secret']}", f"postgres://u:{p['url_password']}@h/db",
                 f"https://x.example.com/a?token={p['query_token']}&page=2", f"apiKey: {p['env_value']}"]
        for c in cases:
            out = self.s(c)
            for v in p.values():
                if len(v) > 8 and v in c:
                    self.assertNotIn(v, out, c)
        self.assertEqual(self.s("CLOUDFLARE_API_TOKEN=abcdefgh12345678"), "CLOUDFLARE_API_TOKEN=[REDACTED_SECRET]")
        self.assertIn("&page=2", self.s(f"https://x.example.com/a?token={p['query_token']}&page=2"))

    def test_private_key_and_high_entropy(self):
        p = F.planted()
        pk = "-----BEGIN OPENSSH PRIVATE" + " KEY-----\n" + p["private_key_body"] + "\n-----END OPENSSH PRIVATE KEY-----"
        self.assertEqual(self.s("key:\n" + pk + "\nok"), "key:\n[REDACTED_PRIVATE_KEY]\nok")
        self.assertEqual(self.s("cut " + pk[:60]), "cut [REDACTED_PRIVATE_KEY]")       # truncated by Beacon's 4 KB cap
        self.assertEqual(self.s(f"blob {p['random_blob']}"), "blob [REDACTED_HIGH_ENTROPY]")
        self.assertEqual(self.s(f"signing {p['hex_secret']}"), "signing [REDACTED_HIGH_ENTROPY]")

    def test_code_and_hashes_survive(self):
        keep = ["for key, value in d.items():", "max_tokens = 4096", "items.sort(key=len)", "token_count = 3",
                "cache_key = make_key(x)", "def test_parse_v2_handles_empty_input_correctly():",
                "class CamelCaseIdentifierWithVersion2:", "sha256:" + "ab" * 32, "commit " + "c0ffee12" * 5,
                'fetch("/api/users/42")', "a/b and/or 10/20/2026", "API_KEY = os.environ['API_KEY']",
                "token = get_token()", "password = None", "secret: ${SECRET}", "author = 'Ada'",
                "https://docs.example.org/guide/install"]
        for k in keep:
            self.assertEqual(self.s(k), k)

    def test_personal_data(self):
        p = F.planted()
        out = self.s(f"mail {p['email']}, call {p['phone']} or {p['phone_intl']}; ship to {p['address']}.")
        self.assertEqual(out, "mail [REDACTED_EMAIL], call [REDACTED_PHONE] or [REDACTED_PHONE]; ship to "
                              "[REDACTED_ADDRESS].")
        self.assertEqual(self.s("PO Box 1234"), "[REDACTED_ADDRESS]")
        self.assertEqual(self.s("+2024"), "+2024")                              # a diff line adding 2024

    def test_home_dirs_collapse_and_outside_paths_go(self):
        s = S.Scrubber(project="C:\\Users\\Jordan Example\\proj", names=["alice-laptop"])
        self.assertEqual(s.scrub("C:\\Users\\Jordan Example\\proj\\src\\app.py:12"), "./src/app.py:12")
        self.assertEqual(s.scrub("C:/Users/Jordan Example/proj/src/app.py"), "./src/app.py")
        self.assertEqual(s.scrub("/c/Users/Jordan Example/proj/src/app.py"), "./src/app.py")
        self.assertEqual(s.scrub("C:\\\\Users\\\\Jordan Example\\\\proj\\\\a.py"), "./a.py")    # JSON-escaped
        self.assertEqual(s.scrub("cd C:\\Users\\Jordan Example"), "cd ~")                 # name from the project
        self.assertEqual(S.Scrubber().scrub("C:\\Users\\bob is here"), "~ is here")
        self.assertEqual(s.scrub("C:\\Users\\Jordan Example\\.ssh\\id_rsa"), "~/[REDACTED_PATH]")
        self.assertEqual(s.scrub("/home/alice/.aws/credentials and /Users/sam/Desktop/x.txt"),
                         "~/[REDACTED_PATH] and ~/[REDACTED_PATH]")
        self.assertEqual(s.scrub("D:\\finance\\taxes.xlsx"), "[REDACTED_PATH]")
        self.assertEqual(s.scrub("/etc/ssl/private/server.key"), "[REDACTED_PATH]")
        self.assertEqual(s.scrub("/usr/lib/python3.12/site-packages/requests/models.py"),
                         "<site-packages>/requests/models.py")
        self.assertEqual(s.scrub("on alice-laptop today"), "on [REDACTED_USER] today")
        self.assertEqual(s.scrub("relative tests/test_x.py stays"), "relative tests/test_x.py stays")

    def test_env_files_lose_every_value(self):
        diff = "--- a/.env.local\n+++ b/.env.local\n@@\n-API_BASE=https://staging.example.com\n+DEBUG=1\n+NAME=x\n"
        out = S.Scrubber().scrub(diff, path="C:/p/.env.local")
        self.assertNotIn("staging.example.com", out)
        self.assertEqual(out.count("[REDACTED_SECRET]"), 3)

    def test_report_counts_never_values(self):
        p = F.planted()
        s = S.Scrubber()
        s.scrub(f"{p['openai']} {p['email']}", field="prompt")
        r = s.report()
        self.assertEqual(r["by_field"]["prompt"], {"email": 1, "openai_key": 1})
        self.assertEqual(r["total"], 2)
        self.assertNotIn(p["openai"], json.dumps(r))
        self.assertNotIn(p["email"], json.dumps(r))

    def test_second_pass(self):
        p = F.planted()
        with self.assertRaises(S.ScrubError) as cm:
            S.second_pass(f"left over {p['github']}")
        self.assertNotIn(p["github"], str(cm.exception))
        with self.assertRaises(S.ScrubError):
            S.second_pass('password = "hunter2hunter2"')                     # the node's own pattern too
        self.assertTrue(S.second_pass(S.Scrubber().scrub(f"x {p['github']} {p['jwt']} {p['aws_id']}")))


class Episodes(unittest.TestCase):
    def test_clean_fix_episode(self):
        eps, agg, stats = episodes_of(F.clean_session())
        self.assertEqual(len(eps), 1)
        self.assertEqual(stats["merged"], 1)                                 # hook + collector copy, one event.id
        ep = eps[0]
        self.assertEqual(ep["check"][0], "pytest")
        self.assertEqual(len(ep["failures"]), 1)
        self.assertEqual(len(ep["edits"]), 1)
        self.assertEqual(ep["prompt"], "The average test is failing, please fix calc.py")

    def test_flaky_and_non_checks_make_no_episode(self):
        eps, agg, _ = episodes_of(F.flaky_session())
        self.assertEqual((len(eps), agg["flaky"]), (0, 1))
        eps, _, _ = episodes_of(F.non_check_session())
        self.assertEqual(eps, [])

    def test_no_exit_code_attempts_and_tool_argument_diffs(self):
        eps, _, _ = episodes_of(F.no_exit_code_session())
        self.assertEqual(len(eps), 1)
        ep = eps[0]
        self.assertEqual((ep["check"][0], len(ep["failures"]), len(ep["edits"])), ("jest", 2, 2))
        self.assertIn("+return subtotal * (1 + TAX_RATE);", ep["edits"][1][0]["diff"])

    def test_status_rules(self):
        ev = lambda cmd, code, out, **kw: F.command("09:00", cmd, code, out, **kw)
        self.assertEqual(E.status_of(ev("pytest -q | tail -5", 0, "FAILED tests/t.py::x\n1 failed")), "fail")
        self.assertEqual(E.status_of(ev("pytest -q", 0, "")), "pass")
        self.assertEqual(E.status_of(ev("pytest -q", 2, "")), "fail")
        self.assertEqual(E.status_of(ev("go test ./...", None, "--- FAIL: TestX")), "fail")
        self.assertEqual(E.status_of(ev("go test ./...", None, "ok  \tpkg\t0.1s")), "pass")
        self.assertIsNone(E.check_of(ev("pip install -r requirements.txt", 1, "")))
        self.assertIsNone(E.check_of(ev("git status", 1, "")))
        self.assertEqual(E.check_of(ev("cd x && FOO=1 python -m pytest -q -x tests/a.py --tb=short", 1, ""))[1],
                         "python -m pytest tests/a.py")

    def test_broader_run_covers(self):
        failing = E.check_of(F.command("09:00", "pytest tests/test_a.py::test_x -q", 1, ""))
        self.assertTrue(E.covers(E.check_of(F.command("09:01", "pytest", 0, "")), failing))
        self.assertTrue(E.covers(E.check_of(F.command("09:01", "pytest tests/", 0, "")), failing))
        self.assertFalse(E.covers(E.check_of(F.command("09:01", "pytest tests/test_b.py", 0, "")), failing))
        self.assertFalse(E.covers(E.check_of(F.command("09:01", "npm test", 0, "")), failing))


class Export(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.outbox = os.path.join(self.dir, "outbox")

    def log(self, evs):
        path = os.path.join(self.dir, "runtime.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(F.jsonl(evs))
        return path

    def test_static_fixture_end_to_end(self):
        s = X.export(STATIC, self.outbox)
        self.assertEqual((s["episodes"], len(s["written"]), s["flaky"], s["refused"]), (2, 2, 1, []))
        items = tx_outbox.items(self.outbox)                    # the outbox tool still reads it (scrub/ is a folder)
        self.assertEqual(len(items), 2)
        for path, item in items:
            t = item["trace"]
            self.assertEqual((t["v"], t["task"], t["privacy"]), ("trace/0.1", "code.repair", "open"))
            self.assertEqual(t["fixed_fields"], ["code"])
            self.assertEqual(t["failure_modes"], {"code": "wrong_answer"})
            self.assertEqual(privacy_leaks(t), [])
            self.assertEqual(t["base_model"]["name"], "claude-sonnet-4-5")
            self.assertTrue(os.path.exists(os.path.join(self.outbox, "scrub", os.path.basename(path))))
            self.assertEqual(item["sent"], None)
            self.assertNotIn('"s-1"', json.dumps(item))                       # the session id stays local, hashed
            self.assertNotIn("claude_code|s-1", json.dumps(item))
        checkers = sorted(i["trace"]["checker"]["id"] for _, i in items)
        self.assertEqual(checkers, ["jest", "pytest"])

    def test_skeleton_default_and_open(self):
        path = self.log(F.clean_session())
        t = load(X.export(path, self.outbox)["written"][0]["file"])["trace"]
        self.assertNotIn("average", t["input"])                                # project identifiers skeletonized
        self.assertNotIn("calc", t["input"])
        self.assertIn("-    return sum(values) / (len(values) + 1)", t["input"])
        self.assertNotIn("please fix", t["input"])                             # no prompt in skeleton mode
        t2 = load(X.export(path, self.outbox + "2", open_text=True)["written"][0]["file"])["trace"]
        self.assertIn("def average(values):", t2["input"])
        self.assertIn("@@ ./calc.py @@", t2["input"])
        self.assertIn("please fix calc.py", t2["input"])
        self.assertNotIn("/home/alice", json.dumps(t2))
        self.assertEqual(t2["created"], "2026-10-20T10:00:15Z")

    def test_planted_secrets_are_all_removed(self):
        p = F.planted()
        path = self.log(F.secrets_session())
        for open_text in (False, True):
            out = os.path.join(self.dir, f"out-{open_text}")
            s = X.export(path, out, open_text=open_text)
            self.assertEqual((len(s["written"]), s["refused"]), (1, []), s)
            text = all_text(out)
            for name, value in p.items():
                self.assertNotIn(value, text, f"{name} leaked (open={open_text})")
                self.assertNotIn(value.lower(), text.lower(), f"{name} leaked (open={open_text})")
            for word in ("Jordan", "samantha", "alice", "taxes", "acme-vpn", "Evergreen", "keychain"):
                self.assertNotIn(word.lower(), text.lower(), f"{word} leaked (open={open_text})")
            report = load(os.path.join(out, "scrub", os.path.basename(s["written"][0]["file"])))
            self.assertEqual((report["outcome"], report["second_pass"]), ("written", "clean"))
            self.assertGreaterEqual(report["total"], 30)
            trace = load(s["written"][0]["file"])["trace"]
            self.assertEqual(privacy_leaks(trace), [])
            self.assertTrue(S.second_pass(json.dumps(trace)))

    def test_refuses_when_a_secret_survives(self):
        p = F.planted()
        path = self.log(F.secrets_session())
        with mock.patch.object(S.Scrubber, "scrub", lambda self, text, field="text", path=None: text):
            s = X.export(path, self.outbox, open_text=True)
        self.assertEqual(len(s["written"]), 0)
        self.assertEqual(len(s["refused"]), 1)
        self.assertEqual(tx_outbox.items(self.outbox), [])                      # nothing in the outbox
        text = all_text(self.outbox)
        self.assertIn("refused", text)
        for value in p.values():
            self.assertNotIn(value, text)                                       # the report names kinds only

    def test_cli_is_local_only_without_submit(self):
        path = self.log(F.clean_session())
        with mock.patch("traceex.client.Client.__init__", side_effect=AssertionError("network")), \
                mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = X.main([path, "--outbox", self.outbox, "--node", "http://127.0.0.1:9", "--address", "0x" + "1" * 40])
        self.assertEqual(rc, 0)
        self.assertIn("dry run: nothing sent", buf.getvalue())
        self.assertEqual(len(tx_outbox.items(self.outbox)), 1)
        self.assertFalse(os.path.exists(os.path.join(self.outbox, "sent")))

    def test_submit_needs_node_and_address(self):
        path = self.log(F.clean_session())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            X.main([path, "--outbox", self.outbox, "--submit"])
        with self.assertRaises(ValueError):
            X.export(path, self.outbox, submit=True)

    def test_bad_lines_are_skipped(self):
        path = os.path.join(self.dir, "bad.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("not json\n[1,2]\n{\"vendor\": \"other\", \"event\": {\"action\": \"x\"}}\n" + F.jsonl(F.clean_session()))
        s = X.export(path, self.outbox)
        self.assertEqual((s["read"]["skipped"], len(s["written"])), (3, 1))


if __name__ == "__main__":
    unittest.main()
