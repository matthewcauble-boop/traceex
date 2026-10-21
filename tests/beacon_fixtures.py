"""Synthetic Agent Beacon runtime logs for tests/test_beacon_export.py. Every session, person, path and key here is
made up. The fake secrets are assembled at run time from pieces and seeded random characters, so no string in this
repository looks like a live credential to a secret scanner; the scrubber only ever sees them in a temp folder.

Events follow Beacon's normalized schema (schema_version 1.0: event.action, session, command, file, gen_ai, ...)."""
import json
import random
import string

_n = [0]


def event(action, ts, session="s-1", harness="claude_code", wd="/home/alice/work/calc", **fields):
    _n[0] += 1
    ev = {"timestamp": f"2026-10-20T10:{ts}.000000000Z", "sequence": _n[0], "vendor": "beacon",
          "product": "endpoint-agent", "schema_version": "1.0",
          "event": {"kind": "agent_runtime", "action": action, "category": action.split(".")[0],
                    "fidelity": "observed"},
          "severity": "info", "endpoint": {"hostname": "alice-laptop", "os": "linux", "agent_version": "0.9.0"},
          "user": {"name": "alice"}, "harness": {"name": harness, "collection_method": "hook"},
          "session": {"id": session, "working_directory": wd}, "model": "claude-sonnet-4-5"}
    for k, v in fields.items():
        if k == "event_id":
            ev["event"]["id"] = v
        elif k == "severity":
            ev["severity"] = v
        else:
            ev[k] = v
    return ev


def command(ts, cmd, exit_code=None, output="", **kw):
    c = {"command": cmd}
    if exit_code is not None:
        c["exit_code"] = exit_code
    if output:
        c["output"] = output
    return event("command.executed", ts, tool={"name": "Bash", "command": cmd}, command=c, **kw)


def edit(ts, path, diff=None, args=None, action="file.modified", **kw):
    f = {"path": path, "operation": "modify", "language": path.rsplit(".", 1)[-1]}
    if diff:
        f["diff"] = diff
        f["diff_bytes"] = len(diff)
    extra = {}
    if args is not None:
        extra["gen_ai"] = {"tool": {"name": "Edit", "call": {"id": f"toolu_{ts}", "arguments": args}}}
    return event(action, ts, tool={"name": "Edit", "path": path}, file=f, **extra, **kw)


def prompt(ts, text, **kw):
    return event("prompt.submitted", ts, prompt={"text": text}, **kw)


CALC_FAIL = """============================= test session starts ==============================
collected 2 items

tests/test_calc.py F.                                                    [100%]

=================================== FAILURES ===================================
_________________________________ test_average _________________________________

    def test_average():
>       assert average([2, 4, 6]) == 4
E       assert 3.0 == 4

tests/test_calc.py:5: AssertionError
FAILED tests/test_calc.py::test_average - assert 3.0 == 4
1 failed, 1 passed in 0.03s"""

CALC_DIFF = """--- a/calc.py
+++ b/calc.py
@@ -1,3 +1,3 @@
 def average(values):
-    return sum(values) / (len(values) + 1)
+    return sum(values) / len(values)
"""


def clean_session():
    """One fix episode (pytest fails, one edit, pytest passes), plus the hook and collector reports of the passing
    run (one event.id, merged), a file read and an `ls` that are not checks."""
    return [
        prompt("00:00", "The average test is failing, please fix calc.py"),
        command("00:05", "cd /home/alice/work/calc && python -m pytest tests/test_calc.py -q", 1, CALC_FAIL),
        event("file.read", "00:06", file={"path": "/home/alice/work/calc/calc.py", "operation": "read"}),
        command("00:07", "ls -la", 0, "calc.py\ntests"),
        edit("00:10", "/home/alice/work/calc/calc.py", CALC_DIFF),
        command("00:15", "python -m pytest tests/test_calc.py -q", 0, "2 passed in 0.02s", event_id="ev-pass-1"),
        command("00:15", "python -m pytest tests/test_calc.py -q", None, "", event_id="ev-pass-1",
                harness={"name": "claude_code", "collection_method": "otlp"}),
    ]


def flaky_session():
    """Fails, passes with no edit in between: flaky, no episode."""
    kw = dict(session="s-flaky", harness="cursor", wd="C:\\Users\\Jordan Example\\proj")
    return [command("01:00", "npm test", 1, "Tests: 1 failed, 4 passed", **kw),
            command("01:30", "npm test", 0, "Tests: 5 passed", **kw)]


def no_exit_code_session():
    """A runtime that reports no exit code (a transcript reader): the failure is read from severity and output, the
    edit from the Edit tool's arguments (no file.diff), the pass from the output. Two failed attempts."""
    kw = dict(session="s-jest", harness="claude_code", wd="/Users/sam/code/shop")
    fail_out = ("FAIL src/cart.test.js\n  ● total › adds tax\n    expect(received).toBe(expected)\n"
                "    Expected: 10.8\n    Received: 10\nTests: 1 failed, 3 passed, 4 total")
    return [
        prompt("02:00", "make the cart tests pass", **kw),
        command("02:01", "npx jest src/cart.test.js", None, fail_out, severity="high", **kw),
        edit("02:02", "/Users/sam/code/shop/src/cart.js", None,
             args={"file_path": "/Users/sam/code/shop/src/cart.js", "old_string": "return subtotal;",
                   "new_string": "return subtotal * 1.0;"}, **kw),
        command("02:03", "npx jest src/cart.test.js", None, fail_out, severity="high", **kw),
        edit("02:04", "/Users/sam/code/shop/src/cart.js", None,
             args={"file_path": "/Users/sam/code/shop/src/cart.js", "old_string": "return subtotal * 1.0;",
                   "new_string": "return subtotal * (1 + TAX_RATE);"}, **kw),
        command("02:05", "npx jest", None, "PASS src/cart.test.js\nTests: 4 passed, 4 total", **kw),
    ]


def non_check_session():
    """`cat` fails then works after a file is created: not a check, so no episode."""
    kw = dict(session="s-cat", wd="/home/alice/work/calc")
    return [command("03:00", "cat notes.txt", 1, "cat: notes.txt: No such file or directory", **kw),
            edit("03:01", "/home/alice/work/calc/notes.txt", "--- /dev/null\n+++ b/notes.txt\n@@\n+todo\n",
                 action="file.created", **kw),
            command("03:02", "cat notes.txt", 0, "todo", **kw)]


# --- planted secrets -----------------------------------------------------------------------------------------------

def _rnd(n, alphabet=string.ascii_letters + string.digits, seed=None):
    r = random.Random(seed if seed is not None else n * 7919)
    while True:
        s = "".join(r.choice(alphabet) for _ in range(n))
        need = [f for f in (str.isdigit, str.isupper, str.islower) if any(f(c) for c in alphabet)]
        if all(any(f(c) for c in s) for f in need):
            return s


HEXD = "0123456789abcdef"


def planted():
    """{name: fake secret or personal value}. Each must be gone from everything the exporter writes."""
    p = {
        "openai": "sk-" + "proj-" + _rnd(48, seed=1),
        "anthropic": "sk-" + "ant-" + "api03-" + _rnd(40, seed=2),
        "github": "gh" + "p_" + _rnd(36, seed=3),
        "github_pat": "github" + "_pat_" + _rnd(22, seed=4) + "_" + _rnd(59, seed=5),
        "aws_id": "AK" + "IA" + _rnd(16, string.ascii_uppercase + string.digits, seed=6),
        "aws_secret": _rnd(40, string.ascii_letters + string.digits + "/+", seed=7),
        "jwt": "ey" + "J" + _rnd(20, seed=8) + ".ey" + "J" + _rnd(30, seed=9) + "." + _rnd(43, seed=10),
        "bearer": _rnd(40, seed=11),
        "fal": "".join(random.Random(12).choice(HEXD) for _ in range(8)) + "-1a2b-4c3d-8e9f-" +
               "".join(random.Random(13).choice(HEXD) for _ in range(12)) + ":" +
               "".join(random.Random(14).choice(HEXD) for _ in range(32)),
        "cloudflare": _rnd(40, string.ascii_letters + string.digits + "_-", seed=15),
        "hf": "h" + "f_" + _rnd(34, seed=16),
        "slack": "xo" + "xb-" + "1234567890-" + _rnd(24, seed=17),
        "google": "AI" + "za" + _rnd(35, seed=18),
        "stripe": "sk" + "_live_" + _rnd(24, seed=19),
        "db_password": "Pw" + _rnd(14, seed=20) + "!9",
        "env_value": "envv" + _rnd(18, seed=21),
        "url_password": "urlpw" + _rnd(12, seed=22),
        "query_token": _rnd(32, seed=23),
        "random_blob": _rnd(44, string.ascii_letters + string.digits + "+/", seed=24),
        "hex_secret": "".join(random.Random(25).choice(HEXD) for _ in range(64)),
        "private_key_body": _rnd(64, string.ascii_letters + string.digits + "+/", seed=26),
        "email": "jordan.example" + "@" + "example.com",
        "email2": "ops+alerts" + "@" + "acme-widgets.io",
        "phone": "(512) 555-" + "0142",
        "phone_intl": "+44 20 7946 " + "0958",
        "address": "742 Evergreen Terrace, Springfield, OR 97403",
        "win_user": "Jordan Example",
        "mac_user": "samantha",
        "linux_user": "alice",
        "host": "alice-laptop",
        "outside_file": "finance/taxes-2025.xlsx",
        "outside_dir": "private/acme-vpn",
    }
    return p


def secrets_session():
    """One fix episode in which every planted value appears somewhere the exporter reads: the prompt, the failing
    command and its output, a .env diff, a config diff and the passing output."""
    s = planted()
    wd = "C:\\Users\\Jordan Example\\projects\\billing"
    kw = dict(session="s-secrets", harness="claude_code", wd=wd)
    pk = ("-----BEGIN RSA PRIVATE" + " KEY-----\n" + s["private_key_body"] + "\n" + s["private_key_body"][::-1] +
          "\n-----END RSA PRIVATE" + " KEY-----")
    prompt_text = (f"Tests fail with auth errors. My key is {s['openai']} and the Anthropic one {s['anthropic']}. "
                   f"Mail me at {s['email']} or call {s['phone']} / {s['phone_intl']}. I live at {s['address']}.\n"
                   f"Use the GitHub token {s['github']} and {s['github_pat']}. Here is my deploy key:\n{pk}\n")
    fail_out = (
        "Traceback (most recent call last):\n"
        f'  File "C:\\Users\\Jordan Example\\projects\\billing\\billing\\client.py", line 12, in charge\n'
        "    resp = session.post(url, headers=headers)\n"
        f'  File "C:\\Users\\Jordan Example\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages'
        f'\\requests\\sessions.py", line 637, in post\n'
        f"requests.exceptions.HTTPError: 401 Unauthorized for url: "
        f"https://api.example.com/v1/charge?api_key={s['query_token']}&page=2\n"
        f"DEBUG headers: {{'Authorization': 'Bearer {s['bearer']}', 'x-api-key': '{s['cloudflare']}'}}\n"
        f"DEBUG fal: Authorization: Key {s['fal']}\n"
        f"DEBUG jwt={s['jwt']}\n"
        f"DEBUG db: postgres://billing:{s['url_password']}@db.internal:5432/billing\n"
        f"DEBUG env: AWS_ACCESS_KEY_ID={s['aws_id']} AWS_SECRET_ACCESS_KEY={s['aws_secret']}\n"
        f"DEBUG blob {s['random_blob']} digest none\n"
        f"DEBUG signing secret {s['hex_secret']}\n"
        f"DEBUG also /Users/{s['mac_user']}/Library/Keychains/login.keychain-db and "
        f"/home/{s['linux_user']}/.ssh/id_ed25519 and D:\\{s['outside_file'].replace('/', chr(92))} and "
        f"/etc/{s['outside_dir']}/client.conf and /c/Users/Jordan Example/.aws/credentials\n"
        f"host {s['host']} user {s['linux_user']}\n"
        f"FAILED tests/test_client.py::test_charge - HTTPError: 401\n1 failed in 0.40s")
    env_diff = ("--- a/.env\n+++ b/.env\n@@ -1,3 +1,4 @@\n"
                f" STRIPE_SECRET={s['stripe']}\n"
                f"-API_BASE=https://staging.example.com\n"
                f"+API_BASE=https://api.example.com\n"
                f"+BILLING_WEBHOOK={s['env_value']}\n"
                f"+HF_TOKEN={s['hf']}\n")
    cfg_diff = ("--- a/billing/client.py\n+++ b/billing/client.py\n@@ -5,8 +5,8 @@\n"
                " import os\n"
                f'-API_KEY = "{s["google"]}"\n'
                '+API_KEY = os.environ["BILLING_API_KEY"]\n'
                f'-SLACK = "{s["slack"]}"\n'
                '+SLACK = os.environ.get("SLACK_TOKEN")\n'
                f'-DB_PASSWORD = "{s["db_password"]}"\n'
                '+DB_PASSWORD = os.environ["DB_PASSWORD"]\n'
                f"-CERT = r\"C:\\Users\\Jordan Example\\certs\\billing.pem\"\n"
                '+CERT = os.path.join(os.path.dirname(__file__), "certs", "billing.pem")\n'
                f'-OWNER = "{s["email2"]}"\n'
                '+OWNER = None\n'
                "-    headers = {'Authorization': 'Bearer ' + token}\n"
                "+    headers = {'Authorization': 'Bearer ' + os.environ['BILLING_TOKEN']}\n")
    return [
        prompt("04:00", prompt_text, **kw),
        command("04:01", f"cd \"{wd}\" && set FAL_KEY={s['fal']} && python -m pytest tests/test_client.py -q",
                1, fail_out, **kw),
        edit("04:02", wd + "\\.env", env_diff, **kw),
        edit("04:03", wd + "\\billing\\client.py", cfg_diff, **kw),
        command("04:04", "python -m pytest tests/test_client.py -q", 0,
                f"1 passed in 0.31s (token {s['github']} cached for {s['email']})", **kw),
    ]


def jsonl(events):
    return "\n".join(json.dumps(e) for e in events) + "\n"
