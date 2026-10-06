"""traceX for pytest: when a test that failed passes again after a code change, that change is a verified fix. This
plugin turns it into a trace (ingestion path A, SPEC 4i).

Opt in (it does nothing otherwise):
    pytest --traceex                          # or, in pytest.ini / pyproject:  traceex = true
    pytest --traceex --traceex-submit --traceex-node https://<node> --traceex-address 0xYourWallet

How it works
  1. A test fails: the plugin notes it in .traceex/state.json with the exception type and message, and keeps a copy of
     the project's Python files as they were (.traceex/blobs/, content-addressed), so a later run can see what changed.
  2. A later run (the same session, the next CI run, an agent's next attempt) finds that test passing. If no file
     changed, it was flaky: no trace. If files changed, the change fixed it: the plugin builds a trace from the
     test, the failure and the diff, all skeletonized (traceex.codeskel), and writes it to .traceex/outbox/ for review
     (the default, --traceex-dry-run) or, with --traceex-submit, sends it to the node.
  3. Review and send what is in the outbox later with `python -m traceex.outbox list | show | send`.

What leaves the machine (only with --traceex-submit or `python -m traceex.outbox send`), and nothing else:
  * the trace JSON, exactly as written in the outbox file's "trace" key:
      task "code.repair", base_model (--traceex-model, TRACEX_MODEL, or "unknown"), checker "pytest@1",
      privacy "open", the producer address you give, a timestamp, and these skeletons:
      - input: the test's file name and test name as skeletons ("{DIR}/test_{ID_1}.py::test_{ID_2}_empty"), the
        exception type (builtin exception names are kept, others become {ID_n}), the failure message's skeleton, the
        test function's skeleton, and the skeleton diff of the change (hunks only, no line numbers, at most 80 lines
        and 5 files);
      - model_output / verified_output: the skeleton lines the change removed / added;
      - feedback: the failure message's skeleton; failure_modes: {"code": wrong_answer | runtime_error | wrong_name |
        syntax_error | timeout}, from the exception type.
  * Skeletons replace every string literal (f-strings and bytes too) with {STR_n}, drop every comment, replace every
    identifier that is not Python vocabulary (keywords, builtins, standard-library modules, built-in types' methods,
    unittest/pytest words) with {ID_n}, every number outside -10..10 with {NUM_n}, and every directory with {DIR}.
  * Before anything is written or sent, the trace is scanned for keys, tokens, passwords, emails and phone numbers
    (traceex.client.privacy_leaks); a trace that fails the scan is dropped, never written.
Never sent: your source files, file paths, project names, string contents, comments, the raw failure message, and
everything in .traceex/ other than the outbox traces (state.json and blobs/ hold raw copies; keep .traceex/ out of
version control: the plugin writes a .gitignore inside it).
"""
import ast
import datetime as dt
import hashlib
import json
import os

import pytest

from .codeskel import CodeSkeleton, changed_sides, KEEP
from .client import privacy_leaks

SKIP_DIRS = {"venv", "env", "node_modules", "__pycache__", "build", "dist", "site-packages"}
MAX_FILES, MAX_BYTES, MAX_CHANGED = 5000, 2_000_000, 5
NO_ADDRESS = "0x" + "0" * 40


def failure_mode(exc_type):
    """The checker's failure mode, from the exception a test raised."""
    name = str(exc_type or "")
    if name == "AssertionError":
        return "wrong_answer"
    if name in ("NameError", "AttributeError", "ImportError", "ModuleNotFoundError", "UnboundLocalError"):
        return "wrong_name"
    if name in ("SyntaxError", "IndentationError", "TabError"):
        return "syntax_error"
    if "Timeout" in name:
        return "timeout"
    return "runtime_error"


def pytest_addoption(parser):
    g = parser.getgroup("traceex", "traceX: turn tests that start passing after a change into traces")
    g.addoption("--traceex", action="store_true", default=False, help="opt in: record failures, build traces for fixes")
    g.addoption("--traceex-dry-run", action="store_true", default=False,
                help="write traces to the local outbox for review (the default)")
    g.addoption("--traceex-submit", action="store_true", default=False, help="send traces to the node")
    g.addoption("--traceex-node", default=None, help="node URL (or ini traceex_node, env TRACEX_NODE)")
    g.addoption("--traceex-address", default=None, help="your wallet address (or ini traceex_address, env TRACEX_ADDRESS)")
    g.addoption("--traceex-model", default=None, help="the model that wrote the code (or ini traceex_model, env "
                                                      "TRACEX_MODEL; default 'unknown')")
    g.addoption("--traceex-dir", default=None, help="state and outbox folder (default <rootdir>/.traceex)")
    parser.addini("traceex", type="bool", default=False, help="traceX: opt in")
    parser.addini("traceex_submit", type="bool", default=False, help="traceX: send traces instead of the outbox")
    for k in ("traceex_node", "traceex_address", "traceex_model", "traceex_dir"):
        parser.addini(k, default="", help=f"traceX: {k[8:]}")


def pytest_configure(config):
    if config.getoption("--traceex") or config.getini("traceex"):
        config.pluginmanager.register(Recorder(config), "traceex-recorder")


def _opt(config, name, env=None):
    v = config.getoption(f"--traceex-{name}") or config.getini(f"traceex_{name}") or (os.environ.get(env) if env else None)
    return v or None


class Snapshots:
    """Content-addressed copies of the project's Python files, kept on this machine only."""

    def __init__(self, root, folder):
        self.root, self.blobs = str(root), os.path.join(folder, "blobs")
        self.cache = {}

    def files(self):
        out = []
        for d, dirs, files in os.walk(self.root):
            dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x not in SKIP_DIRS and not x.endswith(".egg-info"))
            for f in sorted(files):
                if f.endswith(".py"):
                    out.append(os.path.join(d, f))
                    if len(out) >= MAX_FILES:
                        return out
        return out

    def take(self, keep=False):
        """{relative path: sha256} of every Python file now; with keep, their contents are stored as blobs."""
        snap = {}
        for full in self.files():
            try:
                st = os.stat(full)
                if st.st_size > MAX_BYTES:
                    continue
                key = (st.st_mtime_ns, st.st_size)
                if self.cache.get(full, (None,))[0] != key:
                    with open(full, "rb") as fh:
                        data = fh.read()
                    self.cache[full] = (key, hashlib.sha256(data).hexdigest(), data)
                _, sha, data = self.cache[full]
            except OSError:
                continue
            rel = os.path.relpath(full, self.root).replace(os.sep, "/")
            snap[rel] = sha
            if keep:
                path = os.path.join(self.blobs, sha)
                if not os.path.exists(path):
                    os.makedirs(self.blobs, exist_ok=True)
                    with open(path, "wb") as fh:
                        fh.write(data)
        return snap

    def text(self, sha):
        try:
            with open(os.path.join(self.blobs, sha), "rb") as fh:
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return None


def shape_of_test(path, name):
    """(skeleton-ready source, number of parameters, number of asserts) of a test function, from its file."""
    try:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
    except (OSError, SyntaxError, ValueError):
        return "", 0, 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            lines = src.splitlines()[node.lineno - 1:getattr(node, "end_lineno", node.lineno)]
            params = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
            asserts = sum(isinstance(n, ast.Assert) or (isinstance(n, ast.Call) and getattr(n.func, "attr", "")
                                                         .startswith(("assert", "raises"))) for n in ast.walk(node))
            return "\n".join(lines[:40]), len(params), asserts
    return "", 0, 0


class Recorder:
    def __init__(self, config):
        self.config = config
        self.root = str(config.rootpath)
        self.dir = _opt(config, "dir") or os.path.join(self.root, ".traceex")
        self.outbox = os.path.join(self.dir, "outbox")
        self.submit = bool(config.getoption("--traceex-submit") or config.getini("traceex_submit")) and \
            not config.getoption("--traceex-dry-run")
        self.node = _opt(config, "node", "TRACEX_NODE")
        self.address = _opt(config, "address", "TRACEX_ADDRESS")
        self.model = _opt(config, "model", "TRACEX_MODEL") or "unknown"
        self.snaps = Snapshots(self.root, self.dir)
        self.state = self._load()
        self.made, self.flaky, self.refused, self.sent, self.errors = [], 0, 0, 0, []
        self._snap = None                           # this session's snapshot, taken at the first failure

    # --- state ------------------------------------------------------------------------------------------------------
    def _load(self):
        try:
            with open(os.path.join(self.dir, "state.json"), encoding="utf-8") as fh:
                s = json.load(fh)
            return s if s.get("version") == 1 else {"version": 1, "failures": {}}
        except (OSError, ValueError):
            return {"version": 1, "failures": {}}

    def _save(self):
        os.makedirs(self.dir, exist_ok=True)
        ign = os.path.join(self.dir, ".gitignore")
        if not os.path.exists(ign):
            with open(ign, "w") as fh:
                fh.write("# traceX keeps raw copies of your files here to compute diffs: never commit them\n*\n")
        fails = self.state["failures"]
        if len(fails) > 500:                         # oldest first out
            for k in sorted(fails, key=lambda k: fails[k]["at"])[:len(fails) - 500]:
                del fails[k]
        with open(os.path.join(self.dir, "state.json"), "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=1)

    # --- hooks --------------------------------------------------------------------------------------------------------
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        rep = outcome.get_result()
        if rep.when != "call" or hasattr(rep, "wasxfail"):
            return
        nodeid = item.nodeid
        if rep.failed and call.excinfo is not None:
            if self._snap is None:
                self._snap = self.snaps.take(keep=True)
            self.state["failures"][nodeid] = {
                "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "exc": call.excinfo.typename, "message": call.excinfo.exconly()[:2000],
                "snapshot": self._snap, "file": str(item.path), "name": getattr(item, "originalname", None) or item.name}
        elif rep.passed and nodeid in self.state["failures"]:
            was = self.state["failures"].pop(nodeid)
            now = self.snaps.take()
            changed = sorted(p for p in set(was["snapshot"]) | set(now) if was["snapshot"].get(p) != now.get(p))
            if not changed:
                self.flaky += 1                     # passed with no change: flaky, not a fix
                return
            self._fixed(nodeid, was, now, changed)

    def pytest_sessionfinish(self, session):
        self._save()

    def pytest_terminal_summary(self, terminalreporter):
        tr = terminalreporter
        open_fails = len(self.state["failures"])
        if not (self.made or self.flaky or self.refused or open_fails or self.errors):
            return
        tr.section("traceX")
        where = f"sent to {self.node}" if self.submit else f"written to {os.path.relpath(self.outbox, self.root)} " \
                                                           "(dry run: nothing sent; review with python -m traceex.outbox list)"
        if self.made:
            tr.write_line(f"{len(self.made)} test{'s' if len(self.made) != 1 else ''} fixed by a change: "
                          f"{len(self.made)} trace{'s' if len(self.made) != 1 else ''} {where}")
        if self.flaky:
            tr.write_line(f"{self.flaky} test{'s' if self.flaky != 1 else ''} passed again with no change: flaky, no trace")
        if self.refused:
            tr.write_line(f"{self.refused} trace{'s' if self.refused != 1 else ''} dropped by the privacy scan")
        for e in self.errors:
            tr.write_line(f"not sent ({e}); kept in the outbox")
        if open_fails:
            tr.write_line(f"{open_fails} failing test{'s' if open_fails != 1 else ''} remembered: a later run where "
                          f"{'it passes' if open_fails == 1 else 'they pass'} after a change makes a trace")

    # --- the trace ----------------------------------------------------------------------------------------------------
    def build(self, nodeid, was, now, changed):
        """The trace for one fixed test, every piece skeletonized with one shared placeholder table."""
        sk = CodeSkeleton()
        test_file = nodeid.split("::")[0]
        name = (was.get("name") or nodeid.split("::")[-1]).split("[")[0]
        src, nparams, nasserts = shape_of_test(was.get("file") or os.path.join(self.root, test_file), name)
        exc = was.get("exc") or "Exception"
        exc_sk = exc if exc in KEEP else sk.name(exc)
        message = was.get("message") or ""
        msg = message.split(":", 1)[1] if message.startswith(exc + ":") else message
        msg_sk = sk.code("\n".join(msg.strip().splitlines()[:3]))[:600]
        diff = []
        for path in changed[:MAX_CHANGED]:
            old = self.snaps.text(was["snapshot"][path]) if path in was["snapshot"] else ""
            new_full = os.path.join(self.root, path)
            try:
                with open(new_full, encoding="utf-8", errors="replace") as fh:
                    new = fh.read() if path in now else ""
            except OSError:
                new = ""
            if old is None:
                continue
            diff += sk.diff(old, new, label=path)
        if len(changed) > MAX_CHANGED:
            diff.append(f"… {len(changed) - MAX_CHANGED} more changed files not shown")
        removed, added = changed_sides(diff)
        if not removed and not added:
            return None
        params = "" if not nodeid.endswith("]") else "[{STR}]"
        lines = [f"pytest: {sk.path(test_file)}::{sk.words(name)}{params} failed with {exc_sk}, then passed after a "
                 f"change to {len(changed)} file{'s' if len(changed) != 1 else ''}",
                 f"shape: {nparams} parameter{'s' if nparams != 1 else ''}, {nasserts} assert{'s' if nasserts != 1 else ''}",
                 f"failure: {exc_sk}: {msg_sk}"]
        if src:
            lines += ["test:", sk.code(src)]
        lines += ["change:", *diff]
        from .trace import Trace
        return Trace.from_fix(task="code.repair", base_model=self.model, input="\n".join(lines),
                              model_output={"code": "\n".join(removed)}, verified_output={"code": "\n".join(added)},
                              checker="pytest@1", producer=self.address or NO_ADDRESS, privacy="open",
                              feedback=[f"{exc_sk}: {msg_sk}"], failure_modes={"code": failure_mode(exc)},
                              fixed_by={"code": "unknown"})

    def _fixed(self, nodeid, was, now, changed):
        t = self.build(nodeid, was, now, changed)
        if t is None:
            return
        if privacy_leaks(t):
            self.refused += 1
            return
        about = {"source": nodeid, "changed_files": len(changed), "note": "local only: never sent"}
        from . import outbox
        if self.submit and self.node and self.address:
            from .client import Client
            try:
                r = Client(self.node, self.address).submit(t)
                self.sent += 1
                self.made.append(r.get("id"))
                return
            except Exception as e:                  # an unreachable node never loses a trace: it waits in the outbox
                self.errors.append(f"{type(e).__name__}: {str(e)[:80]}")
        elif self.submit:
            self.errors.append("--traceex-submit needs --traceex-node and --traceex-address")
        outbox.write(t, self.outbox, about)
        self.made.append(t.id)
