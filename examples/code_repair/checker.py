"""Checker 'mbpp-tests@1': run a candidate against the task's unit tests and say exactly how it failed.

This is the reward function. It never needs a reference solution: the tests decide. Each failure comes back with a
mode (for search and lots) and a short traceback (for the repair loop and for training data):

  no_code        no code in the reply            syntax_error   doesn't parse
  unsafe         touches the OS, network or files (refused, never run)
  wrong_name     the tests call a function the code doesn't define
  runtime_error  raised while running a test     wrong_answer   ran, returned the wrong value (with got / expected)
  timeout        took longer than the limit

Candidates run in a separate `python -I` process, in an empty temporary directory, with a timeout, after a static
deny-list check. That keeps small-model homework safe on a dev machine; it is not a security sandbox for hostile code.
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile

TIMEOUT = 8
DENY_IMPORTS = {"os", "subprocess", "shutil", "socket", "ctypes", "multiprocessing", "signal", "requests", "urllib",
                "http", "ftplib", "smtplib", "pathlib", "glob", "tempfile", "importlib", "pickle", "marshal", "asyncio"}
DENY_CALLS = {"exec", "compile", "__import__", "open", "input", "breakpoint", "exit", "quit"}
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

HARNESS = r'''
import ast, json, sys
_src = open(sys.argv[1], encoding="utf-8").read()
_tests = json.loads(open(sys.argv[2], encoding="utf-8").read())
_g = {"__name__": "__candidate__"}
_out = {"defined": True, "results": []}
try:
    exec(compile(_src, "candidate.py", "exec"), _g)
except BaseException as e:
    _out["defined"] = False
    _out["error"] = type(e).__name__ + ": " + str(e)[:300]
if _out["defined"]:
    for _t in _tests:
        _r = {"test": _t, "ok": False}
        try:
            exec(_t, _g)
            _r["ok"] = True
        except AssertionError:
            _r["error"] = "AssertionError"
            try:                                   # name what the code actually returned
                _node = ast.parse(_t).body[0].test
                if isinstance(_node, ast.Compare) and len(_node.ops) == 1 and isinstance(_node.ops[0], ast.Eq):
                    _r["got"] = repr(eval(compile(ast.Expression(_node.left), "t", "eval"), _g))[:200]
                    _r["expected"] = repr(eval(compile(ast.Expression(_node.comparators[0]), "t", "eval"), _g))[:200]
            except BaseException:
                pass
        except BaseException as e:
            _r["error"] = type(e).__name__ + ": " + str(e)[:200]
        _out["results"].append(_r)
print("__RESULT__" + json.dumps(_out))
'''


def extract_code(reply):
    blocks = FENCE.findall(reply or "")
    if blocks:
        return strip_scaffolding(max(blocks, key=len).strip())
    text = (reply or "").strip()
    return strip_scaffolding(text) if ("def " in text or "lambda" in text) else ""


def strip_scaffolding(code):
    """Drop the model's own top-level test calls (asserts, prints, bare calls, `if __name__ == "__main__":`), keeping
    imports, definitions and assignments. Applied identically to every model, so scores measure the solution itself."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Assert, ast.Expr)) and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            continue
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            continue
        keep.append(node)
    lines = code.splitlines()
    return "\n".join("\n".join(lines[n.lineno - 1 - len(getattr(n, "decorator_list", [])):n.end_lineno]) for n in keep).strip()


def _unsafe(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in DENY_IMPORTS:
                    return f"import {a.name}"
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] in DENY_IMPORTS:
                return f"from {n.module} import ..."
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in DENY_CALLS:
            return f"call to {n.func.id}()"
        elif isinstance(n, ast.Attribute) and n.attr in {"system", "popen", "rmdir", "unlink", "kill", "rmtree"}:
            return f"use of .{n.attr}"
    return None


def _names_called(tests):
    out = set()
    for t in tests:
        try:
            for n in ast.walk(ast.parse(t)):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    out.add(n.func.id)
        except SyntaxError:
            pass
    return out


def check_many(pairs, workers=8):
    """[(task, reply), ...] -> [check(...)]: each check is a subprocess, so threads parallelise them well."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: check(*p), pairs))


def check(task, reply):
    """-> {"passed": bool, "mode": str|None, "feedback": str, "code": str, "passed_tests": int, "total": int}"""
    tests = list(task["test_list"])
    code = extract_code(reply)
    res = {"passed": False, "mode": None, "feedback": "", "code": code, "passed_tests": 0, "total": len(tests)}
    if not code:
        return dict(res, mode="no_code", feedback="No Python code block found in the reply.")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return dict(res, mode="syntax_error", feedback=f"SyntaxError: {e.msg} (line {e.lineno})")
    bad = _unsafe(tree)
    if bad:
        return dict(res, mode="unsafe", feedback=f"Refused to run: {bad} is not allowed in solutions.")
    defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    defined |= {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    missing = sorted(n for n in _names_called(tests) if n not in defined and n not in dir(__builtins__))
    src = ((task.get("test_setup_code") or "") + "\n" + code).replace("\r\n", "\n").replace("\r", "\n")
    with tempfile.TemporaryDirectory() as d:
        cf, tf, hf = (os.path.join(d, n) for n in ("candidate.py", "tests.json", "harness.py"))
        with open(cf, "w", encoding="utf-8", newline="\n") as f:    # MBPP stores \r\n; doubling it breaks '\' joins
            f.write(src)
        with open(tf, "w", encoding="utf-8") as f:
            json.dump(tests, f)
        with open(hf, "w", encoding="utf-8") as f:
            f.write(HARNESS)
        # -s (no user site) instead of -I, because -I would ignore PYTHONHASHSEED: a fixed seed makes reprs of sets and
        # dicts identical on every run, so the feedback (and every repair prompt built from it) is reproducible
        env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ}
        env.update(PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
        try:
            p = subprocess.run([sys.executable, "-s", "-X", "utf8", hf, cf, tf], cwd=d, capture_output=True,
                               text=True, timeout=TIMEOUT, stdin=subprocess.DEVNULL, env=env)
        except subprocess.TimeoutExpired:
            return dict(res, mode="timeout", feedback=f"Timed out after {TIMEOUT}s (infinite loop or far too slow).")
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("__RESULT__")), None)
    if not line:
        tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["crashed"]
        return dict(res, mode="runtime_error", feedback=tail[0][:300])
    out = json.loads(line[len("__RESULT__"):])
    if not out["defined"]:
        return dict(res, mode="runtime_error", feedback=f"Error while defining the code: {out['error']}")
    results = out["results"]
    ok = sum(r["ok"] for r in results)
    res["passed_tests"] = ok
    if ok == len(tests):
        return dict(res, passed=True)
    first = next(r for r in results if not r["ok"])
    err = first.get("error", "")
    if err.startswith("NameError") and missing:
        mode = "wrong_name"
        fb = f"{err}. The tests call {', '.join(missing)}(), which your code does not define."
    elif err == "AssertionError":
        mode = "wrong_answer"
        fb = f"Failed: {first['test']}"
        if "got" in first:
            fb += f"\n  your function returned {first['got']}, expected {first['expected']}"
    else:
        mode = "runtime_error"
        fb = f"{err}\n  while running: {first['test']}"
    fb += f"\n({ok} of {len(tests)} tests passed)"
    return dict(res, mode=mode, feedback=fb)
