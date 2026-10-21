"""Fix episodes: the part of a Beacon session that is a verified fix.

A fix episode is, inside one session:
  1. a check that fails: a command result (`command.executed`, or `tool.failed` / `tool.completed` carrying a command)
     whose `command.exit_code` is non-zero; or, when the runtime reported no exit code, whose event says failure
     (`tool.failed`, `error.type`, severity medium or worse) or whose output reads as a failure (a traceback,
     `N failed`, `FAILED`, `error TS2304`, `panic:`, ...). A pipe that hides the exit code (`pytest | tail`) is
     judged by the output;
  2. one or more edits after it: `file.modified`, `file.created`, `file.deleted` (with `file.diff`, or a diff rebuilt
     from the edit tool's arguments in `gen_ai.tool.call.arguments`);
  3. the same check passing later: the same command (ignoring output flags such as `-q`, `-v`, `--tb=short`, a
     leading `cd ... &&` and trailing pipes), or a broader run of the same runner (`pytest` after
     `pytest tests/test_a.py`), with exit code 0 (or, without one, an `info` result with no failure in its output).
More failures of the same check before the pass are attempts of the same episode. A check that passes again with no
edit in between was flaky: no episode. Commands that inspect, navigate, install or talk to git (`ls`, `cat`, `grep`,
`cd`, `git`, `pip install`, `npm install`, ...) are never checks.
"""
import difflib
import re
import shlex

from .events import get

EDIT_ACTIONS = {"file.modified", "file.created", "file.deleted", "file.edited", "file.written"}
CHECK_ACTIONS = {"command.executed", "tool.failed", "tool.completed"}
SHELL_TOOLS = {"bash", "shell", "run_terminal_cmd", "exec_command", "run_command", "terminal", "execute_command",
               "powershell", "local_shell", "run_shell_command", "command"}

FAIL_MARK = re.compile(r"Traceback \(most recent call last\)|^\s*E\s{2,}\S|\b[1-9]\d* (?:failed|errors?)\b|"
                       r"\bFAILED\b|^FAIL\b|--- FAIL:|\bnpm ERR!|\berror(?:\[E\d+\]|\s+TS\d+)|^error:|\bpanic:|"
                       r"\b(?:[A-Z][a-z]+)+(?:Error|Exception)\b(?!\s+(?:is|was)\s+raised as expected)|"
                       r"\bAssertionError\b|Segmentation fault|\bTests?:\s+\d+ failed|\bexit (?:status|code) [1-9]",
                       re.M)
PASS_MARK = re.compile(r"\b\d+ passed\b|^OK\b|^ok\s|\bAll tests passed\b|\bTests?:\s+\d+ passed|\bPASS\b|"
                       r"\b0 (?:failed|errors?)\b|^Ran \d+ tests?", re.M)
PIPED = re.compile(r"\|\s*(?:tail|head|grep|rg|less|more|tee|cat|sed|awk|cut|sort|uniq|wc|Select-Object|"
                   r"Out-String|findstr)\b")

NOT_CHECKS = {"ls", "dir", "cat", "head", "tail", "less", "more", "grep", "rg", "ag", "find", "fd", "echo", "printf",
              "pwd", "cd", "pushd", "popd", "which", "where", "type", "git", "gh", "hg", "svn", "mkdir", "rm", "rmdir",
              "cp", "mv", "ln", "touch", "chmod", "chown", "curl", "wget", "ssh", "scp", "rsync", "open", "start",
              "code", "cursor", "kill", "pkill", "taskkill", "ps", "top", "htop", "sleep", "env", "export", "set",
              "unset", "source", ".", "clear", "cls", "history", "du", "df", "wc", "sort", "uniq", "diff", "tree",
              "stat", "file", "date", "whoami", "hostname", "brew", "apt", "apt-get", "yum", "dnf", "choco", "winget",
              "scoop", "sudo", "tar", "zip", "unzip", "gzip", "xargs", "tee", "sed", "awk", "jq", "yq", "man",
              "true", "false", "exit", "read", "nano", "vim", "vi", "emacs", "explorer", "get-childitem",
              "get-content", "set-location", "select-string", "write-host", "write-output", "test-path", "copy",
              "del", "move", "ren", "mklink", "docker-compose", "docker", "kubectl"}
INSTALLS = re.compile(r"^(?:pip3?|uv\s+pip|python3?\s+-m\s+pip|npm|pnpm|yarn|bun|poetry|uv|cargo|go|gem|bundle|"
                      r"composer|conda|mamba)\s+(?:install|i|ci|add|remove|uninstall|update|upgrade|sync|lock|get|"
                      r"mod\s+(?:tidy|download)|init|login|publish|config|cache|list|show|freeze|outdated)\b")
NOISE_FLAGS = re.compile(r"^(?:-q+|-v+|-x|-s|-rA|-r[a-zA-Z]+|--tb=\S+|--color=\S+|--no-header|--quiet|--verbose|"
                         r"--silent|--reporter=\S+|--maxfail=\S+|--disable-warnings|-W\S*|--durations=\S+|"
                         r"--no-cov|-p|no:cacheprovider|--ci|--watch=false|--run|--count=1|-count=1|--nocapture)$")

RUNNERS = [  # (checker id, regex over the normalized command)
    ("pytest", re.compile(r"^(?:(?:uv|poetry|pipenv|hatch|pdm)\s+run\s+)?(?:python3?(?:\.\d+)?\s+-m\s+)?"
                          r"(?:pytest|py\.test)\b")),
    ("unittest", re.compile(r"^(?:python3?(?:\.\d+)?\s+-m\s+unittest)\b")),
    ("go-test", re.compile(r"^go\s+test\b")),
    ("go-build", re.compile(r"^go\s+(?:build|vet)\b")),
    ("cargo-test", re.compile(r"^cargo\s+(?:test|nextest)\b")),
    ("cargo-build", re.compile(r"^cargo\s+(?:build|check|clippy)\b")),
    ("jest", re.compile(r"^(?:npx\s+|pnpm\s+(?:exec\s+)?|yarn\s+)?jest\b")),
    ("vitest", re.compile(r"^(?:npx\s+|pnpm\s+(?:exec\s+)?|yarn\s+)?vitest\b")),
    ("npm-test", re.compile(r"^(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b")),
    ("npm-script", re.compile(r"^(?:npm|pnpm|yarn|bun)\s+run\s+\S+")),
    ("tsc", re.compile(r"^(?:npx\s+)?tsc\b")),
    ("mypy", re.compile(r"^(?:python3?\s+-m\s+)?mypy\b")),
    ("ruff", re.compile(r"^(?:python3?\s+-m\s+)?ruff\b")),
    ("pyright", re.compile(r"^(?:npx\s+)?pyright\b")),
    ("eslint", re.compile(r"^(?:npx\s+)?eslint\b")),
    ("flake8", re.compile(r"^(?:python3?\s+-m\s+)?flake8\b")),
    ("dotnet-test", re.compile(r"^dotnet\s+(?:test|build)\b")),
    ("maven", re.compile(r"^mvn\w*\s")),
    ("gradle", re.compile(r"^(?:\./)?gradlew?\s")),
    ("make", re.compile(r"^make\b")),
    ("python-run", re.compile(r"^python3?(?:\.\d+)?\s+(?!-m\s+pip)\S")),
    ("node-run", re.compile(r"^(?:node|deno|bun|tsx|ts-node)\s+\S")),
]


def command_of(ev):
    cmd = get(ev, "command.command")
    if not cmd and str(get(ev, "tool.name", "")).lower() in SHELL_TOOLS:
        cmd = get(ev, "tool.command")
    if not cmd:
        args = get(ev, "gen_ai.tool.call.arguments")
        if isinstance(args, dict) and str(get(ev, "tool.name", "")).lower() in SHELL_TOOLS:
            cmd = args.get("command") or args.get("cmd")
            if isinstance(cmd, list):
                cmd = " ".join(map(str, cmd))
    return str(cmd).strip() if cmd else None


def output_of(ev):
    out = get(ev, "command.output")
    if out:
        return str(out)
    res = get(ev, "gen_ai.tool.call.result")
    if isinstance(res, str):
        return res
    if isinstance(res, list):
        return "\n".join(str(p.get("text", p) if isinstance(p, dict) else p) for p in res)
    if isinstance(res, dict):
        return str(res.get("output") or res.get("stdout") or res.get("content") or res.get("text") or "")
    return ""


def normalize(cmd):
    """The command as a check: no leading `cd x &&` or env assignments, no trailing pipes or redirects."""
    c = cmd.strip()
    c = re.sub(r"^(?:(?:cd|pushd|Set-Location)\s+(?:\"[^\"]*\"|'[^']*'|\S+)\s*(?:&&|;|\|\|)\s*)+", "", c)
    c = re.sub(r"^(?:(?:set|export|\$env:)\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:\"[^\"]*\"|'[^']*'|\S*)\s*(?:&&|;)\s*)+",
               "", c, flags=re.I)
    c = re.sub(r"^(?:[A-Z_][A-Z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S*)\s+)+", "", c)
    c = re.sub(r"^(?:timeout\s+\d+\s+)", "", c)
    c = re.split(r"\s*(?:\|(?!\|)|2>&1|>\s*\S+|;\s*echo\b|&&\s*echo\b)", c)[0]
    return re.sub(r"\s+", " ", c).strip()


def check_of(ev):
    """(runner, key, targets) when this event is the result of a check command, else None."""
    if get(ev, "event.action") not in CHECK_ACTIONS:
        return None
    cmd = command_of(ev)
    if not cmd:
        return None
    c = normalize(cmd)
    if not c or INSTALLS.match(c):
        return None
    first = re.split(r"[\s/\\]", c.split(" ")[0])[-1].lower() if c else ""
    first = re.sub(r"\.exe$", "", first)
    if first in NOT_CHECKS:
        return None
    runner = next((name for name, rx in RUNNERS if rx.match(c)), None) or "shell"
    try:
        toks = shlex.split(c, posix=True)
    except ValueError:
        toks = c.split()
    kept = [t for t in toks if not NOISE_FLAGS.match(t)]
    targets = tuple(t for t in kept[1:] if not t.startswith("-") and ("/" in t or "\\" in t or "::" in t or
                                                                      re.search(r"\.\w+$", t)))
    return runner, " ".join(kept), targets


def status_of(ev):
    """"fail", "pass" or None (can't tell) for a check result."""
    code = get(ev, "command.exit_code")
    out = output_of(ev)
    piped = bool(PIPED.search(command_of(ev) or ""))
    if isinstance(code, int) and not isinstance(code, bool):
        if code != 0:
            return "fail"
        if piped and FAIL_MARK.search(out):
            return "fail"                              # `pytest | tail` exits 0 whatever pytest did
        return "pass"
    if get(ev, "event.action") == "tool.failed" or get(ev, "error.type"):
        return "fail"
    if str(ev.get("severity", "")).lower() in ("medium", "high", "critical"):
        return "fail"
    if out and FAIL_MARK.search(out):
        return "fail"
    if out and PASS_MARK.search(out):
        return "pass"
    if str(ev.get("severity", "")).lower() == "info" and get(ev, "event.action") == "command.executed":
        return "pass"
    return None


def covers(passing, failing):
    """True when a passing check re-runs a failing one: the same key, or a broader run of the same runner."""
    (pr, pk, pt), (fr, fk, ft) = passing, failing
    if pk == fk:
        return True
    if pr != fr or pr in ("shell", "python-run", "node-run", "npm-script", "make"):
        return False
    if not pt:
        return True
    norm = lambda t: t.replace("\\", "/").split("::")[0].lstrip("./")
    return bool(ft) and all(any(norm(f).startswith(norm(p)) for p in pt) for f in ft)


# --- edits -----------------------------------------------------------------------------------------------------------

def _synth_diff(path, old, new):
    a = (old or "").splitlines()
    b = (new or "").splitlines()
    lines = list(difflib.unified_diff(a, b, fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=2))
    return "\n".join(lines) if lines else None


def edit_of(ev):
    """{path, operation, diff (unified, or None when Beacon kept only its hash), diff_bytes} for a file edit."""
    if get(ev, "event.action") not in EDIT_ACTIONS:
        return None
    path = get(ev, "file.path") or get(ev, "tool.path")
    args = get(ev, "gen_ai.tool.call.arguments")
    if isinstance(args, str):
        try:
            import json
            args = json.loads(args)
        except ValueError:
            args = None
    if not path and isinstance(args, dict):
        path = args.get("file_path") or args.get("path") or args.get("filePath")
    if not path:
        return None
    diff = get(ev, "file.diff")
    if not diff and isinstance(args, dict):
        if "old_string" in args or "new_string" in args:
            diff = _synth_diff(path, args.get("old_string"), args.get("new_string"))
        elif isinstance(args.get("edits"), list):
            parts = [_synth_diff(path, e.get("old_string"), e.get("new_string")) for e in args["edits"]
                     if isinstance(e, dict)]
            diff = "\n".join(p for p in parts if p) or None
        elif "content" in args and get(ev, "event.action") == "file.created":
            diff = _synth_diff(path, "", args.get("content"))
    op = get(ev, "file.operation") or {"file.created": "create", "file.deleted": "delete"}.get(
        get(ev, "event.action"), "modify")
    return {"path": str(path), "operation": op, "diff": diff, "diff_bytes": get(ev, "file.diff_bytes")}


# --- segmentation ----------------------------------------------------------------------------------------------------

def _prompt_text(ev):
    if get(ev, "event.action") != "prompt.submitted":
        return None
    return get(ev, "prompt.text") or None


def find(events):
    """(episodes, stats) for one session's ordered events. An episode is a dict: failures (the failing check events,
    first to last), passed (the passing event), edits (edit dicts with their events), prompt (the last prompt before
    the first failure), prompts_between (how many prompts arrived during the fix), check (runner, key, targets)."""
    stats = {"checks": 0, "failures": 0, "episodes": 0, "flaky": 0, "unresolved": 0}
    open_, episodes, last_prompt = {}, [], None
    for ev in events:
        text = _prompt_text(ev)
        if text is not None:
            last_prompt = text
            for s in open_.values():
                s["prompts_between"] += 1
            continue
        e = edit_of(ev)
        if e is not None:
            for s in open_.values():
                s["edits"].append((e, ev))
            continue
        chk = check_of(ev)
        if chk is None:
            continue
        st = status_of(ev)
        if st is None:
            continue
        stats["checks"] += 1
        if st == "fail":
            stats["failures"] += 1
            if chk[1] in open_:
                open_[chk[1]]["failures"].append(ev)
            else:
                open_[chk[1]] = {"check": chk, "failures": [ev], "edits": [], "prompt": last_prompt,
                                 "prompts_between": 0}
            continue
        for key in [k for k, s in open_.items() if covers(chk, s["check"])]:
            s = open_.pop(key)
            if not s["edits"]:
                stats["flaky"] += 1
                continue
            s["passed"] = ev
            s["passed_check"] = chk
            episodes.append(s)
            stats["episodes"] += 1
    stats["unresolved"] = len(open_)
    return episodes, stats
