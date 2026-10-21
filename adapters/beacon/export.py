"""Agent Beacon -> traceX: turn the fix episodes in a Beacon runtime log into trace submissions (SPEC 4i, path C).

    python -m adapters.beacon runtime.jsonl                    # dry run: traces into .traceex/outbox/ for review
    python -m adapters.beacon runtime.jsonl --open             # keep the scrubbed code instead of skeletons
    python -m adapters.beacon runtime.jsonl --outbox DIR --model claude-sonnet-4-5
    python -m adapters.beacon runtime.jsonl --submit --node https://<node> --address 0xYourWallet

Nothing leaves the machine unless --submit is given (with --node and --address). Without it the exporter only reads
the log and writes files: one trace per fix episode in the outbox (`python -m traceex.outbox list | show | send`
reviews and sends them later), and one scrub report per trace in <outbox>/scrub/. A trace whose scrub fails is not
written; its report goes to <outbox>/refused/ with the kinds of secret that survived (never the values).

The pipeline, per fix episode (see episodes.py for how they are found, MAPPING.md for the field mapping):
  1. scrub (mandatory): every string taken from the session (the command, the failure output, each diff and file
     path, the prompt) goes through scrub.Scrubber: secrets, personal data, home directories, paths outside the
     project (scrub.py lists every rule);
  2. build: a `trace/0.1` with task `code.repair`, the session's model, checker `<runner>@1` (the re-run command
     that passed), the failure output as feedback, and the change's removed / added lines as model_output /
     verified_output. By default (privacy "skeleton" here, written as the trace's "open" privacy level like the pytest
     plugin's) code, the command and the failure output are skeletonized with traceex.codeskel: string literals,
     comments, project identifiers and large numbers become placeholders, and the prompt is left out. With --open the
     scrubbed text is kept as it is, and the prompt's first 1,500 characters go in too;
  3. second pass: the finished trace is scanned again for every secret pattern (this adapter's and the node's); any
     hit refuses the trace. Then traceex.client.privacy_leaks, the check the node itself runs.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(ROOT, "sdk", "python") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "sdk", "python"))

from traceex.trace import Trace                                   # noqa: E402
from traceex.client import privacy_leaks                         # noqa: E402
from traceex.codeskel import CodeSkeleton, KEEP                  # noqa: E402
from traceex import outbox as tx_outbox                          # noqa: E402

from . import events as bev                                      # noqa: E402
from . import episodes as bep                                    # noqa: E402
from .scrub import Scrubber, ScrubError, second_pass             # noqa: E402

NO_ADDRESS = "0x" + "0" * 40
MAX_DIFF_LINES, MAX_FILES, MAX_FAIL_LINES, MAX_PROMPT = 80, 5, 25, 1500


CLI_WORDS = {
    # tools and file types: what a check command is made of
    "python", "python3", "py", "pytest", "unittest", "pip", "uv", "poetry", "npm", "npx", "pnpm", "yarn", "bun", "node",
    "deno", "jest", "vitest", "mocha", "tsc", "ts", "tsx", "js", "jsx", "mjs", "cjs", "json", "go", "cargo", "rustc",
    "rs", "make", "mypy", "ruff", "eslint", "pyright", "flake8", "dotnet", "mvn", "gradle", "java", "kt", "rb", "sh",
    "cs", "cpp", "hpp", "toml", "yaml", "yml", "md", "src", "lib", "spec", "m", "q", "v", "x",
    # test-runner output
    "FAILED", "PASSED", "ERROR", "ERRORS", "FAILURES", "FAIL", "PASS", "OK", "failed", "passed", "errors", "error",
    "collected", "session", "starts", "warnings", "warning", "items", "skipped", "xfailed", "xpassed", "Expected",
    "Received", "Tests", "Test", "Suites", "Ran", "tests", "in", "ok", "panic", "exit", "status", "code", "Traceback",
    "most", "recent", "call", "last", "File", "line", "module", "received", "toBe", "toEqual", "expect"}


class Skeleton(CodeSkeleton):
    """traceex's code skeleton, keeping the scrubber's [REDACTED_*] labels and command / test-runner words readable."""

    def name(self, word):
        if word.startswith("REDACTED_") or word in CLI_WORDS:
            return word
        return super().name(word)


def failure_mode(text):
    """The checker's failure mode from the failure output (the pytest plugin's modes, read from text)."""
    t = text or ""
    if re.search(r"\b(?:SyntaxError|IndentationError|TabError)\b|\bUnexpected token\b|\bsyntax error\b|"
                 r"\berror\[E0(?:0[0-9]{2})\]|expected .* found", t, re.I):
        return "syntax_error"
    if re.search(r"\b(?:NameError|AttributeError|ImportError|ModuleNotFoundError|UnboundLocalError|ReferenceError)\b|"
                 r"\berror TS2(?:304|305|307|339|551|552|724)\b|\bundefined: |\bcannot find (?:name|module|symbol)\b|"
                 r"\bunresolved (?:import|reference)\b|\bis not defined\b|\bhas no attribute\b", t, re.I):
        return "wrong_name"
    if re.search(r"\btimed? ?out\b|\bTimeoutError\b|\bdeadline exceeded\b", t, re.I):
        return "timeout"
    if re.search(r"\bAssertionError\b|^\s*E\s+assert\b|\bassert(?:ion)? failed\b|\bexpected\b.*\b(?:to|but)\b|"
                 r"\b[1-9]\d* failed\b|--- FAIL:|\bFAILED\b|\bnot equal\b", t, re.I | re.M):
        return "wrong_answer"
    return "runtime_error"


def _excerpt(output):
    """The part of a failure output worth keeping: the lines around the first failure marker, then the tail."""
    lines = [ln.rstrip() for ln in (output or "").splitlines() if ln.strip()]
    if len(lines) <= MAX_FAIL_LINES:
        return "\n".join(lines)
    first = next((i for i, ln in enumerate(lines) if bep.FAIL_MARK.search(ln)), None)
    keep = list(range(max(0, first - 3), min(len(lines), first + 12))) if first is not None else []
    tail = list(range(len(lines) - (MAX_FAIL_LINES - len(keep)), len(lines)))
    idx = sorted(set(keep) | set(i for i in tail if i >= 0))
    out, prev = [], None
    for i in idx:
        if prev is not None and i != prev + 1:
            out.append("…")
        out.append(lines[i])
        prev = i
    return "\n".join(out)


def _diff_lines(diff, sk, rel_path):
    """Hunk lines of one unified diff: headers dropped, a path header added; each line skeletonized when sk is set."""
    out = [f"@@ {sk.path(rel_path) if sk else rel_path} @@"]
    for ln in diff.splitlines():
        if ln.startswith(("---", "+++", "diff --git", "index ", "new file mode", "deleted file mode")):
            continue
        if ln.startswith("@@"):
            out.append("@@")
            continue
        if not ln[:1] in ("+", "-", " ") or not ln[1:].strip():
            continue
        body = sk.code(ln[1:]) if sk else ln[1:]
        out.append(ln[0] + body.rstrip())
    return out


def _session_names(evs):
    names = set()
    for ev in evs:
        for f in ("user.name", "endpoint.hostname", "run.actor"):
            v = bev.get(ev, f)
            if v:
                names.add(str(v))
    return names


def _model_of(evs, until):
    model = None
    for ev in evs:
        m = bev.get(ev, "model") or bev.get(ev, "gen_ai.response.model") or bev.get(ev, "gen_ai.request.model")
        if m:
            model = str(m)
        if ev is until:
            break
    return model


def build(episode, session_events, *, producer=None, model=None, open_text=False):
    """(trace or None, report, about) for one episode. Raises ScrubError if a secret survives the scrub."""
    fails, passed = episode["failures"], episode["passed"]
    project = next((bev.get(e, "session.working_directory") for e in [fails[0], passed] + session_events
                    if bev.get(e, "session.working_directory")), None)
    scrub = Scrubber(project=project, names=_session_names(session_events))
    sk = None if open_text else Skeleton()

    runner = episode["check"][0]
    cmd = scrub.scrub(bep.normalize(bep.command_of(fails[0]) or ""), "command")
    raw_out = [bep.output_of(f) for f in fails[:3]]
    excerpts = [scrub.scrub(_excerpt(o), "failure_output") for o in raw_out]
    codes = [bev.get(f, "command.exit_code") for f in fails]
    mode = failure_mode("\n".join(raw_out))

    diff, files, no_diff = [], [], 0
    for e, _ev in episode["edits"]:
        path = e["path"]
        rel = scrub.scrub(path, "file_path")
        if rel not in files:
            files.append(rel)
        if not e["diff"]:
            no_diff += 1
            continue
        if files.index(rel) >= MAX_FILES:
            continue
        diff += _diff_lines(scrub.scrub(e["diff"], "diff", path=path), sk, rel)
    report_extra = {"edits": len(episode["edits"]), "edits_without_diff": no_diff, "files": len(files)}
    if len(diff) > MAX_DIFF_LINES:
        diff = diff[:MAX_DIFF_LINES] + [f"… {len(diff) - MAX_DIFF_LINES} more diff lines not shown"]
    if len(files) > MAX_FILES:
        diff.append(f"… {len(files) - MAX_FILES} more changed files not shown")
    removed = [ln[1:] for ln in diff if ln.startswith("-")]
    added = [ln[1:] for ln in diff if ln.startswith("+")]
    about = {"source": "agent-beacon", "session": hashlib.sha256(episode["session"].encode()).hexdigest()[:16],
             "harness": bev.get(passed, "harness.name"), "attempts": len(fails),
             "failed_event_ids": [bev.get(f, "event.id") for f in fails if bev.get(f, "event.id")],
             "passed_event_id": bev.get(passed, "event.id"), "files": len(files),
             "note": "local only: never sent"}
    if not removed and not added:
        return None, dict(scrub.report(), **report_extra, outcome="skipped: no diff retained"), about

    if sk:
        cmd_sk = sk.code(cmd)
        excerpts = [sk.code(x)[:1500] for x in excerpts]
    else:
        cmd_sk = cmd
        excerpts = [x[:3000] for x in excerpts]
    code_txt = ", ".join(sorted({f"exit {c}" for c in codes if isinstance(c, int)})) or "no exit code"
    harness = re.sub(r"[^a-z0-9_.-]", "", str(bev.get(passed, "harness.name") or "agent").lower()) or "agent"
    lines = [f"beacon: a {harness} session ran `{cmd_sk}` ({runner}); it failed {len(fails)} "
             f"time{'s' if len(fails) != 1 else ''} ({code_txt}), then passed after {len(episode['edits'])} "
             f"edit{'s' if len(episode['edits']) != 1 else ''} to {len(files)} file{'s' if len(files) != 1 else ''}",
             f"failure: {mode}", *excerpts[:1]]
    if open_text and episode.get("prompt"):
        lines += ["task:", scrub.scrub(episode["prompt"], "prompt")[:MAX_PROMPT]]
    lines += ["change:", *diff]
    created = str(passed.get("timestamp") or "")[:19] + "Z" if passed.get("timestamp") else None
    fixed_by = "model" if not episode.get("prompts_between") else "unknown"
    base = model or _model_of(session_events, passed) or "unknown"
    t = Trace.from_fix(task="code.repair", base_model=base, input="\n".join(lines),
                       model_output={"code": "\n".join(removed)}, verified_output={"code": "\n".join(added)},
                       checker=f"{runner}@1", producer=producer or NO_ADDRESS, privacy="open",
                       feedback=excerpts, failure_modes={"code": mode}, fixed_by={"code": fixed_by}, created=created)
    if not t.get("fixed_fields"):
        return None, dict(scrub.report(), **report_extra, outcome="skipped: change made no difference"), about
    report = dict(scrub.report(), **report_extra)
    # second pass: everything the trace holds, serialized, must be free of every secret pattern
    second_pass(json.dumps(dict(t), ensure_ascii=False) + "\n" + t.text())
    leaks = privacy_leaks(t)
    if leaks:
        raise ScrubError([(k, len(str(v))) for k, v in leaks])
    report["outcome"] = "written"
    report["second_pass"] = "clean"
    return t, report, about


def export(path_or_lines, outbox=None, *, producer=None, model=None, open_text=False, submit=False, node=None,
           client=None):
    """Read a Beacon log, write one trace per fix episode to the outbox. Returns a summary dict. Sends only when
    submit is True (and then needs node and producer, or a client)."""
    outbox = outbox or tx_outbox.DEFAULT_DIR
    evs, read_stats = bev.read(path_or_lines)
    summary = {"read": read_stats, "sessions": 0, "episodes": 0, "written": [], "refused": [], "skipped": [],
               "flaky": 0, "sent": [], "outbox": outbox}
    if submit and client is None and not (node and producer):
        raise ValueError("--submit needs --node and --address")
    for key, sess in bev.sessions(evs).items():
        summary["sessions"] += 1
        found, st = bep.find(sess)
        summary["flaky"] += st["flaky"]
        for ep in found:
            ep["session"] = key
            summary["episodes"] += 1
            ref = hashlib.sha256(f"{key}|{bev.get(ep['passed'], 'timestamp')}|{ep['check'][1]}".encode()) \
                .hexdigest()[:24]
            try:
                t, report, about = build(ep, sess, producer=producer, model=model, open_text=open_text)
            except ScrubError as e:
                kinds = sorted({k for k, _ in e.hits})
                _write_json(os.path.join(outbox, "refused", ref + ".json"),
                            {"outcome": "refused: secret patterns survived the second pass", "kinds": kinds,
                             "count": len(e.hits), "at": _now()})
                summary["refused"].append({"episode": ref, "kinds": kinds})
                continue
            if t is None:
                summary["skipped"].append({"episode": ref, "why": report.get("outcome")})
                continue
            path = tx_outbox.write(t, outbox, about)
            name = os.path.basename(path)
            _write_json(os.path.join(outbox, "scrub", name), dict(report, trace=t.id, at=_now()))
            summary["written"].append({"file": path, "id": t.id, "checker": t["checker"]["id"],
                                       "failure": t.get("failure_modes", {}).get("code")})
    if submit and summary["written"]:
        summary["sent"] = _submit(summary["written"], outbox, node, producer, client)
    return summary


def _submit(written, outbox, node, producer, client):
    """Send only the traces this run wrote (not the rest of the outbox), as `traceex.outbox send` would."""
    from traceex.client import Client
    client = client or Client(node, producer)
    sent = []
    for w in written:
        with open(w["file"], encoding="utf-8") as fh:
            item = json.load(fh)
        t = dict(item["trace"])
        if t.get("producer") in (None, "", NO_ADDRESS):
            t["producer"] = producer
        second_pass(json.dumps(t, ensure_ascii=False))
        r = client.submit(t)
        item["sent"] = {"at": _now(), "reply": r}
        _write_json(os.path.join(outbox, "sent", os.path.basename(w["file"])), item)
        os.remove(w["file"])
        sent.append({"file": w["file"], "id": r.get("id"), "failure_id": r.get("failure_id")})
    return sent


def _now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m adapters.beacon",
                                 description="Agent Beacon runtime.jsonl -> traceX traces (local outbox by default)")
    ap.add_argument("log", help="a Beacon runtime.jsonl (e.g. ~/.beacon/endpoint/logs/runtime.jsonl)")
    ap.add_argument("--outbox", default=tx_outbox.DEFAULT_DIR, help="where traces go (default .traceex/outbox)")
    ap.add_argument("--model", help="the model name to record (default: the session's `model` field)")
    ap.add_argument("--open", action="store_true", help="keep the scrubbed code and prompt instead of skeletons")
    ap.add_argument("--submit", action="store_true", help="send the traces to --node (off by default)")
    ap.add_argument("--node", default=os.environ.get("TRACEX_NODE"))
    ap.add_argument("--address", default=os.environ.get("TRACEX_ADDRESS"), help="your producer address")
    a = ap.parse_args(argv)
    if a.submit and not (a.node and a.address):
        ap.error("--submit needs --node and --address")
    s = export(a.log, a.outbox, producer=a.address, model=a.model, open_text=a.open, submit=a.submit, node=a.node)
    r = s["read"]
    print(f"read {r['lines']} lines ({r['skipped']} skipped, {r['merged']} duplicate reports merged), "
          f"{s['sessions']} session{'s' if s['sessions'] != 1 else ''}, {s['episodes']} fix "
          f"episode{'s' if s['episodes'] != 1 else ''}, {s['flaky']} flaky re-run{'s' if s['flaky'] != 1 else ''}")
    for w in s["written"]:
        print(f"  wrote {w['file']}  {w['checker']}  {w['failure']}")
    for x in s["skipped"]:
        print(f"  skipped {x['episode']}: {x['why']}")
    for x in s["refused"]:
        print(f"  REFUSED {x['episode']}: secret patterns survived the scrub ({', '.join(x['kinds'])}); "
              f"report in {os.path.join(s['outbox'], 'refused')}")
    if a.submit:
        print(f"sent {len(s['sent'])} trace{'s' if len(s['sent']) != 1 else ''} to {a.node}")
    else:
        print(f"dry run: nothing sent. Review with: python -m traceex.outbox list --dir {s['outbox']}")
    return 1 if s["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
