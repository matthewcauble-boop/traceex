"""Reading Agent Beacon's runtime log (`~/.beacon/endpoint/logs/runtime.jsonl`): one normalized event per line.

Only the fields the exporter uses are read; see MAPPING.md. The log is read, never written, and nothing here sends
anything anywhere. Lines that are not JSON objects, or not Beacon `agent_runtime` events, are skipped and counted.

Ordering follows Beacon's own rule: by `timestamp` (fixed-width RFC3339 UTC, so string order is time order), then by
`sequence`; never by position in the file, because the hook path and the collector path append at different times.
The same action reported by both paths carries one `event.id`; those copies are merged into one event.
"""
import json
import os


def get(ev, dotted, default=None):
    """ev["a"]["b"]["c"] for "a.b.c", or default."""
    cur = ev
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _merge(a, b):
    """b's fields added to a's, nested, a's non-empty values kept."""
    out = dict(a)
    for k, v in b.items():
        if k not in out or out[k] in (None, "", [], {}):
            out[k] = v
        elif isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
    return out


def read(path_or_lines):
    """(events, stats) from a runtime.jsonl path, an open file, or an iterable of lines."""
    stats = {"lines": 0, "skipped": 0, "merged": 0}
    if isinstance(path_or_lines, (str, os.PathLike)):
        with open(path_or_lines, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    else:
        lines = list(path_or_lines)
    by_id, events = {}, []
    for line in lines:
        line = line.strip() if isinstance(line, str) else line
        if not line:
            continue
        stats["lines"] += 1
        try:
            ev = json.loads(line) if isinstance(line, str) else line
        except ValueError:
            stats["skipped"] += 1
            continue
        if not isinstance(ev, dict) or not isinstance(ev.get("event"), dict) or not ev["event"].get("action"):
            stats["skipped"] += 1
            continue
        if ev.get("vendor", "beacon") != "beacon" or ev["event"].get("kind", "agent_runtime") != "agent_runtime":
            stats["skipped"] += 1
            continue
        eid = ev["event"].get("id")
        if eid and eid in by_id:
            i = by_id[eid]
            events[i] = _merge(events[i], ev)
            stats["merged"] += 1
            continue
        if eid:
            by_id[eid] = len(events)
        events.append(ev)
    events.sort(key=lambda e: (str(e.get("timestamp", "")), int(e.get("sequence") or 0)))
    return events, stats


def session_key(ev):
    """Which session an event belongs to: session.id, else the harness and working directory, else the run."""
    sid = get(ev, "session.id") or get(ev, "gen_ai.conversation.id")
    if sid:
        return f"{get(ev, 'harness.name', '?')}|{sid}"
    wd = get(ev, "session.working_directory")
    if wd:
        return f"{get(ev, 'harness.name', '?')}|wd:{wd}"
    rid = get(ev, "run.run_id")
    return f"{get(ev, 'harness.name', '?')}|run:{rid}" if rid else None


def sessions(events):
    """{session key: [events in order]}; events with no session are dropped (they can't be put in an episode)."""
    out = {}
    for ev in events:
        k = session_key(ev)
        if k:
            out.setdefault(k, []).append(ev)
    return out
