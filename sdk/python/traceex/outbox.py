"""The local outbox: traces the pytest plugin and the OpenTelemetry exporter built, waiting for a person to review them.

Nothing in the outbox has left the machine. Each file is one trace exactly as it would be sent (`trace`), plus a local
note (`about`: where it came from, e.g. the test's node id) that is never sent.

    python -m traceex.outbox list   [--dir .traceex/outbox]
    python -m traceex.outbox show   <file or trace id prefix>
    python -m traceex.outbox send   --node https://<node> --address 0xYourWallet [--dir ...] [--yes]

`send` runs the same privacy check the node runs (secrets, and personal data in skeleton traces), refuses anything
that fails it, sets the producer to your address if the trace was built without one, and moves what was sent to
<dir>/sent/.
"""
import argparse
import datetime as dt
import json
import os
import sys

from .canon import object_id
from .client import Client, privacy_leaks

NO_ADDRESS = "0x" + "0" * 40          # traces built before an address is configured; `send` fills in the real one
DEFAULT_DIR = os.path.join(".traceex", "outbox")


def write(trace, folder=DEFAULT_DIR, about=None):
    """Save one trace for review. Returns its path. The same trace written twice is one file."""
    os.makedirs(folder, exist_ok=True)
    tid = object_id(dict(trace))
    path = os.path.join(folder, tid.split(":")[-1][:24] + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"trace": dict(trace), "about": about or {},
                   "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "sent": None}, f, indent=1, ensure_ascii=False)
    return path


def items(folder=DEFAULT_DIR):
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                out.append((os.path.join(folder, name), json.load(f)))
    return out


def send(folder=DEFAULT_DIR, node=None, address=None, client=None):
    """Submit every trace in the outbox (after the privacy check), then move each sent file to <folder>/sent/."""
    client = client or Client(node, address)
    address = address or client.address
    done = []
    for path, item in items(folder):
        t = dict(item["trace"])
        if t.get("producer") in (None, "", NO_ADDRESS):
            if not address:
                raise ValueError("these traces have no producer: pass --address (where royalties go)")
            t["producer"] = address
        leaks = privacy_leaks(t)
        if leaks:
            done.append({"file": path, "refused": f"privacy check: {leaks[:3]}"})
            continue
        r = client.submit(t)
        item["sent"] = {"at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "reply": r}
        sent_dir = os.path.join(folder, "sent")
        os.makedirs(sent_dir, exist_ok=True)
        with open(os.path.join(sent_dir, os.path.basename(path)), "w", encoding="utf-8") as f:
            json.dump(item, f, indent=1, ensure_ascii=False)
        os.remove(path)
        done.append({"file": path, "id": r.get("id"), "failure_id": r.get("failure_id"), "duplicate": r.get("duplicate")})
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m traceex.outbox", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("list", "show", "send"))
    ap.add_argument("what", nargs="?", default="")
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--node")
    ap.add_argument("--address", default=os.environ.get("TRACEX_ADDRESS"))
    ap.add_argument("--yes", action="store_true", help="send without asking")
    a = ap.parse_args(argv)
    found = items(a.dir)
    if a.cmd == "list":
        for path, item in found:
            t = item["trace"]
            print(f"{os.path.basename(path)}  {t['task']}  {t['base_model']['name']}  "
                  f"{','.join(sorted(set((t.get('failure_modes') or {}).values()))) or '-'}  "
                  f"{(item.get('about') or {}).get('source', '')}")
        print(f"{len(found)} trace{'s' if len(found) != 1 else ''} in {a.dir}; nothing has been sent")
    elif a.cmd == "show":
        for path, item in found:
            if a.what in path or object_id(item["trace"]).startswith(a.what):
                print(json.dumps(item["trace"], indent=1, ensure_ascii=False))
    else:
        if not a.node:
            ap.error("send needs --node")
        if not a.yes and sys.stdin.isatty():
            if input(f"send {len(found)} trace(s) from {a.dir} to {a.node}? [y/N] ").strip().lower() != "y":
                return 1
        for r in send(a.dir, a.node, a.address):
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
