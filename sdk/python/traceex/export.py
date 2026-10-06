"""Turn a lot of traces into training data for open-weight models, in the formats trainers already use.

  to_sft(traces)   chat SFT rows:  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}], "trace": id}
  to_dpo(traces)   preference rows: {"prompt", "chosen", "rejected", "trace"}  (the model's answer is "rejected",
                   the verified fix is "chosen"; TRL's DPOTrainer and most other trainers read this shape)
  to_repair(traces) self-repair rows: the failing answer plus the checker's feedback in, the fix out
  refill(trace, seed)   skeleton traces become concrete again with *synthetic* values: the trainer gets as many
                   realistic pairs as it wants and never sees the person's data
  dataset_card(...) / model_card(...)  provenance a hub page can show: which traces, which producers, which checker,
                   which bounty, which eval, which licence
"""
import json
import random
import re

from .canon import object_id

PH = re.compile(r"\{([A-Z]+)_(\d+)\}")
_FIRST = ["Avery", "Jordan", "Riley", "Sam", "Casey", "Morgan", "Quinn", "Rowan", "Harper", "Emerson"]
_LAST = ["Lee", "Patel", "Garcia", "Okafor", "Nguyen", "Kim", "Silva", "Novak", "Haddad", "Larsen"]
_CITIES = ["Austin", "Denver", "Boston", "Seattle", "Chicago", "Atlanta", "Phoenix", "Portland"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _synth(kind, n, r):
    if kind == "NAME":
        return r.choice(_CITIES) if r.random() < .25 else f"{r.choice(_FIRST)} {r.choice(_LAST)}"
    if kind == "CODE":
        return "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3)) if r.random() < .5 else \
            "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    if kind == "NUM":
        return str(r.randint(10, 4999))
    if kind == "TIME":
        return f"{r.randint(1, 12)}:{r.choice(['05', '15', '20', '35', '40', '55'])} {r.choice(['AM', 'PM'])}"
    if kind == "DATE":
        return f"{r.choice(_MONTHS)} {r.randint(1, 28)}, {r.randint(2026, 2028)}"
    if kind == "MONEY":
        return f"${r.randint(20, 2999):,}.{r.randint(0, 99):02d}"
    if kind == "EMAIL":
        return f"{r.choice(_FIRST).lower()}@example.com"
    if kind == "PHONE":
        return f"(555) {r.randint(200, 999)}-{r.randint(1000, 9999)}"
    if kind == "URL":
        return f"https://example.com/{r.randint(1000, 9999)}"
    return f"value{n}"


def refill(trace, seed=0):
    """A concrete copy of a skeleton trace with synthetic values (the same value for the same placeholder everywhere).
    Open traces come back unchanged."""
    if trace.get("privacy") != "skeleton":
        return dict(trace)
    r, values = random.Random(f"{object_id(dict(trace))}:{seed}"), {}

    def fill(s):
        def one(m):
            key = m.group(0)
            if key not in values:
                values[key] = _synth(m.group(1), m.group(2), r)
            return values[key]
        return PH.sub(one, s) if isinstance(s, str) else s
    out = dict(trace)
    out["input"] = fill(trace["input"])
    out["model_output"] = {k: fill(v) for k, v in trace["model_output"].items()}
    out["verified_output"] = {k: fill(v) for k, v in trace["verified_output"].items()}
    if "feedback" in trace:
        out["feedback"] = [fill(f) for f in trace["feedback"]]
    return out


def _answer(output):
    """How an output is written as an assistant turn: code as a python block, structured fields as JSON."""
    if set(output) == {"code"}:
        return "```python\n" + str(output["code"]).strip() + "\n```"
    return json.dumps(output, ensure_ascii=False, sort_keys=True)


def _concrete(traces, refills):
    for t in traces:
        tid = t.id if hasattr(t, "id") else object_id(dict(t))
        copies = range(refills) if t.get("privacy") == "skeleton" else range(1)
        for s in copies:
            yield tid, refill(t, s)


def to_sft(traces, refills=1):
    return [{"messages": [{"role": "user", "content": c["input"]},
                          {"role": "assistant", "content": _answer(c["verified_output"])}], "trace": tid}
            for tid, c in _concrete(traces, refills)]


def to_dpo(traces, refills=1):
    rows = []
    for tid, c in _concrete(traces, refills):
        rejected = {k: c["model_output"].get(k, "") for k in c["verified_output"]}
        rows.append({"prompt": c["input"], "chosen": _answer(c["verified_output"]), "rejected": _answer(rejected),
                     "trace": tid})
    return rows


def to_repair(traces, refills=1):
    """Self-repair turns: the model's failing answer and the checker's feedback in, the verified fix out."""
    rows = []
    for tid, c in _concrete(traces, refills):
        if not c.get("feedback"):
            continue
        rejected = {k: c["model_output"].get(k, "") for k in c["verified_output"]}
        user = (c["input"] + "\n\nYour previous attempt:\n" + _answer(rejected) + "\nfailed the checks:\n"
                + c["feedback"][0] + "\n\nFix it.")
        rows.append({"messages": [{"role": "user", "content": user},
                                  {"role": "assistant", "content": _answer(c["verified_output"])}], "trace": tid})
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def dataset_card(traces, *, name, license="CC-BY-4.0", source_note=""):
    """A Hugging Face style README for an exported lot, crediting every producer and checker."""
    producers, checkers, models, modes = {}, set(), set(), {}
    for t in traces:
        producers[t["producer"]] = producers.get(t["producer"], 0) + 1
        checkers.add(f"{t['checker']['id']}@{t['checker']['version']}")
        models.add(t["base_model"]["name"])
        for m in (t.get("failure_modes") or {}).values():
            modes[m] = modes.get(m, 0) + 1
    lines = [f"---\nlicense: {license.lower()}\ntags: [trace-exchange, verified-fixes]\n---", f"# {name}", "",
             f"{len(traces)} verified fixes from traceX: each row is a real failure of "
             f"{', '.join(sorted(models))} and the answer a checker ({', '.join(sorted(checkers))}) proved right.", ""]
    if source_note:
        lines += [source_note, ""]
    lines += ["| producer | traces |", "|---|---|"] + [f"| `{p}` | {n} |" for p, n in sorted(producers.items())]
    if modes:
        lines += ["", "Failure modes: " + ", ".join(f"{m} {n}" for m, n in sorted(modes.items(), key=lambda kv: -kv[1]))]
    lines += ["", "Formats: `sft.jsonl` (chat messages), `dpo.jsonl` (prompt / chosen / rejected), `repair.jsonl` "
              "(failing answer + feedback in, fix out). Every row carries its trace id; royalties and credit follow it."]
    return "\n".join(lines) + "\n"


def model_card(learning, *, title, base_model_license, eval_name, bounty=None, producers=None, extra=""):
    """A model card for a released learning: what it improves, how it was measured, and who it came from."""
    a = learning["attestation"]
    lines = [f"---\nbase_model: {learning['base_model']['name']}\nlibrary_name: peft\nlicense: {base_model_license}\n"
             f"tags: [trace-exchange, lora]\n---", f"# {title}", "",
             f"A `{learning['kind']}` learning for **{learning['base_model']['name']}**, built from "
             f"{len(learning['parents'])} verified fixes on traceX.", "",
             f"**Measured by the validator** on {eval_name} (never trained on): {a['metric']} "
             f"{a['before']:.1%} -> {a['after']:.1%}.", ""]
    if bounty:
        lines += [f"Funded by bounty #{bounty['id']} ({bounty.get('title', '')}): its backers' pledges paid the "
                  "solver and the traces it was built from; metered hosted use pays the same family tree.", ""]
    if producers:
        lines += ["Built from fixes contributed by:", ""] + [f"- `{p}`: {n} traces" for p, n in sorted(producers.items())] + [""]
    lines += [f"Learning id `{object_id(dict(learning))}`; artifact `{learning['artifact'].get('hash', '')}`. "
              "The family tree (learning -> traces -> producers) is on the exchange."]
    if extra:
        lines += ["", extra]
    return "\n".join(lines) + "\n"
