"""Adaptation: turn traces into a learning, prove it helps on held-out data, adopt it. This is the loop the exchange
exists to pay for: act -> check -> fix -> trace -> learning -> attested improvement -> every agent adopts it.

v0.1 ships one learning kind that needs no GPU and works with any model, 'routing': from the fields a model keeps
getting wrong on a task, learn which fields to extract on their own with a narrowed context *before* checking. Weight
deltas (kind 'lora') use the same object, attestation and royalty path; only the artifact and the apply step differ.
"""
import hashlib
from collections import Counter

from .canon import canonical, object_id


def routing_from_traces(traces, *, threshold=0.5, min_traces=2):
    """Fields the model fixed *itself* once narrowed, in at least `threshold` of the traces, become 'focus' fields.
    Fields only the checker's rules could fix are left out: narrowing didn't help them, and routing them can hurt.
    Returns (artifact, parents) where parents weights each trace by how many focus fields it taught."""
    if len(traces) < min_traces:
        raise ValueError(f"need at least {min_traces} traces, have {len(traces)}")
    counts = Counter(f for t in traces for f in t["fixed_fields"]
                     if t.get("fixed_by", {}).get(f, "model") in ("model", "unknown"))
    rates = {f: round(n / len(traces), 3) for f, n in sorted(counts.items())}
    focus = sorted(f for f, r in rates.items() if r >= threshold)
    body = {"kind": "routing", "task": traces[0]["task"], "focus": focus, "rates": rates, "n": len(traces)}
    artifact = {"uri": "inline", "hash": object_id(body), "body": body}
    parents = [(t.id if hasattr(t, "id") else object_id(dict(t)), len(set(t["fixed_fields"]) & set(focus)))
               for t in traces]
    return artifact, [(p, w) for p, w in parents if w > 0]


def apply_routing(extract, artifact, narrow):
    """Wrap extract(text, fields) so focus fields are re-extracted alone, on their narrowed context, in the first pass."""
    focus = artifact["body"]["focus"]

    def adapted(text, fields):
        r = dict(extract(text, fields) or {})
        if len(fields) > 1:
            for f in [f for f in fields if f in focus]:
                got = extract(narrow(text, f, r), [f]) or {}
                if got.get(f) not in (None, ""):
                    r[f] = got[f]
        return r
    return adapted


def first_pass_score(docs, extract, check, *, fields, clean=lambda r: r):
    """The metric a validator attests: share of fields verified by the checker on the first pass, no retries."""
    ok = total = 0
    per_doc = {}
    for name, text in docs.items():
        r = clean(extract(text, list(fields)) or {})
        bad = check(text, r)
        per_doc[name] = sorted(bad)
        ok += len(fields) - len(bad)
        total += len(fields)
    return round(ok / total, 4), per_doc


def attest(validator, eval_docs, metric, before, after):
    """A validator's statement that a learning improved a hidden eval. The eval set is committed by hash only.
    'sig' here is a digest; a live network replaces it with the validator's EIP-191 signature over the same bytes."""
    eval_set = eval_docs if isinstance(eval_docs, str) else         "sha256:" + hashlib.sha256(canonical(sorted(eval_docs.items()))).hexdigest()    # docs, or a hash already taken
    a = {"validator": validator, "eval_set": eval_set, "metric": metric, "before": before, "after": after}
    a["sig"] = "digest:" + hashlib.sha256(canonical(a)).hexdigest()
    return a


class AdaptiveAgent:
    """An agent that improves itself from its own verified fixes, and from learnings it buys.

        agent = AdaptiveAgent(extract, check, fields=FIELDS, clean=clean, narrow=context_for, evidence=evidence, ...)
        agent.run(email)          # QA loop; every fix becomes a skeleton trace in agent.traces
        agent.adopt(learning)     # apply a learning (its own, or one bought on the exchange)

    With autopilot=Autopilot(...), every run also uses the exchange on its own: fixes are submitted, unresolved
    failures trigger a search for a proven learning (adopted automatically when this agent can apply it), and
    recurring failures nobody has fixed become bounties.
    """

    def __init__(self, base_extract, check, *, fields, task, base_model, checker, producer,
                 clean=lambda r: r, narrow=None, evidence=None, autopilot=None):
        from .loop import extract_checked
        from .trace import Trace
        self._loop, self._Trace = extract_checked, Trace
        self.base_extract, self.extract = base_extract, base_extract
        self.check, self.fields, self.clean, self.narrow, self.evidence = check, fields, clean, narrow, evidence
        self.task, self.base_model, self.checker, self.producer = task, base_model, checker, producer
        self.traces, self.adopted, self.autopilot = [], [], autopilot

    def run(self, text, *, created=None):
        out = self._loop(text, self.extract, self.check, fields=list(self.fields), clean=self.clean,
                         narrow=self.narrow, evidence=self.evidence)
        if not out["failing"] and out["first"] != out["result"]:
            t = self._Trace.from_fix(task=self.task, base_model=self.base_model, input=text, model_output=out["first"],
                                     verified_output=out["result"], checker=self.checker, producer=self.producer,
                                     created=created, fixed_by={**{f: "model" for f in out["by_model"]},
                                                                **{f: "rule" for f in out["by_rule"]}})
            if t["fixed_fields"]:
                self.traces.append(t)
                out["trace"] = t
        if self.autopilot is not None:
            out["autopilot"] = self.autopilot.on_result(text, out)
            for act in out["autopilot"]:
                L = act.get("learning")
                fresh = L and L.get("id") not in {a.get("id") for a in self.adopted}
                if act["action"] == "adopt" and fresh and L["artifact"].get("body", {}).get("kind") == "routing":
                    act["adopted"] = self.helps(L, text)
                    if act["adopted"]:
                        self.adopt(L)
        return out

    def helps(self, learning, text):
        """Try a learning on this agent's own failing case before adopting it: its first pass has to get more fields
        right than the agent does now. Whatever validators attested, an agent only takes on (and pays for) what helps
        on its own traffic."""
        trial = apply_routing(self.base_extract, learning["artifact"], self.narrow)
        doc = {"case": text}
        now, _ = first_pass_score(doc, self.extract, self.check, fields=self.fields, clean=self.clean)
        then, _ = first_pass_score(doc, trial, self.check, fields=self.fields, clean=self.clean)
        return then > now

    def adopt(self, learning):
        art = learning["artifact"]
        if art["body"]["kind"] != "routing":
            raise NotImplementedError(f"this agent applies 'routing' learnings, not {art['body']['kind']}")
        self.extract = apply_routing(self.base_extract, art, self.narrow)
        self.adopted.append(learning)
