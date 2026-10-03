"""Trace and Learning objects (spec sections 2)."""
import datetime as dt
from .canon import object_id
from .skeleton import skeletonize


def _model(base_model):
    return base_model if isinstance(base_model, dict) else {"name": str(base_model), "hash": None}


def _checker(checker):
    if isinstance(checker, dict):
        return checker
    cid, _, ver = str(checker).partition("@")
    return {"id": cid, "version": ver or "1", "hash": None}


PRIVACY = ("skeleton", "open")
_SEP = "\n␞\n"          # joins input and feedback so one skeleton pass gives them consistent placeholders


class Trace(dict):
    @classmethod
    def from_fix(cls, *, task, base_model, input, model_output, verified_output, checker, producer,
                 private_terms=(), license=None, created=None, fixed_by=None, privacy="skeleton", feedback=None,
                 failure_modes=None):
        """Build a trace from one verified fix.

        privacy="skeleton" (default, for anything personal): every concrete value is replaced by a typed placeholder on
            this machine; raw values never enter the returned object.
        privacy="open" (non-personal domains: code, maths, public documents): the full text is kept, because that is
            what a model needs to learn from. Submission still refuses secrets, keys, emails and phone numbers.
        fixed_by: optional {field: "model" | "rule" | "human"}, how each fix was reached (default "unknown").
        feedback: optional list of checker messages that drove the fix (tracebacks, failed asserts): training signal for
            self-repair. failure_modes: optional {field: mode} from the checker; overrides placeholder inference."""
        if privacy not in PRIVACY:
            raise ValueError(f"privacy must be one of {PRIVACY}")
        model_output = model_output or {}
        fixed = sorted(k for k in verified_output if str(verified_output.get(k)) != str(model_output.get(k)))
        feedback = [str(f) for f in (feedback or [])]
        if privacy == "skeleton":
            joined = _SEP.join([input] + feedback)
            skel, (m_out, v_out), slots = skeletonize(joined, model_output, verified_output, private_terms=private_terms)
            skel, *feedback = skel.split(_SEP)
        else:
            skel, m_out, v_out, slots = input, dict(model_output), dict(verified_output), {}
        t = cls({
            "v": "trace/0.1", "task": task, "base_model": _model(base_model),
            "input": skel, "model_output": m_out, "verified_output": v_out, "fixed_fields": fixed,
            "fixed_by": {f: (fixed_by or {}).get(f, "unknown") for f in fixed}, "slots": slots,
            "checker": _checker(checker), "privacy": privacy,
            "license": license or {"kind": "shared", "max_licensees": 10, "exclusive_days": 0},
            "producer": producer, "created": created or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        if feedback:                    # optional keys only when present, so older traces keep their ids
            t["feedback"] = feedback
        if failure_modes:
            t["failure_modes"] = {f: failure_modes[f] for f in fixed if f in failure_modes}
        return t

    def text(self):
        """Everything that leaves the device, for the pre-send scans."""
        return "\n".join([self["input"], *map(str, self["model_output"].values()),
                          *map(str, self["verified_output"].values()), *self.get("feedback", [])])

    @property
    def id(self):
        return object_id(dict(self))

    @property
    def lot(self):
        return f"{self['task']}|{self['base_model']['name']}|{self['checker']['id']}@{self['checker']['version']}"


class Learning(dict):
    DEFAULT_SPLIT = {"traces": 0.60, "trainer": 0.25, "checkers": 0.10, "validators": 0.05}

    KINDS = ("routing", "rule", "prompt_patch", "decoding", "lora", "full_finetune", "checker", "package")

    @classmethod
    def build(cls, *, kind, task, base_model, artifact, parents, trainer, attestation, per_call_micros, split=None,
              release=None):
        """release: None (licensed: buyers get the artifact, every metered use pays royalties) or "open" (the artifact
        is published for anyone, e.g. open weights; it can't be metered once public, so it is funded up front by a
        bounty and earns only from uses that are metered, such as hosted inference)."""
        if kind not in cls.KINDS:
            raise ValueError(f"kind must be one of {cls.KINDS}")
        total = sum(w for _, w in parents) or 1.0
        L = cls({
            "v": "learning/0.1", "kind": kind, "task": task, "base_model": _model(base_model), "artifact": artifact,
            "parents": [{"trace": t, "weight": round(w / total, 9)} for t, w in parents],
            "trainer": trainer, "attestation": attestation,
            "royalty": {"per_call_micros": per_call_micros, "split": split or dict(cls.DEFAULT_SPLIT)},
        })
        if release:
            L["release"] = release
        return L

    @property
    def id(self):
        return object_id(dict(self))
