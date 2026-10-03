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


class Trace(dict):
    @classmethod
    def from_fix(cls, *, task, base_model, input, model_output, verified_output, checker, producer,
                 private_terms=(), license=None, created=None, fixed_by=None):
        """Build a skeleton trace from one verified fix. Raw values never enter the returned object.
        fixed_by: optional {field: "model" | "rule" | "human"}, how each fix was reached (default "unknown")."""
        fixed = sorted(k for k in verified_output if str(verified_output.get(k)) != str((model_output or {}).get(k)))
        skel, (m_out, v_out), slots = skeletonize(input, model_output or {}, verified_output, private_terms=private_terms)
        t = cls({
            "v": "trace/0.1", "task": task, "base_model": _model(base_model),
            "input": skel, "model_output": m_out, "verified_output": v_out, "fixed_fields": fixed, "fixed_by": {f: (fixed_by or {}).get(f, "unknown") for f in fixed}, "slots": slots,
            "checker": _checker(checker), "privacy": "skeleton",
            "license": license or {"kind": "shared", "max_licensees": 10, "exclusive_days": 0},
            "producer": producer, "created": created or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        return t

    @property
    def id(self):
        return object_id(dict(self))

    @property
    def lot(self):
        return f"{self['task']}|{self['base_model']['name']}|{self['checker']['id']}@{self['checker']['version']}"


class Learning(dict):
    DEFAULT_SPLIT = {"traces": 0.60, "trainer": 0.25, "checkers": 0.10, "validators": 0.05}

    @classmethod
    def build(cls, *, kind, task, base_model, artifact, parents, trainer, attestation, per_call_micros, split=None):
        total = sum(w for _, w in parents) or 1.0
        return cls({
            "v": "learning/0.1", "kind": kind, "task": task, "base_model": _model(base_model), "artifact": artifact,
            "parents": [{"trace": t, "weight": round(w / total, 9)} for t, w in parents],
            "trainer": trainer, "attestation": attestation,
            "royalty": {"per_call_micros": per_call_micros, "split": split or dict(cls.DEFAULT_SPLIT)},
        })

    @property
    def id(self):
        return object_id(dict(self))
