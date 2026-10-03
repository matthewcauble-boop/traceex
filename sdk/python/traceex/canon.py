"""Canonical JSON and content ids: the same object always has the same id, on any machine."""
import hashlib
import json


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def object_id(obj) -> str:
    return "sha256:" + hashlib.sha256(canonical(obj)).hexdigest()
