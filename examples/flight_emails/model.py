"""The base model: Cactus Needle (26M params, on-device) when installed, else its recorded answers.

Every call is keyed by sha256(text + fields). Live runs add to recorded.json, so the demo replays real Needle output
on any machine, with no model download. `pip install cactus-needle` and delete recorded.json to re-record.
"""
import hashlib
import json
import os
import warnings

from flight import schema

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "recorded.json")


class Model:
    name = "needle3"

    def __init__(self, live=None):
        self.cache = {}
        if os.path.exists(RECORD):
            with open(RECORD, encoding="utf-8") as f:
                self.cache = json.load(f)
        self.calls = 0
        self.needle = None
        if live is not False:
            try:
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                warnings.filterwarnings("ignore")
                import needle
                self.needle = needle
            except ImportError:
                if live:
                    raise

    @staticmethod
    def key(text, fields):
        return hashlib.sha256((text + "\x00" + ",".join(fields)).encode()).hexdigest()[:24]

    def __call__(self, text, fields):
        self.calls += 1
        k = self.key(text, fields)
        if k not in self.cache:
            if not self.needle:
                raise KeyError(f"no recorded answer for this call and Needle isn't installed ({k})")
            self.cache[k] = self.needle.extract(text, schema(fields), strict=False) or {}
            with open(RECORD, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=0, sort_keys=True)
        return dict(self.cache[k])
