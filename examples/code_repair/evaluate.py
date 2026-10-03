"""The validator: greedy pass@1 on the held-out MBPP test split (500 problems nobody trains on), base vs base+LoRA.

    python examples/code_repair/evaluate.py base                 # score the base model
    python examples/code_repair/evaluate.py runs/lora-v1         # score base + adapter, compare with base

The eval set is named only by its hash in attestations. Results go to runs/eval-<name>.json with every task's result,
so the comparison is paired: which problems the learning newly solves, which it breaks, and an exact sign test.
"""
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "sdk", "python")]
import checker  # noqa: E402
import tasks  # noqa: E402
from lm import LM  # noqa: E402

SPLIT = "test"


def eval_set_hash(split=SPLIT):
    items = [[t["task_id"], t["test_list"]] for t in tasks.load(split)]
    return "sha256:" + hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def score(adapter=None, split=SPLIT, live=None, record="eval", log=print):
    ts = tasks.load(split)
    lm = LM(adapter=adapter, live=live, record=record)
    t0 = time.time()
    outs = lm.generate([tasks.prompt(t) for t in ts])
    rs = checker.check_many([(t, o[0]) for t, o in zip(ts, outs)])
    per = {str(t["task_id"]): {"passed": r["passed"], "mode": r["mode"]} for t, r in zip(ts, rs)}
    k = sum(r["passed"] for r in rs)
    log(f"  {'base' if not adapter else os.path.basename(adapter)}: {k}/{len(ts)} = {k / len(ts):.1%} "
        f"({lm.generated} generated, {time.time() - t0:.0f}s)")
    return {"eval_set": eval_set_hash(split), "split": split, "n": len(ts), "passed": k, "rate": k / len(ts),
            "adapter": lm.adapter, "per_task": per}


def score_with_repair(adapter=None, split=SPLIT, live=None, record="eval", log=print):
    """The agent workflow: a first try, then one repair round with the checker's feedback (greedy both times).
    Returns per-task results for both stages, so first-try and after-feedback rates are paired across models."""
    ts = tasks.load(split)
    lm = LM(adapter=adapter, live=live, record=record)
    first = lm.generate([tasks.prompt(t) for t in ts])
    r0 = checker.check_many([(t, o[0]) for t, o in zip(ts, first)])
    fails = [(t, r) for t, r in zip(ts, r0) if not r["passed"]]
    rep = lm.generate([tasks.repair_prompt(t, r["code"] or "(no code)", r["feedback"]) for t, r in fails])
    r1 = checker.check_many([(t, o[0]) for (t, _), o in zip(fails, rep)])
    fixed = {t["task_id"] for (t, _), r in zip(fails, r1) if r["passed"]}
    per = {str(t["task_id"]): {"passed": r["passed"] or t["task_id"] in fixed, "first": r["passed"]} for t, r in zip(ts, r0)}
    k0, k1 = sum(r["passed"] for r in r0), sum(v["passed"] for v in per.values())
    log(f"  {'base' if not adapter else os.path.basename(adapter)}: first try {k0}/{len(ts)} = {k0 / len(ts):.1%}; "
        f"after one round of feedback {k1}/{len(ts)} = {k1 / len(ts):.1%} ({len(fixed)} of {len(fails)} failures fixed)")
    return {"eval_set": eval_set_hash(split), "split": split, "n": len(ts), "passed": k1, "rate": k1 / len(ts),
            "first_rate": k0 / len(ts), "repaired": len(fixed), "failed_first": len(fails), "adapter": lm.adapter,
            "per_task": per}


def sign_test(wins, losses):
    """Exact two-sided sign test on the discordant pairs (McNemar's exact test)."""
    n, k = wins + losses, min(wins, losses)
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(base, other):
    b, o = base["per_task"], other["per_task"]
    wins = sorted(int(t) for t in b if o[t]["passed"] and not b[t]["passed"])
    losses = sorted(int(t) for t in b if b[t]["passed"] and not o[t]["passed"])
    return {"before": base["rate"], "after": other["rate"], "delta": other["rate"] - base["rate"],
            "newly_solved": len(wins), "newly_broken": len(losses), "p_value": sign_test(len(wins), len(losses)),
            "newly_solved_ids": wins, "newly_broken_ids": losses}


if __name__ == "__main__" and "--repair" in sys.argv:
    # second metric (added after the pre-registered greedy pass@1): first try + one round of checker feedback
    out = {}
    for which in ["base", "runs/lora-v1", "runs/lora-v2"]:
        path = None if which == "base" else os.path.join(HERE, which)
        name = "base" if which == "base" else os.path.basename(path)
        out[name] = score_with_repair(path, record="eval-" + name)
        if name != "base":
            out[name]["vs_base"] = compare(out["base"], out[name])
            c = out[name]["vs_base"]
            print(f"  {name} vs base after feedback: {c['before']:.1%} -> {c['after']:.1%} ({c['delta']:+.1%}); newly solved "
                  f"{c['newly_solved']}, newly broken {c['newly_broken']}, sign test p = {c['p_value']:.3g}")
    with open(os.path.join(HERE, "runs", "eval-repair.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0)
    sys.exit(0)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    if which == "base":
        res = score(None, record="eval-base")
        name = "base"
    else:
        path = which if os.path.isabs(which) else os.path.join(HERE, which)
        res = score(path, record="eval-" + os.path.basename(path))
        name = os.path.basename(path)
        base_path = os.path.join(HERE, "runs", "eval-base.json")
        if os.path.exists(base_path):
            with open(base_path, encoding="utf-8") as f:
                res["vs_base"] = compare(json.load(f), res)
            c = res["vs_base"]
            print(f"  vs base: {c['before']:.1%} -> {c['after']:.1%} ({c['delta']:+.1%}); newly solved {c['newly_solved']}, "
                  f"newly broken {c['newly_broken']}, sign test p = {c['p_value']:.3g}")
    with open(os.path.join(HERE, "runs", f"eval-{name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=0)
