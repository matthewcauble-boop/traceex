"""Producers: three agents run an open-weight model on their share of the training problems. Every failure the checker
catches goes back to the model with its traceback (up to two repair rounds); if it still fails, the agent samples
8 more attempts. The first attempt that passes the tests is a verified fix, and becomes an `open` trace.

    python examples/code_repair/produce.py            # replays recorded generations; live on a GPU when missing

Writes runs/traces.jsonl (the traces) and runs/produce.json (what happened to every problem).
"""
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "sdk", "python")]
from traceex import Trace  # noqa: E402
import checker  # noqa: E402
import tasks  # noqa: E402
from lm import LM, BASE_MODEL  # noqa: E402

PRODUCERS = ["0x" + c * 40 for c in "abe"]          # three agents, three wallets
REPAIR_ROUNDS, SAMPLES, TEMPERATURE = 2, 8, 0.8
ALTERNATIVES = 4                                    # distinct passing samples kept per failure, each a verified fix
CREATED = "2026-10-03T00:00:00Z"


def produce(lm=None, train=None, log=print, base_model=BASE_MODEL):
    lm = lm or LM()
    train = train or tasks.load("train")
    t0 = time.time()
    state = {t["task_id"]: {"task": t, "attempts": [], "fix": None, "how": None} for t in train}

    def run(prompts_by_id, temperature=0.0, n=1):
        ids = list(prompts_by_id)
        outs = lm.generate([prompts_by_id[i] for i in ids], temperature=temperature, n=n)
        checks = checker.check_many([(state[i]["task"], o) for i, os_ in zip(ids, outs) for o in os_])
        res, k = {}, 0
        for i, os_ in zip(ids, outs):
            res[i] = checks[k:k + len(os_)]
            k += len(os_)
        return res

    # round 0: greedy on everything
    first = run({i: tasks.prompt(s["task"]) for i, s in state.items()})
    for i, (r,) in first.items():
        state[i]["first"] = r
        state[i]["attempts"].append(r)
        if r["passed"]:
            state[i]["how"] = "first_try"
    log(f"  round 0: {sum(s['how'] == 'first_try' for s in state.values())}/{len(state)} pass on the first try "
        f"({time.time() - t0:.0f}s)")

    # repair rounds: the failing code goes back with its traceback
    for rnd in range(1, REPAIR_ROUNDS + 1):
        todo = {i: s for i, s in state.items() if s["how"] is None}
        if not todo:
            break
        res = run({i: tasks.repair_prompt(s["task"], s["attempts"][-1]["code"] or "(no code)", s["attempts"][-1]["feedback"])
                   for i, s in todo.items()})
        for i, (r,) in res.items():
            state[i]["attempts"].append(r)
            if r["passed"]:
                state[i]["fix"], state[i]["how"] = r, f"repair_{rnd}"
        log(f"  repair round {rnd}: {sum(1 for i in todo if state[i]['how'] == f'repair_{rnd}')}/{len(todo)} fixed "
            f"from the traceback ({time.time() - t0:.0f}s)")

    # still failing: sample, keep the first attempt that passes
    todo = {i: s for i, s in state.items() if s["how"] is None}
    if todo:
        res = run({i: tasks.prompt(s["task"]) for i, s in todo.items()}, temperature=TEMPERATURE, n=SAMPLES)
        for i, rs in res.items():
            oks, seen = [], set()
            for r in rs:
                if r["passed"] and r["code"].strip() not in seen:
                    seen.add(r["code"].strip())
                    oks.append(r)
            if oks:
                state[i]["fix"], state[i]["how"], state[i]["alts"] = oks[0], "sampled", oks[1:ALTERNATIVES]
        log(f"  sampling {SAMPLES} at T={TEMPERATURE}: {sum(1 for i in todo if state[i]['how'] == 'sampled')}/{len(todo)} "
            f"fixed ({time.time() - t0:.0f}s)")

    state_dump = {i: {"how": s["how"], "attempts": [{k: a[k] for k in ("passed", "mode", "feedback", "code")}
                                                     for a in s["attempts"]],
                      "fixes": [f["code"] for f in ([s["fix"]] if s["fix"] else []) + s.get("alts", [])]}
                  for i, s in state.items()}
    traces, summary = [], []
    for n, (i, s) in enumerate(sorted(state.items())):
        producer = PRODUCERS[n % len(PRODUCERS)]
        summary.append({"task_id": i, "producer": producer, "how": s["how"] or "unsolved",
                        "first_mode": s["first"]["mode"]})
        if s["fix"] is None:
            continue
        fb = [a["feedback"] for a in s["attempts"] if not a["passed"]][:REPAIR_ROUNDS + 1]
        for k, fix in enumerate([s["fix"]] + s.get("alts", [])):      # each distinct passing answer is its own fix
            t = Trace.from_fix(task=tasks.TASK, base_model=base_model, input=tasks.prompt(s["task"]),
                               model_output={"code": s["first"]["code"]}, verified_output={"code": fix["code"]},
                               checker=tasks.CHECKER, producer=producer, created=CREATED, privacy="open",
                               fixed_by={"code": "model_repair" if s["how"].startswith("repair") else "model_sample"},
                               feedback=fb, failure_modes={"code": s["first"]["mode"]})
            t["source"] = {"dataset": "mbpp", "task_id": i, "split": "train", "alternative": k}
            traces.append(t)
    produce.last_state = state_dump                 # every attempt, for the audit trail
    return traces, summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="run base + this LoRA (the next round of the loop: the adopted model produces)")
    ap.add_argument("--tag", default="", help="suffix for runs/traces<tag>.jsonl and runs/produce<tag>.json")
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    if args.adapter:
        path = args.adapter if os.path.isabs(args.adapter) else os.path.join(HERE, args.adapter)
        lm = LM(adapter=path, record="produce-" + os.path.basename(path))
        traces, summary = produce(lm, base_model={"name": BASE_MODEL, "hash": lm.adapter})
    else:
        traces, summary = produce()
    with open(os.path.join(HERE, "runs", f"traces{args.tag}.jsonl"), "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, sort_keys=True) + "\n")
    with open(os.path.join(HERE, "runs", f"produce{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=0)
    with open(os.path.join(HERE, "runs", f"attempts{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(produce.last_state, f, indent=0)
    hows = Counter(s["how"] for s in summary)
    print(f"{len(traces)} verified fixes -> traces; outcomes {dict(hows)}")
    print("first-try failure modes:", dict(Counter(s["first_mode"] for s in summary if s["how"] != "first_try")))
