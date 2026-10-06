"""The pre-registered analysis (PREREGISTRATION.md) of a run.py output. No GPU, no model: the graphs are in the file.

    python analyze.py results/run_r0.json [results/run_r1.json ...] --out results/summary.json
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sdk", "python"))
import countdown as cd  # noqa: E402
from traceex import tropic as T  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mcnemar(b, c):
    """Exact two-sided McNemar: b, c discordant counts."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(x, y, reps=10_000, seed=0):
    rng = random.Random(seed)
    n = len(x)
    diffs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(x[i] - y[i] for i in idx) / n)
    diffs.sort()
    return [round(diffs[int(0.025 * reps)], 4), round(diffs[int(0.975 * reps) - 1], 4)]


def reachable_solution(g, allowed):
    """Is there a complete root -> passing path using only `allowed` edges? Exhaustive (the graph is a DAG)."""
    out = defaultdict(list)
    for k in allowed:
        out[g.edges[k]["source"]].append(k)
    seen, stack = set(), [g.root]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        for k in out[s]:
            if g.edges[k]["success"]:
                return True
            stack.append(g.edges[k]["target"])
    return False


def agent_edges(g, agent):
    return [k for k, e in g.edges.items() if any(p == agent for _, p in e["owners"])]


def min_cover(g, path, agents):
    need = [{p for _, p in g.edges[k]["owners"]} for k in path]
    for r in range(1, len(agents) + 1):
        from itertools import combinations
        for combo in combinations(agents, r):
            if all(n & set(combo) for n in need):
                return r
    return None


def posthoc_graph(pid, problem, iso_graphs, attempts):
    """Every isolated-condition fragment of a problem in one graph, filed in the order they were made."""
    g = T.StepGraph(cd.root(problem), pid, top_l=4, max_edges=4096, max_solutions=64)
    for a in attempts:
        if a["problem"] != pid or a["id"].startswith("poo"):
            continue
        src = iso_graphs[a["agent"]]
        start = None
        if a["prefix"]:
            start = src.nodes[src.edges[a["prefix"][-1]]["target"]]["state"]
        g.add_fragment(a["id"], a["agent"], a["steps"], start_state=start, start_depth=a["start_depth"],
                       prefix=a["prefix"])
    scores = {}
    for src in iso_graphs.values():
        scores.update({k: e["log_prob"] for k, e in src.edges.items() if e["log_prob"] is not None})
    g.set_scores(scores)
    verify = lambda path: (T.replay(g, path, cd.step, cd.check)
                           and cd.expression_check(problem, [g.edges[k]["action"] for k in path]))
    T.admit(g, verify)
    for _ in range(4):                      # compose until nothing new (each refresh adds at most 64 candidates)
        if not T.compose(g, verify, limit=64):
            break
    return g


def analyze(path):
    d = json.load(open(path, encoding="utf-8"))
    probs, agents = d["problems"], list(d["config"]["agents"])
    G = {cond: {k: T.StepGraph.from_dict(v).reindex() for k, v in gs.items()} for cond, gs in d["graphs"].items()}
    by_problem = defaultdict(list)
    for a in d["attempts"]:
        by_problem[a["problem"]].append(a)
    rows = {}
    for pid, prob in probs.items():
        iso_graphs = {a: G["isolated"][f"{pid}|{a}"] for a in agents}
        pg = G["pooled"][f"{pid}|*"]
        w0 = [a for a in by_problem[pid] if a["id"].startswith("w0")]
        r = {"size": len(prob["nums"]),
             "direct": any(a["end"] == "pass" for a in w0),
             "direct_by_agent": {ag: any(a["end"] == "pass" for a in w0 if a["agent"] == ag) for ag in agents},
             "iso": any(g.solutions for g in iso_graphs.values()),
             "iso_by_agent": {ag: bool(g.solutions) for ag, g in iso_graphs.items()},
             "iso_exhaustive": any(reachable_solution(g, list(g.edges)) for g in iso_graphs.values()),
             "pooled": bool(pg.solutions),
             "pooled_exhaustive": reachable_solution(pg, list(pg.edges))}
        single = {ag: reachable_solution(pg, agent_edges(pg, ag)) for ag in agents}
        r["pooled_single_agent_capable"] = single
        r["cross_only"] = r["pooled"] and not any(single.values())
        ph = posthoc_graph(pid, prob, iso_graphs, by_problem[pid])
        r["posthoc"] = bool(ph.solutions)
        for name, g in (("pooled", pg), ("posthoc", ph)):
            best = g.best_path()
            if best:
                r[f"{name}_best_mixed"] = len(set(g.producers(best))) > 1
                r[f"{name}_best_cover"] = min_cover(g, best, agents)
        if r["cross_only"]:
            best = pg.best_path()
            src = sorted(set(T.step_sources(pg, best)))          # provenance: who first filed each step
            r["cross"] = {"path": [pg.edges[k]["action"] for k in best], "steps": len(best),
                          "agents_needed": min_cover(pg, best, agents),
                          "credited_fragments": len(src),
                          "credited_failed_fragments": sum(1 for f in src if pg.fragments[f]["outcome"] == "fail"),
                          "step_sources_by_agent": dict(Counter(pg.fragments[f]["producer"] for f in T.step_sources(pg, best))),
                          "paid_by_agent": T.producer_credit(pg, best),   # payment: passing traces, per whole trace
                          "origins": sorted({o for p, o in pg.solutions.items()})}
        rows[pid] = r
    first = {}
    for ev in d["events"]:
        if ev["cond"] == "pooled" and ev["problem"] not in first:
            first[ev["problem"]] = ev
    return d, rows, first


def summarize(d, rows, first):
    n = len(rows)
    ids = sorted(rows)
    col = lambda k: [int(bool(rows[i][k])) for i in ids]
    iso, pooled, posthoc, direct = col("iso"), col("pooled"), col("posthoc"), col("direct")
    b = sum(1 for x, y in zip(pooled, iso) if x and not y)
    c = sum(1 for x, y in zip(pooled, iso) if y and not x)
    cross = [i for i in ids if rows[i]["cross_only"]]
    rate = lambda xs: {"k": sum(xs), "n": len(xs), "rate": round(sum(xs) / len(xs), 4), "ci95": wilson(sum(xs), len(xs))}
    out = {
        "problems": n,
        "solved": {"direct (one attempt, any agent, wave 0)": rate(direct), "isolated (any agent's own graph)": rate(iso),
                   "posthoc (isolated steps pooled afterwards)": rate(posthoc), "pooled (shared graph)": rate(pooled)},
        "solved_by_agent": {
            "direct": {a: sum(rows[i]["direct_by_agent"][a] for i in ids) for a in d["config"]["agents"]},
            "isolated": {a: sum(rows[i]["iso_by_agent"][a] for i in ids) for a in d["config"]["agents"]}},
        "primary": {"pooled_minus_isolated": round((sum(pooled) - sum(iso)) / n, 4),
                    "ci95_bootstrap": paired_bootstrap(pooled, iso), "pooled_only": b, "isolated_only": c,
                    "mcnemar_p": round(mcnemar(b, c), 6)},
        "posthoc_minus_isolated": sum(posthoc) - sum(iso),
        "exhaustive_checks": {"iso_exhaustive_equals_iso": all(rows[i]["iso_exhaustive"] == rows[i]["iso"] for i in ids),
                              "pooled_exhaustive_minus_pooled": sum(rows[i]["pooled_exhaustive"] for i in ids) - sum(pooled)},
        "cross_only": {"k": len(cross), "share_of_problems": rate([int(i in cross) for i in ids]),
                       "share_of_pooled_solved": wilson(len(cross), max(sum(pooled), 1)),
                       "agents_needed": dict(Counter(rows[i]["cross"]["agents_needed"] for i in cross)),
                       "step_source_fragments": dict(Counter(rows[i]["cross"]["credited_fragments"] for i in cross)),
                       "using_a_failed_attempts_steps": sum(rows[i]["cross"]["credited_failed_fragments"] > 0 for i in cross),
                       "failed_step_source_fragments_total": sum(rows[i]["cross"]["credited_failed_fragments"] for i in cross),
                       "step_source_fragments_total": sum(rows[i]["cross"]["credited_fragments"] for i in cross),
                       "paid_passing_traces": dict(Counter(len(rows[i]["cross"]["paid_by_agent"]) for i in cross))},
        "best_path_mixes_agents": {
            name: {"k": sum(1 for i in ids if rows[i].get(f"{name}_best_mixed")),
                   "of_solved": sum(1 for i in ids if f"{name}_best_mixed" in rows[i])} for name in ("pooled", "posthoc")},
        "pooled_first_solve_origin": dict(Counter(e["origin"] for e in first.values())),
        "pooled_first_solve_mixed_agents": sum(1 for e in first.values() if len(set(e["producers"])) > 1),
        "by_size": {s: {k: sum(rows[i][k] for i in ids if rows[i]["size"] == s)
                        for k in ("direct", "iso", "posthoc", "pooled", "cross_only")} | {"n": sum(1 for i in ids if rows[i]["size"] == s)}
                    for s in sorted({r["size"] for r in rows.values()})},
        "compute": d["compute"],
        "examples": {i: rows[i]["cross"] for i in cross[:5]},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    summary = {}
    for p in args.runs:
        d, rows, first = analyze(p)
        name = os.path.splitext(os.path.basename(p))[0]
        summary[name] = summarize(d, rows, first)
        print(json.dumps({k: v for k, v in summary[name].items() if k != "examples"}, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
