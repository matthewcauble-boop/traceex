"""Sample three agents on Countdown and keep every step, for the cross-agent composition test (see PREREGISTRATION.md).

    python run.py --per-size 100 --out results/run.json              # one RTX 3060: about half an hour
    python run.py --per-size 5 --seed 999 --out results/pilot.json   # a feasibility pilot on other problems

Three agents share one open model (Qwen2.5-0.5B-Instruct, inference only, Hugging Face transformers) and differ in
system prompt, temperature and seed. Wave 0: every agent makes `--n` attempts per problem from the start. Then two
conditions get the same further budget, `--waves` waves of `--n` attempts per agent per problem, each wave from one
frontier state per agent (TROPIC's restart rule):
  isolated  each agent keeps its own step graph: frontier, restarts and composition only over its own steps;
  pooled    the three agents share one step graph (traceX): frontier chosen from everyone's steps, composition over
            everyone's steps.
Wave 0 is shared by both conditions. After every wave each graph is rescored under one scoring model (the base model
with agent A's prompt, on a canonical completion), passing attempts are replayed and archived, and the L best prefixes
are joined with the L best suffixes, replayed and verified (TROPIC's composition). The output holds every graph, so
analyze.py needs no GPU.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sdk", "python"))
import countdown as cd  # noqa: E402
from traceex import tropic as T  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TROPIC_INSTRUCTION = (   # TROPIC's Countdown instruction (config/_4_countdown_tropic.yaml), verbatim
    "You are solving the Countdown puzzle one operation at a time. You see the target and the numbers that are "
    "currently available.\nYour answer must be exactly ONE operation between exactly TWO numbers taken from the current "
    "list, written as \"a + b\", \"a - b\", \"a * b\" or \"a / b\", nothing else.\nThe result replaces the two numbers; you "
    "then get the new list and answer again. Reach the target by the time every number has been used exactly once.\n"
    "Fractions and negative intermediate values are allowed; division by zero is not.\nInvalid answers END the episode: "
    "a full expression with several operators (e.g. \"15/5 * 17 + 7\"), a number that is not in the current list, "
    "decimals, or text.\nExample with numbers [3, 5, 2] and target 4: first response <think>2 + 5 = 7, then 7 - 3 = "
    "4.</think><answer>2 + 5</answer>; the list becomes [3, 7]; next response <answer>7 - 3</answer>.")
AGENTS = {
    "A": {"system": TROPIC_INSTRUCTION, "temperature": 1.0, "max_new": 96, "seed": 11},
    "B": {"system": "You play the Countdown numbers game one step at a time. Reply with exactly one operation on two "
                    "numbers from the current list, inside answer tags, for example <answer>7 - 3</answer>. Do not "
                    "explain.", "temperature": 0.7, "max_new": 24, "seed": 22},
    "C": {"system": "You are a careful arithmetic planner playing Countdown. Combine two of the current numbers with "
                    "+, -, * or / so that the remaining numbers can still reach the target. Think in one short "
                    "sentence, then give the single operation as <answer>a op b</answer>.",
          "temperature": 0.9, "max_new": 64, "seed": 33},
}
SCORER = "A"
CANONICAL = "<answer>{}</answer>"


class Model:
    def __init__(self, batch=256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.batch = torch, batch
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
        self.calls = self.tokens = 0

    def chat(self, system, user):
        return self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            add_generation_prompt=True, tokenize=False)

    def generate(self, prompts, temperature, max_new, seed):
        out = []
        for i in range(0, len(prompts), self.batch):
            chunk = prompts[i:i + self.batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to("cuda")
            self.torch.manual_seed(seed + i)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temperature,
                                          top_p=1.0, top_k=0, pad_token_id=self.tok.pad_token_id)
            gen = gen[:, enc.input_ids.shape[1]:]
            self.calls += len(chunk)
            self.tokens += int((gen != self.tok.pad_token_id).sum())
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
        return out

    def score(self, pairs):
        """log p(canonical completion | scorer prompt) for each (state, action), summed over completion tokens."""
        torch, out = self.torch, []
        sysmsg = AGENTS[SCORER]["system"]
        for i in range(0, len(pairs), 128):
            chunk = pairs[i:i + 128]
            prompts = [self.chat(sysmsg, cd.render(s)) for s, _ in chunk]
            comps = [CANONICAL.format(a) + "<|im_end|>" for _, a in chunk]
            p_ids = [self.tok(p, add_special_tokens=False).input_ids for p in prompts]
            c_ids = [self.tok(c, add_special_tokens=False).input_ids for c in comps]
            width = max(len(p) + len(c) for p, c in zip(p_ids, c_ids))
            ids = torch.full((len(chunk), width), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros_like(ids)
            for r, (p, c) in enumerate(zip(p_ids, c_ids)):
                seq = p + c
                ids[r, width - len(seq):] = torch.tensor(seq)
                att[r, width - len(seq):] = 1
            ids, att = ids.cuda(), att.cuda()
            keep = max(len(c) for c in c_ids) + 1            # completions sit at the right edge (left padding)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=att, logits_to_keep=keep).logits.float()
            logp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, width - keep + 1:, None])[..., 0]
            for r, c in enumerate(c_ids):
                out.append(float(logp[r, keep - 1 - len(c):].sum()))
        return out


def run_attempts(model, episodes, log, sample_seed=0):
    """Roll every episode forward until it fails, passes or uses every number, batching all live episodes of an agent
    per step. An episode: {agent, problem, state, depth, steps: [...], raw: [...]}."""
    live = [e for e in episodes]
    while live:
        for name, spec in AGENTS.items():
            mine = [e for e in live if e["agent"] == name]
            if not mine:
                continue
            prompts = [model.chat(spec["system"], cd.render(e["state"])) for e in mine]
            seed = spec["seed"] * 1_000_003 + model.calls + 7_919 * sample_seed
            replies = model.generate(prompts, spec["temperature"], spec["max_new"], seed)
            for e, reply in zip(mine, replies):
                action = cd.parse(reply)
                nxt = cd.step(e["state"], action) if action else None
                e["raw"].append(reply[:300])
                if nxt is None:
                    e["done"], e["end"] = True, "invalid"
                    continue
                passed = cd.check(nxt)
                e["steps"].append({"action": action, "state": nxt, "success": passed})
                e["state"], e["depth"] = nxt, e["depth"] + 1
                if passed or not cd.alive(nxt) or e["depth"] >= cd.MAX_STEPS:
                    e["done"], e["end"] = True, "pass" if passed else "dead_end"
        live = [e for e in live if not e.get("done")]
        log(f"  {len(live)} episodes still running, {model.calls} model calls so far")
    return episodes


def verifier(problem, graph):
    def verify(path):
        return (T.replay(graph, path, cd.step, cd.check)
                and cd.expression_check(problem, [graph.edges[k]["action"] for k in path]))
    return verify


def refresh(model, graphs, cache, problems, events, cond, wave):
    """Score new steps under the one scorer, archive verified passing attempts, then compose."""
    todo = {}
    for (pid, _), g in graphs.items():
        for k, e in g.edges.items():
            if e["log_prob"] is None:
                ck = (pid, e["source"], e["action"])
                if ck not in cache:
                    todo[ck] = (g.nodes[e["source"]]["state"], e["action"])
    if todo:
        keys = list(todo)
        for ck, lp in zip(keys, model.score([todo[k] for k in keys])):
            cache[ck] = lp
    for (pid, owner), g in graphs.items():
        g.set_scores({k: cache[(pid, e["source"], e["action"])] for k, e in g.edges.items() if e["log_prob"] is None})
        verify = verifier(problems[pid], g)
        for path, origin in T.admit(g, verify) + T.compose(g, verify, limit=64):
            events.append({"cond": cond, "graph": owner, "problem": pid, "wave": wave, "origin": origin,
                           "path": list(path), "producers": g.producers(path),
                           "credit": T.producer_credit(g, path, "steps")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n", type=int, default=6, help="attempts per agent per problem per wave")
    ap.add_argument("--waves", type=int, default=3, help="frontier waves after wave 0, per condition")
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--sample-seed", type=int, default=0, help="replicate: shifts every agent's sampling seed")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "run.json"))
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    probs = {f"p{i:04d}": p for i, p in enumerate(cd.problems(args.per_size, args.seed))}
    model = Model()
    agents = list(AGENTS)
    cache, events, attempts = {}, [], []
    new_graph = lambda pid: T.StepGraph(cd.root(probs[pid]), pid, top_l=4, max_edges=4096, max_solutions=64)
    graphs = {"isolated": {(pid, a): new_graph(pid) for pid in probs for a in agents},
              "pooled": {(pid, "*"): new_graph(pid) for pid in probs}}

    def file(cond, episodes, wave):
        for e in episodes:
            key = (e["problem"], e["agent"] if cond == "isolated" else "*")
            g = graphs[cond][key]
            g.add_fragment(e["id"], e["agent"], e["steps"], start_state=e["start"], start_depth=e["start_depth"],
                           prefix=e["prefix"])

    def episodes_for(cond, wave):
        eps = []
        for pid in probs:
            for a in agents:
                if wave == 0:
                    prefix, (state, depth) = (), (cd.root(probs[pid]), 0)
                else:
                    g = graphs[cond][(pid, a if cond == "isolated" else "*")]
                    prefix = g.choose_frontier(cd.MAX_STEPS, args.eta, alive=cd.alive)
                    state, depth = g.end_state(prefix)
                for i in range(args.n):
                    eps.append({"id": f"{'w0' if wave == 0 else cond[:3]}:{wave}:{pid}:{a}:{i}", "agent": a,
                                "problem": pid, "state": state, "depth": depth, "start": state if depth else None,
                                "start_depth": depth, "prefix": list(prefix), "steps": [], "raw": []})
        return eps

    log(f"{len(probs)} problems, agents {agents}, wave 0")
    w0 = run_attempts(model, episodes_for("isolated", 0), log, args.sample_seed)
    attempts += w0
    for cond in graphs:
        file(cond, w0, 0)
        refresh(model, graphs[cond], cache, probs, events, cond, 0)
    spent = {}
    for cond in graphs:
        before = model.calls
        for wave in range(1, args.waves + 1):
            log(f"{cond}: wave {wave}")
            eps = run_attempts(model, episodes_for(cond, wave), log, args.sample_seed)
            attempts += eps
            file(cond, eps, wave)
            refresh(model, graphs[cond], cache, probs, events, cond, wave)
        spent[cond] = model.calls - before
    out = {"config": {**vars(args), "model": MODEL, "agents": AGENTS, "scorer": SCORER, "top_l": 4,
                      "max_candidates": 64, "max_steps": cd.MAX_STEPS},
           "problems": probs, "events": events,
           "attempts": [{k: e[k] for k in ("id", "agent", "problem", "start_depth", "prefix", "steps", "raw", "end")}
                        for e in attempts],
           "graphs": {cond: {f"{pid}|{a}": g.to_dict() for (pid, a), g in gs.items()} for cond, gs in graphs.items()},
           "compute": {"model_calls": model.calls, "restart_calls_by_condition": spent, "generated_tokens": model.tokens, "scored_steps": len(cache),
                       "seconds": round(time.time() - t0, 1)}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    log(f"wrote {args.out}: {model.calls} model calls, {model.tokens} generated tokens")


if __name__ == "__main__":
    main()
