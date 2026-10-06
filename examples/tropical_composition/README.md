# Does stitching fragments across agents solve problems no single agent solved?

A small, pre-registered test of the idea behind traceX's step traces (SPEC 4j), using the method of **TROPIC**
("Tropical Reinforcement Learning", Asadulaev, Djuhera, Salta, Boche, Karray, Takac, arXiv:2610.02478,
[code](https://github.com/machinestein/Tropical-Reinforcement-Learning), MIT; see [NOTICE](NOTICE)). Inference only:
nothing is trained here.

**Short answer.** Yes, in the narrow sense: in a step graph shared by three agents, 25 of 400 problems (6.3%, 95% CI
4.3-9.1%) were solved only by joining steps from two agents (33 in the replicate), and every one of those joins used a
step from an attempt that failed. No, in the sense that matters for a budget: at equal sampling cost the shared graph did not solve more
problems than the same three agents running the same loop alone (16.3% against 15.0%; +1.3 points, 95% CI -2.3 to
+4.8; McNemar p = 0.57). The pre-registered hypothesis (H1) is **not supported**. A second replicate agrees (+0.5
points, CI -3.3 to +4.5, p = 0.90; 33 cross-agent-only solves).

## Setup ([PREREGISTRATION.md](PREREGISTRATION.md), written before the run)

- **Task:** Countdown exactly as TROPIC generates it (its generator, ported in `countdown.py`; seed 2027 for the
  training split, excluding the seed-2026 validation split), 100 problems each with 3, 4, 5 and 6 numbers. One exact
  operation per step on the live numbers; a solution must replay step by step and pass an independent check (the
  actions folded into one expression over the original inputs, evaluated exactly).
- **Agents:** Qwen2.5-0.5B-Instruct three times (transformers, bf16, one RTX 3060): A with TROPIC's own Countdown
  instruction at temperature 1.0, B "answer only" at 0.7, C "one sentence, then answer" at 0.9. The prompt holds the
  current state only (TROPIC's single-turn mode). One scorer for every step: the base model with A's prompt on a
  canonical `<answer>a op b</answer>` completion.
- **Budget:** wave 0, 6 attempts per agent per problem from the start (shared); then 3 waves per condition of 6
  attempts per agent per problem, each from one frontier state per agent (TROPIC's rule). **isolated:** one step graph
  per agent, frontier and composition over its own steps only. **pooled:** one graph per problem for all three.
  Composition: the 4 best prefixes joined with the 4 best suffixes at every shared state, replayed and verified.
- Compute per replicate: about 59,000 generations (1.8M tokens), 29 minutes; the two conditions' restarts spent
  25,111 and 24,571 model calls (replicate 0).

## Results

Replicate 0 decides H1; replicate 1 (every agent's sampling seed shifted) is the pre-registered robustness check.

| problems solved (of 400) | replicate 0 | replicate 1 |
|---|---|---|
| in one attempt, any agent (wave 0) | 12 (3.0%) | 10 (2.5%) |
| isolated: some agent's own TROPIC loop | 60 (15.0%, CI 11.8-18.8) | 63 (15.8%, CI 12.5-19.6) |
| post-hoc: the isolated runs' steps pooled afterwards | 60 | 63 |
| **pooled: one shared step graph** | **65 (16.3%, CI 13.0-20.2)** | **65 (16.3%, CI 13.0-20.2)** |
| pooled minus isolated (paired, 95% bootstrap CI) | **+1.3 points (-2.3, +4.8)** | **+0.5 points (-3.3, +4.5)** |
| solved only pooled / only isolated (McNemar p) | 27 / 22 (0.57) | 32 / 30 (0.90) |
| **cross-agent only:** pooled, and no agent's own steps hold a solution | **25 (6.3%, CI 4.3-9.1)** | **33 (8.3%, CI 5.9-11.4)** |
| ...of the pooled solves | 38% (CI 28-51) | 51% (CI 39-63) |
| ...needing steps from 2 agents / 3 agents | 25 / 0 | 32 / 1 |
| ...whose best path uses a failed attempt's step | 25 of 25 (34 of 59 step sources) | 33 of 33 (47 of 80) |
| ...passing traces that path would pay (traceX's rule) | 1 in all 25 | 1 in all 33 |
| pooled problems whose best path mixes agents | 34 of 65 | 38 of 65 |
| post-hoc problems whose best path mixes agents | 7 of 60 | 6 of 63 |

By size (replicate 0; isolated / pooled / cross-agent only): 3 numbers 30 / 41 / 12; 4 numbers 20 / 18 / 10; 5
numbers 10 / 4 / 2; 6 numbers 0 / 2 / 1. Replicate 1: 38 / 41 / 17; 24 / 18 / 11; 1 / 4 / 4; 0 / 2 / 1.

Checks that came out as predicted: pooling the isolated runs' steps after the fact added **exactly zero** first solves
in both replicates (a restart starts at a state its own agent reached, so post-hoc joins can only add alternative
paths, here 6-7 problems got a better mixed path); an exhaustive search of every graph found no solution the
composition missed (isolated and pooled alike).

## What it shows, and what it doesn't

- **Cross-agent joins are real and they reuse failures.** A third to a half of what the shared graph solved, it
  solved only by putting two agents' steps together, and every such path is built partly from an attempt that failed
  by itself (in replicate 0, 34 of the 59 attempts these paths take steps from had failed). Under traceX's rule those
  failed attempts are unpaid reports: they made the join possible (a frontier to restart from) but earn nothing, and
  each such path pays exactly one passing trace, the restart that finished it. Examples: `16 + 20`, `8 + 4`, `36 + 12` (two steps from C, one from A);
  `20 - 13`, `9 + 19`, `20 + 28`, `48 + 7` (three from C, one from B).
- **But sharing did not raise the solve rate at this budget.** The shared graph solves *different* problems, not more:
  27 problems only pooled, 22 only isolated (replicate 0). By size it did better on 3-number problems in both
  replicates (41 against 30 and 38), worse on 4-number problems in both (18 against 20 and 24), and the 5-6 number
  problems are too few to read (6 against 10, then 6 against 1). A plausible reading, not tested here: with one
  frontier, the three agents' restarts go to the same graph's least-explored states, so the shared graph explores
  one region per problem more deeply where three separate graphs would explore three regions.
- **Joins come from restarts, not from offline stitching.** Every pooled first solve came from an attempt from the
  start (12) or a restart from a frontier state (53); TROPIC's prefix-suffix joins added 9 further verified paths and
  no first solve. The cross-agent solves are agent X restarting from a state agent Y reached (and filing it), which is
  exactly what the node's `/frontier` endpoint serves. The post-hoc condition shows that pooling fragments that were
  collected independently is worth nothing for first solves; the value, where there is any, is in sharing *where to
  restart*.
- **Limits.** One small model (most attempts end on an illegal first move: 93% of all attempts), one task, one budget,
  agents that differ only in prompt and temperature, two replicates whose solved sets overlap only partly (26 of 65
  pooled problems in common), so per-problem outcomes are noisy and a 2-3 point effect is below what 400 problems can
  detect. Nothing was trained, so this says nothing about TROPIC's training gains. The lenient answer parser and the
  fixed scorer differ from TROPIC (see NOTICE).

## Run it

```bash
python run.py --per-size 100 --sample-seed 0 --out results/run_r0.json    # about 30 minutes on an RTX 3060
python run.py --per-size 100 --sample-seed 1 --out results/run_r1.json
python analyze.py results/run_r0.json results/run_r1.json --out results/summary.json   # no GPU
```

`results/summary.json` holds every number above. The raw runs (every attempt, reply and step graph, 34 MB each) are
not in the repository. `countdown.py` (generator, environment, verifiers), `run.py` (sampling), `analyze.py`
(the pre-registered analysis); the graph, values, frontier, composition and credit are `traceex.tropic`, the same code
the node runs.
