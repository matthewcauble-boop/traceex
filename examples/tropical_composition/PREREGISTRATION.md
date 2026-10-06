# Pre-registration: does stitching fragments across agents solve problems no single agent solved?

Written 2026-10-06, before the main run. The only data seen before writing it is a feasibility pilot on 20 *other*
problems (`--seed 999 --per-size 5`, not part of any result below), run to check throughput and answer formats; its
numbers are not reported as results.

## Question

Three agents attempt the same problems. Each attempt leaves a trail of valid steps, most of them from attempts that
fail. If the agents put their steps into one shared step graph (what traceX would do for a failure), and TROPIC's
composition joins prefixes and suffixes at shared states, replays them and verifies them, are more problems solved
than when each agent runs the same loop on its own, at the same sampling budget? And how many first solves exist only
because steps from two or more agents were joined?

## Setup (fixed before the run)

- **Task.** Countdown as in the TROPIC reference code: its generator (`generate.py`, ported), seed 2026 for the
  validation split and 2027 for training (excluding validation), **100 problems for each size n = 3, 4, 5, 6: 400
  problems.** One exact binary operation per step over the live numbers, rational arithmetic, success when the last
  number equals the target, at most 5 steps. State = target + sorted multiset of live numbers; depth = step count.
- **Verifier.** A path counts only if it replays from the start in the step environment with every recorded next
  state and goal flag reproduced, and an independent check (fold the actions into one expression over the original
  inputs, evaluate exactly) uses every input once and equals the target.
- **Agents.** Qwen2.5-0.5B-Instruct, inference only (transformers, bf16, one RTX 3060), three system prompts and
  temperatures: A = TROPIC's own Countdown instruction, T 1.0; B = terse "answer only", T 0.7; C = "think in one
  sentence, then answer", T 0.9. Answers are parsed identically for all three (first `a op b` inside the last
  `<answer>` tag, else in the reply); an unparseable or illegal operation ends the attempt.
- **Scoring model (one for every step).** log p(`<answer>a op b</answer>` | agent A's prompt and the state) under the
  same base model. Used for tropical values (path choice, frontier choice); never for verification.
- **Budget.** Wave 0: 6 attempts per agent per problem from the start (shared by both conditions). Then 3 further
  waves per condition, 6 attempts per agent per problem per wave, each wave starting from one frontier state per agent
  chosen by TROPIC's rule (least-explored depth, best prefix value minus eta x log(1 + visits), eta = 1; states that
  can already finish are not eligible; the root if none is eligible). top-L = 4, at most 64 join candidates per
  problem per refresh. Total: 24 attempts per agent per problem per condition.
  - **isolated:** one step graph per agent per problem; frontier, restarts and composition use only that agent's steps.
  - **pooled:** one step graph per problem shared by the three agents; frontier and composition use everyone's steps.

## Outcomes and the comparison

Per problem, binary:

- `direct`: some wave-0 attempt (any agent) solved it in one go.
- `iso`: some agent's isolated graph holds a verified path after all waves (TROPIC run alone, three times).
- `pooled`: the pooled graph holds a verified path after all waves.
- `posthoc`: the isolated runs' steps pooled after the fact (no shared frontier), then composed.
- `cross_only`: `pooled`, and no single agent's own steps in the pooled graph contain a complete verified path
  (checked exhaustively, not just with top-L, so this count is conservative).

**Primary metric:** the paired difference in solve rate, `pooled - iso`, over the 400 problems, with a 95% bootstrap
confidence interval (10,000 resamples of problems) and an exact two-sided McNemar test on the discordant problems.
**Hypothesis H1:** pooled > iso. It is supported only if the CI's lower bound is above 0 and McNemar p < 0.05.

**Secondary (reported whatever they show):**

1. `cross_only` count and its share of all problems and of pooled-solved problems (Wilson 95% CIs).
2. For each `cross_only` problem, on its best verified path: how many agents' steps it needs (minimum cover), how many
   distinct fragments it is credited to, and how many of those fragments come from attempts that failed.
3. `posthoc - iso`. **Predicted to be exactly 0:** a restart starts at a state its own agent reached, so pooling
   independently collected fragments after the fact can add new paths but never a first solve. If this fails, the
   code is wrong.
4. Share of solved problems whose best (highest tropical value) verified path mixes agents, pooled and post-hoc.
5. Everything by problem size, and the model calls each condition spent.
6. A second replicate (same problems, every agent's sampling seed shifted, `--sample-seed 1`), reported as a
   robustness check of the primary difference; H1 is decided on replicate 0 alone.

## What this can and can't show

It is one small model, one task, one budget, and three agents that differ only in prompt and temperature. A positive
result shows that, at equal sampling cost, a shared step graph with a shared frontier solves problems that the same
agents running alone do not, and how often a solve needs more than one agent's steps; it does not show that training on
those paths helps (no training is done here), and it does not separate "shared fragments" from "shared frontier"
except through the post-hoc condition.
