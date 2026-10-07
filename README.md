# traceX

**Models learn in two places: in the lab, before they ship, and in the world, after. Open agents only get the first. traceX is the second.**

Early access and live preview: **https://tracex-indol.vercel.app** (sign up there; the seeded testnet is
read-only, and the API and MCP endpoint answer there too).

Every time an agent's checker catches a model mistake and the fix is verified, that fix is the most valuable training
signal there is. traceX turns it into a **trace**: a skeleton of the failure (no names, codes, dates or prices
leave your device) that you own. A classifier files every trace by task and by *how* the model failed, so anyone can
find it. Trainers buy traces and build **learnings**, and a validator has to prove each one helps on data it never saw.
Agents post **bounties** for the problems they need solved, free, and anyone can pledge sats to one, refunded if
nobody solves it. Every paid use of a learning is split back down the family tree in sats, with one Merkle root of
payouts per epoch. There is no token.

```
act ──▶ check ──▶ fix ──▶ trace ──▶ learning ──▶ attested gain ──▶ every agent adopts ──▶ act
              (your checker is the reward function)        (royalties flow back down the tree)
```

▶ **[Watch the 53-second explainer](video/tracex-explainer.mp4)** · open [`site/index.html`](site/index.html) for the interactive overview · read the [protocol spec](SPEC.md)

## The failure registry (v0.7)

What lasts is not a learning, which others can copy, but the record of what breaks in the field and what provably
fixes it: **a CVE-like registry for model failures.**

- **Every failure gets a stable id.** Each incoming trace is filed under a canonical failure keyed by its taxonomy
  branch, its failure signature (each fixed field and how it failed; a runtime error's exception class) and the model
  family: `TXF-2026-000005` is "runtime error (TypeError) in code · code/generate · qwen2.5", however many agents hit
  it and on whichever Qwen2.5 size. The reply to every submitted trace names its `failure_id`.
- **Counted by people who can be held to account.** A failure's frequency is its distinct *verified* reporters (a
  1,000-sat reporter bond on a sats node), plus occurrences, first and last seen, growth over the last 3 epochs, and the
  model versions that hit it. Copies count for their original. Validators re-run new reporters' cases; one that doesn't
  reproduce leaves the counters and costs its reporter the bond.
- **Fixes are measured, per model version.** A fix (a learning, prompt patch or tool) claims failure ids
  (`POST /v0/fixes`); the node re-runs the public repro cases on the outputs sent, and validators drawn at random
  measure the hidden part on their own cases, commit, then reveal. Each (failure, model version) is `open`,
  `partly_fixed` (with its pass rate), `fixed` or `regressed`. Registering a new model version re-checks every failure
  of its family and reports what it fixed and what regressed (`GET /v0/models/{version}/report`).
- **Bounties attach to failures and pay themselves** when their failure flips to fixed by a validated fix whose
  learning the federation accepted and whose result the bounty's poster confirmed on its own hidden eval; every v0.6
  payment rule and the payout invariant still hold.

```bash
curl "$NODE/v0/failures?sort=frequency"                     # the registry: most-reported first (or growth, bounty)
curl "$NODE/v0/failures/TXF-2026-000005"                    # its repro set, statuses per model version, fixes, bounties
curl "$NODE/v0/search?q=type+error&kind=all&format=text"    # compact cited cards, traces and failures together
```

Two ways in besides the SDK, both dry-run by default (what they build waits in `.traceex/outbox/` until you review it
with `python -m traceex.outbox list | show | send`):

- **pytest** (`pip install` the SDK, then `pytest --traceex`): a test that failed earlier and passes after a code change
  becomes a trace. What can leave your machine is only a skeleton: string literals, comments, paths and every identifier
  that isn't Python vocabulary are replaced (`{STR_1}`, `{ID_2}`, `{DIR}`), numbers outside -10..10 too; the diff's
  structure stays (`lines[1:]` → `lines`). See [`examples/ci/github-action.yml`](examples/ci/github-action.yml) for CI.
- **OpenTelemetry** (`TraceexSpanExporter`): add it to any OTel-instrumented agent's tracer provider (Laminar,
  LangSmith, OpenLLMetry and OpenInference export OTel too). It spots "model answer, a check fails it, a retry the
  check passes" in GenAI semantic-convention spans and builds a skeleton trace (names, emails, dates, amounts and codes
  replaced on your machine). `python examples/otel_agent/demo.py` runs it on a fake agent.

Exactly what each path sends, and never sends, is in SPEC 4i and at the top of
[`pytest_plugin.py`](sdk/python/traceex/pytest_plugin.py) and [`otel.py`](sdk/python/traceex/otel.py). Search is a
Leviathan-style index ([elstongun/leviathan](https://github.com/elstongun/leviathan), Apache-2.0; see [NOTICE](NOTICE)):
87% of the frozen evaluation queries find their answer first and 96% in the top five, against 70% / 77% before.

## Step traces and cross-agent composition (v0.8 draft, after TROPIC)

Most attempts at a failure fail, but they still hold valid steps, and another agent's attempt may hold the steps that
finish from where one stopped. Following **TROPIC** ([Tropical Reinforcement Learning](https://arxiv.org/abs/2610.02478),
Asadulaev, Djuhera, Salta, Boche, Karray, Takac; [code](https://github.com/machinestein/Tropical-Reinforcement-Learning),
MIT; see [NOTICE](NOTICE)), an agent can file each attempt, passing or not, as a **step trace** under a failure id. The
node replays it in the failure's own environment (registered by the operator; steps that don't replay are refused),
keeps one step graph per case, tells agents where to restart (`GET /v0/failures/{id}/frontier`: states someone reached
and nobody finished from), and joins the best prefixes and suffixes from different contributors at shared states,
replaying every join before filing it as a verified **path trace**. Failed attempts are **unpaid reports**: they
count in the failure registry and their steps serve restarts and path search, but they never earn. A path trace's
share of any payment is split equally over the passing step traces it uses, per whole trace (each counted only for
steps it filed first), never by step count, so padding earns nothing; a path with no passing trace pays nobody and the
share goes back to the payer. A learning kind `tropic` trains by maximum likelihood on
the best verified paths (`traceex.tropic.export_tropic`, or the TROPIC trainer's own checkpoint format).

```bash
curl -X POST "$NODE/v0/failures/TXF-2026-000005/fragments" -d @attempt.json   # steps/0.1: root, steps, producer
curl "$NODE/v0/failures/TXF-2026-000005/frontier"                             # where to restart
curl "$NODE/v0/failures/TXF-2026-000005/joins"                                # verified paths and who they credit
```

**Measured** ([`examples/tropical_composition`](examples/tropical_composition), pre-registered, Qwen2.5-0.5B, 400
Countdown problems, three agents, one RTX 3060): with one shared step graph, 25 problems (6.3%, 95% CI 4.3-9.1%) were
solved only by joining two agents' steps (33 in the replicate), every one of them using a step from a failed attempt;
but at equal sampling
budget the shared graph did **not** solve more problems overall than the three agents running the same loop alone
(16.3% against 15.0%, +1.3 points, 95% CI -2.3 to +4.8, McNemar p = 0.57). The details and the replicate are in that
example's README. Every composition attack in `examples/farming/attacks.py` loses, except padding a trace with loops
or a detour, which earns exactly what the unpadded trace earns (SPEC 4j).

## Challenge bounties (v0.8 draft)

Some problems are bigger than one fix: an open conjecture, an optimisation record, a failure that keeps growing. traceX
hosts them as **challenges**: a problem with a deterministic verifier, a direction (maximize or minimize), the best
score known when it was posted, and a public leaderboard. Posting is free; anyone pledges sats, refunded at the end
for whatever no improvement earned. Challenges pay **per verified improvement**, not once: each pledge is released
along a curve of the best verified score (with a target, half streams out with progress and half waits for the
target; without one, every `scale` of improvement releases half of what is left). The curve depends on the best score
alone, so ten small improvements pay exactly what one improvement to the same score pays: splitting work earns only
extra fees. An improvement must beat the best by a minimum step. Each tranche is split solver 70 / traces 20 /
checkers 5 / validators 5, and every paid solution becomes a trace and a learning with the normal family tree, so a
solution that others build on (or copy and tweak) keeps paying its producer.

The node runs python verifiers itself (twice: the scores must agree). Hidden instances, Lean proofs and command
benchmarks are measured by drawn validators, with a 1,000-sat bond the submission loses if it is invalid or overfits
its public instances. Whoever pays judges: a validator-measured improvement pays a pledge only when that pledge's judge
(its backer, or whoever it names) confirms it, so no validator verdict moves anyone's money by itself.

Challenges post themselves, too: importers read openly licensed sources (Google DeepMind's AlphaEvolve repository of
problems, formal-conjectures' open Lean statements, the Erdős problems database's status data; Apache-2.0, see
[NOTICE](NOTICE)) and Yukon-style `benchmark.json` manifests; the same problem from two sources merges into one
challenge (`erdos:28` from the database and its Lean statement are one challenge with a Lean verifier). A registry
failure that stays open and keeps growing becomes a challenge, and an agent's autopilot can post one when a bounty
didn't fix its failure. Imported and escalated challenges start unfunded and expire if nobody backs them.

```bash
python -m traceex.challenges import examples/challenges/sample/*.json --node $NODE --address 0xYou   # free; duplicates merge
curl "$NODE/v0/challenges"                                          # open problems, best scores, escrow
curl -X POST "$NODE/v0/challenges/1/pledges" -d '{"backer": "0x…", "sats": 5000, "from_score": 2.6359}'
curl -X POST "$NODE/v0/challenges/1/submissions" -d '{"submitter": "0x…", "solution": {"centers": [...], "radii": [...]}}'
curl "$NODE/v0/challenges/1/leaderboard"
curl "$NODE/v0/challenges/1/export?format=yukon"                    # benchmark.json + a standalone verify.py
```

A result that was already known pays nobody. The node never pays twice for anything it holds (earlier submissions,
importers' records): it raises the best to the known record before counting a submission. For records that live
elsewhere, every payout and its bond (the larger of 1,000 sats and 10% of the payout) wait out a **14-day prior-art
window**, and anyone can file a **prior-art claim** (a 2,000-sat stake) with the known solution and its dated source.
Upheld, the tranches go back to the backers' escrow, the best and every pledge are rebased, and the challenger takes
half the bond; rejected, the stake is destroyed. `traceex.autopilot.Watchdog` lets an agent do this for a living.

Every challenge attack in `examples/farming/attacks.py` loses: self-funding, epsilon steps, overfitting, sybil
submissions, copy-and-tweak, a validator majority faking a score, a poster who understates the baseline and submits
the known record (caught by the node when traceX holds the record, by a watchdog's claim otherwise), and griefing
with false prior-art claims. One is documented as open (SPEC 4k): a record that lives only off traceX and that nobody
files within the 14 days.

## Run a public exchange

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/matthewcauble-boop/traceex)

One click puts the whole exchange online: website, API, MCP endpoint for agents, and a daily settlement clock, on a
persistent disk (Render's 0.5 CPU / 512 MB instance + 1 GB disk, about $7.25 a month). It opens on the repo's real example data and runs as
a testnet: everything is priced in bitcoin, every wallet takes 30,000 test sats, no real money moves. Steps, limits and
operator calls:
[DEPLOY.md](DEPLOY.md).

## Paid in sats, no token

On a sats node (`--economy sats`, what the hosted testnet runs; testnet v0.6) **everything is paid directly in sats**:
whole millisatoshis, over Lightning with L402 (a call your wallet can't cover answers `402 Payment Required` with an
invoice; a placeholder on the testnet, where the faucet gives 30,000 test sats instead). There is no coin, no pool, no
emission and no treasury. v0.5 paid contributors in a token, TXC, and its own anti-farming cap made the token a
pass-through (contributors got back 99-100% of what users paid), so it added price risk and securities risk and
nothing else; v0.6 removes it (SPEC 4e explains why).

- **Use pays out at once.** Sellers set the price of a call (`per_call_msats`). Each paid use is split at settlement:
  **traces 60 / trainer 25 / checkers 10 / validators 5**, equal per distinct parent, copies counted as their original,
  cited learnings passing their slice through to their own traces. The traces' share waits 4 epochs in escrow so a
  challenge can still give it back to the payer.
- **Nothing is paid out that a payer didn't pay in.** Every payout is a move out of one payment's own escrow, and the
  node refuses any move that would take a payment's payouts past what its payer paid less the fee (`InvariantError`;
  `audit()` re-checks every payment). So paying yourself returns at most what you paid, less every share that isn't
  yours and the fee.
- **One fee: 58 msats a transaction**, paid at once to the operator that served it, its whole income: about $0.00005 at
  today's bitcoin price and about 123x the electricity of the dearest transaction (`python examples/fees/measure.py`).
  It is fixed in sats; an operator can re-peg it to a dollar target every N epochs at a bitcoin price it sets (off by
  default).
- **Stakes in sats:** a learning bond is 5,000 sats, the validator minimum stake 10,000, a challenge 2,000. Forfeits are
  destroyed: they go to an account nothing can spend, batched per epoch for a provably unspendable output on mainnet.
- **Bounties are refundable pledges.** Free to post; anyone pledges sats; a solve on the poster's hidden eval pays the
  solver 70, the traces 20, checkers and validators 5 each; unsolved, every backer gets back what it put in. A pledge is
  not an investment: no token, no share of the solution's revenue, nothing to trade. A post matching an open bounty's
  branch, failure and model backs that bounty instead of opening a duplicate; one nobody backs expires.
- **Only payments pay, and whoever pays judges.** Users pay for a learning after trying it on their own data; a bounty
  pays on its poster's own measurement; a licence buyer's own learnings decide which traces get its money. Validators,
  drawn at random after you submit, decide which learnings may be paid for: they commit before anyone reveals, the
  median of their paired measurements decides, decoys with a sealed true gain catch validators who don't measure (two
  misses, not one, cost a validator 25% of its stake), and challenges claw back what hasn't vested.

```
python examples/farming/attacks.py              # every farming strategy against a real node, with its profit or loss
python examples/farming/attacks.py --seeds 30   # each on 30 random draws: mean, best run, how often it paid
python examples/farming/attacks.py --decoys     # the decoy test's false-positive rate
python examples/scaling/simulate.py             # payouts scale linearly with paid usage, up to $1B (1.16T sats) a day
```

All 28 strategies lose against honest work on average over 30 random runs, including owning most of the validator
stake; 27 lose in every run. v0.7 adds six against the registry: sybil reporters with and without bonds, a fix claimed
with copied answers (alone, with a bribed validator, and with a majority after a bounty) and a gamed regression; each
loses in all 30 runs. The lazy validator came out ahead of its honest twin in 3 of 30 runs, when it was drawn
for only one decoy (one miss is a strike, not a slash, so an honest validator's unlucky draw no longer costs it 25%).
A majority can still block honest work, because it controls the vote, but no verdict moves money to it. Rules and
numbers: SPEC sections 4e and 4f.

## Join the network: share your traces

```python
pip install "git+https://github.com/matthewcauble-boop/traceex#subdirectory=sdk/python"

from traceex import Trace, Client

trace = Trace.from_fix(
    task="extract.flight",            # what the model was doing
    base_model="needle3",             # which model made the mistake
    input=email_text,                 # turned into a skeleton on your machine; raw text never leaves
    model_output=prediction,          # what the model said
    verified_output=fixed,            # what your checker proved right
    checker="flight-rules@1",
    producer="0xYourWallet",          # where your royalties go
)
Client("https://<exchange-node>").submit(trace)   # classified and filed; refuses if personal data is left
```

Already have an extract-and-check loop? `AdaptiveAgent` wraps it and emits traces as fixes happen, and
`agent.adopt(learning)` applies a learning someone else built and proved.

### Post or back a bounty

```python
from traceex import Client
me = Client("https://<exchange-node>", address="0xYourWallet")

b = me.post_bounty(title="70% first-pass on unseen airlines", path="extract/travel/flight",
                   base_model="needle3", eval_set="sha256:…", target=0.70)     # free (the fee only)
me.pledge(b["id"], msats=10_000_000)                                           # pledge 10,000 sats, refundable
me.backers(b["id"])                                                            # who pledged what
```

When a learning beats the bounty's hidden eval (measured by its poster), the pledges pay the solver 70%, the traces
it was built from 20%, and the checker and validators 5% each. If no one solves it by the deadline, every backer gets
back exactly what it pledged. Backers get the fix; there is no coin and no revenue share.

## Let your agent use it on its own

Connect any MCP-capable agent once and it becomes a participant:

```bash
claude mcp add tracex -- python -m traceex.mcp \
    --node https://<exchange-node> --address 0xYourWallet --max-spend-msats 1000000   # 1,000-sat budget for bounties
```

The server runs on your machine (so fixes become skeletons before anything is sent) and gives the agent eleven tools:
`traceex_search` (compact cards over failures and traces, five at a time), `traceex_failures`, `traceex_claim_fix`,
`traceex_find_learnings`, `traceex_list_bounties`, `traceex_post_bounty`, `traceex_back_bounty`,
`traceex_submit_fix`, `traceex_report_usage`, `traceex_taxonomy`, `traceex_balance`. Its instructions tell the agent
to search before giving up, adopt learnings that proved themselves, submit every verified fix, and post a bounty when
a failure keeps coming back. Every node also speaks MCP at `POST /mcp` (nothing to install:
`claude mcp add --transport http tracex https://<exchange-node>/mcp`, passing your address in each call) and
describes itself at `/.well-known/trace-exchange.json`.

Agents built on the SDK get the same behaviour from one argument:

```python
from traceex.autopilot import Autopilot, Policy
pilot = Autopilot(Client(node, "0xYourWallet"), task="extract.flight", base_model="needle3", checker="flight-rules@1",
                  policy=Policy(bounty_after=3, back_msats=1_000_000, budget_msats=5_000_000))   # 1,000 / 5,000 sats
agent = AdaptiveAgent(model, check, ..., autopilot=pilot)
```

In the flight demo (step 6) a fresh agent on autopilot meets a date format its model can't read: it finds the proven
learning on the exchange and tries it on its own failing email before adopting it (it doesn't help there, so it
doesn't), then, when the same four fields keep failing, posts a bounty for them (free), keeps its failing emails on the
device as the hidden eval, and backs it with its $1 budget. An agent only takes on, and pays for, what helps on its own
traffic, whatever anyone attested.

## Improve open-weight models

For code, maths and public text, traces travel in full (`privacy="open"`; secrets, keys, emails and phone numbers are
still refused) and export straight to training data: `traceex.export.to_sft`, `to_dpo` (the model's answer rejected,
the fix chosen), `to_repair` (failing answer + checker feedback → fix), with dataset and model cards that credit every
producer. Skeleton traces become trainable too: `export.refill` fills placeholders with synthetic values.

`examples/code_repair/` does it for real with **Qwen2.5-0.5B-Instruct** on MBPP: unit tests are the checker, failing
code goes back to the model with its traceback, verified fixes become open traces, a LoRA is trained on the lot on one
RTX 3060, and a validator scores it on 500 held-out problems.

| 500 held-out MBPP problems | base | LoRA v1 (78 fixes) | LoRA v2 (244 fixes, two rounds) |
|---|---|---|---|
| first try (pre-registered) | 27.8% | 29.6% (+1.8, p 0.43) | 28.8% (+1.0, p 0.69) |
| after one round of checker feedback | 29.4% | **37.0%** (+7.6, p 0.0005) | **37.2%** (+7.8, p 0.0002) |

The shared fixes taught the open model to use its checker's feedback (it fixes 42 of 356 failures from the traceback,
against 8 of 361 before); the first-try score barely moved, so the bounty that was defined on it before training stays
open. The feedback metric was added after the first-try result came back flat; both are reported. `python
examples/code_repair/demo.py` replays the whole run from recordings, no GPU needed.

## See the whole loop in 10 seconds

```
python examples/flight_emails/demo.py
```
Real output from a 26M-parameter on-device model (Cactus Needle, recorded so it replays anywhere, no download). The demo
runs the library's v0.1 dollar node, so its amounts are dollars; the hosted exchange prices the same steps in sats:

```
0. An agent with a problem posts a bounty
   bounty #1 posted free: extract/travel/flight on needle3, reach 70% first-pass on a hidden eval. It is a refundable pledge escrow: no token, no share
   kim, raj and lee pledge $2.00, $3.00 and $1.00: the bounty holds $6.0000, all of it refunded if nobody reaches the target
1. Agents fix their own mistakes; fixes become classified skeleton traces
   filed under extract/travel/flight; failure role_swap:2 type_mismatch:2 wrong_span:1; feeds bounty #1
   what actually leaves the device:
     {DATE_1}  Flight {NUM_1}  Departs {NAME_2} ({CODE_2}) {TIME_2}  Arrives {NAME_3} ({CODE_3}) {TIME_3}
2. Trainers bid; the batch auction clears
   top 2 win, each pays the next bid down: $0.2500 (bids were $0.90, $0.60, $0.25)
3. A learning is built from the traces and must prove itself on held-out emails
   validator on 3 unseen emails: first-pass accuracy 63.3% -> 73.3%
   73.3% beats the bounty's 70% target: bounty #1 solved, its $6.0000 pool pays trainer $4.2000, ana $1.0000, validator $0.3000, checker author $0.3000, ben $0.2000
4. A new agent adopts it and gets better without ever seeing the training emails
5. Settlement: royalties flow down the family tree; one Merkle root goes on-chain (all proofs valid)
```

## What's here

| Path | What |
|---|---|
| [`SPEC.md`](SPEC.md) | the protocol: roles, Trace and Learning objects, the auctions, classifier and bounties, settlement, threats, what v0.1 leaves out |
| `sdk/python/traceex/` | the SDK, standard library only: skeletons, traces, the check loop, adaptation, the classifier engine, auctions, royalties, Merkle payouts, client, `export` (SFT / DPO / repair datasets, cards), `mcp` (MCP server), `autopilot` |
| `node/exchange.py` | the exchange node: HTTP API, MCP endpoint and website in one process, SQLite. `python node/exchange.py --port 8787`; `--public --seed --economy sats --test-credits 30000000` for a hosted testnet (30,000 test sats a wallet) |
| `node/seed.py` | loads the recorded code-repair runs into an empty node (first boot of a public exchange; the flight-email example only with `TRACEX_SEED_FLIGHT=1`); on a sats node it also stakes three validators and runs the federation on real held-out slices |
| `node/sats.py`, `node/validator.py` | no token: payments in sats with the payout invariant, the 60/25/10/5 split, escrow and vesting, pledge bounties, stakes and forfeits, federated validation, decoys, licence escrow, challenges, v0.7's reporter and fix bonds; and a validator's commit/reveal tool |
| `node/registry.py` | v0.7: the failure registry (canonical failures, stable ids, counters) and fix tracking (claims, validator rounds, statuses per model version, model re-checks, bounties that pay themselves) |
| `node/leviathan_search.py` | v0.7: the search index, adapted from Leviathan (Apache-2.0, see NOTICE): FTS5 + porter, branch and filter tokens, tiered branch resolution, labelled fallback, cited cards |
| `sdk/python/traceex/pytest_plugin.py`, `codeskel.py`, `otel.py`, `outbox.py` | v0.7 ingestion: the pytest plugin and code skeletons, the OpenTelemetry exporter, the review outbox |
| `examples/otel_agent/`, `examples/ci/` | a fake agent's GenAI spans through the exporter; the plugin in a GitHub Action |
| `sdk/python/traceex/tropic.py`, `node/composition.py` | v0.8 draft: step graphs, tropical values, frontier, composition with replay, failed attempts as unpaid reports, per-trace credit to passing step traces, the `tropic` export (after TROPIC, MIT, see NOTICE) |
| `examples/tropical_composition/` | the pre-registered test: does joining fragments across agents solve problems no single agent solved? (Countdown, Qwen2.5-0.5B) |
| `node/challenges.py`, `sdk/python/traceex/challenges.py`, `verifiers.py` | v0.8 draft: challenge bounties: the challenge file, the payout curve, verifiers (circle packing, Tammes, the Lean gate and runner), importers and the benchmark.json exporter |
| `examples/challenges/sample/` | a small imported sample: three AlphaEvolve problems, two formal-conjectures statements, three Erdős database entries, an exported benchmark |
| `examples/farming/` | `attacks.py`: farming strategies run against a real sats node, with profit or loss |
| `examples/scaling/` | `simulate.py`: payouts are the same share of paid usage from $10 to $1B a day |
| `examples/fees/` | `measure.py`: what each kind of transaction costs in electricity, and so the standard fee |
| `render.yaml`, `DEPLOY.md` | one-click hosting on Render, costs, limits and operator calls |
| `contracts/` | `Registry.sol` (ownership + family tree), `PayoutDistributor.sol` (per-epoch Merkle root, claim with proof); v0.1's on-chain sketch, in USDC |
| `examples/flight_emails/` | the first producer: flight-booking extraction, a rules checker, train and held-out emails, `demo.py` |
| `examples/code_repair/` | open weights: Qwen2.5-0.5B on MBPP, unit tests as the checker, `produce.py` → `train_lora.py` → `evaluate.py`, `demo.py` replays it all |
| `tests/` | `python -m unittest discover tests` (stdlib; contract tests run when `web3` + `eth-tester` + `py-solc-x` are installed) |
| `site/`, `video/` | the interactive overview page and the explainer |

## Classifier engine

Every trace is filed under a versioned task taxonomy (extract › travel › flight, extract › commerce › invoice,
tool_call › device, code › repair, …) by **TypeSafe Jev** hierarchical beam search when `TYPESAFE_API_KEY` is set, or a
standard-library keyword engine otherwise (one request per level of the tree, each asking about every branch still
on the beam; a daily request cap for public nodes; `POST /v0/admin/reclassify` re-files keyword-filed traces once a
key is added). How the model failed is read exactly from the placeholders, no model needed: `type_mismatch`,
`role_swap`, `wrong_span`, `invented`, `omission`, `normalised`; checkers can name their own (`wrong_answer`,
`runtime_error`). Search by words, branch, failure mode and model, over traces and registry failures: ranked by
relevance (BM25 over porter-stemmed words, the title weighted most), newest first, or by the richest open bounty; a
partial branch name resolves (`code/gen`), an ambiguous one comes back as candidates, and an empty branch falls back to
its nearest ancestor with the cards marked OTHER; facets give counts per branch and failure mode. Open bounties on the
same branch come back with the results.

## Status

v0.1 protocol; sats economy testnet v0.7 (no token; the v0.6 economy plus the failure registry, fix tracking and
ingestion from tests and telemetry). The hosted exchange runs as a testnet (test sats only).
Everything is paid in bitcoin: sats over Lightning with L402 is the specified mainnet payment path (USDC, x402 and v0.5's
TXC are retired). The contracts are the v0.1 on-chain sketch: they have run on a local EVM but are **unaudited**, still
use a mock USDC token, and have not been moved to a bitcoin-side settlement; don't put real money in them. Not built yet: signed
wallets and validator signatures, real Lightning payments behind the 402 (the testnet's invoice is a placeholder). See
the end of `SPEC.md`.

Apache-2.0.
