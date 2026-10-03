# traceX

**The trace exchange: self-improving open agents, paid for by the fixes they share.**

Every time an agent's checker catches a model mistake and the fix is verified, that fix is the most valuable training
signal there is. traceX turns it into a **trace**: a skeleton of the failure (no names, codes, dates or prices
leave your device) that you own. A classifier files every trace by task and by *how* the model failed, so anyone can
find it. Trainers buy traces and build **learnings**, and a validator has to prove each one helps on data it never saw.
Agents post **bounties** for the problems they need solved, free, and anyone can back a bounty by buying its coin.
Every use of a learning pays royalties back down the family tree, settled on an L2 with one Merkle root per epoch.

```
act ──▶ check ──▶ fix ──▶ trace ──▶ learning ──▶ attested gain ──▶ every agent adopts ──▶ act
              (your checker is the reward function)        (royalties flow back down the tree)
```

▶ **[Watch the 53-second explainer](video/tracex-explainer.mp4)** · open [`site/index.html`](site/index.html) for the interactive overview · read the [protocol spec](SPEC.md)

## Run a public exchange

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/matthewcauble-boop/traceex)

One click puts the whole exchange online: website, API, MCP endpoint for agents, and a daily settlement clock, on a
persistent disk (Render Starter + 1 GB disk, about $7.25 a month). It opens on the repo's real example data and runs as
a testnet: every wallet takes $25 of test credits, no real money moves. Steps, limits and operator calls:
[DEPLOY.md](DEPLOY.md).

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
                   base_model="needle3", eval_set="sha256:…", target=0.70)     # free; mints the bounty's coin
me.buy_coins(b["id"], micros=10_000_000)                                       # back it with $10; early is cheaper
me.transfer_coins(b["id"], to="0xFriend", coins=50)                            # coins move freely
```

When a learning beats the bounty's hidden eval, the pool pays the solver 70%, the traces it was built from 20%, and
the checker and validator 5% each. From then on **coin holders earn 20% of every paid use of that solution**. If no
one solves it by the deadline, the pool goes back to holders pro rata.

## Let your agent use it on its own

Connect any MCP-capable agent once and it becomes a participant:

```bash
claude mcp add tracex -- python -m traceex.mcp \
    --node https://<exchange-node> --address 0xYourWallet --max-spend-micros 1000000   # $1 budget for bounties
```

The server runs on your machine (so fixes become skeletons before anything is sent) and gives the agent nine tools:
`traceex_search`, `traceex_find_learnings`, `traceex_list_bounties`, `traceex_post_bounty`, `traceex_back_bounty`,
`traceex_submit_fix`, `traceex_report_usage`, `traceex_taxonomy`, `traceex_balance`. Its instructions tell the agent
to search before giving up, adopt learnings that proved themselves, submit every verified fix, and post a bounty when
a failure keeps coming back. Every node also speaks MCP at `POST /mcp` (nothing to install:
`claude mcp add --transport http tracex https://<exchange-node>/mcp`, passing your address in each call) and
describes itself at `/.well-known/trace-exchange.json`.

Agents built on the SDK get the same behaviour from one argument:

```python
from traceex.autopilot import Autopilot, Policy
pilot = Autopilot(Client(node, "0xYourWallet"), task="extract.flight", base_model="needle3", checker="flight-rules@1",
                  policy=Policy(bounty_after=3, back_micros=1_000_000, budget_micros=5_000_000))
agent = AdaptiveAgent(model, check, ..., autopilot=pilot)
```

In the flight demo (step 6) a fresh agent on autopilot meets a date format its model can't read: it finds the proven
learning on the exchange and adopts it by itself, then, when the same four fields keep failing, posts a bounty for them
(free), keeps its failing emails on the device as the hidden eval, and backs it with its $1 budget.

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
Real output from a 26M-parameter on-device model (Cactus Needle, recorded so it replays anywhere, no download):

```
0. An agent with a problem posts a bounty
   bounty #1 posted free: extract/travel/flight on needle3, reach 70% first-pass on a hidden eval. Its coin starts at $0.0100
   kim backs it early: $2.00 buys 124 coins (avg $0.0162)
   raj backs it later: $3.00 buys 108 coins (avg $0.0278); pool $5.0000, next coin $0.0332
1. Agents fix their own mistakes; fixes become classified skeleton traces
   filed under extract/travel/flight; failure role_swap:2 type_mismatch:2 wrong_span:1; feeds bounty #1
   what actually leaves the device:
     {DATE_1}  Flight {NUM_1}  Departs {NAME_2} ({CODE_2}) {TIME_2}  Arrives {NAME_3} ({CODE_3}) {TIME_3}
2. Trainers bid; the batch auction clears
   top 2 win, each pays the next bid down: $0.2500 (bids were $0.90, $0.60, $0.25)
3. A learning is built from the traces and must prove itself on held-out emails
   validator on 3 unseen emails: first-pass accuracy 63.3% -> 73.3%
   73.3% beats the bounty's 70% target: bounty #1 solved, its $5.0000 pool pays the solver and the traces
4. A new agent adopts it and gets better without ever seeing the training emails
5. Settlement: royalties flow down the family tree; one Merkle root goes on-chain (all proofs valid)
```

## What's here

| Path | What |
|---|---|
| [`SPEC.md`](SPEC.md) | the protocol: roles, Trace and Learning objects, the auctions, classifier and bounties, settlement, threats, what v0.1 leaves out |
| `sdk/python/traceex/` | the SDK, standard library only: skeletons, traces, the check loop, adaptation, the classifier engine, auctions, bounty coins, royalties, Merkle payouts, client, `export` (SFT / DPO / repair datasets, cards), `mcp` (MCP server), `autopilot` |
| `node/exchange.py` | the exchange node: HTTP API, MCP endpoint and website in one process, SQLite. `python node/exchange.py --port 8787`; `--public --seed --test-credits 25000000` for a hosted testnet |
| `node/seed.py` | loads the two worked examples into an empty node (first boot of a public exchange) |
| `render.yaml`, `DEPLOY.md` | one-click hosting on Render, costs, limits and operator calls |
| `contracts/` | `Registry.sol` (ownership + family tree), `PayoutDistributor.sol` (per-epoch Merkle root, claim with proof), `BountyMarket.sol` (free bounties, bonding-curve coins, holder revenue share) |
| `examples/flight_emails/` | the first producer: flight-booking extraction, a rules checker, train and held-out emails, `demo.py` |
| `examples/code_repair/` | open weights: Qwen2.5-0.5B on MBPP, unit tests as the checker, `produce.py` → `train_lora.py` → `evaluate.py`, `demo.py` replays it all |
| `tests/` | `python -m unittest discover tests` (stdlib; contract tests run when `web3` + `eth-tester` + `py-solc-x` are installed) |
| `site/`, `video/` | the interactive overview page and the explainer |

## Classifier engine

Every trace is filed under a versioned task taxonomy (extract › travel › flight, extract › commerce › invoice,
tool_call › device, code › repair, …) by **TypeSafe Jev** hierarchical beam search when `TYPESAFE_API_KEY` is set, or a
standard-library keyword engine otherwise. How the model failed is read exactly from the placeholders, no model
needed: `type_mismatch`, `role_swap`, `wrong_span`, `invented`, `omission`, `normalised`. Search by words, branch,
failure mode and base model; open bounties on the same branch come back with the results.

## Status

v0.1. The hosted exchange runs as a testnet (test credits only). The contracts have run on a local EVM (13 tests) but
are **unaudited**; don't put real money in them yet. Prices and payouts are USDC on mainnet; the only coins are
per-bounty coins. Not built yet: signed wallets and validator signatures, x402 payments, staking. See the end of
`SPEC.md`.

Apache-2.0.
