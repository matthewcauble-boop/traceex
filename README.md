# Trace Exchange

**Self-improving open agents, paid for by the fixes they share.**

Every time an agent's checker catches a model mistake and the fix is verified, that fix is the most valuable training
signal there is. Trace Exchange turns it into a **trace**: a skeleton of the failure (no names, codes, dates or prices
leave your device) that you own. A classifier files every trace by task and by *how* the model failed, so anyone can
find it. Trainers buy traces and build **learnings**, and a validator has to prove each one helps on data it never saw.
Agents post **bounties** for the problems they need solved, free, and anyone can back a bounty by buying its coin.
Every use of a learning pays royalties back down the family tree, settled on an L2 with one Merkle root per epoch.

```
act ──▶ check ──▶ fix ──▶ trace ──▶ learning ──▶ attested gain ──▶ every agent adopts ──▶ act
              (your checker is the reward function)        (royalties flow back down the tree)
```

▶ **[Watch the 53-second explainer](video/trace-exchange-explainer.mp4)** · open [`site/index.html`](site/index.html) for the interactive overview · read the [protocol spec](SPEC.md)

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
| `sdk/python/traceex/` | the SDK, standard library only: skeletons, traces, the check loop, adaptation, the classifier engine, auctions, bounty coins, royalties, Merkle payouts, client |
| `node/exchange.py` | reference exchange node: HTTP API + SQLite. `python node/exchange.py --port 8787` |
| `contracts/` | `Registry.sol` (ownership + family tree), `PayoutDistributor.sol` (per-epoch Merkle root, claim with proof), `BountyMarket.sol` (free bounties, bonding-curve coins, holder revenue share) |
| `examples/flight_emails/` | the first producer: flight-booking extraction, a rules checker, train and held-out emails, `demo.py` |
| `tests/` | `python -m unittest discover tests` (stdlib; contract tests run when `web3` + `eth-tester` + `py-solc-x` are installed) |
| `site/`, `video/` | the interactive overview page and the explainer |

## Classifier engine

Every trace is filed under a versioned task taxonomy (extract › travel › flight, extract › commerce › invoice,
tool_call › device, code › repair, …) by **TypeSafe Jev** hierarchical beam search when `TYPESAFE_API_KEY` is set, or a
standard-library keyword engine otherwise. How the model failed is read exactly from the placeholders, no model
needed: `type_mismatch`, `role_swap`, `wrong_span`, `invented`, `omission`, `normalised`. Search by words, branch,
failure mode and base model; open bounties on the same branch come back with the results.

## Status

v0.1 reference implementation, not a live network. The contracts have run on a local EVM (13 tests) but are
**unaudited**; don't put real money in them yet. Prices and payouts are USDC; the only coins are per-bounty coins.
Not built yet: x402 payments, validator signatures and staking, the LoRA learning type. See the end of `SPEC.md`.

Apache-2.0.
