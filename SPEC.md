# traceX — the trace exchange protocol, spec v0.1

**The goal is self-improving open agents.** An agent running any open model should get better from its own mistakes,
and from everyone else's, without a lab in the loop. traceX is the protocol for that loop, plus the market that
pays for it: any model's **verified fixes** become owned, tradable training data, and the **learnings** built from
them earn per-use royalties that flow back to everyone whose work went into them.

**The registry (v0.7).** The durable asset is not a learning, which others can copy, but the record of what breaks in
the field and what provably fixes it: a CVE-like registry for model failures. Every trace is filed under a canonical
failure with a stable public id (`TXF-2026-000123`), counted by distinct verified reporters, and tracked per model
version (open, partly fixed, fixed, regressed) as fixes are measured against it; bounties attach to failure ids and
pay when their failure is fixed (4g, 4h). Traces arrive from agents and SDKs, from test suites (a pytest plugin) and
from telemetry (an OpenTelemetry exporter), skeletonized on the machine they come from (4i).

## 0. The adaptive loop
```
act ──▶ check ──▶ fix ──▶ trace ──▶ learning ──▶ attested gain on held-out data ──▶ every agent adopts ──▶ act
```
- **The checker is the reward function.** It does not need to know the answer, only to say which output is wrong and
  why. Once a fix passes it, the pair (state, the model's action, the verified action) is exactly a preference pair,
  the raw material for DPO / RL fine-tuning. Verified fixes are the scarce signal that open agents lack; the exchange
  makes them flow from every deployment to every trainer.
- **Adaptation is a learning.** v0.1 ships `routing` (no GPU, any model): fields the model fixed *itself* once its
  context was narrowed get narrowed up front. The roadmap: `rule` (checker evidence promoted to a fallback),
  `prompt_patch`, `lora` (trained on refilled skeletons), and `checker` (a better reward function is itself a learning
  that earns). All share the same object, attestation and royalty path.
- **Nothing is adopted on faith.** A learning is only listed with a validator's before/after on a held-out eval it
  never trained on, and an agent can re-measure on its own held-out traffic before switching.
- **Measured (examples/flight_emails, 26M-parameter Needle on-device):** three fixed emails from two agents produced
  a routing learning that raised first-pass field accuracy on three unseen airline formats from **63.3% to 73.3%**,
  with no field getting worse. A first version that also routed fields only the *rules* could fix scored 70.0% and
  broke a field that was right before; recording how each fix was made (`fixed_by`) is what separated the two.

```
model in production ──fails──▶ checker proves the right answer ──▶ TRACE (skeleton, never raw)
        ▲                                                              │
        │                                                   [1] trace auction (trainer agents bid)
        │                                                              ▼
   per-use royalties ◀── [2] learnings auction / metering ◀── LEARNING (adapter, patch, rule, checker)
   split down the family tree                                  + validator's measured eval gain
```

## 1. Roles
| Role | Does | Earns |
|---|---|---|
| **Producer** | runs any model; when its checker proves a fix, submits a trace | trace sales + royalties from every learning built on it |
| **Checker author** | publishes the rules/tests that prove fixes (e.g. `flight-rules@1`) | a cut of every sale of a trace it verified |
| **Trainer** (usually an agent) | buys trace lots, builds learnings | learning sales and royalties |
| **Validator** | runs hidden, rotating evals; signs eval-gain attestations; probe-scores trace lots | a fee per attestation + a royalty cut |
| **Consumer** (model owner / agent) | buys or meters learnings | better models |

## 2. Objects
All objects are canonical JSON (sorted keys, no whitespace); **id = `sha256:` of the canonical bytes**. Payloads that
are not public (trace bodies, weights) are stored encrypted off-chain (IPFS / Arweave / HF); only ids, licenses,
parents and settlements touch the chain.

### Trace (`trace/0.1`)
```json
{
  "v": "trace/0.1",
  "task": "extract.flight",
  "base_model": {"name": "needle3", "hash": "sha256:…"},
  "input": "Flight {NUM_1} - {DATE_1} - {CITY_1} ({CODE_1}) Depart {TIME_1} …",
  "model_output":    {"origin_airport_code": "{CODE_2}", "outbound_flight_number": "{NUM_2}"},
  "verified_output": {"origin_airport_code": "{CODE_1}", "outbound_flight_number": "{NUM_1}"},
  "fixed_fields": ["origin_airport_code", "outbound_flight_number"],
  "fixed_by": {"origin_airport_code": "rule", "outbound_flight_number": "model"},   // model | rule | human | unknown
  "slots": {"NUM_1": "flight_number", "CODE_1": "airport_code", "DATE_1": "date", "TIME_1": "time", …},
  "checker": {"id": "flight-rules", "version": "1", "hash": "sha256:…"},
  "privacy": "skeleton",
  "license": {"kind": "shared", "max_licensees": 10, "exclusive_days": 0},
  "producer": "0xProducerAddress",
  "created": "2026-10-03T18:00:00Z"
}
```
**Skeletons.** Every concrete value (names, emails, phones, money, dates, times, codes, numbers, and any value that
appears in either output) is replaced by a typed placeholder, consistently across input and outputs, on the producer's
device. The mapping placeholder → real value never leaves the device. A trainer refills placeholders with synthetic
values to generate as many concrete training pairs as it wants, so the *structure of the failure* is sold, never the
user's data. Nodes reject traces whose text still matches PII detectors.

### Learning (`learning/0.1`)
```json
{
  "v": "learning/0.1",
  "kind": "routing | rule | prompt_patch | decoding | lora | full_finetune | checker | package",
  "task": "extract.flight",
  "base_model": {"name": "needle3", "hash": "sha256:…"},
  "artifact": "sha256:…",              // encrypted weights / patch, off-chain
  "parents": [{"trace": "sha256:…", "weight": 0.0012}, …],   // contribution weights sum to 1
  "trainer": "0xTrainer",
  "attestation": {"validator": "0xVal", "eval_set": "sha256:…", "metric": "field_accuracy",
                  "before": 0.67, "after": 0.79, "sig": "…"},
  "royalty": {"per_call_msats": 20, "split": {"traces": 0.60, "trainer": 0.25, "checkers": 0.10, "validators": 0.05}}
}
```
A learning must cite only traces its trainer holds a license for, and **cannot be listed without an attestation whose
`after` beats `before`**. Contribution weights come from the validator (batch leave-one-out on the probe eval in v0.1;
influence estimates later).

## 3. Market 1 — trace auctions
- Traces are grouped into **lots** by `(task, base_model.name, checker)`; a lot's metadata, a redacted sample and the
  validator's **probe score** (eval gain from a throwaway adapter trained on the lot) are public.
- **Frequent batch auctions:** bids are sealed and the book clears every epoch (default 5 min), so ordering speed buys
  nothing and settlement is one batch.
- **Shared licence (non-rival):** `max_licensees = k`. The top *k* bids at or above the reserve win, and every winner
  pays the **(k+1)-th highest bid** (or the reserve if fewer bids): a uniform-price *k*-unit Vickrey auction, so
  bidding your true value is optimal for unit demand.
- **Exclusive window:** single-unit second-price sealed bid; the winner gets sole access for `exclusive_days`.
- Proceeds per trace: **producer 85% / checker author 10% / validators 5%**.
- The decryption key for a lot is released to winners by escrow after payment clears.

## 4. Market 2 — learnings
- **Exclusive / early access:** second-price sealed bid, same epochs.
- **Metered:** consumers report usage through a payment channel; each call accrues `per_call_msats` (millisatoshis;
  everything is priced in bitcoin, 4e; the retired v0.1 dollar node used `per_call_micros`). Royalties split by the
  learning's `split`, and the `traces` share is divided by parent weights
  (and on to each trace's producer). Learnings can cite other learnings as parents, so royalties propagate up the tree
  with the same rule.

## 4b. Classifier engine, search, and bounties (market 3)
Traces must be correctly categorised and easy to search, and agents must be able to post bounties for problems that
get solved with learnings or packages that are then sold. The classifier engine is the backbone.

- **Every trace is classified on arrival** (`traceex/classify.py`):
  - *Where it belongs:* a path down a versioned task taxonomy (`taxonomy/0.1`: extract > travel > flight,
    extract > commerce > invoice, tool_call > device, code > repair, …). Engines are interchangeable: **TypeSafe Jev**
    hierarchical beam search when a key is present (one request per level, holding a `choice` question for each
    branch still on the beam, beam 2; the method of TypeSafe's hierarchical-classification cookbook; about 0.5 s and
    560 input tokens per level), and a standard-library keyword engine otherwise. A hosted engine that fails, or a
    node's daily request cap, never blocks a submission: the node falls back to the rules engine, records which engine
    filed each trace, and classifies outside its write lock. `POST /v0/admin/reclassify` re-files keyword-filed traces
    once a key is added (a node with a key does this by itself on startup). Measured on the 247 seeded traces: the
    keyword engine misfiled 7 of the 244 code traces (6 MBPP problems under extract > commerce > receipt, 1 under
    code > repair); Jev (jev-1.13.0) re-filed all 247 with 497 requests and 279k input tokens, about 2 requests and
    1,100 tokens a trace, and moved exactly those 7 to code > generate.
  - *How it failed:* read exactly from the placeholders, no model: `type_mismatch` (a code where a number belongs),
    `role_swap` (a value that belongs to another field), `wrong_span`, `invented`, `omission`, `normalised`. The
    signature (`role_swap:2 type_mismatch:1`) is searchable. On the demo traces the dominant failure is role swaps,
    which is exactly what the routing learning fixes.
- **Search** (`GET /v0/search?q=&path=&failure=&model=&kind=&sort=&offset=&format=&facets=`; v0.7): a Leviathan-style
  index (`node/leviathan_search.py`, adapted from Leviathan, https://github.com/elstongun/leviathan, Apache-2.0,
  credited in NOTICE) inside the node's own SQLite database, over traces and registry failures alike (`kind`: trace,
  the default; failure; all). SQLite FTS5 with porter stemming: any word may match, ranked by BM25 with the title
  weighted 2x (text and labels 1x, branch names 0.5x, the fix's code 0.3x), newest first on ties; stopwords are dropped
  and every word re-quoted, so query syntax typed by a person or an agent is inert. The taxonomy branch (one token per
  ancestor, so a branch scopes its subtree) and every filter (failure mode, model or model family, checker, kind,
  failure id) are synthetic tokens in a `tags` column, so scoping and filtering are posting-list intersections inside
  the match, not post-filters. `path` resolves in tiers, exact > case-insensitive > description > key contains >
  description contains > fuzzy (Sorensen-Dice >= 0.6): `code/gen` finds code/generate, `travel` the extract/travel
  subtree, and when several unrelated branches match, the reply is `status: ambiguous_branch` with the candidates and
  no results: it never guesses. When nothing matches inside the branch, cards from its nearest ancestor (then from
  anywhere) come back in `other_branches`, marked OTHER, with a note to say so. `format=text` returns compact cited
  cards ("shown 1-5 of 72", each card's id, branch, failure id and the sentence that matched, then `next: GET
  /v0/traces/<id>`), which is what the MCP search tool returns by default, five at a time. Every index row is written
  in the same transaction as the trace or failure it indexes. `sort=new` (the default without words) is newest first;
  `sort=bounty` puts what feeds the richest open bounty first; `facets=1` adds counts per branch and per failure mode.
  Results come back with the open bounties on the branch and, with words, the top matching failures, so a trainer
  sees supply, demand and the registry together. Measured on the 247 seeded traces with 54 frozen queries (relevance
  labelled from trace metadata only; 47 answerable): **87% top-1 and 96% top-5 from the words alone, 89% / 96% with
  the filters an agent knows to pass**; the v0.6 search scored 70% / 77% and 66% / 66%, and its JSON answers were about
  2.7 times as long (median 946 tokens against 349 for five cards). `GET /v0/taxonomy` gives the tree with trace counts
  per branch.
- **Bounties are refundable pledge escrows, free to post** (the transaction fee only). A bounty names a taxonomy branch
  (optionally a failure mode and base model), a hidden eval set by hash, and the score a solution must reach.
  - *Backing:* anyone pledges sats to it (`POST /v0/bounties/{id}/pledges`; micro-dollars on the retired v0.1 dollar
    node). Each pledge waits in escrow. A pledge buys nothing: no token, no share of the solution's revenue, nothing to
    sell or transfer. Backers pay for the fix, the way a maintainer pays a contractor.
  - *One bounty per problem:* a post whose branch, failure mode and base model (the classifier's key) match an open
    bounty backs that bounty instead of opening a duplicate (the reply says `merged`); a bounty nobody pledges to within
    3 epochs expires and drops out of the default search, so the board shows problems someone will pay to have solved.
  - *Matching:* new traces that match an open bounty are flagged on submission, so producers see where their fixes
    are wanted and trainers see the demand next to the supply in search.
  - *Solving:* a learning claims the bounty when it is accepted by the validator federation and the bounty's poster
    measures it on the hidden eval at the target. The pledges pay **solver 70 / the traces it was built from 20 /
    checkers 5 / validators 5**, through the family tree, vesting 4 epochs so a challenge can still give them back.
    The winning learning (or `package`: code, a tool, a prompt pack) stays on the exchange to be sold and metered, so
    solving one agent's problem keeps earning from every agent with the same problem, for its traces and its trainer.
  - *Unsolved:* at the deadline every backer gets back exactly what it pledged.
  - v0.5's bounty coins (a bonding curve, buy / sell / transfer, and 20% of the solution's revenue to holders) are
    gone in v0.6: a revenue share sold to fund work is a security, and it bought nothing a refundable pledge doesn't.
    `contracts/BountyMarket.sol`, the coin's contract, is removed with it.

## 4c. Open-weight models
The same loop that improves one agent improves the open-weight models everyone shares. A verified fix is exactly
the training example open models lack at scale: a real input, the model's real failure, and an answer a checker
proved right.

- **Two privacy levels.** `skeleton` (the default, for anything personal): every value replaced by a typed
  placeholder on the producer's device. `open` (non-personal domains: code, maths, public documents): the full text
  travels, because that is what a model learns from. Every trace at every level is scanned for keys, tokens and
  passwords; open traces also for emails and phone numbers; skeleton traces for any personal data left. The client
  refuses to send and the node refuses to accept anything that fails.
- **The checker is a verifiable reward.** Unit tests, answer checkers and schema rules say what is wrong without
  knowing the answer, which is what RL with verifiable rewards needs. Traces carry the checker's `feedback` (the
  traceback, the failed assert with what the code returned) and its `failure_modes` (`wrong_answer`,
  `runtime_error`, `wrong_name`, `syntax_error`, `timeout`…), which the classifier files and search filters by.
- **Traces export to the formats trainers use** (`traceex.export`): chat SFT rows (prompt → verified answer), DPO
  pairs (the model's answer rejected, the fix chosen), and self-repair turns (failing answer + feedback → fix), with
  a dataset card crediting every producer and checker. Skeleton traces become trainable through `refill`: synthetic
  values per placeholder type, as many concrete pairs per trace as a trainer wants, never the person's data.
- **Learnings that are weights, or decoding.** Kinds `lora` and `full_finetune` carry a weights hash and location;
  `decoding` carries an inference-time recipe that changes no weights, for example contrastive decoding across the
  recurrent passes of looped transformers (LoopCD, arXiv 2610.02185: Huginn on HumanEval 22.6% → 31.7%). All are
  judged the same way: the validator scores base and learning on the hidden eval and reports the paired result,
  problems newly solved against problems newly broken, with an exact sign test.
- **Open release, and how it still pays.** Weights released openly can't be metered per call, so an open learning
  (`release: "open"`) is paid for differently: a **bounty funds it up front** (backers pledge sats; the pledges pay
  the solver and the traces when the attested weights reach the target, and come back if they don't); **metered uses
  still pay** (inference providers that serve the weights report usage, and those payments are split down the tree);
  **trace lots can be licensed** before release; and the **model card** carries the family tree, so every producer is
  credited wherever the weights go.
- **Iterate.** The adopted model becomes the next producer: its remaining failures become the next lot. Across the
  network that is expert iteration on real deployments, with every round attested on data nobody trained on.
- **Measured (examples/code_repair, Qwen2.5-0.5B-Instruct on MBPP, one RTX 3060):** three agents produced 78
  verified fixes; LoRA v1 trained on them, then base + v1 produced 94 more (the loop's second round) and v2 trained on
  both. On 500 held-out problems the pre-registered first-try score moved +1.8 (v1, p 0.43) and +1.0 (v2, p 0.69), so
  the bounty defined on it before training stays open. Measured afterwards, the agent workflow (first try plus one
  round of checker feedback) went from 29.4% to 37.0% (v1, p 0.0005) and 37.2% (v2, p 0.0002): the fixes taught the
  model to use its checker. Every generation is recorded, so the run replays without a GPU.

## 4d. Agents that use the exchange on their own
The exchange only works if agents use it without a human wiring each step, so the protocol ships the parts that make
an agent a participant by default.

- **MCP server.** `python -m traceex.mcp --node <url> --address <wallet>` (or `traceex-mcp`) gives any MCP-capable
  agent the exchange as tools: `traceex_taxonomy`, `traceex_search`, `traceex_find_learnings`, `traceex_list_bounties`,
  `traceex_post_bounty`, `traceex_back_bounty` (a refundable pledge), `traceex_submit_fix`, `traceex_report_usage`, `traceex_balance`. The
  server's instructions tell the agent when to act: search before giving up, adopt proven learnings, submit every
  verified fix, post a bounty when a failure keeps recurring. It runs on the agent's machine, so fixes become skeletons
  before anything is sent, and bounty pledges are capped by the owner's budget (default 0).
- **Remote MCP and discovery.** Every node also answers MCP at `POST /mcp` (streamable HTTP, JSON responses) and
  describes itself at `/.well-known/trace-exchange.json`. The remote server refuses raw personal text: personal fixes
  must be turned into skeletons on the agent's own machine.
- **Autopilot** (`traceex.autopilot`). Attached to an agent's check loop (`AdaptiveAgent(..., autopilot=…)`), it:
  submits every verified fix; classifies every failure the loop can't fix and searches `GET /v0/learnings` for an
  attested learning on that branch and base model, handing the best untried one back to adopt; counts unresolved
  failures per kind (per field, or the checker's own mode) and, when one keeps recurring with no learning to fix it,
  pledges to a matching open bounty or posts a new one for free (a post the node finds already covered backs the open
  one), keeping the failing cases on the device as the hidden eval (only their hash is published); and never spends
  past the owner's budget.
- **Learning search.** `GET /v0/learnings?path=&model=&kind=&min_gain=` returns attested learnings, biggest measured
  gain first, each with its branch (from the traces it was built from), release, price and artifact;
  `GET /v0/learnings/{id}` returns the whole learning for adoption.
- **Measured (examples/flight_emails, step 6):** a fresh agent on autopilot meets a date format its model can't read.
  On the first failure it finds the routing learning on the exchange and tries it on that email before adopting it
  (`AdaptiveAgent.helps`: the first pass must get more fields right); it doesn't help there, so it isn't adopted. When
  the same four fields keep failing, it posts bounty #2 for them, free, with its two failing emails kept on the device
  as the hidden eval, and pledges $1.00 of its $1.00 budget to it (on the v0.1 dollar node the example runs on). On the
  third email it sees the bounty already stands.

## 4e. The sats economy (testnet v0.6): no token, every payment in sats

There is no network token. Everything is paid directly in sats, and every payout is a split of a real payment. Run a
node with `--economy sats` (`node/sats.py`); the v0.1 dollar node stays in the library for its tests and examples but
is retired as a mainnet path. v0.6 opens a fresh testnet: a node refuses a v0.5 (TXC) database, and any database from
an older node. v0.7 keeps this economy unchanged and opens a v0.6 database as it is, adding the failure registry and
the search index and filing every stored trace in them (4g); its only new money is bonds (a 2,000-sat fix bond, a
1,000-sat reporter bond), held and destroyed under the same rules as the others.

**Why no token.** v0.5 priced everything in sats but paid contributors in a coin, TXC, minted against burned payments.
Its own anti-farming cap made the coin a pass-through: contributors got back 99-100% of what users burned, so the coin
added only price risk, a permanently thin pool, a hoarding failure mode and most of the securities risk. The record
agrees. Revenue-sharing coins sold to fund work meet every prong of the Howey test (the SEC won against LBRY, Kik and
Telegram); protocol-owned liquidity and issuance tied to activity failed elsewhere (Olympus, Bittensor's dTAO, Filecoin's
baseline minting); and the one rule that keeps farming unprofitable is Ocean Protocol's lesson, "fees must exceed
rewards": if payouts are only ever splits of real payments, paying yourself can never return more than you paid. The
owner's decision: "If the txc coin doesn't make sense then it doesn't make sense." v0.6 removes TXC, the pool (AMM),
burn and mint, the time-weighted price, the halvings, the operator's emission share, credits and every part of the fee
re-peg that touched the coin.

The eight rules:

1. **One unit: whole millisatoshis, paid over Lightning with L402.** A call the account can't cover answers
   **402 Payment Required** with `WWW-Authenticate: L402 macaroon="...", invoice="..."` and a JSON body
   `{"error", "l402": {"scheme": "L402", "amount_msats", "amount_sats", "invoice", "payment_hash", "macaroon"}}` for
   exactly what is missing; the client pays the invoice and retries with `Authorization: L402 <macaroon>:<preimage>`.
   The SDK raises `PaymentRequired` with the challenge (`.l402`, `.challenge`). On the testnet the 402 is real but its
   invoice is a placeholder (`invoice_is_placeholder: true`), and each wallet takes **30,000 test sats** once from
   `POST /v0/faucet`. Every amount on the wire is an integer that says its unit: `_msats` or `_sats`. A `_micros`
   amount is refused, so nobody pays in sats believing they paid dollars. Dollars appear only as labelled
   approximations (`_usd_approx`) at the operator's reference bitcoin price (`--btc-usd`, default $85,962, Coinbase
   spot on 2026-10-05); no amount is ever computed from them. No token, no pool, no emission, no treasury: the only
   money on the node is what payers put in.
2. **Use pays out at once.** A learning's seller sets its price (`royalty.per_call_msats`). Each paid use (calls x that
   price) is its own payment, held in its own escrow and split at the epoch's settlement: **traces 60 / trainer 25 /
   checkers 10 / validators 5**, whatever split the trainer asked for. The traces' share goes to the learning's
   parents with equal weight per distinct parent (a copy counts as its original); a learning cited by another passes
   its whole slice through to its own traces and checkers (so wrapping someone's traces in a learning of one's own
   diverts nothing); learnings nest at most 32 deep. The trainer and the validators who agreed with the verdict are
   paid at settlement; the parents' shares wait in the payment's escrow for **4 epochs**, so a challenge can still claw
   them back, to the payer.
3. **Nothing is ever paid out that a payer didn't pay in.** This is the anti-farming guarantee, and it is an invariant
   in code. Every payment (a use, a pledge, a licence) records what its payer paid (`gross`) and the fee taken from it;
   the rest sits in the payment's own escrow account (`pay:<id>`). Every payout is a move out of one payment's escrow,
   and `_disburse()` refuses (`InvariantError`) any move that would take that payment's payouts past `gross - fee` or
   past what the escrow holds; `_split()` checks the shares against what the payment holds before anything moves. The
   ledger is double-entry, so it always sums to zero, and `audit()` re-checks every payment: paid out plus still held
   equals paid in less the fee, never more. `tests/test_sats.py` ends every test with that audit and tries to break it
   (an over-payment, a share function that would mint money, a spend from the burn account): each is refused. So
   paying yourself returns at most what you paid, less every share that isn't yours and the fee.
4. **One fee: 58 msats a transaction** (`exchange.TX_FEE_MSATS`, about $0.00005 at $85,962 a bitcoin), paid at once to
   the operator that served the transaction: its whole income. Every transaction pays it (a trace, a bid, a learning,
   a usage report, a pledge, a validator's stake, commitment or reveal, a challenge, a claim). `examples/fees/measure.py`
   runs each kind of transaction on the reference node and prices its electricity in dollars (CPU at 10 W a busy
   core, bytes moved at 0.02 kWh/GB, bytes stored in three copies, PUE 1.4, $0.15/kWh), then converts at `--btc-usd`.
   The dearest, registering a learning, comes to about $0.0000004 (0.47 msats); the fee is about 123 times that
   (measured 2026-10-05). The margin is the point: it pays the operator and prices out spam at machine scale (a billion
   junk transactions cost 58 million sats, about $50,000). The fee is fixed in sats, so its dollar value floats with
   bitcoin. An operator who wants it held in dollars sets `Params.fee_repeg_epochs = N`: every N epochs the node resets
   it to `fee_target_usd_nanos` ($0.00005) at the bitcoin price the operator sets (`POST /v0/admin/btc-price`), rounded
   to whole msats. It is off by default, because that price is a trusted input. Operator actions (checkers, clearing,
   settling, decoys) don't pay.
5. **Stakes are sats.** Learning bond **5,000 sats**, validator minimum stake **10,000 sats**, challenge stake
   **2,000 sats**; slashing takes sats. **Forfeits are destroyed:** a lost bond, slashed stake or failed challenge moves
   to the burn account (`burn:unspendable`), which no call on the node can spend from (the ledger refuses), and each
   epoch's forfeits are recorded as one batch with a digest (`GET /v0/economy` shows the batches). On the testnet the
   burn account is the end of the road. **Mainnet path:** the node holds forfeits in its Lightning balance until the
   epoch settles, then its settlement transaction pays the batch to an `OP_RETURN` output committing to the batch's
   digest, which no key can ever spend; anyone can check the output's value against the published batch. A validator
   slashed under the minimum loses its seat at once. v0.5's grace period for validators a price drop pushed under the
   minimum is gone: there is no price to drop.
6. **Validation as 4f**: commit-reveal, median minus 2 standard errors, decoys, challenges at any time. v0.6 fixes the
   decoy test's false positives (below).
7. **Bounties are refundable pledge escrows.** Posting is free (the fee only). Anyone pledges sats (`POST
   /v0/bounties/{id}/pledges`), each pledge its own payment in its own escrow. A solve, measured by the poster on its
   hidden eval at the target, pays **trainer 70 / traces 20 / checkers 5 / validators 5** out of each pledge, down the
   learning's family tree, vesting 4 epochs (a challenge gives it back to the backers). Unsolved at the deadline, every
   backer gets back what it pledged (what is left of each pledge, pro rata, if a clawed-back solve already paid some
   out). Backers get **no tradable claim and no revenue share**: v0.5's bounty coins, bonding curve, buy / sell /
   transfer and the 20% holder share are gone. At scale: a post whose taxonomy branch, failure mode and base model (the
   classifier's key) match an open bounty backs that bounty instead of opening a duplicate (the reply says `merged`;
   the existing poster's hidden eval decides the solve), and a bounty nobody pledges to within 3 epochs expires and
   drops out of the default search.
8. **Sellers set prices.** Each learning sets `per_call_msats`; licences to trace lots clear in sats in the batch
   auction (section 3), and the buyer's payment waits in its escrow until the buyer's own learnings cite the traces it
   used (or the buyer names them, `POST /v0/licences/direct`); those traces share it producer 85 / checker 10 /
   validators 5. Junk no buyer uses earns nothing.

**Scale.** Payouts are the same share of what users pay at every size, because they are splits of it.
`examples/scaling/simulate.py` runs one learning on a fresh node at five usage levels, one epoch each, and settles
until every escrowed share is paid:

| usage a day | paid in | traces 60% | trainer 25% | checkers 10% | validators 5% | paid out / paid in |
|---|---|---|---|---|---|---|
| $10 | 11,633 sats | 6,980 | 2,908 | 1,163 | 582 | 0.9999998 |
| $10,000 | 11.6M sats | 6.98M | 2.91M | 1.16M | 582k | 1.0000000 |
| $1M | 1.16B sats | 698M | 291M | 116M | 58.2M | 1.0000000 |
| $100M | 116B sats | 69.8B | 29.1B | 11.6B | 5.82B | 1.0000000 |
| $1B (machine economy) | 1.163T sats | 698B | 291B | 116B | 58.2B | 1.0000000 |

The remainder is a few msats of rounding, refunded to the payer. **The machine economy, 10 billion machines x 100 paid
uses x $0.001 = $1B a day, is 1,163,304,716,037 sats a day (11,633 BTC) at $85,962 a bitcoin**: about 698 billion sats a
day to traces, 291 billion to trainers, 116 billion to checker authors and 58 billion to validators. Operators' fees are
on top: 580 million sats a day if each machine batches its day's calls into one usage report, 58 billion if every call
were its own transaction. There is no price to crash and nothing to hoard; at half or twice today's bitcoin price the
same dollars are twice or half the sats, and every share moves with them.

**Storage at machine scale.** Per-transaction detail (usage reports, finished payments, forfeit rows, ledger rows,
per-address Merkle claims) is kept through the challenge window (`keep_epochs`, 6), then folded: each account's older
ledger rows become one carried-forward row, so balances stay exact to the msat. Every epoch keeps its Merkle root, its
totals and its burn batch for good; traces and learnings stay.

## 4f. Validation by federation, and why farming loses

The rule everything else follows: **only payments pay, and whoever pays judges.** A federation of staked validators
decides which learnings are accepted, and only accepted learnings can be paid for; but no verdict moves money by
itself. Users pay for a learning after trying it on their own data, a bounty's poster measures solutions on its own
hidden eval, and a licence buyer's own learnings decide which traces get its money. So a federation captured by a
majority of stake can still block honest work, but it has nothing to print and nothing to take. And since v0.6 pays
only splits of real payments, there is nothing to print anywhere.

1. **Validators you can't pick.** Each learning gets `quorum` validators (3 on the testnet), drawn by stake-weighted
   rendezvous hashing over a beacon published at the settlement *after* it was submitted, so nobody can grind a
   learning's content for friendly validators.
2. **Commit, then reveal.** Every assigned validator commits `sha256(attestation + salt)` before any reveal opens, so
   nobody can copy another's score (Bittensor's weight-copying problem).
3. **Robust aggregation, as in federated learning.** Each validator measures on its own private held-out data and
   reports the paired standard error of its gain. The median gain counts. *Accepted* if
   `median - 2 x SE(median) >= 1 point`; *inconclusive* if the median clears 1 point but not that bound (bond back
   minus 10%); *rejected* below it (bond forfeited). A claim more than twice the measured gain, beyond the noise on both
   sides, is rejected as an overclaim, so a bribed vote that lifts the median a little still costs the whole bond.
4. **Forfeits are destroyed.** Bonds (5,000 sats), challenge stakes (2,000 sats) and slashed stake go to the burn
   account and the epoch's burn batch (4e, rule 5), never to anyone, so a verdict is never worth buying, or faking, for
   the money it moves.
5. **Validators earn from what they vouch for.** A validator's pay is its 5% of what users pay for the learnings it
   agreed with. Disagreeing is no fault (slashing for it would let a majority punish the honest minority). Stake is
   slashed for not revealing (5%), for agreeing with a gain a challenge round couldn't reproduce (25%), and for scoring
   **decoys** without measuring them (25%): the operator submits learnings whose true gain it has sealed
   (`sha256(gain|salt)`), indistinguishable from real ones until validators reveal. A score further from the truth than
   four of the validator's own standard errors is a **strike**; **two strikes inside 30 epochs cost 25% of its stake**
   (v0.6). v0.5 slashed on the first miss, and an honest measurement lands beyond 4 standard errors about 6 times in
   100,000 (measured: 65 of 1,000,000 simulated honest decoy measurements, 6.5e-5; the normal tail is 6.3e-5), which
   once in 30 runs cost an honest validator 25% of its stake. With two strikes, an honest validator that measures 100
   decoys in a window is slashed with probability 2.1e-5 (v0.5's rule: 6.5e-3), and one that measures 1,000 with
   2.0e-3 (v0.5: 6.3e-2); across the 30-run attack table below, honest validators made 1,015 decoy measurements (about
   508 distinct draws: each honest twin replays its attack's noise) and drew 2 strikes (one draw 4.08 standard errors
   out, seen in both twins) and no slash; under v0.5's rule that draw cost an honest validator 25% of its stake. A validator that never measures repeats the claim, lands far from the
   truth every time, and is slashed on its second decoy. The cost of the fix: a lazy validator drawn for only one decoy
   escapes the slash, so decoys must be frequent enough that every validator meets two (`python
   examples/farming/attacks.py --decoys` prints the rates). A validator slashed under the minimum stake loses its seat
   at once.
6. **Challenges, any time.** Anyone can stake 2,000 sats to challenge an accepted learning; fresh validators re-measure
   it on new eval sets and look at its parents. Upheld: the escrowed shares go back to whoever paid them (users, bounty
   backers), the bond is destroyed, the challenger gets its stake back, and the validators who accepted it lose 25%. An
   audit challenge (padding) sends the parents' escrowed share back to the payers and destroys half the bond. Failed:
   the challenger's stake is destroyed. While a challenge runs, the learning's payouts wait.
7. **Paying yourself loses.** Every payout is a split of a real payment (4e, rule 3), so wash usage, a self-funded
   bounty or licensing one's own traces returns at most what it paid, less the shares that aren't yours (at least the
   validators' 5%) and the fee. That is Ocean Protocol's lesson, "fees must exceed rewards", built in: there are no
   rewards but other people's payments.
8. **Copies and padding earn nothing.** Traces that differ only in placeholder numbering, spacing or case share one
   slot, and so do traces with the same distinctive fix (a verified output of 20+ characters, placeholders aside)
   whose inputs share 30% of their words: a reworded copy. The same weights can't be registered twice, even citing
   the first. Every reveal audits at least 10 parents; if the median audit finds more than 10% junk, the parents'
   share goes back to the payers and half the bond is destroyed.
9. **The standard fee: 58 msats a transaction**, paid to the operator (4e, rule 4). For the attacks that earn nothing
   (spam, copies, stuffing a lot, tiny pledges) it is the whole loss: 1,000 junk traces cost 58 sats.
10. **Users judge.** The SDK's `AdaptiveAgent` tries a learning on its own failing cases before adopting it
    (`helps`), and the MCP instructions tell agents to do the same: an attested gain is where to look, not proof it
    helps you.

`examples/farming/attacks.py` runs each strategy against a real sats node (7 validators staking 15,000 sats each,
quorum 3) with honest neighbours: a producer, a user who pays 10,000 sats for an accepted learning only when it helps
on its own traffic, a watchdog that challenges what it can show is fake, bounty posters and licence buyers. Each
attack that does real work is compared with its honest twin: the same run, where the attacker's real learning draws
the same validation noise. `--seeds 30` runs each on 30 different random draws; `tests/test_sats.py` fails the build if
any stops losing. v0.5's attacks on the token (pumping or dumping the pool, the time-weighted-price lag, cheap credits,
inflating a capped mint, the sats curve, a coin's pump and dump, the operator's emission share, pushing validators
under a price-denominated minimum) no longer apply: there is no token. Re-run for v0.6 on 2026-10-05, in sats, with an
approximate dollar column at $85,962 a bitcoin:

| attack | mean vs honest work, 30 runs | about | best run for the attacker | runs it paid |
|---|---|---|---|---|
| trace spam (1,000 junk traces) | -58 sats | -$0.05 | -58 sats | 0 of 30 |
| 1,000 junk traces stuffed into an honest lot | -58 sats | -$0.05 | -58 sats | 0 of 30 |
| 20 near-copies of honest traces | -1.16 sats | -$0.000997 | -1.16 sats | 0 of 30 |
| 20 reworded copies of honest traces | -1.16 sats | -$0.000997 | -1.16 sats | 0 of 30 |
| fake learning (+30 points claimed, true gain 0) | -5,000 sats | -$4.30 | -5,000 sats | 0 of 30 |
| fake learning, 1 bribed validator | -5,000 sats | -$4.30 | -5,000 sats | 0 of 30 |
| 12 fake learnings, 2 of 7 validators bribed | -66,306 sats | -$57.00 | -40,002 sats | 0 of 30 |
| wash usage: 100,000 sats of one's own usage of one's own learning | -7,501 sats | -$6.45 | -7,501 sats | 0 of 30 |
| self-funded bounty (50,000 sats) solved with one's own learning | -15,000 sats | -$12.89 | -15,000 sats | 0 of 30 |
| duplicate of an honest bounty, to solve and confirm its own copy | -5,000 sats | -$4.30 | -5,000 sats | 0 of 30 |
| refund gaming: a lure bounty, an honest backer's 60,000 sats, 1,000 one-msat pledges | -58.12 sats | -$0.05 | -58.12 sats | 0 of 30 |
| real learning padded with 200 junk parents, honest audits | -2,512 sats | -$2.16 | -2,512 sats | 0 of 30 |
| real learning padded with 200 junk parents, lazy audits | -2,512 sats | -$2.16 | -2,512 sats | 0 of 30 |
| real learning citing a wrapper of one's own (100% to itself) around honest traces | -16.72 sats | -$0.01 | -0.058 sats | 0 of 30 |
| a validator that never measures (6 decoys among 24 learnings) | -3,066 sats | -$2.64 | +450 sats | 3 of 30 |
| challenge griefing: challenges 3 honest learnings | -6,000 sats | -$5.16 | -6,000 sats | 0 of 30 |
| fee evasion: 10,000 fixes in 25 traces, batched usage | -52.73 sats | -$0.05 | -52.73 sats | 0 of 30 |
| sybil: 10 trainers, 10 payers, 5 minimum-stake validators | -4,019 sats | -$3.46 | -3,847 sats | 0 of 30 |
| **4 of 7 validator seats (57% of stake):** fake learnings | -24,201 sats | -$20.80 | -3.60 sats | 0 of 30 |
| 4 of 7 seats: wash usage of its own accepted fake | -5,002 sats | -$4.30 | -5,002 sats | 0 of 30 |
| 4 of 7 seats: claim an honest bounty with a fake | -1.45 sats | -$0.001246 | -1.45 sats | 0 of 30 |
| 4 of 7 seats: block honest work | -2,814 sats | -$2.42 | -1.28 sats | 0 of 30 |
| registry: 20 sybil reporters, no bonds | -1.16 sats | -$0.000997 | -1.16 sats | 0 of 30 |
| registry: 20 bonded sybil reporters (1 -> 21 -> 1 verified reporters) | -20,002 sats | -$17.19 | -20,002 sats | 0 of 30 |
| registry: claim a fix one didn't make (public answers copied) | -2,000 sats | -$1.72 | -2,000 sats | 0 of 30 |
| registry: the same, 1 bribed validator | -2,000 sats | -$1.72 | -2,000 sats | 0 of 30 |
| registry: game a regression to be paid twice for one fix | -2,000 sats | -$1.72 | -2,000 sats | 0 of 30 |
| 4 of 7 seats: mark a failure fixed to take its bounty | -1.74 sats | -$0.001496 | -1.74 sats | 0 of 30 |

The validator that never measures now loses on average but not in every run: in 3 of the 30 runs it was drawn for only
one of the 6 decoys, took one strike and no slash, and kept the 50 sats of GPU time a measurement it never spent (best
run +450 sats). That is the price of not slashing an honest validator for one unlucky draw; the remedy is operator
policy, enough decoys that every validator meets at least two inside each 30-epoch window.

Every attack loses on average. The majority rows are what a federation captured by most of the stake can still do: it
controls the vote, so it can reject honest learnings or claw them back with challenges, and their trainers lose bonds;
it gains nothing by it. That griefing is the remaining limit, and the reason a real network still wants many
independent validators. Not covered on the testnet yet: signatures (the operator relays validator and poster
messages), a public randomness beacon (the testnet's comes from each epoch's payout root; mainnet would use drand),
decoys from someone other than the operator, real Lightning payments (the L402 challenge is issued, but its invoice is
a placeholder and no preimage is checked), and the burn batch's on-chain `OP_RETURN` output (the testnet only records
it).

**v0.7's registry attacks** (the last six rows, re-run on 2026-10-05 with the 23 above unchanged to the sat). Inflating a
failure's frequency with sybils buys nothing without bonds (unbonded reports are listed apart and never counted) and
costs each bonded sybil its whole bond at the re-check (4g). A fix claimed with the public answers copied passes the
node's public repro and nothing else: validators run it on their own cases, the claim is rejected and its 2,000-sat bond
destroyed; one bribed validator doesn't move the median. A solver paid once for a fix can't reopen the failure to be
paid again: fixes can only improve a failure's status (only a model version's re-check, operator-registered and
validator-measured, can say regressed), and a learning is paid at most once per failure. A captured majority can mark
a failure fixed (the record is only as honest as the federation, the same limit as learning verdicts), but the bounty
pays only on its poster's own measurement, so no money moves.

## 4g. The failure registry (v0.7)

A CVE-like registry for model failures: the record of what breaks in the field. It fills itself: every trace is filed
under a canonical failure as it is accepted (`node/registry.py`).

- **One failure, one id.** A failure's key is the classifier's taxonomy branch, the **failure signature** (each fixed
  field with its failure mode, sorted; for a runtime error, the exception class the checker reported first:
  `code:runtime_error/TypeError`) and the **model family** (the organisation dropped, then the leading name and version:
  `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct` are both `qwen2.5`; a trace may name its family). The
  first time a key is seen the failure gets a stable public id, `TXF-<year>-<sequence>`, which never changes and is
  never reused. A trace re-filed under a better branch (a hosted classifier re-filing keyword-filed traces) moves to
  the failure of its new key; both ids stay. The reply to `POST /v0/traces` names the trace's `failure_id`.
- **What a failure holds.** Its branch, signature, failure modes, family and a readable title; its **reproduction
  set**: the skeleton cases of its traces (public: `GET /v0/failures/{id}` lists the 50 newest; anyone with the checker
  can re-run them) and a hidden part (each validator's own private cases of the same failure, of which only pass
  counts are ever published); the checkers that judged its cases; and live counters: **distinct verified reporters**,
  unverified reporters (listed apart, never counted), **occurrences** (distinct cases: a near-duplicate counts as its
  original, a refuted case not at all), raw reports, first and last seen (time and epoch), **growth** (cases from
  verified reporters in the last 3 epochs, minus the 3 epochs before), the **model versions** that hit it, the open
  bounties on it, and its status per model version (4h).
- **Demand.** Frequency (verified reporters, then occurrences) and growth are the demand signal: `GET /v0/failures`
  sorts by `frequency`, `growth`, `bounty` (the richest open bounty) or `new`, and filters by `path` (a subtree),
  `failure` (mode), `model` (a version or a family) and `status`. Bounties attach to failure ids: `POST /v0/bounties`
  with `failure_id` takes the failure's branch and mode, and a second post on the same failure backs the open bounty
  instead of opening a duplicate (`merged`, as in v0.6).
- **Verified reporters.** A reporter counts when the node can hold it to account. On a sats node that is an address
  with a **reporter bond** of 1,000 sats in escrow (`POST /v0/reporters`; refundable: `POST /v0/reporters/withdraw`, and it
  comes back `vest_epochs` later, so a reporter can't dodge a re-check by leaving). On the retired dollar node with
  wallets, an address that opened one (the faucet allows 3 a day per network). Counts follow the original: a copy
  filed by someone else adds no reporter and no case.
- **Re-checks.** Validators re-run a failure's new cases on the base model and say which reproduce (`POST
  /v0/failures/{id}/repro`, operator-relayed until validator messages are signed). A case that a majority of at least
  two validators (on a quorum of 3) finds does not reproduce leaves the counters, with every copy of it, and its
  reporter's **whole bond is destroyed**.
- **What inflating a failure costs.** To add k to a failure's verified reporters, an attacker needs k bonded addresses
  (k x 1,000 sats held in escrow for as long as they are to count), k distinct cases (copies count for the original's
  reporter) and k x 58 msats of fees; every case that a re-check refutes costs its address the whole bond. If a
  fraction q of new reporters' cases is re-checked, inflating by k costs k x q x 1,000 sats in expectation (plus the
  fees and the locked capital); the testnet's watchdog policy re-checks every new reporter's cases (q = 1), which is
  cheap, one model call a case. Measured: 20 bonded sybils lift a failure from 1 to 21 verified reporters, then back
  to 1 at the re-check, for -20,002 sats in every one of 30 runs; unbonded, they never count (-1.16 sats of fees).
  And inflating buys no money: frequency moves no payment, and a bounty pays only on its poster's own measurement.

## 4h. Fix tracking (v0.7)

A fix (a learning, a prompt patch, a tool, or a new model version) claims failure ids, and the node keeps the record of
what it fixed, per model version.

- **Claims.** `POST /v0/fixes {claimant, kind: learning | prompt_patch | tool, claims: [TXF-…], model, learning?,
  artifact?, outputs?}`. `model` is the model version the fix applies to; `learning`, the learning that carries it
  (needed for a bounty to pay; only its trainer may claim with it); `outputs`, the fixed model's answers on the
  claimed failures' public cases. On a sats node a claim pays the fee and holds a **2,000-sat bond**, destroyed if the
  validators find it fixes none of its claims, returned otherwise (and when there were too few cases to call).
- **The public part.** The node runs each claimed failure's public cases through the checkers it can run on the
  claimant's outputs: by default the verified output must match; an operator can register stronger runners in Python
  (never over HTTP: they run code). Public answers can be copied, so the public repro can only hold a fix back
  (`fixed` needs it at 90% when outputs are sent); it never fixes anything by itself.
- **The hidden part.** Validators are drawn for each claim like a learning's: stake-weighted rendezvous hashing over
  the beacon published after the claim, never the claimant. Each commits `sha256(measurement + salt)`, then, once all
  have committed, reveals `{results: {TXF-…: {passed, n}}}` on its own private cases of each failure (`POST
  /v0/fixes/{id}/commits`, `.../reveals`, operator-relayed until signed). The median pass rate decides; fewer than 10
  hidden cases in all is **inconclusive**: recorded in the history, not counted.
- **Status per (failure, model version).** `open` (under 10%), `partly_fixed` (with its pass rate), `fixed` (90% or
  more), `regressed` (it had been fixed, and a later model version fails it). `GET /v0/fixes/{id}` shows each claim's
  public repro, the hidden measurements and its status; `GET /v0/failures/{id}/history` every measurement, in order. A
  failure's own status is the newest verified state of the model: a model version's re-check sets it outright, and a
  fix can only improve it, so a weaker (or sabotaged) claim never hides a stronger one.
- **New model versions.** `POST /v0/models {version, family?, parent?, outputs?}` (operator) re-checks every tracked
  failure of the version's family, open or fixed, the same way (public cases on any outputs sent, validators on the
  hidden part). `GET /v0/models/{version}/report` then lists what it **fixed**, what stayed fixed, what **regressed**,
  what is partly fixed or still open, what is pending, and what got worse.
- **Bounties pay themselves.** A bounty attached to a failure pays when that failure's status flips to `fixed` by a
  validated fix carrying a learning the federation accepted, once the bounty's poster has measured that fix on its own
  hidden eval at the target (`POST /v0/bounties/{id}/measurements {fix, attestation}`, the poster as validator,
  operator-relayed until signed): whichever comes last triggers the payout, and settlement retries any that became
  payable (a learning accepted later, say). Every v0.6 payment rule holds: the pledges split trainer 70 / traces 20 /
  checkers 5 / validators 5 down the learning's family tree, vesting 4 epochs, clawed back to the backers if a challenge
  is upheld, refunded if the bounty ends unsolved; every payout is still a split of a real payment (the invariant is
  unchanged). A learning is paid at most once per failure. A failure fixed upstream by a new model version pays no
  one (there is no learning and no tree): its bounty refunds its backers at the deadline.
- **Limits.** The record is as honest as the validator federation: a majority of the stake can mark failures fixed or
  block real fixes, as it can learnings, but moves no money by it. Validators who measure fixes are drawn from the
  staked federation but not yet paid for that work, nor slashed for not revealing (a round that runs out of time
  settles with the majority that revealed). Repro re-checks are not commit-reveal yet. Validator and poster messages
  are relayed by the operator until they are signed.
- **The seeded preview** (`node/seed.py`, real recorded runs, nothing made up): 247 traces filed under 13 failures;
  LoRA v2 claims the 8 code failures it was built from and the three validators measure it on their own third of the
  held-out problems the base model failed with each failure's mode (first try: 6 partly fixed at about 13%, 2
  inconclusive on 5 and 3 hidden cases); LoRA v1, registered as a model version, re-checks all 10 Qwen2.5 failures
  (8 partly fixed at 13-14%, 2 inconclusive); the flight routing learning claims the 3 flight failures and is
  inconclusive (a few fields per airline); the maintainer's bounty on the TypeError failure stays open (its poster
  measured LoRA v2 at 14% against a 50% target). Runtime-error failures share their hidden cases (the recorded eval
  keeps each problem's mode, not its exception), so they share a pass rate.

## 4i. Ingestion: from tests and telemetry (v0.7)

Two more ways in, both dry-run by default: what they build waits in a local outbox (`.traceex/outbox/`) until a person
reviews it and sends it (`python -m traceex.outbox list | show | send`), or they send it when told to.

- **A. pytest** (`sdk/python/traceex/pytest_plugin.py`, entry point `pytest11`; opt in with `--traceex` or `traceex =
  true` in the ini). A test that failed earlier (in the same session, a previous CI run, or a coding agent's previous
  attempt) and passes now after a code change is a verified fix: the plugin builds a trace (task `code.repair`,
  checker `pytest@1`, failure mode from the exception: AssertionError is `wrong_answer`, NameError / AttributeError /
  ImportError `wrong_name`, SyntaxError `syntax_error`, timeouts `timeout`, anything else `runtime_error`). A test
  that passes again with no change is flaky and makes no trace. `examples/ci/github-action.yml` runs it in CI, keeping
  the failure memory in the repository's Actions cache and uploading the outbox as an artifact (sending is opt in).
  **What leaves the machine** (only with `--traceex-submit` or `outbox send`): the trace JSON as written in the outbox:
  the model name you give (`--traceex-model`, default `unknown`), your address, a timestamp, and skeletons
  (`traceex.codeskel`) of the test's file and test name, the exception type (builtin names kept), the failure message,
  the test function, and the diff of the change (hunks only, no line numbers, at most 80 lines and 5 files), with the
  removed and added lines as the model's output and the verified output. A skeleton replaces every string literal
  (f-strings and bytes too) with `{STR_n}`, drops every comment, replaces every identifier that is not Python
  vocabulary (keywords, builtins, standard-library modules, built-in types' methods, unittest and pytest words) with
  `{ID_n}`, every number outside -10..10 with `{NUM_n}` and every directory with `{DIR}`, consistently across the
  pieces, so `lines[1:]` -> `lines` still reads as the fix. The trace is scanned for keys, tokens, passwords, emails and
  phone numbers before it is written, and dropped if anything is found. **Never sent:** source files, paths, project
  and function names, string contents, comments, the raw failure message, and `.traceex/` itself (which keeps raw
  copies of your files to compute diffs; the plugin writes a `.gitignore` into it).
- **B. OpenTelemetry** (`sdk/python/traceex/otel.py`, `TraceexSpanExporter`, a `SpanExporter`). It reads GenAI
  semantic-convention spans, so it works with any OTel-instrumented agent and the tools that export OTel (Laminar,
  LangSmith, OpenLLMetry, OpenInference: add its processor to their provider). It finds a model call
  (`gen_ai.operation.name` chat / text_completion / generate_content), a check that fails it (an `execute_tool` span,
  a `gen_ai.evaluation.result` event or `gen_ai.evaluation.*` attributes, a span named check / eval / validate /
  verify / test, or `traceex.check.passed`; failed when the evaluation says so, its score is under 0.5, or the span
  ended in error), then the retry the next check passes, and turns the pair into a trace: the prompt's last user
  message as input, the failed and the passing answers as the model's output and the verified output, the failed
  check's message as feedback. Spans are held per trace until the root span ends. `examples/otel_agent/demo.py` runs
  a fake agent through the real OTel SDK. **What leaves the machine** (only with `submit=True` or `outbox send`): the
  task name, the model name, the checker's name, your address, a timestamp, and skeletons (`traceex.skeleton`: every
  name, email, phone number, URL, date, time, amount, code and number replaced by a typed placeholder, consistently)
  of the last user message (at most 4,000 characters), both answers and the checker's message. **Never sent:** system
  prompts, other messages, tool arguments, span and resource attributes, trace and span ids, timings; and a trace
  that still holds personal data or a secret after skeletonizing is dropped.

## 5. Ownership and settlement at near-zero cost
- **On-chain (L2, e.g. Base):** `Registry` (trace and learning ids, owners, licences, parents, attestations) and
  `PayoutDistributor` (one Merkle root of `(address, amount)` per epoch). One transaction per epoch, no matter how many
  millions of calls or bids it settles; payees claim with a Merkle proof whenever they like.
- **Off-chain:** bids, usage metering (payment channels), clearing, and the encrypted payloads. Anyone can recompute
  an epoch's root from the published batch and challenge it.
- **Agents pay over HTTP with L402, in bitcoin over Lightning** (4e): an agent calls a node endpoint, gets
  `402 Payment Required` with `WWW-Authenticate: L402 macaroon="...", invoice="..."` for what it is short, pays the
  Lightning invoice, and retries with `Authorization: L402 <macaroon>:<preimage>`. No accounts to open, no card, no
  chargebacks; a payment settles in seconds and can be a few sats. The testnet issues the challenge with a placeholder
  invoice and funds wallets from its faucet instead.
- **Everything is paid in sats, and there is no token.** Prices, fees, bonds, stakes, pledges and test wallets are
  sats (integer msats on the wire); every payout is a split of a real payment (4e). Each epoch's payouts are one Merkle
  root of `(address, msats)`, and each epoch's forfeits one burn batch, destined for an `OP_RETURN` output on mainnet.
  *Retired:* v0.1's USDC settlement and x402, and v0.5's TXC. The contracts in `contracts/` (`Registry`,
  `PayoutDistributor`, and their `MockUSDC` test token) are the v0.1 on-chain sketch and have not been moved to a
  bitcoin-side settlement yet; v0.5's `BountyMarket.sol` (bounty coins) is removed.

## 6. Abuse and trust
| Threat | Defence |
|---|---|
| Junk or farmed traces | pay-for-improvement (probe scores, attestations), producer deposits slashed for rejected lots, near-duplicate detection on skeleton hashes |
| Poisoned learnings / backdoors | hidden rotating eval sets, multiple validators, canary prompts, trainer deposits; provenance lets everything downstream of a bad trace be pulled |
| Data leakage | skeletons only, on-device; node-side PII detectors; raw traces only with explicit consent, encrypted end to end |
| Eval gaming | evals rotate and stay hidden; attestations name the eval-set hash and expire |
| Licence violations | traces carry the base model; registry blocks closed-model outputs whose terms forbid training competitors |
| Sybil validators | validator deposits; attestations need k-of-n agreement for high-value learnings |
| Inflated failure frequency (sybil reports) | counters count distinct bonded reporters; a copy counts for its original; validators re-run new reporters' cases, and one that doesn't reproduce leaves the counters and destroys its reporter's bond (4g) |
| False fix claims | the public repro can only hold a fix back; validators measure the hidden part on their own cases; a claim that fixes nothing loses its 2,000-sat bond; fixes can only improve a failure's status (4h) |
| Code or prompts leaking through ingestion | skeletons on the producer's machine (code: no strings, comments, project identifiers or paths; text: typed placeholders), a secrets and contact-details scan before anything is written, dry run by default (4i) |

## 7. Reference implementation (this repo)
- `sdk/python/traceex/` — client, skeletoniser, the extract → check → retry loop that produces traces, `adapt`
  (routing learnings, first-pass scoring, attestations, `AdaptiveAgent`), the classifier engine, auction and royalty
  maths, Merkle payouts, `export` (SFT / DPO / repair datasets, refill, dataset and model cards), `mcp`
  (MCP server, stdio and the node's `/mcp`), `autopilot` (agents that use the exchange on their own); v0.7:
  `pytest_plugin` and `codeskel` (ingestion from tests), `otel` (ingestion from OpenTelemetry), `outbox` (review
  before sending).
- `node/` — a reference exchange node (Python stdlib + SQLite) exposing the HTTP API below: `exchange.py` (the API,
  the retired v0.1 dollar node), `sats.py` (every payment in sats, the payout invariant, stakes, the federation;
  v0.7 adds reporter and fix bonds), `registry.py` (v0.7: the failure registry and fix tracking),
  `leviathan_search.py` (v0.7: the search index, adapted from Leviathan), `seed.py`, `validator.py`.
- `examples/otel_agent/` — a fake agent's GenAI spans through the real OpenTelemetry SDK into the exporter;
  `examples/ci/github-action.yml` — the pytest plugin in CI.
- `contracts/` — Solidity 0.8.24+: `Registry.sol`, `PayoutDistributor.sol` (compiles clean with solc 0.8.26; leaf
  layout matches `merkle.py`, checked in tests). v0.1's on-chain sketch, in USDC; not yet moved to bitcoin.
- `examples/flight_emails/` — a flight-email extraction loop as the first producer, end to end (`demo.py`).
- `examples/code_repair/` — open weights: Qwen2.5-0.5B-Instruct on MBPP, unit tests as the checker, tracebacks fed
  back, `produce.py` → `train_lora.py` (LoRA on one RTX 3060) → `evaluate.py` (500 held-out problems, paired sign
  test); every generation recorded so `demo.py` replays the run without a GPU.

**Not yet in v0.1 (deliberately):** exclusive-licence clearing in the node; enforcing that a learning's parents are
licensed to its trainer; probe scores; leave-one-out contribution weights (v0.1 weights a trace by how many focus
fields it taught); real validator signatures (attestations carry a digest); real Lightning payment behind the L402
challenge, and escrowed decryption keys; deposits and slashing; DPO and RL trainers (the export formats are there, the example trains SFT). Known skeleton gap: an output value the model normalised
(`2026-10-15` for "Thursday, October 15, 2026", `1284.4` for "$1,284.40") gets its own placeholder instead of the
input's, so the trainer loses that link; next step is value normalisers per slot type. Possible tie-in: JEV hierarchical search (TypeSafe; open-source
host app `extend-hq/jevbox`) for sorting traces into task lots and helping agents find the learning for a task.

### HTTP API (node)
| Method | Path | Body / result |
|---|---|---|
| POST | `/v0/traces` | trace → `{id, lot, classified, failure_id, bounties}` (rejects secrets, personal data, duplicates) |
| GET | `/v0/traces/{id}` | the trace (an id prefix of 15+ characters works), its branch and its `failure_id` |
| GET | `/v0/lots` | lots with counts, probe score, reserve |
| POST | `/v0/bids` | `{lot, bidder, price_msats (or price_sats), license: shared|exclusive}` |
| POST | `/v0/epochs/clear` | clears all auctions → licences + payouts for the epoch |
| POST | `/v0/learnings` | learning → `{id}` (parents must be licensed to the trainer; attestation must show a gain) |
| POST | `/v0/usage` | `{learning, consumer, calls}` → accrued royalties |
| POST | `/v0/epochs/settle` | → `{epoch, root, payouts}` (what `PayoutDistributor` receives) |
| GET | `/v0/provenance/{id}` | the family tree under a learning |
| GET | `/v0/balances/{address}` | earnings, with Merkle proofs per epoch |
| GET | `/v0/taxonomy` | the task tree with trace counts per branch, and which engine is classifying |
| GET | `/v0/search` | `q`, `path` (resolved in tiers; ambiguous → candidates), `failure`, `model`, `kind` (trace / failure / all), `sort` (relevant / new / bounty), `limit`, `offset`, `format` (json / text cards), `facets` → ranked results, total, `other_branches` (labelled fallback), open bounties, top failures |
| GET | `/v0/failures` | the registry: `path`, `failure`, `model`, `status` filters; `sort` frequency / growth / bounty / new → each failure's counters, status and open bounty |
| GET | `/v0/failures/{id}` | one failure: counters, public repro set, checkers, status per model version, fixes, bounties |
| GET | `/v0/failures/{id}/history` | every measurement of the failure, in order |
| POST | `/v0/failures/{id}/repro` | operator-relayed validator message: `{validator, results: {trace: reproduced?}}` → refuted cases leave the counters |
| POST | `/v0/fixes` | `{claimant, kind, claims, model, learning?, artifact?, outputs?}` → the fix, its public repro and the validators drawn (sats node: 2,000-sat bond) |
| GET | `/v0/fixes`, `/v0/fixes/{id}` | fixes (filter by `failure`, `status`); one fix with each claim's public repro, hidden measurements and status |
| POST | `/v0/fixes/{id}/commits`, `.../reveals` | operator-relayed validator messages: a commitment, then `{measurement: {results}, salt}` |
| POST | `/v0/models` | operator: `{version, family?, parent?, outputs?}` → re-checks every failure of the family |
| GET | `/v0/models`, `/v0/models/{version}/report` | model versions; what a version fixed and regressed (URL-encode a version's `/`) |
| POST | `/v0/bounties/{id}/measurements` | operator-relayed: a failure bounty's poster measured a fix, `{fix, attestation}` → pays if the failure is fixed |
| POST | `/v0/admin/reclassify` | operator: `{only: rules|all, limit?}` → re-file traces with the node's classifier in the background |
| POST | `/v0/bounties` | `{poster, title, path or failure_id, eval_set, target, seed_msats?, failure?, base_model?, epochs?}` → free (the fee only); a post matching an open bounty's branch, failure and model (or its failure id) backs it (`merged`) |
| GET | `/v0/bounties` | `path`, `status` filters; each with `pool_msats` (in escrow), `pledged_msats`, backers, deadline |
| POST | `/v0/bounties/{id}/pledges` | `{backer, msats}` (or `sats`) → a refundable pledge into the bounty's escrow |
| GET | `/v0/bounties/{id}/backers` | what each backer pledged, what the bounty holds |
| POST | `/v0/bounties/{id}/claims` | operator-relayed: `{learning, attestation}` with the poster's own measurement on the hidden eval → the pledges vest to the solver and the tree |

On the retired v0.1 dollar node the same amounts are `_micros` (and a claim needs no poster measurement). Any call a
wallet can't cover answers `402 Payment Required` with an L402 challenge (4e). v0.5's token routes (`/v0/coin`,
`/v0/swap`, `/v0/credits`, `/v0/quote`, `/v0/bounties/{id}/buy|sell|transfer|holders`) answer `410 Gone`. A sats node
(`--economy sats`, sections 4e and 4f) adds:

| Method | Path | Body / result |
|---|---|---|
| GET | `/v0/economy` | where the sats are: paid in, paid out, refunded, fees, escrow (payments, vesting, pledges, bonds, challenge stakes), staked, forfeits destroyed and the burn batches, the rules |
| GET | `/v0/fees` | the standard fee in msats, its approximate dollar value, the re-peg setting, what it has paid the operator |
| POST | `/v0/admin/btc-price` | operator: `{usd_per_btc}` → the reference for approximate dollar figures and the optional fee re-peg |
| GET / POST | `/v0/validators` | the federation; operator-relayed: `{address, stake_msats}` (or `stake_sats`) stakes a validator |
| POST | `/v0/learnings/{id}/commits`, `.../reveals` | operator-relayed until signed: a validator's commitment, then its attestation and salt |
| POST | `/v0/learnings/{id}/challenges` | `{challenger}` → stakes 2,000 sats; fresh validators re-measure |
| GET | `/v0/learnings/{id}/verdict` | who was drawn, their reveals, the outcome |
| POST | `/v0/licences/direct` | operator-relayed: `{lot, buyer, traces}` → the buyer's licence payment to the traces it used |
| POST | `/v0/decoys`, `/v0/decoys/unseal` | operator: `{learning, digest, funder}`, then `{learning, gain, salt}` → a strike for each validator far from the truth; a second strike in 30 epochs costs 25% of stake |
| POST | `/v0/reporters`, `/v0/reporters/withdraw` | v0.7: `{address}` → a 1,000-sat reporter bond, so its reports count as a verified reporter's; withdrawn, it returns after 4 epochs |
| GET | `/v0/reporters/{address}` | the bond, whether it counts, its cases and how many were refuted |
