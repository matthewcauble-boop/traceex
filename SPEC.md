# traceX — the trace exchange protocol, spec v0.1

**The goal is self-improving open agents.** An agent running any open model should get better from its own mistakes,
and from everyone else's, without a lab in the loop. traceX is the protocol for that loop, plus the market that
pays for it: any model's **verified fixes** become owned, tradable training data, and the **learnings** built from
them earn per-use royalties that flow back to everyone whose work went into them.

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
  "royalty": {"per_call_micros": 20, "split": {"traces": 0.60, "trainer": 0.25, "checkers": 0.10, "validators": 0.05}}
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
- **Metered:** consumers report usage through a payment channel; each call accrues `per_call_micros` (1 micro =
  $0.000001 USDC). Royalties split by the learning's `split`, and the `traces` share is divided by parent weights
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
- **Search** (`GET /v0/search?q=&path=&failure=&model=&sort=&offset=&facets=`): SQLite FTS5 over skeleton text,
  path, signature, model and task, filtered by taxonomy branch and failure mode. `sort=relevant` (the default with
  words) ranks by BM25 with a match in the branch name weighted 4x, the failure label 3x, model and task 2x, the text
  1x; when every word together finds nothing, any word will do. `sort=new` (the default without words) is newest
  first; `sort=bounty` puts traces that feed the richest open bounty first. `facets=1` adds counts per branch and per
  failure mode, so a browse tree can be built from one call. Results come back with the open bounties on the same
  branch, so a trainer sees supply and demand together. `GET /v0/taxonomy` gives the tree with trace counts per
  branch.
- **Bounties are free to post and each one mints a coin.** Buying the coin stakes the bounty, and everyone who holds
  it shares in the solution. A bounty names a taxonomy branch (optionally a failure mode and base model), a hidden eval set by
  hash, and the score a solution must reach.
  - *Backing:* anyone buys the bounty's coin on a linear bonding curve (`price = $0.01 + $0.0001 × supply`); every
    dollar goes into the bounty's pool. Early backers pay less per coin, so they hold a bigger share of the solution's
    revenue. While the bounty is open, holders can sell coins back for what they paid for them, never more (any profit
    there could only come out of later backers' money); unsolved at the deadline, the pool goes back to the backers by
    what each put in. Coins transfer freely at any time (they carry what they cost), so a coin can be sold on, holder
    to holder, when its value rises.
  - *Matching:* new traces that match an open bounty are flagged on submission, so producers see where their fixes
    are wanted and trainers see the demand next to the supply in search.
  - *Solving:* a learning claims the bounty when its validator attestation is on that same eval set and reaches the
    target. The pool pays **solver 70 / the traces it was built from 20 / checkers 5 / validators 5**, through the
    family tree. The winning learning (or `package`: code, a tool, a prompt pack) stays on the exchange to be sold
    and metered, so solving one agent's problem keeps earning from every agent with the same problem.
  - *The coin becomes a share of the solution:* **20% of every metered use of the winning learning goes to the
    coin's holders**, pro rata, paid at each settlement to whoever holds the coins then. In the demo, backers Kim,
    Raj and Lee split $0.20 of the first $1.00 of usage.
  - *Unsolved:* at the deadline the pool is returned to holders pro rata.
  - On-chain: `contracts/BountyMarket.sol` (curve buy / sell with slippage limits, transfer, solve, per-coin
    revenue accrual that survives transfers, withdraw, expire, redeem). Compiles clean with solc 0.8.26; not yet
    exercised on a test chain.

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
  (`release: "open"`) is paid for differently: a **bounty funds it up front** (backers buy the coin; the pool pays
  the solver and the traces when the attested weights reach the target); **metered uses still pay** (inference
  providers that serve the weights report usage, and those royalties flow down the tree with the coin holders'
  20%); **trace lots can be licensed** before release; and the **model card** carries the family tree, so every
  producer is credited wherever the weights go.
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
  `traceex_post_bounty`, `traceex_back_bounty`, `traceex_submit_fix`, `traceex_report_usage`, `traceex_balance`. The
  server's instructions tell the agent when to act: search before giving up, adopt proven learnings, submit every
  verified fix, post a bounty when a failure keeps recurring. It runs on the agent's machine, so fixes become skeletons
  before anything is sent, and coin purchases are capped by the owner's budget (default 0).
- **Remote MCP and discovery.** Every node also answers MCP at `POST /mcp` (streamable HTTP, JSON responses) and
  describes itself at `/.well-known/trace-exchange.json`. The remote server refuses raw personal text: personal fixes
  must be turned into skeletons on the agent's own machine.
- **Autopilot** (`traceex.autopilot`). Attached to an agent's check loop (`AdaptiveAgent(..., autopilot=…)`), it:
  submits every verified fix; classifies every failure the loop can't fix and searches `GET /v0/learnings` for an
  attested learning on that branch and base model, handing the best untried one back to adopt; counts unresolved
  failures per kind (per field, or the checker's own mode) and, when one keeps recurring with no learning to fix it,
  backs a matching open bounty or posts a new one for free, keeping the failing cases on the device as the hidden eval
  (only their hash is published); and never spends past the owner's budget.
- **Learning search.** `GET /v0/learnings?path=&model=&kind=&min_gain=` returns attested learnings, biggest measured
  gain first, each with its branch (from the traces it was built from), release, price and artifact;
  `GET /v0/learnings/{id}` returns the whole learning for adoption.
- **Measured (examples/flight_emails, step 6):** a fresh agent on autopilot meets a date format its model can't read.
  On the first failure it finds the routing learning on the exchange and tries it on that email before adopting it
  (`AdaptiveAgent.helps`: the first pass must get more fields right); it doesn't help there, so it isn't adopted. When
  the same four fields keep failing, it posts bounty #2 for them, free, with its two failing emails kept on the device
  as the hidden eval, and backs it with $1.00 of its $1.00 budget. On the third email it sees the bounty already
  stands.

## 4e. The coin economy (testnet v0.4)

Contributors earn a network coin, TXC; everything is paid for in credits, which hold their dollar value; the market
prices the coin. Run a node with `--economy coin` (`node/coin.py`); the v0.1 dollar economy stays the default for the
library and its tests. v0.4 opens a fresh testnet: a node refuses a v0.3 database.

- **Two units.** *TXC* is the asset: rewards, stakes, bonds and bounty pools. It has 18 decimals (`UNIT = 10**18`):
  a credit's worth of TXC at $1,000,000,000 a TXC is 10^-15 TXC, so 15 decimals is the least that pays every credit
  there, and 18 leaves a thousand times headroom. *Credits* are what everything is paid in: 1 credit = $0.000001, whole
  numbers, the same at every TXC price, so a machine always knows what a call costs. Credits are made only by burning
  TXC: dollars (test dollars here, USDC on mainnet) buy TXC from the pool and burn it in the same step, one credit per
  micro-dollar (`POST /v0/credits`, or automatically when a payment needs topping up), or a holder burns TXC at the
  lower of the pool's spot and reference prices. Credits can't be moved between accounts and never turn back into TXC
  or dollars. Every TXC amount is stored as TEXT (SQLite integers stop at 2^63) and travels in JSON as a decimal
  string; every key ending `_units` is one.
- **Burn and mint.** Usage, licences and the transaction fee are paid in credits, and spent credits are gone. Each
  epoch mints at most a fixed emission: 50,000 TXC, halving every 180 epochs five times, then 1,562.5 TXC an epoch for
  good. 90% goes to the people whose work was paid for that epoch (on the protocol's split, below; a solved bounty's
  coin holders first take 20% of its learning's usage), 10% to the node operator for the transactions it served,
  each by its share of the credits burned on its work.
- **The anti-farming cap.** Nobody is minted more TXC than the credits burned on their own work were worth at the
  epoch's *time-weighted* pool price (never the spot), and never more than the TXC those credits burned at the
  pool's price before its spread. Emission nobody earned is never minted and never rolls over. So paying yourself
  returns at most what you burned, less every share that isn't yours, the spread, and the fee.
- **What the price does.** Burning pushes the price up to where an epoch's burns match its emission, P* = credits
  burned / work emission, and there the network mints exactly what it burns: TXC follows the *rate* of paid usage and
  the halvings, not the sum of everything ever paid. Below P* the emission is over-subscribed and shared out pro rata,
  so wash usage earns less than it costs; above P* the cap binds. The protocol never pushes the price down (that would
  take minting TXC nobody earned, which is exactly what farming wants); only holders selling does.
  `examples/scaling/simulate.py` runs v0.3 and v0.4 side by side: usage growing from $10,000 to $1B a day, a 90%
  crash and a recovery, and a machine economy at $1B a day from day one (10 billion machines x 100 paid uses x
  $0.001). When contributors sell what they earn, v0.4 tracks P* to within 2% through growth ($711k a TXC at $1B a
  day after five halvings, $81k after the crash, where v0.3 compounds to $2.2T), and contributors realise about 100% of
  what users pay. If contributors hoard instead, *both* versions climb far above P*, because nobody sells into the
  burn; that is holders' choice and harmless to payers (credits still cost $1 per million), and 18 decimals keep every
  amount payable at the prices reached ($10T a TXC is 10^-5 dollars per base unit). `--top-up` shows an option the
  node does *not* implement: minting unearned emission into the protocol-owned pool, which holds the price near P*
  even when contributors hoard, at the cost of diluting holders with TXC nobody earned.
- **The pool.** A constant-product pool (Uniswap v2 maths) holds protocol-owned liquidity from genesis: 1,000,000 TXC
  beside $10,000, so TXC opens at $0.01. It keeps a 0.3% spread on each swap; nobody collects it. Prices are kept as
  micro-dollars per TXC times 10^12, exact from a billionth of a dollar to beyond a trillion dollars a TXC. The
  time-weighted price accumulates price x milliseconds between every change to the reserves; each settlement fixes the
  epoch's TWAP, which becomes the *reference price* for the next.
- **Dollar-priced, whatever TXC does.** Bonds ($5), the validator minimum stake ($10), challenge stakes ($2) and the
  bounty curve (first coin $0.01, each one sold adds $0.0001, as on the dollar node) are priced in dollars and charged
  in TXC at the reference price, so they keep their real cost at any TXC price. Backing a bounty with dollars buys TXC
  on the way in and counts for no more than that TXC is worth at the reference price (pumping the pool first buys no
  extra coins). A validator the price pushes under the minimum keeps its seat for 4 epochs to top up, so a dump can't
  empty the federation and hand every seat to whoever stakes right after it.
- **One standard transaction fee: $0.00005** (`exchange.TX_FEE_NANOS` = 50,000 nano-dollars). Every transaction (a
  trace, a swap, making credits, a backing or sale, a bid, a learning, a usage report, a validator's commitment or
  reveal, a challenge, a claim) pays it, in credits, burned; it is the operator's claim on its 10% of the emission.
  `examples/fees/measure.py` runs each kind of transaction on the reference node and prices its electricity: CPU at
  10 W a busy core, bytes moved at 0.02 kWh/GB, bytes stored in three copies (traces and learnings for ten years,
  everything else through the challenge window), PUE 1.4, $0.15/kWh. The dearest, registering a learning, comes to
  about $0.0000004; the fee is about 125 times that. The margin is the point: the fee funds the network and prices out
  spam at machine scale (a billion junk transactions cost $50,000, not $400). 100x to 200x the measured electricity
  keeps both jobs. Fees accrue per account and are billed each epoch in whole credits; the fraction carries over. An
  account can't spend the dollars that cover what it owes, so nobody can drain a wallet before the fee bill. Operator
  actions (checkers, clearing, settling, decoys) don't pay.
- **Storage at machine scale.** Per-transaction detail (usage reports, payments, burns, ledger rows, per-address
  Merkle claims) is kept through the challenge window (`keep_epochs`, 6), then folded: each account's older ledger
  rows become one carried-forward row, so balances stay exact to the unit. Every epoch keeps its Merkle root and total
  for good; traces and learnings stay.
- **The protocol's split.** Usage goes traces 60, trainer 25, checkers 10, validators 5, with equal weight per
  distinct parent (a copy counts as its original), whatever split the trainer asked for. A learning cited by another
  learning takes no cut of its own there: its slice passes through to its own traces and checkers, so wrapping
  someone's traces in a learning of one's own diverts nothing. The trainer's and validators' mint is paid at once; the
  parents' part vests over four epochs, so an audit challenge can claw it back. Learnings nest at most 32 deep.
- **Licences.** A licence is paid in credits and burned at clearing; the credits count as burned on the work of the
  traces its buyer shows it used: the parents its own learnings cite, once validators are done with them, or a list it
  sends (`POST /v0/licences/direct`). Those traces share it (producer 85, checker 10, validators 5).
- **Bounties** stay free to post. The pool holds TXC; the curve is priced in dollars. Selling back returns the TXC the
  coins cost, never more, and an unsolved bounty refunds its backers by the TXC each put in, so nobody can cash out
  later backers' money (the same rule holds on the dollar node and in `contracts/BountyMarket.sol`). A bounty pays when
  its poster measures an accepted learning on the bounty's hidden eval at the target; the pool vests to the solver and
  down the family tree (trainer 70, traces 20, checkers 5, validators 5).

## 4f. Validation by federation, and why farming loses

The rule everything else follows: **only payments pay, and whoever pays judges.** A federation of staked validators
decides which learnings are accepted, and only accepted learnings can earn; but no verdict moves money by itself.
Users pay for a learning after trying it on their own data, a bounty's poster measures solutions on its own hidden
eval, and a licence buyer's own learnings decide which traces get its money. So a federation captured by a majority
of stake can still block honest work, but it has nothing to print and nothing to take.

1. **Validators you can't pick.** Each learning gets `quorum` validators (3 on the testnet), drawn by stake-weighted
   rendezvous hashing over a beacon published at the settlement *after* it was submitted, so nobody can grind a
   learning's content for friendly validators.
2. **Commit, then reveal.** Every assigned validator commits `sha256(attestation + salt)` before any reveal opens, so
   nobody can copy another's score (Bittensor's weight-copying problem).
3. **Robust aggregation, as in federated learning.** Each validator measures on its own private held-out data and
   reports the paired standard error of its gain. The median gain counts. *Accepted* if
   `median - 2 x SE(median) >= 1 point`; *inconclusive* if the median clears 1 point but not that bound (bond back
   minus 10%); *rejected* below it (bond burned). A claim more than twice the measured gain, beyond the noise on both
   sides, is rejected as an overclaim, so a bribed vote that lifts the median a little still costs the whole bond.
4. **Forfeits burn.** Bonds ($5 of TXC), challenge stakes ($2 of TXC) and slashed stake go to nobody, so a verdict is
   never worth buying, or faking, for the money it moves.
5. **Validators earn from what they vouch for.** A validator's pay is its 5% of what the learnings it agreed with go
   on to earn. Disagreeing is no fault (slashing for it would let a majority punish the honest minority). Stake is
   slashed for not revealing (5%), for agreeing with a gain a challenge round couldn't reproduce (25%), and for scoring
   a **decoy** without measuring it (25%): the operator submits learnings whose true gain it has sealed
   (`sha256(gain|salt)`), indistinguishable from real ones until validators reveal; whoever is further from the truth
   than four of its own standard errors never measured it. Decoys catch validators who repeat the claim.
6. **Challenges, any time.** Anyone can stake $2 of TXC to challenge an accepted learning; fresh validators re-measure it
   on new eval sets and look at its parents. Upheld: unvested rewards stop (bounty pools go back to their backers,
   escrowed royalties burn), the bond burns, the challenger gets its stake back, and the validators who accepted it
   lose 25%. An audit challenge (padding) takes the parents' share and half the bond. Failed: the challenger's stake
   burns.
7. **Paying yourself loses.** Nobody is minted more than the credits burned on their own work were worth at the
   epoch's time-weighted price, nor more than the TXC those credits burned (4e), so wash usage, a self-funded bounty or
   licensing one's own traces returns at most what it burned, less the shares that aren't yours, the spread and the
   fee; below the equilibrium price it returns much less (Ocean Protocol's wash-consume lesson: fees must exceed rewards).
8. **Copies and padding earn nothing.** Traces that differ only in placeholder numbering, spacing or case share one
   slot, and so do traces with the same distinctive fix (a verified output of 20+ characters, placeholders aside)
   whose inputs share 30% of their words: a reworded copy. The same weights can't be registered twice, even citing
   the first. Every reveal audits at least 10 parents; if the median audit finds more than 10% junk, the parents' share
   is withheld (burned) and half the bond burns.
9. **The standard fee: $0.00005 a transaction, in credits, burned** (4e), about 125 times the electricity of the
   dearest transaction. For the attacks that earn nothing (spam, copies, stuffing a lot) it is the whole loss: 1,000
   junk traces cost $0.05. The operator is minted for the fees it collects, never more than they burned, so
   self-dealing to claim more of its 10% loses too.
10. **Users judge.** The SDK's `AdaptiveAgent` tries a learning on its own failing cases before adopting it
    (`helps`), and the MCP instructions tell agents to do the same: an attested gain is where to look, not proof it
    helps you.

`examples/farming/attacks.py` runs each strategy against a real coin node (7 validators, quorum 3) with honest
neighbours: a producer, a user who pays $10 for an accepted learning only when it helps on its own traffic, a watchdog
that challenges what it can show is fake, bounty posters and licence buyers. Each attack is compared with its honest
twin: the same run, where the attacker's real learning draws the same validation noise. `--seeds 30` runs each on 30
different random draws; `tests/test_coin.py` fails the build if any stops losing. Strategies 13 and 14 came from an
independent red-team pass; both paid before the fixes in 4e and 4f.

| attack | mean vs honest work, 30 runs | best run for the attacker | runs it paid |
|---|---|---|---|
| trace spam (1,000 junk traces) | -$0.05 | -$0.05 | 0 of 30 |
| 1,000 junk traces stuffed into an honest lot | -$0.05 | -$0.05 | 0 of 30 |
| 20 near-copies, or reworded copies, of honest traces | -$0.001 | -$0.001 | 0 of 30 |
| fake learning (+30 points claimed, true gain 0) | -$5.13 | -$5.13 | 0 of 30 |
| fake learning, 1 bribed validator | -$4.98 | -$0.53 | 0 of 30 |
| 12 fake learnings, 2 of 7 validators bribed | -$67.24 | -$45.58 | 0 of 30 |
| wash usage ($100 of one's own usage), price above equilibrium | -$8.54 | -$8.54 | 0 of 30 |
| self-funded bounty solved with one's own learning | -$13.88 | -$13.88 | 0 of 30 |
| real learning padded with 200 junk parents (honest or lazy audits) | -$2.57 | -$2.57 | 0 of 30 |
| 13. real learning citing a wrapper of one's own (100% to itself) around honest traces | -$0.05 | -$0.03 | 0 of 30 |
| 14. pump and dump: back one's own bounty first, sell into an honest backer's $60 | -$0.01 | -$0.01 | 0 of 30 |
| a validator that never measures (6 decoys among 24 learnings) | -$6.65 | -$3.46 | 0 of 30 |
| **4 of 7 validator seats (57% of stake):** fake learnings | -$30.87 | -$15.46 | 0 of 30 |
| 4 of 7 seats: wash usage of its own accepted fake | -$16.47 | -$7.05 | 0 of 30 |
| 4 of 7 seats: claim an honest bounty with a fake | -$0.08 | -$0.08 | 0 of 30 |
| 4 of 7 seats: block honest work | -$2.48 | -$0.001 | 0 of 30 |
| *v0.4:* wash usage below equilibrium ($2,000 of honest usage that epoch) | -$97.03 | -$97.03 | 0 of 30 |
| *v0.4:* wash usage while the price climbs 23% above its TWAP | -$5.76 | -$5.76 | 0 of 30 |
| *v0.4:* pump the pool, then burn held TXC for cheap credits | -$1.01 | -$1.01 | 0 of 30 |
| *v0.4:* dump 300,000 TXC to hold the price 41% down for an epoch, pay itself, buy back | -$23.17 | -$23.17 | 0 of 30 |
| *v0.4:* operator sends 2,000 transactions of its own for more of the 10% | -$0.0003 | -$0.0003 | 0 of 30 |
| *v0.4:* fee evasion: pack 10,000 fixes in 25 traces, batch usage, drain before the bill | -$0.08 | -$0.08 | 0 of 30 |
| *v0.4:* back one's own bounty with the pool pumped (dollar curve) | -$0.52 | -$0.52 | 0 of 30 |
| *v0.4:* a large holder dumps 300,000 TXC into the pool at once | -$720.53 | -$720.53 | 0 of 30 |
| *v0.4:* sybil: 10 trainers, 10 payers, 5 minimum-stake validators | -$7.72 | -$7.51 | 0 of 30 |
| *v0.4:* dump to push honest validators under the minimum, stake 3 of its own | -$695.60 | -$687.83 | 0 of 30 |

The v0.4 rows are the attacks the new design invites. Two needed fixes before they lost: the time-weighted price alone
let a payer whose credits were made as the price climbed be minted more TXC than it burned (the TWAP lags the spot),
so the cap is also bounded by the TXC the credits burned; and a dump that pushed every honest validator under the
dollar minimum handed all 12 seats on its fakes to the attacker's 3 fresh validators (+$2,364), so a validator the
price pushes under keeps its seat for 4 epochs to top up (now 1 of 12 seats, all 4 fakes rejected). Backing a bounty
with dollars into a pumped pool also counts only what the TXC is worth at the reference price.

Decoys are spot checks: a validator that is never drawn for one keeps its savings (about one run in 30 at this decoy
rate), but on average it loses. The last row is what a majority of stake can still do. It controls the vote, so it can
reject honest learnings or claw them back with challenges, and their trainers lose bonds ($20.61 for four learnings in
the default run); it gains nothing by it. That griefing is the remaining limit, and the reason a real network still
wants many independent validators. Not covered on the testnet yet: signatures (the operator relays validator and
poster messages), a public randomness beacon (the testnet's comes from each epoch's payout root; mainnet would use
drand), and decoys from someone other than the operator.

## 5. Ownership and settlement at near-zero cost
- **On-chain (L2, e.g. Base):** `Registry` (trace and learning ids, owners, licences, parents, attestations) and
  `PayoutDistributor` (one Merkle root of `(address, amount)` per epoch). One transaction per epoch, no matter how many
  millions of calls or bids it settles; payees claim with a Merkle proof whenever they like.
- **Off-chain:** bids, usage metering (payment channels), clearing, and the encrypted payloads. Anyone can recompute
  an epoch's root from the published batch and challenge it.
- **Agents pay over HTTP with x402** (USDC on Base): an agent calls a node endpoint, gets `402 Payment Required` with
  the price, pays, and retries. No accounts or invoices.
- **No exchange-wide token in v0.1.** Prices, deposits and payouts are USDC. The only coins are per-bounty coins
  (section 4b), priced and redeemed in USDC.

## 6. Abuse and trust
| Threat | Defence |
|---|---|
| Junk or farmed traces | pay-for-improvement (probe scores, attestations), producer deposits slashed for rejected lots, near-duplicate detection on skeleton hashes |
| Poisoned learnings / backdoors | hidden rotating eval sets, multiple validators, canary prompts, trainer deposits; provenance lets everything downstream of a bad trace be pulled |
| Data leakage | skeletons only, on-device; node-side PII detectors; raw traces only with explicit consent, encrypted end to end |
| Eval gaming | evals rotate and stay hidden; attestations name the eval-set hash and expire |
| Licence violations | traces carry the base model; registry blocks closed-model outputs whose terms forbid training competitors |
| Sybil validators | validator deposits; attestations need k-of-n agreement for high-value learnings |

## 7. Reference implementation (this repo)
- `sdk/python/traceex/` — client, skeletoniser, the extract → check → retry loop that produces traces, `adapt`
  (routing learnings, first-pass scoring, attestations, `AdaptiveAgent`), the classifier engine, auction and royalty
  maths, bounty coins, Merkle payouts, `export` (SFT / DPO / repair datasets, refill, dataset and model cards), `mcp`
  (MCP server, stdio and the node's `/mcp`), `autopilot` (agents that use the exchange on their own).
- `node/` — a reference exchange node (Python stdlib + SQLite) exposing the HTTP API below.
- `contracts/` — Solidity 0.8.24+: `Registry.sol`, `PayoutDistributor.sol` (compiles clean with solc 0.8.26; leaf
  layout matches `merkle.py`, checked in tests).
- `examples/flight_emails/` — a flight-email extraction loop as the first producer, end to end (`demo.py`).
- `examples/code_repair/` — open weights: Qwen2.5-0.5B-Instruct on MBPP, unit tests as the checker, tracebacks fed
  back, `produce.py` → `train_lora.py` (LoRA on one RTX 3060) → `evaluate.py` (500 held-out problems, paired sign
  test); every generation recorded so `demo.py` replays the run without a GPU.

**Not yet in v0.1 (deliberately):** exclusive-licence clearing in the node; enforcing that a learning's parents are
licensed to its trainer; probe scores; leave-one-out contribution weights (v0.1 weights a trace by how many focus
fields it taught); real validator signatures (attestations carry a digest); x402 payment and escrowed decryption
keys; deposits and slashing; DPO and RL trainers (the export formats are there, the example trains SFT). Known skeleton gap: an output value the model normalised
(`2026-10-15` for "Thursday, October 15, 2026", `1284.4` for "$1,284.40") gets its own placeholder instead of the
input's, so the trainer loses that link; next step is value normalisers per slot type. Possible tie-in: JEV hierarchical search (TypeSafe; open-source
host app `extend-hq/jevbox`) for sorting traces into task lots and helping agents find the learning for a task.

### HTTP API (node)
| Method | Path | Body / result |
|---|---|---|
| POST | `/v0/traces` | trace → `{id, lot, classified, bounties}` (rejects secrets, personal data, duplicates) |
| GET | `/v0/lots` | lots with counts, probe score, reserve |
| POST | `/v0/bids` | `{lot, bidder, price_micros, license: shared|exclusive}` |
| POST | `/v0/epochs/clear` | clears all auctions → licences + payouts for the epoch |
| POST | `/v0/learnings` | learning → `{id}` (parents must be licensed to the trainer; attestation must show a gain) |
| POST | `/v0/usage` | `{learning, consumer, calls}` → accrued royalties |
| POST | `/v0/epochs/settle` | → `{epoch, root, payouts}` (what `PayoutDistributor` receives) |
| GET | `/v0/provenance/{id}` | the family tree under a learning |
| GET | `/v0/balances/{address}` | earnings, with Merkle proofs per epoch |
| GET | `/v0/taxonomy` | the task tree with trace counts per branch, and which engine is classifying |
| GET | `/v0/search` | `q`, `path`, `failure`, `model`, `sort` (relevant / new / bounty), `limit`, `offset`, `facets` → ranked traces, total, facets, open bounties on that branch |
| POST | `/v0/admin/reclassify` | operator: `{only: rules|all, limit?}` → re-file traces with the node's classifier in the background |
| POST | `/v0/bounties` | `{poster, title, path, eval_set, target, seed_micros?, failure?, base_model?, epochs?}` → free; mints the coin |
| GET | `/v0/bounties` | `path`, `status` filters; each with pool, supply, current coin price |
| POST | `/v0/bounties/{id}/buy` | `{buyer, micros}` → coins on the curve |
| POST | `/v0/bounties/{id}/sell` | `{seller, coins}` → back for what they cost, while open |
| POST | `/v0/bounties/{id}/transfer` | `{from, to, coins}` |
| GET | `/v0/bounties/{id}/holders` | holders, pool, supply, price |
| POST | `/v0/bounties/{id}/claims` | `{learning}` → paid if the attestation is on the bounty's eval set and meets the target; on a coin node, `{learning, attestation}` with the poster's own measurement |

A coin-economy node (`--economy coin`, sections 4e and 4f) adds:

| Method | Path | Body / result |
|---|---|---|
| GET | `/v0/coin` | price, supply, burns, vesting, licence money waiting, stake, rules |
| POST | `/v0/swap` | `{account, side: buy|sell, amount}` → test dollars for TXC and back, at the pool's price |
| GET | `/v0/quote` | `side`, `amount` → what a swap would get now |
| GET / POST | `/v0/validators` | the federation; operator: `{address, stake_units}` stakes a validator |
| POST | `/v0/learnings/{id}/commits`, `.../reveals` | operator-relayed until signed: a validator's commitment, then its attestation and salt |
| POST | `/v0/learnings/{id}/challenges` | `{challenger}` → stakes 200 TXC; fresh validators re-measure |
| GET | `/v0/learnings/{id}/verdict` | who was drawn, their reveals, the outcome |
| POST | `/v0/licences/direct` | operator-relayed: `{lot, buyer, traces}` → the buyer's licence money to the traces it used |
| POST | `/v0/decoys`, `/v0/decoys/unseal` | operator: `{learning, digest, funder}`, then `{learning, gain, salt}` → slashes validators who never measured |
