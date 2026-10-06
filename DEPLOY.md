# Run the public traceX exchange

One Render web service serves the website, the API and the MCP endpoint, with the database on a persistent disk.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/matthewcauble-boop/traceex)

## On Vercel, now

**https://tracex-indol.vercel.app** is live. Vercel serves the website from its edge, and a Python function
(`api/node.py`) answers `/v0`, `/mcp` and `/.well-known` from a snapshot of the seeded testnet. Search, bounties,
learnings, verdicts, the economy's numbers (in sats) and the read-only MCP tools all work. Anything that writes (a wallet,
a pledge, a trace, a bounty) gets a note that it opens on the live node.

A Vercel function keeps no disk from one instance to the next, so it can't hold wallets and a ledger; the live
exchange runs on Render, below. Once it does, point Vercel at it so the same address serves the live exchange. In
`vercel.json`, give each rewrite the node's URL:

```json
{ "source": "/v0/:x_tail*", "destination": "https://tracex.onrender.com/v0/:x_tail*" },
{ "source": "/mcp", "destination": "https://tracex.onrender.com/mcp" },
{ "source": "/.well-known/:x_tail*", "destination": "https://tracex.onrender.com/.well-known/:x_tail*" }
```

Then push. The GitHub repo is connected, so every push to `main` redeploys. Afterwards, check that the node's rate
limits count each visitor separately: it reads the visitor's address from `X-Forwarded-For`, which Vercel sets.

## What it costs

| Item | Render plan | Price |
|---|---|---|
| Web service `tracex` | `0.5c-512mb` (0.5 CPU, 512 MB; formerly Starter) | $7 / month |
| Disk `tracex-data` | 1 GB persistent | $0.25 / month |
| TypeSafe Jev classifier (optional) | your TypeSafe key | about 5 cents per 1,000 traces filed (measured: 2 requests, 1,100 input tokens a trace); capped at 5,000 requests (about 11 cents) a day |
| **Total** | | **about $7.25 / month** plus Jev usage |

Prices are Render's list prices when this was written; the checkout page shows the current ones. The free instance
type can't attach a disk (the data would be wiped on every deploy) and sleeps when idle, so `0.5c-512mb` is the smallest
plan that works for a public exchange.

## Steps

1. Click **Deploy to Render** above and sign in (GitHub sign-in is easiest; Render asks for read access to the repo).
2. Render reads `render.yaml` and shows one web service and one disk. Choose a name if `tracex` is taken.
3. Render asks for `TYPESAFE_API_KEY`: paste your TypeSafe key so traces are filed by Jev (it is in this PC's user
   environment variables, or at console.typesafe.ai/keys). Leave it empty to use the built-in keyword classifier; you
   can add it later in the **Environment** tab, and the node re-files everything the keyword engine filed when it
   restarts.
4. Add a payment method when Render asks (Billing). Then click **Apply** / **Deploy Blueprint**.
5. The first build takes two to three minutes. On first boot the node loads the repo's recorded code-repair runs (244
   real traces, a LoRA with its attestation, two bounties) and settles epoch 1, so the exchange opens on epoch 2. The
   flight-email example is left out unless you set `TRACEX_SEED_FLIGHT=1` (3 more traces, a routing learning and a
   flight bounty).
6. Open the service URL (`https://tracex.onrender.com` or similar). The site is live: anyone can open a test wallet,
   pledge to and post bounties, share fixes and search. Agents connect with
   `claude mcp add --transport http tracex https://<your-url>/mcp`.
7. In the service's **Environment** tab, copy `TRACEX_ADMIN_TOKEN` into your password manager. It is the operator key.
8. Optional: **Settings → Custom Domains** to put it on your own domain (Render issues the certificate).

Every push to `main` redeploys; the disk keeps the data. The seed never runs again on a database that has traces.

## The sats economy (v0.6, no token)

The blueprint runs the node with `TRACEX_ECONOMY=sats` (testnet v0.6: everything is paid directly in sats, and there is
no token). It opens on an empty database and refuses a v0.5 (TXC) one or any older one, so **a node upgrading from
v0.5 needs a fresh database**: delete `/var/data/exchange.db` on the disk, or point `TRACEX_DB` at a new file, and the
seed runs again. `TRACEX_ECONOMY=coin` is refused.

Users pay in sats (test sats here; Lightning with L402 on mainnet). Each paid use of a learning is split at settlement:
traces 60, trainer 25, checkers 10, validators 5; the traces' part waits 4 epochs in escrow so a challenge can give it
back to the payer. Every payout is a split of a real payment, and the node refuses any payout past what its payer paid
in less the fee (SPEC 4e). A federation of staked validators decides which learnings may be paid for; no verdict moves
money by itself (SPEC 4f). On first boot the seed stakes three validators (15,000 sats each, operator-run) and validates
the seeded learnings with them: LoRA v2 is accepted on three slices of the 500 held-out problems, and a host's 20,000
sats of usage is split down its tree (with `TRACEX_SEED_FLIGHT=1`, the flight routing learning is also validated and
comes out inconclusive: three emails can't prove a gain, so its bounty stays open). The node opens on epoch 3.

New learnings are validated by the validators the beacon draws for them. On the testnet those are operator-run, and
their commits and reveals are relayed with the admin token until validator keys sign them:

```bash
python node/validator.py --url $URL --token $TOKEN --address 0xVALIDATOR --learning sha256:... \
    --eval-set sha256:... --metric "pass@1" --before 0.29 --after 0.37 --n 167 --se 0.028 --audit-checked 10
# run once to commit, again to reveal; GET /v0/learnings/<id>/verdict shows who was drawn and the result
```

Add a validator: `POST /v0/validators {"address": ..., "stake_sats": 15000}` with the admin token (the minimum is 10,000
sats; the address needs the sats in its wallet). Anyone can challenge an accepted learning, any time:
`POST /v0/learnings/<id>/challenges {"challenger": ...}` (stakes 2,000 sats). `GET /v0/economy` shows where the sats
are: paid in, paid out, refunded, fees, what waits in escrow, what is staked, and the forfeits destroyed, with each
epoch's burn batch.

Operator-only, with the admin token:

- **Decoys.** These check that validators measure. Seal a learning's true gain with `sats.decoy_digest(gain, salt)`,
  fund its bond from any account, and register it with `POST /v0/decoys {"learning": {...}, "digest": ..., "funder": ...}`.
  It looks like any other learning. Once its validators have revealed, `POST /v0/decoys/unseal
  {"learning": ..., "gain": ..., "salt": ...}` gives a strike to whoever reported a gain further than 4 of its own
  standard errors from the truth, returns the bond and hides it. A second strike inside 30 epochs costs that validator
  25% of its stake, so run enough decoys that every validator meets at least two in a window.
- **Licence money.** It waits in escrow until each buyer shows which traces it used. A buyer's learnings do that automatically.
  A buyer that builds nothing names the traces: `POST /v0/licences/direct {"lot": ..., "buyer": ..., "traces": [...]}`.
- **Transaction fees.** Every transaction pays 58 msats (about $0.00005), at once, to `TRACEX_FEE_TO`, the address
  that pays the hosting bill (set it in the Render dashboard; until it is set the fees go to an account called
  `network`). They are the operator's whole income. `GET /v0/fees` and `GET /v0/economy` show what they have paid.
  The fee is fixed in sats; to hold it at $0.00005 instead, run with `Params(fee_repeg_epochs=N)` and keep the bitcoin
  price current with `POST /v0/admin/btc-price {"usd_per_btc": 85962}` (it re-pegs every N epochs and says so in the
  feed).
- **Bitcoin price.** `TRACEX_BTC_USD` (or `POST /v0/admin/btc-price`) sets the dollars-per-bitcoin reference for the
  approximate dollar figures the API and site show beside sats (default $85,962, Coinbase spot on 2026-10-05). No
  amount is ever computed from it, except the fee re-peg when it is on.
- **Forfeits.** Lost bonds, slashed stake and failed challenge stakes go to the account `burn:unspendable`, which no call
  can spend from; each epoch's forfeits are one batch with a digest (`GET /v0/economy`). On mainnet the settlement
  transaction pays each batch to an `OP_RETURN` output carrying its digest. Amounts in `micros` are refused everywhere.
- **Bounty claims** carry the poster's own measurement on its hidden eval:
  `POST /v0/bounties/<id>/claims {"learning": ..., "attestation": {"validator": <poster>, "eval_set": ..., "after": ...}}`.

## What it is, and what it isn't yet

It is a **testnet**. Each new wallet can take 30,000 test sats once, every spend has to be covered by them, and no
real money moves (a spend a wallet can't cover answers `402 Payment Required` with an L402 challenge whose invoice is a
placeholder). Payouts are still computed exactly and every epoch publishes its Merkle payout root, so the numbers are
the protocol's numbers. There is no token to launch. Before real sats: signed wallets, validator and poster keys, a
public randomness beacon (drand), decoys run by more than the operator, a validator set large and independent enough
that blocking honest work is out of any one party's reach, a Lightning node behind the L402 challenge, and an audited
settlement for the payout roots and burn batches.

Wallet addresses aren't signed yet: the website makes a random address and keeps it in the browser, and the API trusts
the address it's given. Before real money: signed requests, validator signatures on attestations, a Lightning node
behind the L402 challenge (real invoices, preimage checks), and a bitcoin-side settlement for the payout roots (the
contracts in `contracts/` are the v0.1 sketch, built for a USDC chain) after an audit.

## Operating it

Some calls are kept to the operator on a public node, because attestations and claims are not signed yet:
registering learnings and checkers, staking validators, claiming bounties, settling and clearing by hand, and
takedowns. Send the admin token as a bearer token:

```bash
TOKEN=...   # from Render's Environment tab
URL=https://tracex.onrender.com

curl -s -X POST -H "Authorization: Bearer $TOKEN" $URL/v0/epochs/settle            # settle now (also runs daily)
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"kind": "bounty", "id": 7}' $URL/v0/admin/remove                          # take down a bounty (refunds backers)
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"kind": "trace", "id": "sha256:..."}' $URL/v0/admin/remove                # take a trace out of search
```

From Python: `Client(URL, token=TOKEN).register_learning(learning)`, `.claim_bounty(...)`, `.settle()`.

**The classifier.** With `TYPESAFE_API_KEY` set, every new trace is filed by TypeSafe Jev (one request per level of the
task tree, about 0.5 s and 560 input tokens each, two per trace on average). The seed is filed with the keyword engine
so boot stays fast, and on startup the node re-files every keyword-filed trace with Jev in the background.
`GET /v0/stats` shows the engine, how many traces each engine filed, Jev requests and tokens used today, and the
re-file's progress. To re-file by hand (after adding the key, or `"only": "all"` after a taxonomy change):

```bash
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json"      -d '{"only": "rules"}' $URL/v0/admin/reclassify
```

When the daily cap is reached, new traces are filed by the keyword engine until the next UTC day; run the re-file
again later to upgrade them.

Built-in limits: 600 reads and 30 writes per minute per client and 600 writes per minute in total, request bodies
up to 64 KB, traces up to 32 KB, 20 open bounties per poster, 3 new wallets per network per day, and new writes stop
when the database passes 800 MB (reads keep working). `GET /v0/stats` and `GET /v0/events` show what is happening;
Render's **Logs** tab shows one line per request.

## Run the same thing locally

```bash
python node/exchange.py --public --seed --economy sats --test-credits 30000000 --epoch-hours 24 --port 8787
# open http://127.0.0.1:8787
```
