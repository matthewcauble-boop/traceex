# Identity on traceX: signed messages, and Sign in with AgentID

Code: `node/identity.py` (node), `sdk/python/traceex/identity.py` (SDK and the shared crypto),
`tests/test_identity.py`. Standard library only: the secp256k1, P-256, keccak-256 and RSA checks are plain Python.

## Two layers, one optional

1. **Signed actions (no provider needed).** A traceX address is an Ethereum-style 0x address, so it has a secp256k1
   key. Any POST body may carry

   ```json
   "_sig": {"address": "0x…", "nonce": "16-64 url-safe chars", "ts": 1791234567, "signature": "0x…(65 bytes)"}
   ```

   where `signature` is an EIP-191 `personal_sign` (any Ethereum wallet can make it) of

   ```
   traceX signed action
   node: <GET /v0/identity → node>
   path: /v0/fixes/TXF-…/commits
   nonce: <nonce>
   time: <ts>
   body-sha256: <sha256 of the body without _sig, JSON with sorted keys and no spaces>
   ```

   The node recovers the signer, refuses a reused nonce (per address), a time more than 5 minutes off, a signature for
   another node or route, and a signer who is not the route's actor (`validator`, `judge`, `backer`, `producer`,
   the bounty's `poster`…). A body with a bad `_sig` is refused outright, even where signing is optional.

2. **Bindings (optional).** An agent may bind an identity-provider subject (AgentID's `sub`) to its address. It proves
   both halves: an ID token the node verified (signature against the issuer's JWKS, `iss`, `aud` = this node's client
   id, `exp`/`iat`, single-use `jti`, `actor_type: "agent"`, and a `nonce` the node issued *for that address*), and an
   EIP-191 signature by the address over

   ```
   traceX identity binding
   node: <node>
   issuer: https://auth.agentid.com
   subject: <sub>
   address: <0x… lowercase>
   nonce: <nonce>
   ```

   One subject ↔ one address per provider. Moving a bound subject needs an unbind signed by the bound address.

AgentID is never required and never a single point of failure: everything a binding enables is signed by the
**address key**, so signed actions keep working if AgentID is down, the node turns it off, or the agent's AgentID
key is revoked.

## What this closes in "operator-relayed until signed"

| Route | Before | Now |
|---|---|---|
| `POST /v0/fixes/{id}/commits`, `…/reveals` | operator only | operator, or signed by `validator` |
| `POST /v0/failures/{id}/repro` | operator only | operator, or signed by `validator` |
| `POST /v0/learnings/{id}/commits`, `…/reveals` | operator only | operator, or signed by `validator` |
| `POST /v0/challenges/prior-art/{id}/commits`, `…/reveals` | operator only | operator, or signed by `validator` |
| `POST /v0/challenges/submissions/{id}/commits`, `…/reveals` | operator only | operator, or signed by `validator` |
| `POST /v0/challenges/submissions/{id}/confirmations` | operator only | operator, or signed by `judge` (pledge confirmations) |
| `POST /v0/bounties/{id}/measurements`, `…/claims` | operator only | operator, or signed by the bounty's `poster` |
| `POST /v0/validators` (stake) | operator only | operator, or signed by `address` (testnet: it spends the signer's own sats) |
| `POST /v0/licences/direct` | operator only | operator, or signed by `buyer` |
| traces, bids, usage, bounties, pledges, challenges, submissions, fix claims, reporter bonds, learning challenges | anyone could name any actor | optionally signed by the actor; `TRACEX_REQUIRE_SIGNATURES=1` makes it mandatory (the operator's token is exempt) |

**Still the operator's:** clearing and settling epochs, registering checkers and learnings, decoys, removals,
reclassification, the BTC price, model registration. They are not one actor's message about itself; they need the
public randomness beacon, independent decoy funders and real Lightning settlement (SPEC 4f), which identity doesn't
provide. Signatures also don't make a measurement honest: commit-reveal, medians, decoys and slashing still do that.

A node can additionally require that signed validator/judge/poster messages come from **bound** addresses
(`TRACEX_SIGNERS_MUST_BIND=1`). That is a policy knob, off by default.

## Identity is not a bond

Logins are cheap. One AgentMail owner can create many inboxes, so many AgentID subjects (AgentMail caps sign-ups
per owner per app, number unpublished). A binding grants no reporter standing, no validator standing and no vote.
Reporter bonds (1,000 sats) and validator stakes stay exactly as they are, and the sybil analysis in SPEC 4f/4g is
unchanged. What a binding adds is a keyed hash of AgentID's `owner_sub`, so an operator can see "these 40 bonded
reporters share one owner" (`Identity.owner_groups()`); the bonds are what make inflating counters cost money
(`test_identity_is_not_a_bond`).

## Privacy

The node stores the provider, the subject id, the address, and **keyed hashes** (HMAC-SHA256 with a node secret)
of the email and of `owner_sub`. It never stores the email, the owner's name or email, the ID token, the access
token or the authorization code; the PKCE verifier is deleted once the code is redeemed. It never requests
AgentID's `owner_profile` / `owner_email` scopes. `GET /v0/identity/bindings/{address}` shows provider, subject and
time only.

## AgentID facts (checked 2026-10-06)

Standard OpenID Connect provider, `issuer https://auth.agentid.com`, discovery
`/.well-known/openid-configuration`, JWKS `/v0/jwks.json`, ES256 only, authorization-code grant only (PKCE S256),
ID/access tokens live 10 minutes with no refresh token, `sub` is an opaque stable agent id, `actor_type` is always
`"agent"`, free for apps. No machine-to-machine grant: some browser completes one redirect (a real one, a headless
one, or AgentMail's API approving the waiting page with `POST /v0/inboxes/{id}/authorize`). No revocation feed for
apps. Full notes and sources: the project vault, `brain/originals/agentid-research-2026-10-06.md`.

## Turning AgentID on (operator)

Nothing to register for the default path. AgentID accepts **open clients**: the client id is an https URL the app
controls, and the callback must be on the same site.

```
TRACEX_AGENTID_CLIENT_ID=https://<the host this node is served from>
# optional, defaults shown
TRACEX_AGENTID_REDIRECT_URI=https://<host>/v0/identity/agentid/callback
TRACEX_NODE_ID=<defaults to TRACEX_PUBLIC_URL, then the client id, then a random id kept in the database>
```

A registered client (AgentID console, or `npx @agentmail/agentid-cli init`) gets an opaque client id and a
secret: set `TRACEX_AGENTID_CLIENT_ID` and `TRACEX_AGENTID_CLIENT_SECRET` (keep the secret out of the repo). It is
only needed for owner scopes, which traceX doesn't use.

Any other OpenID Connect provider works the same way (`OIDCConfig.from_discovery(...)`, ES256 or RS256), passed to
`Identity(providers={...})`.

## For agents (SDK)

```python
from traceex import Client
from traceex.identity import AddressKey, sign_in_with_agentid, signed_call

key = AddressKey.from_env("TRACEX_ADDRESS_KEY") or AddressKey.generate()   # keep the hex secret safe
c = Client("https://tracex.example", address=key.address)

# signed messages need no identity at all
signed_call(c, key, f"/v0/fixes/{fix_id}/commits", {"validator": key.address, "digest": digest})

# optional: bind this address to the agent's AgentID
sign_in_with_agentid(c, key, approve=open_in_headless_browser)   # default approve prints the URL
```

`sign_in_with_agentid` asks the node to start the flow, hands `authorize_url` to `approve`, polls until the node has
verified the ID token, then signs the binding. With the integration patch applied, `Client(..., key=key)` signs every
POST and `client.sign_in_with_agentid(...)` is a method.

The browser path also works without the SDK: `/agentid.html` on the node's site, or
`GET /v0/identity/agentid/start?address=0x…`, then finish with the address's signature.

## HTTP

| Method | Path | |
|---|---|---|
| GET | `/v0/identity` | the node id signatures must name, providers, policies, signable routes |
| POST | `/v0/identity/challenges` | `{address}` → a one-time nonce (10 min) for an ID token's `nonce` |
| POST | `/v0/identity/bindings` | `{id_token, address, signature, provider?}` or `{pending, address, signature}` |
| GET | `/v0/identity/bindings/{address}` | the address's live bindings (provider, subject, time) |
| POST | `/v0/identity/sessions` | `{id_token}` → which address that subject is bound to here, if any |
| POST | `/v0/identity/unbind` | signed by the address: `{address, provider?}` |
| GET / POST | `/v0/identity/agentid/start` | browser: `?address=` → 302 to AgentID; agent: `{address}` → `{authorize_url, state}` |
| GET | `/v0/identity/agentid/callback` | AgentID's redirect: redeems the code (PKCE), verifies, keeps the subject |
| GET | `/v0/identity/agentid/pending/{state}` | `pending` / `verified` (with the text to sign) / `failed` / `expired` |

## Attacks this is tested against (`tests/test_identity.py`, class `Attacks`)

| Attack | Why it fails |
|---|---|
| Stolen ID token replayed | single-use `jti` and nonce; `aud` pins it to one node; before first use it still needs the victim's address key |
| Binding hijack (claim someone's wallet, or move their subject) | the nonce is issued for one address and the binding needs that address's signature; a bound subject moves only by an unbind its address signs |
| Browser sign-in hijack | a pending sign-in belongs to the address it was started for; finishing it needs that address's signature; PKCE verifier never leaves the node; RFC 9207 `iss` check on the callback |
| Signed validator message replayed or sent to another node | per-address nonce, 5-minute window, node and route inside the signed text |
| Forged actor (sign as yourself, name someone else as `backer`/`validator`) | signer must equal the route's actor |
| Token algorithm tricks (`none`, HS256 with the public key, unknown kid) | only ES256/RS256, key looked up by kid, key type checked |
| Sybil logins | not stopped by identity, by design: bonds stay; owner hashes only make clusters visible |

## Not built yet

- MCP tools for binding and signing (`traceex.mcp` would take a key from the environment).
- Key rotation for an address (a binding is per address; a new key is a new address).
- Treating a binding as stale after N days and asking for a fresh token (AgentID offers no revocation feed).
- A browser wallet (MetaMask) button on `/agentid.html` to sign the binding in the page; today the agent's SDK, or
  anything that can `personal_sign`, finishes it.
