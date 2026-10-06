"""HTTP client for an exchange node (spec section 7). Standard library only."""
import json
import urllib.error
import urllib.parse
import urllib.request

from .skeleton import find_pii, find_secrets, find_open_risks


def privacy_leaks(trace):
    """The check both the client (before sending) and the node (before accepting) run. Secrets are refused at every
    privacy level; skeleton traces must also be free of personal data; open traces of emails and phone numbers."""
    text = "\n".join([trace.get("input", ""), *map(str, trace.get("model_output", {}).values()),
                      *map(str, trace.get("verified_output", {}).values()), *trace.get("feedback", [])])
    if trace.get("privacy") == "open":
        return find_open_risks(text)
    return find_secrets(text) + find_pii(text)


class Client:
    def __init__(self, endpoint: str, address: str = None, timeout: float = 30, token: str = None):
        """token: the operator's admin token, for the calls a public node keeps to its operator (settle, clear,
        register learnings, claim bounties)."""
        self.endpoint, self.address, self.timeout, self.token = endpoint.rstrip("/"), address, timeout, token

    def _call(self, method, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.endpoint + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code == 402:   # L402: a Lightning invoice and a macaroon; a Lightning-backed client pays and retries
                raise PaymentRequired(json.loads(e.read() or b"{}"), e.headers.get("WWW-Authenticate", ""))
            raise RuntimeError(f"{e.code}: {e.read().decode()[:300]}")

    def submit(self, trace):
        leaks = privacy_leaks(trace)
        if leaks:
            raise ValueError(f"refusing to send: trace still contains {leaks[:3]}")
        return self._call("POST", "/v0/traces", dict(trace))

    def lots(self):
        return self._call("GET", "/v0/lots")

    def bid(self, lot, price_micros=0, license="shared", *, price_msats=0):
        """A sealed bid: price_msats on a sats node, price_micros on the retired v0.1 dollar node."""
        price = {"price_msats": int(price_msats)} if price_msats else {"price_micros": price_micros}
        return self._call("POST", "/v0/bids", dict(price, lot=lot, bidder=self.address, license=license))

    def clear(self):
        return self._call("POST", "/v0/epochs/clear", {})

    def register_learning(self, learning):
        return self._call("POST", "/v0/learnings", dict(learning))

    def report_usage(self, learning_id, calls):
        return self._call("POST", "/v0/usage", {"learning": learning_id, "consumer": self.address, "calls": calls})

    def settle(self):
        return self._call("POST", "/v0/epochs/settle", {})

    def taxonomy(self):
        return self._call("GET", "/v0/taxonomy")

    def search(self, q="", path="", failure="", model="", limit=20, sort="", offset=0, facets=False):
        """sort: "relevant" (default with words), "new" (default without), or "bounty"."""
        qs = urllib.parse.urlencode({k: v for k, v in dict(q=q, path=path, failure=failure, model=model, limit=limit,
                                                           sort=sort, offset=offset, facets=int(facets)).items() if v})
        return self._call("GET", f"/v0/search?{qs}")

    def post_bounty(self, *, title, path, eval_set, target, seed_micros=0, failure="", base_model="", epochs=4,
                    seed_msats=0):
        """Free to post (the transaction fee only). A bounty is a refundable pledge escrow: seed_msats (a sats node) or
        seed_micros (the dollar node) makes the poster's first pledge. A post matching an open bounty's branch, failure
        and model backs that bounty instead (the reply says `merged`)."""
        seed = {"seed_msats": int(seed_msats)} if seed_msats else ({"seed_micros": seed_micros} if seed_micros else {})
        return self._call("POST", "/v0/bounties", dict(seed, poster=self.address, title=title, path=path,
                                                       eval_set=eval_set, target=target, failure=failure,
                                                       base_model=base_model, epochs=epochs))

    def pledge(self, bounty_id, msats=0, *, micros=0):
        """Pledge to a bounty: msats on a sats node (micros on the retired v0.1 dollar node). Refunded in full if the
        bounty ends unsolved; a solve pays it to the solver and the traces. No token, no share, nothing to trade."""
        amount = {"msats": int(msats)} if msats else {"micros": int(micros)}
        return self._call("POST", f"/v0/bounties/{bounty_id}/pledges", dict(amount, backer=self.address))

    def find_learnings(self, path="", model="", kind="", min_gain=0.0, limit=20):
        qs = urllib.parse.urlencode({k: v for k, v in dict(path=path, model=model, kind=kind, min_gain=min_gain,
                                                           limit=limit).items() if v})
        return self._call("GET", f"/v0/learnings?{qs}")

    def learning(self, learning_id):
        return self._call("GET", f"/v0/learnings/{learning_id}")

    def describe(self):
        return self._call("GET", "/.well-known/trace-exchange.json")

    def backers(self, bounty_id):
        return self._call("GET", f"/v0/bounties/{bounty_id}/backers")

    def bounties(self, path="", status="open"):
        qs = urllib.parse.urlencode({k: v for k, v in dict(path=path, status=status).items() if v})
        return self._call("GET", f"/v0/bounties?{qs}")

    def claim_bounty(self, bounty_id, learning_id, attestation=None):
        """On a sats node the claim needs the bounty poster's own measurement on its hidden eval (`attestation`, signed
        by the poster as `validator`), unless the learning already carries it."""
        body = {"learning": learning_id}
        if attestation:
            body["attestation"] = attestation
        return self._call("POST", f"/v0/bounties/{bounty_id}/claims", body)

    def provenance(self, object_id):
        return self._call("GET", f"/v0/provenance/{object_id}")

    def balance(self, address=None):
        return self._call("GET", f"/v0/balances/{address or self.address}")

    # --- the sats economy (nodes run with --economy sats, v0.6) --------------------------------------------------------
    def economy(self):
        """Where the sats are: paid in, paid out, refunded, fees, escrow, stakes, forfeits destroyed. No token."""
        return self._call("GET", "/v0/economy")

    def fees(self):
        return self._call("GET", "/v0/fees")

    def validators(self):
        return self._call("GET", "/v0/validators")

    def stake(self, msats):
        """Stake sats as a validator (at least 10,000 sats on the testnet; operator-relayed until signed)."""
        return self._call("POST", "/v0/validators", {"address": self.address, "stake_msats": int(msats)})

    def verdict(self, learning_id):
        return self._call("GET", f"/v0/learnings/{learning_id}/verdict")

    def commit(self, learning_id, digest, round=None):
        return self._call("POST", f"/v0/learnings/{learning_id}/commits",
                          {"validator": self.address, "digest": digest, "round": round})

    def reveal(self, learning_id, attestation, salt, round=None):
        return self._call("POST", f"/v0/learnings/{learning_id}/reveals",
                          {"validator": self.address, "attestation": attestation, "salt": salt, "round": round})

    def challenge(self, learning_id):
        return self._call("POST", f"/v0/learnings/{learning_id}/challenges", {"challenger": self.address})

    def direct_licence(self, lot, traces):
        """Send this buyer's licence money for a lot to the traces it used (learnings it registers do this for it)."""
        return self._call("POST", "/v0/licences/direct", {"lot": lot, "buyer": self.address, "traces": list(traces)})

    def register_decoy(self, learning, digest, funder):
        """Operator: a learning whose true gain is sealed (sats.decoy_digest); unseal it once validators have revealed."""
        return self._call("POST", "/v0/decoys", {"learning": dict(learning), "digest": digest, "funder": funder})

    def unseal_decoy(self, learning_id, gain, salt):
        return self._call("POST", "/v0/decoys/unseal", {"learning": learning_id, "gain": gain, "salt": salt})

    def faucet(self, address=None):
        """On a testnet node: open a wallet with test money (30,000 test sats on a sats node; no real money)."""
        return self._call("POST", "/v0/faucet", {"address": address or self.address})

    def wallet(self, address=None):
        return self._call("GET", f"/v0/wallets/{address or self.address}")

    def stats(self):
        return self._call("GET", "/v0/stats")

    def events(self, limit=30):
        return self._call("GET", f"/v0/events?limit={int(limit)}")


class PaymentRequired(Exception):
    """402 from a node: `terms["l402"]` holds the Lightning invoice (a placeholder on the testnet), the amount in msats
    and the macaroon. Pay the invoice, then retry the call with `Authorization: L402 <macaroon>:<preimage>`."""

    def __init__(self, terms, challenge=""):
        super().__init__(f"payment required: {terms}")
        self.terms, self.challenge = terms, challenge
        self.l402 = (terms or {}).get("l402") or {}
