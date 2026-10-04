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
            if e.code == 402:   # x402: the node names a price; a wallet-backed client pays and retries
                raise PaymentRequired(json.loads(e.read() or b"{}"))
            raise RuntimeError(f"{e.code}: {e.read().decode()[:300]}")

    def submit(self, trace):
        leaks = privacy_leaks(trace)
        if leaks:
            raise ValueError(f"refusing to send: trace still contains {leaks[:3]}")
        return self._call("POST", "/v0/traces", dict(trace))

    def lots(self):
        return self._call("GET", "/v0/lots")

    def bid(self, lot, price_micros, license="shared"):
        return self._call("POST", "/v0/bids", {"lot": lot, "bidder": self.address, "price_micros": price_micros, "license": license})

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

    def post_bounty(self, *, title, path, eval_set, target, seed_micros=0, failure="", base_model="", epochs=4):
        """Free to post; mints the bounty's coin. seed_micros optionally buys the first coins."""
        return self._call("POST", "/v0/bounties", {"poster": self.address, "title": title, "path": path,
                                                   "eval_set": eval_set, "target": target, "seed_micros": seed_micros,
                                                   "failure": failure, "base_model": base_model, "epochs": epochs})

    def buy_coins(self, bounty_id, micros):
        return self._call("POST", f"/v0/bounties/{bounty_id}/buy", {"buyer": self.address, "micros": micros})

    def sell_coins(self, bounty_id, coins):
        return self._call("POST", f"/v0/bounties/{bounty_id}/sell", {"seller": self.address, "coins": coins})

    def transfer_coins(self, bounty_id, to, coins):
        return self._call("POST", f"/v0/bounties/{bounty_id}/transfer", {"from": self.address, "to": to, "coins": coins})

    def find_learnings(self, path="", model="", kind="", min_gain=0.0, limit=20):
        qs = urllib.parse.urlencode({k: v for k, v in dict(path=path, model=model, kind=kind, min_gain=min_gain,
                                                           limit=limit).items() if v})
        return self._call("GET", f"/v0/learnings?{qs}")

    def learning(self, learning_id):
        return self._call("GET", f"/v0/learnings/{learning_id}")

    def describe(self):
        return self._call("GET", "/.well-known/trace-exchange.json")

    def holders(self, bounty_id):
        return self._call("GET", f"/v0/bounties/{bounty_id}/holders")

    def bounties(self, path="", status="open"):
        qs = urllib.parse.urlencode({k: v for k, v in dict(path=path, status=status).items() if v})
        return self._call("GET", f"/v0/bounties?{qs}")

    def claim_bounty(self, bounty_id, learning_id):
        return self._call("POST", f"/v0/bounties/{bounty_id}/claims", {"learning": learning_id})

    def provenance(self, object_id):
        return self._call("GET", f"/v0/provenance/{object_id}")

    def balance(self, address=None):
        return self._call("GET", f"/v0/balances/{address or self.address}")

    # --- coin economy (nodes run with --economy coin) -----------------------------------------------------------------
    def coin(self):
        return self._call("GET", "/v0/coin")

    def swap(self, side, amount):
        """side "buy": spend `amount` dollar micros on TXC; side "sell": sell `amount` TXC units for dollars."""
        return self._call("POST", "/v0/swap", {"account": self.address, "side": side, "amount": int(amount)})

    def back_with_coins(self, bounty_id, coins):
        return self._call("POST", f"/v0/bounties/{bounty_id}/buy", {"buyer": self.address, "coins": coins})

    def validators(self):
        return self._call("GET", "/v0/validators")

    def stake(self, units):
        return self._call("POST", "/v0/validators", {"address": self.address, "stake_units": int(units)})

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

    def faucet(self, address=None):
        """On a testnet node: open a wallet with test credits (no real money)."""
        return self._call("POST", "/v0/faucet", {"address": address or self.address})

    def wallet(self, address=None):
        return self._call("GET", f"/v0/wallets/{address or self.address}")

    def stats(self):
        return self._call("GET", "/v0/stats")

    def events(self, limit=30):
        return self._call("GET", f"/v0/events?limit={int(limit)}")


class PaymentRequired(Exception):
    def __init__(self, terms):
        super().__init__(f"payment required: {terms}")
        self.terms = terms
