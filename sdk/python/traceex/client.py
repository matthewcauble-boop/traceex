"""HTTP client for an exchange node (spec section 7). Standard library only."""
import json
import urllib.error
import urllib.parse
import urllib.request

from .skeleton import find_pii


class Client:
    def __init__(self, endpoint: str, address: str = None, timeout: float = 30):
        self.endpoint, self.address, self.timeout = endpoint.rstrip("/"), address, timeout

    def _call(self, method, path, body=None):
        req = urllib.request.Request(self.endpoint + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code == 402:   # x402: the node names a price; a wallet-backed client pays and retries
                raise PaymentRequired(json.loads(e.read() or b"{}"))
            raise RuntimeError(f"{e.code}: {e.read().decode()[:300]}")

    def submit(self, trace):
        leaks = find_pii(trace["input"])
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

    def search(self, q="", path="", failure="", model="", limit=20):
        qs = urllib.parse.urlencode({k: v for k, v in dict(q=q, path=path, failure=failure, model=model,
                                                           limit=limit).items() if v})
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


class PaymentRequired(Exception):
    def __init__(self, terms):
        super().__init__(f"payment required: {terms}")
        self.terms = terms
