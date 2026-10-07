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

    def _call(self, method, path, body=None, text=False):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.endpoint + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return {"text": raw.decode()} if text else json.loads(raw or b"{}")
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

    def search(self, q="", path="", failure="", model="", limit=20, sort="", offset=0, facets=False, kind="", fmt=""):
        """sort: "relevant" (default with words), "new" (default without), or "bounty". kind: "trace" (the node's
        default), "failure" or "all". fmt="text": compact cited cards (returned as {"text": ...})."""
        qs = urllib.parse.urlencode({k: v for k, v in dict(q=q, path=path, failure=failure, model=model, limit=limit,
                                                           sort=sort, offset=offset, facets=int(facets), kind=kind,
                                                           format=fmt).items() if v})
        return self._call("GET", f"/v0/search?{qs}", text=fmt == "text")

    # --- the failure registry and fix tracking (v0.7) ----------------------------------------------------------------
    def failures(self, path="", failure="", model="", status="", sort="frequency", limit=20, offset=0):
        """Known failures (TXF ids), sorted by frequency (distinct verified reporters), growth, bounty or new."""
        qs = urllib.parse.urlencode({k: v for k, v in dict(path=path, failure=failure, model=model, status=status,
                                                           sort=sort, limit=limit, offset=offset).items() if v})
        return self._call("GET", f"/v0/failures?{qs}")

    def failure(self, failure_id):
        return self._call("GET", f"/v0/failures/{failure_id}")

    def failure_history(self, failure_id):
        return self._call("GET", f"/v0/failures/{failure_id}/history")

    # --- v0.8: step traces and composition ----------------------------------------------------------------------------
    def submit_fragment(self, fragment):
        """File one attempt's steps (traceex.tropic.fragment) under its failure; the node replays it in the failure's
        environment, composes, and answers with any new verified paths. Refused here if it carries secrets (or, at
        privacy 'skeleton', personal data)."""
        from .tropic import fragment_leaks
        leaks = fragment_leaks(fragment)
        if leaks:
            raise ValueError(f"refusing to send a step trace with secrets or personal data: {leaks[:3]}")
        return self._call("POST", f"/v0/failures/{fragment['failure_id']}/fragments", fragment)

    def step_graphs(self, failure_id):
        return self._call("GET", f"/v0/failures/{failure_id}/graphs")

    def frontier(self, failure_id, root="", k=5):
        """States worth restarting from on a failure's cases (another agent's attempt got there, nobody finished)."""
        return self._call("GET", f"/v0/failures/{failure_id}/frontier?" + urllib.parse.urlencode({"root": root, "k": k}))

    def joins(self, failure_id):
        """The failure's verified paths, each with the step traces (and producers) it credits."""
        return self._call("GET", f"/v0/failures/{failure_id}/joins")

    def trace(self, trace_id):
        return self._call("GET", f"/v0/traces/{trace_id}")

    def claim_fix(self, claims, model, kind="learning", learning=None, artifact=None, outputs=None):
        """Claim that a fix (learning, prompt_patch or tool) applied to `model` fixes these failure ids. Validators
        drawn at random measure it on their own cases; on a sats node the claim holds a 2,000-sat bond, destroyed if it
        fixes none of them. outputs: {trace id: the fixed model's output} on the claimed failures' public cases."""
        body = {"claimant": self.address, "kind": kind, "claims": list(claims), "model": model}
        for k, v in (("learning", learning), ("artifact", artifact), ("outputs", outputs)):
            if v:
                body[k] = v
        return self._call("POST", "/v0/fixes", body)

    def fix(self, fix_id):
        return self._call("GET", f"/v0/fixes/{fix_id}")

    def fixes(self, failure="", status=""):
        qs = urllib.parse.urlencode({k: v for k, v in dict(failure=failure, status=status).items() if v})
        return self._call("GET", f"/v0/fixes?{qs}")

    def register_model(self, version, family=None, parent=None, outputs=None):
        """Operator: a new model version; every tracked failure of its family is re-checked."""
        body = {"version": version}
        for k, v in (("family", family), ("parent", parent), ("outputs", outputs)):
            if v:
                body[k] = v
        return self._call("POST", "/v0/models", body)

    def model_report(self, version):
        return self._call("GET", f"/v0/models/{urllib.parse.quote(version, safe='')}/report")

    def commit_fix(self, fix_id, digest):
        return self._call("POST", f"/v0/fixes/{fix_id}/commits", {"validator": self.address, "digest": digest})

    def reveal_fix(self, fix_id, measurement, salt=""):
        return self._call("POST", f"/v0/fixes/{fix_id}/reveals",
                          {"validator": self.address, "measurement": measurement, "salt": salt})

    def measure_fix_for_bounty(self, bounty_id, fix_id, attestation):
        """A failure bounty's poster: its own measurement of a fix on its hidden eval (operator-relayed)."""
        return self._call("POST", f"/v0/bounties/{bounty_id}/measurements", {"fix": fix_id, "attestation": attestation})

    def repro_check(self, failure_id, results):
        """A validator: {trace id: True if it reproduced on the base model} for a failure's public cases."""
        return self._call("POST", f"/v0/failures/{failure_id}/repro", {"validator": self.address, "results": results})

    def reporter_bond(self):
        """Sats node: hold the 1,000-sat reporter bond, so your reports count toward failures' verified reporters."""
        return self._call("POST", "/v0/reporters", {"address": self.address})

    def withdraw_reporter(self):
        return self._call("POST", "/v0/reporters/withdraw", {"address": self.address})

    def reporter(self, address=None):
        return self._call("GET", f"/v0/reporters/{address or self.address}")

    def post_bounty(self, *, title, path="", eval_set, target, seed_micros=0, failure="", base_model="", epochs=4,
                    seed_msats=0, failure_id=""):
        """Free to post (the transaction fee only). A bounty is a refundable pledge escrow: seed_msats (a sats node) or
        seed_micros (the dollar node) makes the poster's first pledge. A post matching an open bounty's branch, failure
        and model backs that bounty instead (the reply says `merged`)."""
        seed = {"seed_msats": int(seed_msats)} if seed_msats else ({"seed_micros": seed_micros} if seed_micros else {})
        body = dict(seed, poster=self.address, title=title, path=path, eval_set=eval_set, target=target,
                    failure=failure, base_model=base_model, epochs=epochs)
        if failure_id:                    # v0.7: a bounty on a registry failure; it pays when that failure is fixed
            body["failure_id"] = failure_id
        return self._call("POST", "/v0/bounties", body)

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

    # --- challenge bounties (v0.8, SPEC 4k): big open problems, paid per verified improvement ----------------------
    def challenges(self, status="open", path="", q="", origin="", limit=50):
        """Open problems with a verifier, a direction and a leaderboard (status "all" for every one)."""
        qs = urllib.parse.urlencode({k: v for k, v in dict(status=status, path=path, q=q, origin=origin,
                                                           limit=limit).items() if v})
        return self._call("GET", f"/v0/challenges?{qs}")

    def get_challenge(self, challenge_id):
        """One challenge bounty (not to be confused with challenge(), which disputes a learning): its file (statement, metric, verifier, sources), escrow, best score and top of the board."""
        return self._call("GET", f"/v0/challenges/{int(challenge_id)}")

    def leaderboard(self, challenge_id, limit=50):
        return self._call("GET", f"/v0/challenges/{int(challenge_id)}/leaderboard?limit={int(limit)}")

    def challenge_export(self, challenge_id, fmt="yukon"):
        """The challenge as a Yukon-style benchmark: benchmark.json (+ verify.py for a python verifier)."""
        return self._call("GET", f"/v0/challenges/{int(challenge_id)}/export?format={fmt}")

    def post_challenge(self, challenge, seed_msats=0):
        """Post a challenge/0.1 file (traceex.challenges.normalize), free (the fee only). A post whose key or alias
        matches an open challenge backs it instead (`merged`)."""
        body = dict(challenge, poster=self.address)
        if seed_msats:
            body["seed_msats"] = int(seed_msats)
        return self._call("POST", "/v0/challenges", body)

    def pledge_challenge(self, challenge_id, msats, from_score=None, judge=None):
        """A refundable pledge, released to verified improvements along the challenge's curve; from_score: pay only for
        progress beyond this score (default: the best now); judge: who confirms validator-measured improvements for
        this pledge (default: you)."""
        body = {"backer": self.address, "msats": int(msats)}
        if from_score is not None:
            body["from_score"] = float(from_score)
        if judge:
            body["judge"] = judge
        return self._call("POST", f"/v0/challenges/{int(challenge_id)}/pledges", body)

    def submit_solution(self, challenge_id, solution=None, *, artifact=None, outputs=None, public_score=None,
                        parents=None, model=None, per_call_msats=None):
        """Submit to a challenge. A verifier the node runs scores it at once; otherwise validators measure it (a
        1,000-sat bond is held, destroyed if it is invalid or overfits its public instances)."""
        body = {"submitter": self.address}
        for k, v in (("solution", solution), ("artifact", artifact), ("outputs", outputs),
                     ("public_score", public_score), ("parents", parents), ("model", model),
                     ("per_call_msats", per_call_msats)):
            if v is not None:
                body[k] = v
        return self._call("POST", f"/v0/challenges/{int(challenge_id)}/submissions", body)

    def confirm_solution(self, submission_id, score):
        """As a pledge's judge: confirm a validator-measured improvement with your own measurement (operator-relayed)."""
        return self._call("POST", f"/v0/challenges/submissions/{int(submission_id)}/confirmations",
                          {"judge": self.address, "measurement": {"score": float(score)}})

    def prior_art(self, challenge_id, reference, provenance, score=None):
        """Dispute paid progress on a challenge as already known (2,000-sat stake): reference is the known solution,
        provenance {kind: "tracex"} (it was on traceX's boards before the challenge) or {kind: "external", url, date}."""
        body = {"challenger": self.address, "reference": reference, "provenance": provenance}
        if score is not None:
            body["score"] = float(score)
        return self._call("POST", f"/v0/challenges/{int(challenge_id)}/prior-art", body)

    def prior_claim(self, claim_id):
        return self._call("GET", f"/v0/challenges/prior-art/{int(claim_id)}")

    def submission(self, submission_id):
        return self._call("GET", f"/v0/challenges/submissions/{int(submission_id)}")

    def commit_solution(self, submission_id, digest):
        return self._call("POST", f"/v0/challenges/submissions/{int(submission_id)}/commits",
                          {"validator": self.address, "digest": digest})

    def reveal_solution(self, submission_id, measurement, salt=""):
        return self._call("POST", f"/v0/challenges/submissions/{int(submission_id)}/reveals",
                          {"validator": self.address, "measurement": measurement, "salt": salt})

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
