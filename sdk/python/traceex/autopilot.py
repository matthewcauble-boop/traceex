"""Autopilot: an agent that uses the exchange on its own.

Hook it to the agent's check loop and it acts without being asked:
  - every verified fix is submitted (turned into a skeleton on this machine first, unless the domain is open);
  - every failure the loop couldn't fix is classified, and the exchange is searched for an attested learning for that
    branch and base model; the best one is handed back for the agent to adopt;
  - when the same kind of failure keeps coming back and nothing fixes it, the autopilot checks the open bounties: it
    backs a matching one, or posts a new one for free. The failing cases stay on this machine as the bounty's hidden
    eval; only their hash is published;
  - it never spends more than the owner's budget (default: nothing).

    pilot = Autopilot(Client(node, address), task="extract.flight", base_model="needle3", checker="flight-rules@1")
    agent = AdaptiveAgent(..., autopilot=pilot)       # or call pilot.on_result(text, out) from your own loop
"""
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass

from .canon import canonical
from .classify import classify, default_engine
from .skeleton import skeletonize


@dataclass
class Policy:
    submit_fixes: bool = True
    privacy: str = "skeleton"        # "open" only for non-personal domains (code, maths, public text)
    adopt_min_gain: float = 0.02     # adopt a learning only if its attested gain is at least this
    bounty_after: int = 3            # unresolved failures of the same kind before acting on a bounty
    bounty_target: float = 0.6       # a solution must fix this share of the hidden failing cases
    bounty_epochs: int = 4
    back_micros: int = 0             # back each bounty it posts or joins with this much
    budget_micros: int = 0           # never spend more than this in total


class Autopilot:
    def __init__(self, client, *, task, base_model, checker, policy=None, engine=None):
        self.c, self.task, self.base_model, self.checker = client, task, base_model, checker
        self.policy, self.engine = policy or Policy(), engine or default_engine()
        self.unresolved = defaultdict(list)       # (path, failure) -> local failing cases: the hidden eval
        self.bounties, self.spent, self.events, self.tried = {}, 0, [], set()

    # -- hooks -------------------------------------------------------------------------------------------------------
    def on_result(self, text, out, failure=None):
        """Call after every run of the check loop. Returns the actions taken (a learning to adopt, if one was found)."""
        acts = []
        if out.get("trace") is not None and self.policy.submit_fixes:
            r = self.c.submit(out["trace"])
            acts.append({"action": "submitted_fix", "trace": r["id"], "path": r["classified"]["path_str"],
                         "feeds_bounties": r.get("bounties", [])})
        if out.get("failing"):
            acts += self.on_unresolved(text, out, failure)
        self.events += acts
        return acts

    def on_unresolved(self, text, out, failure=None):
        """A failure the loop couldn't fix. Every kind of failure is counted (per field, or the checker's own mode);
        first look for a proven learning not tried yet, then, once a failure keeps recurring, act on a bounty."""
        path, kinds = self._classify(text, out, failure)
        case = {"input": text, "result": out.get("result"), "failing": out.get("failing")}
        for k in kinds:
            self.unresolved[(path, k)].append(case)
        best = self._best_learning(path)
        if best:
            self.tried.add(best["id"])
            return [{"action": "adopt", "learning": self.c.learning(best["id"]), "gain": best["gain"], "path": path}]
        keys = [(path, k) for k in kinds]
        due = [k for k in keys if len(self.unresolved[k]) >= self.policy.bounty_after and k not in self.bounties]
        if due:
            return [self._bounty(path, due)]
        if all(k in self.bounties for k in keys):
            return [{"action": "already_posted", "bounty": self.bounties[keys[0]], "path": path}]
        return [{"action": "noted", "path": path, "failures": kinds, "seen": max(len(self.unresolved[k]) for k in keys)}]

    def tick(self):
        """Call now and then: proven learnings, not yet tried, for the kinds of failure this agent still has."""
        found = []
        for path in sorted({p for p, _ in self.unresolved}):
            best = self._best_learning(path)
            if best:
                found.append({"action": "adopt", "learning": self.c.learning(best["id"]), "gain": best["gain"], "path": path})
        return found

    def hidden_eval(self, bounty_id):
        """The failing cases behind a bounty this agent posted, for the validator (shared privately, never published)."""
        cases = [c for k, b in self.bounties.items() if b == bounty_id for c in self.unresolved[k]]
        return list({json.dumps(c, sort_keys=True): c for c in cases}.values())

    # -- internals ---------------------------------------------------------------------------------------------------
    def _classify(self, text, out, failure):
        fields = sorted(out.get("failing") or [])
        if self.policy.privacy == "skeleton":
            text, _, _ = skeletonize(text)              # classify the skeleton, never the raw text
        probe = {"task": self.task, "input": text, "verified_output": {f: "" for f in fields}, "fixed_fields": fields}
        c = classify(probe, self.engine)
        return c["path_str"], ([failure] if failure else ["unresolved:" + f for f in fields])

    def _best_learning(self, path):
        found = self.c.find_learnings(path=path, model=self._model_name(), min_gain=self.policy.adopt_min_gain)
        fresh = [L for L in found["learnings"] if L["id"] not in self.tried]
        return fresh[0] if fresh else None

    def _title(self, path, failure, n):
        """A title a person can read on the board: what keeps failing, on which model, where it is filed."""
        where = f"({path}, {n} unresolved case{'s' if n != 1 else ''})"
        if failure.startswith("unresolved:"):
            fields = [f.replace("_", " ") for f in failure.split(":", 1)[1].split(",") if f]
            what = fields[0] if len(fields) == 1 else ", ".join(fields[:-1]) + " and " + fields[-1]
            return f"Get {what} right on {self._model_name()} {where}"[:200]
        return f"Fix {failure.replace('_', ' ')} failures on {self._model_name()} {where}"[:200]

    def _model_name(self):
        return self.base_model["name"] if isinstance(self.base_model, dict) else str(self.base_model)

    def _bounty(self, path, due):
        cases = list({json.dumps(c, sort_keys=True): c for k in due for c in self.unresolved[k]}.values())
        kinds = [k for _, k in due]
        if len(kinds) == 1:
            failure = kinds[0]
        else:                                   # several fields crossed together: one bounty covering them all
            failure = "unresolved:" + ",".join(sorted(k.split(":", 1)[1] for k in kinds if ":" in k))
        eval_set = "sha256:" + hashlib.sha256(canonical(cases)).hexdigest()
        open_ = [b for b in self.c.bounties(path=path, status="open")["bounties"]
                 if b["path"] == path and (not b["base_model"] or b["base_model"] == self._model_name())
                 and (not b["failure"] or b["failure"] == failure)]
        if open_:
            bid = open_[0]["id"]
            if bid in set(self.bounties.values()):          # already behind this one: just count these failures in
                for k in due:
                    self.bounties[k] = bid
                return {"action": "already_backed", "bounty": bid, "path": path, "failure": failure}
            act = {"action": "backed_existing_bounty"}
        else:
            r = self.c.post_bounty(title=self._title(path, failure, len(cases)),
                                   path=path, eval_set=eval_set, target=self.policy.bounty_target, failure=failure,
                                   base_model=self._model_name(), epochs=self.policy.bounty_epochs)
            bid, act = r["id"], {"action": "posted_bounty", "eval_set": eval_set, "target": self.policy.bounty_target}
        for k in due:
            self.bounties[k] = bid
        back = min(self.policy.back_micros, self.policy.budget_micros - self.spent)
        if back > 0:
            self.c.buy_coins(bid, back)
            self.spent += back
            act["backed_micros"] = back
        return dict(act, bounty=bid, path=path, failure=failure, cases=len(cases))
