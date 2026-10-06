"""Step graphs and tropical composition: joining fragments of different agents' attempts into verified solutions.

Adapted from TROPIC, "Tropical Reinforcement Learning" (Asadulaev, Djuhera, Salta, Boche, Karray, Takac; arXiv
2610.02478; https://github.com/machinestein/Tropical-Reinforcement-Learning, MIT licence; see NOTICE). The memory graph,
the L-best prefix/suffix recursion in the max-plus semiring, frontier choice, prefix-suffix joins with replay and
verification, and the basis of verified paths trained on by maximum likelihood follow `ragen/tropic/graph.py`. What
traceX changes:

  * Edges are environment steps (state, action, next state) that anyone can file, not one actor's token sequences.
    Every step remembers the fragments that filed it, in filing order, so a join knows whose work it is built from.
  * Scores. TROPIC rescores every step under its frozen actor. A node has no model, and log-probabilities a producer
    reports about its own steps can't be trusted (they would steer both path choice and credit), so a node scores every
    step -1 (the tropical value of a path is then minus its length: the best verified path is the shortest) unless the
    operator registers a scorer. A trainer composing locally sets `log_prob` from its own model, as TROPIC does.
  * Payment. Failed attempts are unpaid reports: their steps stay in the graph (frontier restarts, path search,
    training) but never earn. A verified path pays the passing traces it uses, per whole trace, equally (`credit`); a
    passing trace is a parent only for transitions it filed first. A path that revisits a state is also filed with its loops
    cut (`shortcut`), and looping paths stay out of the training basis.

Everything here is the standard library. The graph is time-unrolled (a node is a state at a step count), so it is
acyclic and both value passes are one sweep each, as in the paper.
"""
import hashlib
import json
import math
from collections import defaultdict

from .canon import canonical

DEFAULT_LOG_PROB = -1.0          # a step nobody scored: unit cost, so the tropical value is minus the path's length


def state_key(state, depth):
    """A node of the time-unrolled graph: the same environment state reached after another number of steps is another
    node, which keeps the graph acyclic (TROPIC puts the step count in its state for the same reason)."""
    return hashlib.sha256(canonical({"s": state, "d": int(depth)})).hexdigest()[:40]


def transition_key(state, next_state):
    """A transition between environment states, at any step count: who filed it first is who is credited for it."""
    return hashlib.sha256(canonical([state, next_state])).hexdigest()[:40]


def path_key(path):
    return hashlib.sha256(json.dumps(list(path)).encode()).hexdigest()


class Edge(dict):
    """One step: source and target node keys, the action, whether the goal check passed on the target, its score, and
    the fragments that filed it ([fragment id, producer], first filer first)."""

    @property
    def key(self):
        return edge_key(self["source"], self["action"], self["target"])


def edge_key(source, action, target):
    return hashlib.sha256(canonical([source, action, target])).hexdigest()[:40]


class StepGraph:
    """Every valid step seen on one problem (one root state), from passing and failing attempts alike, with the
    verified complete paths (the archive) and where each step came from."""

    def __init__(self, root_state, problem="", top_l=4, max_edges=8192, max_solutions=64, max_edges_per_producer=0):
        if min(top_l, max_edges, max_solutions) < 1:
            raise ValueError("graph limits must be positive")
        self.problem = str(problem)
        self.root = state_key(root_state, 0)
        self.nodes = {self.root: {"state": root_state, "depth": 0}}
        self.edges = {}
        self.transitions = {}            # transition_key -> fragment ids in filing order (credit follows the first)
        self.fragments = {}              # fragment id -> {"producer", "edges", "start", "outcome", "order"}
        self.solutions = {}              # path (tuple of edge keys) -> origin
        self.failed_joins = set()
        self.frontier_visits, self.depth_visits, self.rehearsals = {}, {}, {}
        self.top_l, self.max_edges, self.max_solutions = top_l, max_edges, max_solutions
        self.max_edges_per_producer = max_edges_per_producer
        self.rejected_edges = 0

    # --- memory -------------------------------------------------------------------------------------------------------
    def has_state(self, state, depth):
        return state_key(state, depth) in self.nodes

    def _producer_edges(self, producer):
        return sum(1 for e in self.edges.values() if e["owners"] and e["owners"][0][1] == producer)

    def add_fragment(self, fragment_id, producer, steps, start_state=None, start_depth=0, prefix=(), count_visit=False):
        """File one attempt's steps. `steps` is [{"action", "state" (the state after it), "success" (the goal check
        passed there)}]; the attempt starts at the root, or at `start_state` after `start_depth` steps, which must be a
        state the graph already holds (a restart from the frontier). `prefix` is the path the attempt was restarted
        after, if any, and makes its success a candidate together with that prefix. Returns the edge keys, in order.
        A fragment is filed once; a step two fragments share keeps both, in filing order. `count_visit` counts a
        restart against its start state, for `frontier_states` (when the frontier was not chosen by `choose_frontier`)."""
        if fragment_id in self.fragments:
            return self.fragments[fragment_id]["edges"]
        depth = int(start_depth)
        source = self.root if start_state is None and depth == 0 else state_key(start_state, depth)
        if source not in self.nodes:
            raise ValueError("a fragment must start at the root or at a state the graph already holds")
        start, keys, quota = source, [], self.max_edges_per_producer
        if count_visit and source != self.root:
            self.frontier_visits[source] = self.frontier_visits.get(source, 0) + 1
            self.depth_visits[depth] = self.depth_visits.get(depth, 0) + 1
        for i, s in enumerate(steps):
            if "action" not in s or "state" not in s:
                raise ValueError("each step needs an action and the state it led to")
            success = bool(s.get("success", False))
            if success and i != len(steps) - 1:
                raise ValueError("an attempt ends at its first success")
            target = state_key(s["state"], depth + 1)
            key = edge_key(source, s["action"], target)
            edge = self.edges.get(key)
            if edge is None:
                if len(self.edges) >= self.max_edges or (quota and self._producer_edges(producer) >= quota):
                    self.rejected_edges += len(steps) - i
                    break
                edge = Edge(source=source, target=target, action=s["action"], success=success, log_prob=None,
                            owners=[])
                self.edges[key] = edge
                self.nodes.setdefault(target, {"state": s["state"], "depth": depth + 1})
            elif edge["success"] != success:
                raise ValueError("this step was filed before with the opposite goal check")
            if [fragment_id, producer] not in edge["owners"]:
                edge["owners"].append([fragment_id, producer])
            owners = self.transitions.setdefault(transition_key(self.nodes[source]["state"], s["state"]), [])
            if fragment_id not in owners:
                owners.append(fragment_id)
            keys.append(key)
            source, depth = target, depth + 1
        self.fragments[fragment_id] = {"producer": producer, "edges": keys, "order": len(self.fragments),
                                       "start": start,
                                       "outcome": "pass" if keys and self.edges[keys[-1]]["success"] else "fail",
                                       "prefix": list(prefix)}
        return keys

    def score(self, path):
        total = 0.0
        for k in path:
            lp = self.edges[k]["log_prob"]
            lp = DEFAULT_LOG_PROB if lp is None else float(lp)
            if not math.isfinite(lp):
                return -math.inf
            total += lp
        return total

    def set_scores(self, scores):
        """{edge key: log-probability} from a scorer (one model for every step, whoever filed it)."""
        for k, v in scores.items():
            if k in self.edges:
                self.edges[k]["log_prob"] = float(v)

    # --- tropical values (max-plus): L best prefixes into, and suffixes out of, every state -----------------------------
    def _top(self, paths):
        return sorted(set(paths), key=lambda p: (-self.score(p), p))[:self.top_l]

    def paths(self, edges=None):
        """The L best prefixes (root -> state) and suffixes (state -> a verified ending) of every state: TROPIC's two
        linear passes, max over alternatives and sum of log-probabilities along each path. `edges` restricts the graph
        (to one producer's steps, say)."""
        allowed = self.edges if edges is None else {k: self.edges[k] for k in edges if k in self.edges}
        outgoing = defaultdict(list)
        for key, e in allowed.items():
            outgoing[e["source"]].append(key)
        ordered = sorted(self.nodes, key=lambda k: (self.nodes[k]["depth"], k))
        prefixes = {self.root: [()]}
        for state in ordered:
            for key in outgoing[state]:
                target = allowed[key]["target"]
                options = prefixes.get(target, []) + [p + (key,) for p in prefixes.get(state, [])]
                prefixes[target] = self._top(options)
        terminals = {self.edges[p[-1]]["target"] for p in self.solutions if p and all(k in allowed for k in p)}
        suffixes = {s: [()] for s in terminals}
        for state in reversed(ordered):
            options = suffixes.get(state, [])
            for key in outgoing[state]:
                options = options + [(key,) + p for p in suffixes.get(allowed[key]["target"], [])]
            if options:
                suffixes[state] = self._top(options)
        return prefixes, suffixes

    def values(self):
        """Each state's tropical prefix value (log-probability of its best way in) and suffix value (of its best verified
        way out); -inf where there is none."""
        prefixes, suffixes = self.paths()
        best = lambda ps: self.score(ps[0]) if ps else -math.inf
        return {s: {"prefix": best(prefixes.get(s, [])), "suffix": best(suffixes.get(s, []))} for s in self.nodes}

    def candidates(self, limit=64):
        """Joins of the L best prefixes and L best suffixes at every shared state that are not yet archived or refused,
        those adding the most steps no verified path uses first, then the most likely."""
        prefixes, suffixes = self.paths()
        certified = {e for p in self.solutions for e in p}
        joined = {p + q for state in prefixes for p in prefixes[state] for q in suffixes.get(state, []) if p and q}
        joined.difference_update(self.solutions)
        joined = {p for p in joined if path_key(p) not in self.failed_joins}
        return sorted(joined, key=lambda p: (-len(set(p) - certified), -self.score(p), p))[:limit]

    def is_complete(self, path):
        current = self.root
        for i, k in enumerate(path):
            e = self.edges.get(k)
            if e is None or e["source"] != current or (e["success"] and i != len(path) - 1):
                return False
            current = e["target"]
        return bool(path) and self.edges[path[-1]]["success"]

    def reject_join(self, path):
        if len(self.failed_joins) < self.max_edges:
            self.failed_joins.add(path_key(path))

    def certify(self, path, origin):
        """Archive a complete path. Call only after it replayed from the root and passed the checker. Returns True for a
        new solution (a new action sequence)."""
        path = tuple(path)
        if not self.is_complete(path):
            raise ValueError("only complete paths ending in a passing check can be certified")
        signature = tuple(self.edges[k]["action"] for k in path)
        novel = True
        for old in list(self.solutions):
            if tuple(self.edges[k]["action"] for k in old) == signature:
                if old == path or self.score(old) >= self.score(path):
                    return False
                del self.solutions[old]
                novel = False
                break
        if len(self.solutions) >= self.max_solutions:
            return False
        self.solutions[path] = origin
        return novel

    def best_path(self):
        return max(self.solutions, key=lambda p: (self.score(p), p), default=None)

    # --- exploration --------------------------------------------------------------------------------------------------
    def choose_frontier(self, budget, eta=1.0, alive=None):
        """Where the next attempts should start: a state the graph can reach but not yet finish from, at the least
        explored depth, the one with the best way in less a visit penalty (eta x log(1 + visits)). `alive(state)` can
        rule out dead ends. Returns the best prefix to it, () for the root."""
        prefixes, suffixes = self.paths()
        eligible = [s for s in prefixes if s != self.root and s not in suffixes and prefixes[s]
                    and self.nodes[s]["depth"] < budget and (alive is None or alive(self.nodes[s]["state"]))]
        if not eligible:
            return ()
        depth = min({self.nodes[s]["depth"] for s in eligible}, key=lambda d: (self.depth_visits.get(d, 0), d))
        state = min((s for s in eligible if self.nodes[s]["depth"] == depth),
                    key=lambda s: (-(self.score(prefixes[s][0]) - eta * math.log1p(self.frontier_visits.get(s, 0))), s))
        self.depth_visits[depth] = self.depth_visits.get(depth, 0) + 1
        self.frontier_visits[state] = self.frontier_visits.get(state, 0) + 1
        return prefixes[state][0]

    def frontier_states(self, budget, k=5, eta=1.0, alive=None):
        """The `k` best places for an agent to restart, without choosing one: states with a way in but no verified way
        out yet, least-explored depth first, then best way in less the visit penalty. What a node serves to agents
        working on a failure, so one agent picks up where another's attempt stopped."""
        prefixes, suffixes = self.paths()
        eligible = [s for s in prefixes if s != self.root and s not in suffixes and prefixes[s]
                    and self.nodes[s]["depth"] < budget and (alive is None or alive(self.nodes[s]["state"]))]
        rank = lambda s: (self.depth_visits.get(self.nodes[s]["depth"], 0), self.nodes[s]["depth"],
                          -(self.score(prefixes[s][0]) - eta * math.log1p(self.frontier_visits.get(s, 0))), s)
        return [{"node": s, "state": self.nodes[s]["state"], "depth": self.nodes[s]["depth"],
                 "prefix_value": self.score(prefixes[s][0]), "visits": self.frontier_visits.get(s, 0),
                 "prefix": [self.edges[e]["action"] for e in prefixes[s][0]]}
                for s in sorted(eligible, key=rank)[:k]]

    def best_prefix(self, state, depth):
        """The best known way from the root to a state (the prefix a restart there is credited after)."""
        prefixes, _ = self.paths()
        ps = prefixes.get(state_key(state, depth))
        return ps[0] if ps else None

    def end_state(self, path):
        node = self.nodes[self.edges[path[-1]]["target"] if path else self.root]
        return node["state"], node["depth"]

    # --- training: the basis of verified paths ---------------------------------------------------------------------------
    def loops(self, path):
        """Does the path visit some environment state twice (a detour that ends where it began)?"""
        seen = {canonical(self.nodes[self.root]["state"])}
        for k in path:
            c = canonical(self.nodes[self.edges[k]["target"]]["state"])
            if c in seen:
                return True
            seen.add(c)
        return False

    def select_basis(self, size=2, coverage=True, loop_free=True):
        """Up to `size` archived paths: the best one, then those covering steps trained on least (TROPIC's basis).
        `loop_free` leaves out paths that revisit a state (when any path doesn't), which would otherwise be picked for
        "covering" their detour steps."""
        frag = lambda p: {(self.edges[k]["source"], self.edges[k]["action"]) for k in p}
        remaining = sorted(self.solutions, key=lambda p: (-self.score(p), p))
        if loop_free and any(not self.loops(p) for p in remaining):
            remaining = [p for p in remaining if not self.loops(p)]
        selected, covered = [], set()
        while remaining and len(selected) < size:
            if selected and coverage:
                remaining.sort(key=lambda p: (-sum(1 / math.sqrt(1 + self.rehearsals.get(json.dumps(f), 0))
                                                   for f in frag(p) - covered), -self.score(p), p))
            path = remaining.pop(0)
            selected.append(path)
            covered.update(frag(path))
        for f in covered:
            self.rehearsals[json.dumps(f)] = self.rehearsals.get(json.dumps(f), 0) + 1
        return selected

    # --- provenance and credit -------------------------------------------------------------------------------------------
    def owner(self, key):
        """Provenance, not payment: the first fragment to file a step's transition (source -> target), whatever words its
        action used, so re-wording someone's step or filing it again earns nothing."""
        e = self.edges[key]
        return self.transitions[transition_key(self.nodes[e["source"]]["state"], self.nodes[e["target"]]["state"])][0]

    def reindex(self):
        """Rebuild who filed each transition first, from the fragments in filing order."""
        self.transitions = {}
        for fid, f in sorted(self.fragments.items(), key=lambda kv: kv[1]["order"]):
            for k in f["edges"]:
                e = self.edges[k]
                owners = self.transitions.setdefault(
                    transition_key(self.nodes[e["source"]]["state"], self.nodes[e["target"]]["state"]), [])
                if fid not in owners:
                    owners.append(fid)
        return self

    def producers(self, path):
        return [self.fragments[self.owner(k)]["producer"] for k in path]

    def to_dict(self):
        return {"problem": self.problem, "root": self.root, "nodes": self.nodes,
                "edges": {k: dict(e) for k, e in self.edges.items()}, "transitions": self.transitions,
                "fragments": self.fragments, "solutions": [[list(p), o] for p, o in self.solutions.items()],
                "failed_joins": sorted(self.failed_joins), "frontier_visits": self.frontier_visits,
                "depth_visits": {str(k): v for k, v in self.depth_visits.items()}, "rehearsals": self.rehearsals,
                "limits": [self.top_l, self.max_edges, self.max_solutions, self.max_edges_per_producer],
                "rejected_edges": self.rejected_edges}

    @classmethod
    def from_dict(cls, d):
        g = cls.__new__(cls)
        g.problem, g.root, g.nodes = d["problem"], d["root"], d["nodes"]
        g.edges = {k: Edge(e) for k, e in d["edges"].items()}
        g.transitions, g.fragments = d["transitions"], d["fragments"]
        g.solutions = {tuple(p): o for p, o in d["solutions"]}
        g.failed_joins = set(d["failed_joins"])
        g.frontier_visits = d["frontier_visits"]
        g.depth_visits = {int(k): v for k, v in d["depth_visits"].items()}
        g.rehearsals = d["rehearsals"]
        g.top_l, g.max_edges, g.max_solutions, g.max_edges_per_producer = d["limits"]
        g.rejected_edges = d["rejected_edges"]
        return g


# --- replay, composition ---------------------------------------------------------------------------------------------
def replay(graph, path, step, check):
    """Re-run a path from the root in the failure's own environment: `step(state, action)` returns the next state (None
    for an invalid action), `check(state)` whether the goal is met. Every recorded next state and goal flag must come
    out exactly as filed, and only the last step may pass. The environment is the verifier's, never the filer's."""
    if not graph.is_complete(path):
        return False
    state, depth = graph.nodes[graph.root]["state"], 0
    for i, k in enumerate(path):
        e = graph.edges[k]
        if state_key(state, depth) != e["source"]:
            return False
        nxt = step(state, e["action"])
        if nxt is None:
            return False
        depth += 1
        if state_key(nxt, depth) != e["target"]:
            return False
        passed = bool(check(nxt))
        if passed != bool(e["success"]) or (passed and i != len(path) - 1):
            return False
        state = nxt
    return bool(check(state))


def admit(graph, verify):
    """Certify the attempts that passed: those from the root as they are, restarts after the prefix they started from.
    Returns the new solutions [(path, origin)]."""
    out = []
    for fid, f in sorted(graph.fragments.items(), key=lambda kv: kv[1]["order"]):
        if f["outcome"] != "pass" or f.get("admitted"):
            continue
        f["admitted"] = True
        path = tuple(f.get("prefix") or ()) + tuple(f["edges"])
        origin = "restart" if f.get("prefix") else "root"
        if graph.is_complete(path) and verify(path) and graph.certify(path, origin):
            out.append((path, origin))
    return out


def compose(graph, verify, limit=64):
    """Join the L best prefixes with the L best suffixes at every shared state, replay and verify each join, archive the
    ones that pass and remember the ones that don't. Returns the new solutions [(path, "composition")]."""
    out = []
    for path in graph.candidates(limit):
        if verify(path):
            if graph.certify(path, "composition"):
                out.append((path, "composition"))
        else:
            graph.reject_join(path)
    return out


def shortcut(graph, path, step, check, producer="node:shortcut"):
    """Cut every loop out of a verified path (a stretch that comes back to a state it already visited), replay what is
    left from the root in the environment and, if it passes, file it as the node's own fragment and archive it. Its
    steps are transitions someone already filed, so they stay credited to whoever filed them first; the loop's steps
    drop out. Returns the shorter path (origin "shortcut"), or None."""
    states = [graph.nodes[graph.root]["state"]] + [graph.nodes[graph.edges[k]["target"]]["state"] for k in path]
    actions = [graph.edges[k]["action"] for k in path]
    keep, seen = [], {}
    for i, st in enumerate(states):
        c = canonical(st)
        if c in seen:                                 # back where it was: drop everything since then
            keep = keep[:seen[c]]
            seen = {cc: j for cc, j in seen.items() if j <= seen[c]}
        else:
            seen[c] = len(keep)
        if i < len(actions):
            keep.append(i)
    if len(keep) == len(actions):
        return None
    state, steps = states[0], []
    for i in keep:
        state = step(state, actions[i])
        if state is None:
            return None
        steps.append({"action": actions[i], "state": state, "success": bool(check(state))})
    if not steps or not steps[-1]["success"] or any(s["success"] for s in steps[:-1]):
        return None
    fid = "shortcut:" + path_key(path)[:24]
    keys = tuple(graph.add_fragment(fid, producer, steps))
    if graph.is_complete(keys) and replay(graph, keys, step, check):
        graph.certify(keys, "shortcut")
        graph.fragments[fid]["admitted"] = True
        return keys
    return None


def is_paid(graph, fragment_id):
    """A fragment that can be paid: a passing attempt filed by a contributor (not the node's own loop-free re-filing).
    Failed attempts are unpaid reports: their steps help (frontier restarts, path search, training) but earn nothing."""
    f = graph.fragments[fragment_id]
    return f["outcome"] == "pass" and not str(f["producer"]).startswith("node:")


def step_sources(graph, path):
    """Who first filed each step of a path, failed attempts included (provenance, not payment)."""
    return [graph.owner(k) for k in path]


def credit(graph, path):
    """Who a verified path pays: {fragment id: weight}, equal weights summing to 1, per WHOLE passing trace, never per
    step. The candidates are the fragments that first filed one of the path's transitions (state -> next state,
    however worded); of those, only passing attempts are paid. A step first filed by a failed attempt pays nobody, and
    re-filing it later inside a passing trace (or rewording it) doesn't make it yours: only new work counts. A path
    whose new steps all came from failed attempts returns {}: it can still be trained on, but its share is paid to
    nobody (the node gives it back to the payer). Padding or detours change nothing: a trace counts once."""
    parents = []
    for k in path:
        f = graph.owner(k)
        if is_paid(graph, f) and f not in parents:
            parents.append(f)
    return {f: 1.0 / len(parents) for f in sorted(parents)} if parents else {}


def producer_credit(graph, path):
    """The same split, summed per producer address."""
    out = defaultdict(float)
    for f, v in credit(graph, path).items():
        out[graph.fragments[f]["producer"]] += v
    return dict(out)


# --- the learning kind "tropic": maximum likelihood on the best verified paths ----------------------------------------
def export_tropic(graphs, basis_size=2, coverage=True, tokens=None):
    """Training rows for a tropic learning: for each problem with a verified path, its basis (the best path plus paths
    covering rarely trained steps), one row per step {problem, path, depth, state, action, weight}. Weights follow
    TROPIC's `basis_rows`: each problem counts once, each of its paths equally, normalised by the path's length in
    tokens (`tokens(action)`, default 1 per step), so the loss is -sum(weight x log p(action | state)) with no reward,
    baseline or negative term."""
    tokens = tokens or (lambda a: 1)
    positive = [(g, g.select_basis(basis_size, coverage)) for g in graphs]
    positive = [(g, ps) for g, ps in positive if ps]
    rows = []
    for g, paths in positive:
        for i, p in enumerate(paths):
            n = sum(max(int(tokens(g.edges[k]["action"])), 1) for k in p)
            for d, k in enumerate(p):
                e = g.edges[k]
                rows.append({"problem": g.problem, "path": i, "depth": d, "state": g.nodes[e["source"]]["state"],
                             "action": e["action"], "weight": 1.0 / (len(positive) * len(paths) * n)})
    return rows


def to_tropic_checkpoint(graphs, prompt_ids, completion_ids):
    """The graphs in the collector checkpoint format of the TROPIC reference code (`TropicCollector.load_state_dict`,
    graphs as `FragmentGraph.to_dict`), so a TROPIC run can start from traceX's archive. `prompt_ids(state)` and
    `completion_ids(action)` are the trainer's tokenizer applied to its own prompt template."""
    out = []
    for g in graphs:
        nodes, keymap, edges = {}, {}, []
        for k, n in g.nodes.items():
            nodes[k] = {"key": k, "depth": n["depth"], "snapshot": {"state": n["state"]},
                        "observation": json.dumps(n["state"]), "prompt_ids": list(prompt_ids(n["state"]))}
        for k, e in g.edges.items():
            p, c = list(nodes[e["source"]]["prompt_ids"]), list(completion_ids(e["action"]))
            act = e["action"] if isinstance(e["action"], (int, str)) else json.dumps(e["action"])
            fields = [e["source"], e["target"], act, p, c]
            keymap[k] = hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()
            edges.append({"source": e["source"], "target": e["target"], "action": act, "prompt_ids": p,
                          "completion_ids": c, "response": str(act), "raw_response": str(act),
                          "reward": 1.0 if e["success"] else 0.0, "success": bool(e["success"]),
                          "rollout_ids": [o[0] for o in e["owners"]][:8], "behavior_log_probs": [None] * min(len(e["owners"]), 8),
                          "log_prob": e["log_prob"], "policy_version": -1, "stop_reason": "eos"})
        out.append({"problem_id": g.problem, "root": g.root, "nodes": list(nodes.values()), "edges": edges,
                    "solutions": [[[keymap[k] for k in p], o] for p, o in g.solutions.items()], "rehearsals": [],
                    "frontier_visits": {}, "depth_visits": {}, "max_edges": g.max_edges,
                    "max_solutions": g.max_solutions, "top_l": g.top_l, "rejected_edges": 0, "failed_joins": []})
    return {"version": 1, "cursor": 0, "graphs": out}


# --- step traces on the exchange -------------------------------------------------------------------------------------
FRAGMENT_VERSION = "steps/0.1"


def fragment(*, failure_id, task, base_model, checker, root, steps, producer, start=None, privacy="open",
             created=None):
    """A step trace for `POST /v0/failures/{id}/fragments`: one attempt on one case of a failure, passing or not.

    root: the case's start state; steps: [{"action", "state" (after it), "success" (the goal check passed there)}];
    start: {"state", "depth"} when the attempt restarted from a state the failure's step graph already holds (see
    Client.frontier), else it starts at the root. States and actions travel as given, because the node replays them in
    the failure's environment: send only what that environment runs on, which for anything personal means skeletons
    (the client and the node both refuse secrets, and personal data in skeleton fragments)."""
    from .trace import _checker, _model
    import datetime as dt
    f = {"v": FRAGMENT_VERSION, "failure_id": str(failure_id), "task": task, "base_model": _model(base_model),
         "checker": _checker(checker), "privacy": privacy, "root": root,
         "steps": [{"action": s["action"], "state": s["state"], "success": bool(s.get("success", False))} for s in steps],
         "producer": producer,
         "created": created or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if start is not None:
        f["start"] = {"state": start["state"], "depth": int(start["depth"])}
    return f


def fragment_text(f):
    """Everything a fragment sends, for the pre-send scans."""
    parts = [json.dumps(f.get("root"), ensure_ascii=False)]
    if f.get("start"):
        parts.append(json.dumps(f["start"].get("state"), ensure_ascii=False))
    for s in f.get("steps") or []:
        parts += [json.dumps(s.get("action"), ensure_ascii=False), json.dumps(s.get("state"), ensure_ascii=False)]
    return "\n".join(parts)


def fragment_leaks(f):
    """The check the client (before sending) and the node (before accepting) run on a step trace, as for traces."""
    from .skeleton import find_secrets, find_pii, find_open_risks
    text = fragment_text(f)
    return find_open_risks(text) if f.get("privacy") == "open" else find_secrets(text) + find_pii(text)
