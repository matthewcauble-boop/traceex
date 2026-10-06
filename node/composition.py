"""traceX v0.8: step graphs and cross-agent composition (SPEC 4j).

Adapted from TROPIC (arXiv 2610.02478, MIT; see NOTICE and sdk/python/traceex/tropic.py).

Step traces
  * An agent's attempt at one case of a failure can be filed as a step trace (`steps/0.1`): the case's start state and
    every step it took (action, next state, whether the goal check passed), whether the attempt passed or not. It is
    filed under a failure id, in the step graph of its case (one graph per distinct start state).
  * The node replays every step trace on arrival in the failure's own environment, which only the operator registers
    (in Python, like checker runners): every recorded next state and goal flag must come out exactly. A step trace
    that doesn't replay is refused (and has paid its fee). A restart (an attempt that started mid-way) must start at a
    state the graph already holds; GET /v0/failures/{id}/frontier says which ones are worth restarting from.

Composition
  * After each step trace the node certifies passing attempts (a restart after the best known way into its start
    state), then joins the L best prefixes with the L best suffixes at every shared state, replays each join from the
    start in the environment and keeps the ones that pass: TROPIC's composition, over everyone's steps.
  * Failed attempts are unpaid reports. Their steps stay in the graph (frontier restarts and path search, which is
    where shared graphs helped in examples/tropical_composition) and each counts as a case of its failure in the
    registry (reporter, occurrence, repro set), but no failed attempt is ever a payment parent.
  * Every verified path is filed as a trace (task, model family and checker of its failure; its verified output the
    action sequence). Its producer share of any payment (a learning's traces share, a bounty's, a licence) is split
    equally over the passing step traces that first filed one of its transitions, per whole trace (re-filing or
    rewording a step someone filed earlier, even a failed attempt's, earns nothing). A path that uses no passing trace can be trained on, but its share is paid to nobody and
    goes back to the payer. The composer earns nothing. Splits use the same integer apportioning as every other share,
    so nothing is paid out that a payer didn't pay in.
  * Steps are scored -1 each (the best path is the shortest) unless the operator registers a scorer for the checker:
    log-probabilities producers report about their own steps are never used. A verified path that revisits a state is
    also filed without its loops (the node's own replay), so padding a path with detours that come back buys nothing.
"""
import json

from traceex import canonical, object_id
from traceex.tropic import (FRAGMENT_VERSION, StepGraph, admit, compose, credit, fragment_leaks, producer_credit,
                            replay, shortcut, state_key, step_sources)

COMP_SCHEMA = """
CREATE TABLE IF NOT EXISTS fragments  (id TEXT PRIMARY KEY, failure TEXT, root TEXT, producer TEXT, checker TEXT,
                                       outcome TEXT, steps INT, accepted INT, note TEXT, epoch INT, body TEXT);
CREATE INDEX IF NOT EXISTS fragments_failure ON fragments(failure, root);
CREATE TABLE IF NOT EXISTS step_graphs(failure TEXT, root TEXT, checker TEXT, task TEXT, privacy TEXT, body TEXT,
                                       epoch INT, PRIMARY KEY (failure, root));
CREATE TABLE IF NOT EXISTS joins      (trace TEXT PRIMARY KEY, failure TEXT, root TEXT, path TEXT, origin TEXT,
                                       fragments TEXT, producers TEXT, rule TEXT, epoch INT);
CREATE INDEX IF NOT EXISTS joins_failure ON joins(failure);
"""
LIMITS = {"steps": 64, "fragment_bytes": 32 * 1024, "graphs_per_failure": 256, "edges_per_graph": 4096,
          "edges_per_producer": 512, "join_candidates": 16, "top_l": 4, "max_depth": 64}


class Composition:
    """Mixed into the exchange node (and so the sats node)."""

    def _open_composition(self):
        self.db.executescript(COMP_SCHEMA)
        self.envs = getattr(self, "envs", {})
        self.scorers = getattr(self, "scorers", {})

    # --- operator registrations (Python only, never over HTTP: they run code) ------------------------------------------
    def register_env(self, checker, step, check, alive=None):
        """The environment a failure's step traces replay in: step(state, action) -> next state or None (invalid),
        check(state) -> the goal is met, alive(state) -> worth continuing from (frontier). Keyed by checker ('id' or
        'id@version')."""
        self.envs[str(checker)] = (step, check, alive)

    def register_scorer(self, checker, score):
        """One scoring model for every step of a checker's graphs: score(state, action) -> log-probability. With one,
        path choice (and so what a tropic learning trains on) is the most likely verified path. It never moves money."""
        self.scorers[str(checker)] = score

    def _env(self, ck):
        cid = ck.split("@")[0]
        env = self.envs.get(ck) or self.envs.get(cid)
        if not env:
            raise ValueError(f"this node has no replay environment for checker {ck}: step traces are composed only where "
                             "the operator registered the failure's environment")
        return env

    # --- graphs -------------------------------------------------------------------------------------------------------
    def _graph(self, failure, root_key):
        r = self.db.execute("SELECT body FROM step_graphs WHERE failure=? AND root=?", (failure, root_key)).fetchone()
        return StepGraph.from_dict(json.loads(r[0])) if r else None

    def _save_graph(self, failure, g, ck, task, privacy):
        self.db.execute("INSERT OR REPLACE INTO step_graphs VALUES (?,?,?,?,?,?,?)",
                        (failure, g.root, ck, task, privacy, json.dumps(g.to_dict(), separators=(",", ":")), self.epoch))

    def _check_fragment(self, failure_id, f):
        if not isinstance(f, dict) or f.get("v") != FRAGMENT_VERSION:
            raise ValueError(f"a step trace is {FRAGMENT_VERSION}")
        from exchange import need_address, LIMITS as XL, UNSAFE
        need_address(f.get("producer"), "producer")
        if f.get("privacy") not in ("skeleton", "open"):
            raise ValueError("rejected: privacy must be 'skeleton' or 'open'")
        steps = f.get("steps")
        if not isinstance(steps, list) or not 0 < len(steps) <= LIMITS["steps"]:
            raise ValueError(f"rejected: a step trace has 1 to {LIMITS['steps']} steps")
        for s in steps:
            if not isinstance(s, dict) or "action" not in s or "state" not in s:
                raise ValueError("rejected: each step needs an action and the state it led to")
        for what, val, cap in (("task", f.get("task"), XL["task"]),
                               ("base_model.name", (f.get("base_model") or {}).get("name"), XL["model"]),
                               ("checker.id", (f.get("checker") or {}).get("id"), XL["checker"])):
            if not isinstance(val, str) or not val or len(val) > cap or UNSAFE.search(val):
                raise ValueError(f"rejected: {what} must be plain text up to {cap} characters")
        if len(canonical(f)) > LIMITS["fragment_bytes"]:
            raise ValueError(f"rejected: a step trace is limited to {LIMITS['fragment_bytes'] // 1024} KB")
        leaks = fragment_leaks(f)
        if leaks:
            raise ValueError(f"rejected: secrets or personal data: {leaks[:3]}")
        start = f.get("start")
        if start is not None and (not isinstance(start, dict) or "state" not in start
                                  or not 0 < int(start.get("depth", 0)) < LIMITS["max_depth"]):
            raise ValueError("rejected: start is {state, depth} with 0 < depth, or absent for the root")
        fid = str(failure_id or f.get("failure_id") or "").upper()
        if str(f.get("failure_id", fid)).upper() != fid or not self._failure_row(fid):
            raise KeyError(f"failure {fid}")
        return fid

    def submit_fragment(self, failure_id, f):
        """File one attempt's steps under a failure (POST /v0/failures/{id}/fragments). Pays the standard fee whether or
        not it replays. Returns what was filed and every new verified path the node composed from it."""
        fid = self._check_fragment(failure_id, f)
        ck = f"{f['checker']['id']}@{f['checker'].get('version', '1')}"
        step, check, alive = self._env(ck)
        frag_id = object_id(f)
        root_key = state_key(f["root"], 0)
        self._room()
        with self.lock:
            if self.db.execute("SELECT 1 FROM fragments WHERE id=?", (frag_id,)).fetchone():
                return {"id": frag_id, "failure_id": fid, "duplicate": True}
            g = self._graph(fid, root_key)
            if g is None:
                n = self.db.execute("SELECT COUNT(*) FROM step_graphs WHERE failure=?", (fid,)).fetchone()[0]
                if n >= LIMITS["graphs_per_failure"]:
                    raise ValueError(f"failure {fid} already holds {n} step graphs (cases); file attempts on those")
                g = StepGraph(f["root"], f"{fid}:{root_key[:12]}", top_l=LIMITS["top_l"],
                              max_edges=LIMITS["edges_per_graph"], max_solutions=64,
                              max_edges_per_producer=LIMITS["edges_per_producer"])
            self._tx_fee(f["producer"])
            note = self._replay_fragment(g, f, step, check)
            if note:
                self._store_fragment(frag_id, fid, root_key, f, ck, accepted=0, note=note)
                self.db.commit()
                return {"id": frag_id, "failure_id": fid, "accepted": False, "reason": note}
            start = f.get("start")
            prefix = g.best_prefix(start["state"], start["depth"]) if start else ()
            keys = g.add_fragment(frag_id, f["producer"], f["steps"], start_state=start and start["state"],
                                  start_depth=start["depth"] if start else 0, prefix=prefix or (), count_visit=True)
            scorer = self.scorers.get(ck) or self.scorers.get(ck.split("@")[0])
            if scorer:
                g.set_scores({k: scorer(g.nodes[e["source"]]["state"], e["action"])
                              for k, e in g.edges.items() if e["log_prob"] is None})
            verify = lambda path: replay(g, path, step, check)
            new = admit(g, verify) + compose(g, verify, limit=LIMITS["join_candidates"])
            for path, _ in list(new):                # a path with a loop in it: file the loop-free one too
                short = shortcut(g, path, step, check)
                if short:
                    new.append((short, "shortcut"))
            joins = [self._file_join(fid, g, path, origin, f, ck) for path, origin in new]
            self._save_graph(fid, g, ck, f["task"], f["privacy"])
            self._store_fragment(frag_id, fid, root_key, f, ck, accepted=1,
                                 note=f"{len(keys)} of {len(f['steps'])} steps filed" if len(keys) < len(f["steps"]) else "")
            failed = not (keys and g.edges[keys[-1]]["success"])
            if failed:                                # an unpaid report: a case of the failure, for its counters
                self._report_attempt(frag_id, fid, root_key, f, ck)
            if joins:
                self._event(f"{len(joins)} verified path{'s' if len(joins) != 1 else ''} composed on {fid}")
            self.db.commit()
        return {"id": frag_id, "failure_id": fid, "accepted": True, "graph": root_key, "steps_filed": len(keys),
                "outcome": "pass" if keys and g.edges[keys[-1]]["success"] else "fail", "joins": joins}

    def _replay_fragment(self, g, f, step, check):
        """Why a step trace doesn't replay in the failure's environment, or '' when it does."""
        start = f.get("start")
        state, depth = (start["state"], int(start["depth"])) if start else (f["root"], 0)
        if start and state_key(state, depth) not in g.nodes:
            return "a restart must start at a state this case's graph already holds (see the frontier)"
        if g.edges and not start and state_key(state, 0) != g.root:
            return "root mismatch"
        for i, s in enumerate(f["steps"]):
            nxt = step(state, s["action"])
            if nxt is None:
                return f"step {i + 1}: the environment refuses this action"
            if canonical(nxt) != canonical(s["state"]):
                return f"step {i + 1}: the environment does not reach the recorded state"
            passed = bool(check(nxt))
            if passed != bool(s.get("success", False)):
                return f"step {i + 1}: the goal check does not match"
            if passed and i != len(f["steps"]) - 1:
                return "an attempt ends at its first success"
            state, depth = nxt, depth + 1
        return ""

    def _report_attempt(self, frag_id, fid, root_key, f, ck):
        """A failed attempt counts toward the failure registry like a reported case: its reporter, and one case per
        start state (later failed attempts on the same case are its copies). Validators can re-check it like any case
        (POST /v0/failures/{id}/repro with the step trace's id). It is never a payment parent."""
        first = self.db.execute("SELECT id FROM fragments WHERE failure=? AND root=? AND accepted=1 AND outcome='fail' "
                                "AND id != ? ORDER BY rowid LIMIT 1", (fid, root_key, frag_id)).fetchone()
        now = self._now()
        self.db.execute("INSERT OR IGNORE INTO occurrences (trace, failure, reporter, model, checker, epoch, at, canonical)"
                        " VALUES (?,?,?,?,?,?,?,?)", (frag_id, fid, f["producer"], f["base_model"]["name"], ck,
                                                      self.epoch, now, first[0] if first else frag_id))
        self.db.execute("UPDATE failures SET last_at=?, last_epoch=? WHERE id=?", (now, self.epoch, fid))
        self._index_failure(fid)

    def _fragment_cases(self, fid):
        """A failure's cases reported as failed step traces: the public part of its repro set for replayable failures."""
        rows = self.db.execute("SELECT f.id, f.root, f.producer, o.reproduced FROM fragments f JOIN occurrences o ON "
                               "o.trace = f.id WHERE f.failure=? AND o.rejected=0 AND o.canonical = o.trace "
                               "ORDER BY f.rowid DESC LIMIT 50", (fid,)).fetchall()
        return [{"step_trace": i, "root": r, "reporter": p, "reproduced": None if rep is None else bool(rep)}
                for i, r, p, rep in rows]

    def _store_fragment(self, frag_id, fid, root_key, f, ck, accepted, note):
        out = "pass" if f["steps"] and f["steps"][-1].get("success") else "fail"
        self.db.execute("INSERT INTO fragments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (frag_id, fid, root_key, f["producer"], ck, out, len(f["steps"]), accepted, note, self.epoch,
                         canonical(f).decode()))

    def _file_join(self, fid, g, path, origin, f, ck):
        """A verified path, filed as a trace whose producer share belongs to the passing step traces it uses."""
        fr = self._failure_row(fid)
        actions = [g.edges[k]["action"] for k in path]
        parts = credit(g, path)                  # the passing traces it uses, equally, per whole trace
        who = producer_credit(g, path)
        t = {"v": "trace/0.1", "task": f["task"], "base_model": {"name": fr["family"], "family": fr["family"], "hash": None},
             "input": json.dumps(g.nodes[g.root]["state"], ensure_ascii=False, sort_keys=True),
             "model_output": {}, "verified_output": {"steps": json.dumps(actions, ensure_ascii=False)},
             "fixed_fields": ["steps"], "fixed_by": {"steps": "model"}, "slots": {}, "checker": f["checker"],
             "privacy": f["privacy"], "license": {"kind": "shared", "max_licensees": 10, "exclusive_days": 0},
             "producer": f["producer"],
             "composed": {"failure_id": fid, "graph": g.root, "origin": origin, "rule": "passing_traces",
                          "parents": [{"fragment": k, "weight": round(v, 9)} for k, v in sorted(parts.items())]}}
        tid = object_id(t)
        if self.db.execute("SELECT 1 FROM traces WHERE id=?", (tid,)).fetchone():
            return {"trace": tid, "duplicate": True}
        lot = f"{t['task']}|{t['base_model']['name']}|{ck}"
        self.db.execute("INSERT INTO traces VALUES (?,?,?,?,?,?)",
                        (tid, lot, t["producer"], f["checker"]["id"], canonical(t).decode(), self.epoch))
        c = {"path_str": fr["path"], "confidence": 1.0, "engine": "composition", "signature": fr["signature"],
             "failure_modes": {"steps": fr["modes"][0]} if fr["modes"] else {}}
        self.db.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?,?)",
                        (tid, c["path_str"], c["confidence"], c["engine"], c["signature"], json.dumps(c["failure_modes"]),
                         t["base_model"]["name"], t["task"]))
        self._index_trace(tid, t, c, fid)
        self.db.execute("INSERT INTO joins VALUES (?,?,?,?,?,?,?,?,?)",
                        (tid, fid, g.root, json.dumps(list(path)), origin, json.dumps(parts), json.dumps(who),
                         "passing_traces", self.epoch))
        return {"trace": tid, "origin": origin, "steps": len(path), "actions": actions, "fragments": parts,
                "producers": who}

    def _join_credit(self, trace_info):
        """A path trace's producer share goes to the producers of the passing traces it uses, equally per trace. Its
        nominal producer (whoever's step trace triggered the composition) gets nothing for composing, and a path with no
        passing trace has nobody to pay: {} places nothing, and the payer gets that share back (SPEC 4j)."""
        for tid, who in self.db.execute("SELECT trace, producers FROM joins").fetchall():
            if tid in trace_info:
                trace_info[tid] = dict(trace_info[tid], producer={a: v for a, v in json.loads(who).items() if v > 0})
        return trace_info

    # --- reads --------------------------------------------------------------------------------------------------------
    def _graphs_of(self, failure_id):
        fid = str(failure_id).upper()
        if not self._failure_row(fid):
            raise KeyError(f"failure {fid}")
        return fid, [(StepGraph.from_dict(json.loads(b)), ck) for b, ck in self.db.execute(
            "SELECT body, checker FROM step_graphs WHERE failure=? ORDER BY rowid", (fid,)).fetchall()]

    def step_graphs(self, failure_id):
        fid, graphs = self._graphs_of(failure_id)
        out = []
        for g, ck in graphs:
            best = g.best_path()
            producers = sorted({f["producer"] for f in g.fragments.values()})
            out.append({"root": g.root, "state": g.nodes[g.root]["state"], "checker": ck, "states": len(g.nodes),
                        "steps": len(g.edges), "fragments": len(g.fragments), "producers": len(producers),
                        "failed_fragments": sum(1 for f in g.fragments.values() if f["outcome"] == "fail"),
                        "verified_paths": len(g.solutions),
                        "best": [g.edges[k]["action"] for k in best] if best else None,
                        "best_value": g.score(best) if best else None})
        refused = self.db.execute("SELECT COUNT(*) FROM fragments WHERE failure=? AND accepted=0", (fid,)).fetchone()[0]
        return {"failure_id": fid, "graphs": out, "refused_fragments": refused}

    def frontier(self, failure_id, root="", k=5):
        """Where agents should restart on a failure's cases: per graph, states with a way in and no verified way out."""
        fid, graphs = self._graphs_of(failure_id)
        out = []
        for g, ck in graphs:
            if root and not g.root.startswith(root):
                continue
            alive = None
            try:
                alive = self._env(ck)[2]
            except ValueError:
                pass
            out.append({"root": g.root, "solved": bool(g.solutions),
                        "frontier": g.frontier_states(LIMITS["max_depth"], max(1, min(int(k), 50)), alive=alive)})
        return {"failure_id": fid, "graphs": out,
                "how": "restart from a frontier state: file a step trace with start = {state, depth}"}

    def joins(self, failure_id):
        fid = str(failure_id).upper()
        if not self._failure_row(fid):
            raise KeyError(f"failure {fid}")
        rows = self.db.execute("SELECT trace, root, path, origin, fragments, producers, rule, epoch FROM joins WHERE "
                               "failure=? ORDER BY rowid", (fid,)).fetchall()
        out = []
        for tid, root, path, origin, frags, who, rule, epoch in rows:
            body = json.loads(self.db.execute("SELECT body FROM traces WHERE id=?", (tid,)).fetchone()[0])
            out.append({"trace": tid, "root": root, "origin": origin, "steps": len(json.loads(path)),
                        "actions": json.loads(body["verified_output"]["steps"]), "fragments": json.loads(frags),
                        "producers": json.loads(who), "rule": rule, "epoch": epoch})
        return {"failure_id": fid, "joins": out}

    def tropic_basis(self, failure_id, size=2):
        """What a tropic learning trains on: for each of the failure's cases, the best verified path and, up to `size`,
        paths covering steps the others don't (TROPIC's basis), loop-free when any path is. Returns the path traces to
        cite as the learning's parents, with their actions."""
        fid, graphs = self._graphs_of(failure_id)
        by_path = {(r, p): t for t, r, p in self.db.execute("SELECT trace, root, path FROM joins WHERE failure=?",
                                                               (fid,)).fetchall()}
        out = []
        for g, _ in graphs:
            for path in g.select_basis(size):
                t = by_path.get((g.root, json.dumps(list(path))))
                if t:
                    out.append({"trace": t, "root": g.root, "actions": [g.edges[k]["action"] for k in path],
                                "value": g.score(path)})
        return {"failure_id": fid, "basis": out}

    def _join_of(self, tid):
        r = self.db.execute("SELECT failure, fragments, producers, origin FROM joins WHERE trace=?", (tid,)).fetchone()
        return {"failure_id": r[0], "fragments": json.loads(r[1]), "producers": json.loads(r[2]), "origin": r[3]} if r else None

    def composition_stats(self):
        one = lambda sql: self.db.execute(sql).fetchone()[0]
        return {"step_graphs": one("SELECT COUNT(*) FROM step_graphs"),
                "fragments": one("SELECT COUNT(*) FROM fragments WHERE accepted=1"),
                "refused_fragments": one("SELECT COUNT(*) FROM fragments WHERE accepted=0"),
                "verified_paths": one("SELECT COUNT(*) FROM joins"),
                "cross_producer_paths": sum(1 for (w,) in self.db.execute("SELECT producers FROM joins").fetchall()
                                            if len(json.loads(w)) > 1)}
