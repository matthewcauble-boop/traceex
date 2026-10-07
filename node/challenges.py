"""traceX v0.8: challenge bounties (SPEC 4k). Big open problems, posted free, backed by refundable pledges, paid per
verified improvement. Sats node only (node/sats.py): the retired dollar node has no challenges.

What a challenge is
  * A problem with a deterministic verifier, a direction (maximize or minimize), a baseline (the best score known when
    it was posted), an optional target and a public leaderboard: a `challenge/0.1` file (traceex.challenges), which can
    also be read from or written as a Yukon-style benchmark.json. One problem, one challenge: a post whose key or any
    alias matches an open challenge's, judged the same way (same verifier, instance and direction, or one side with
    no verifier), backs that challenge instead (`merged`), and the second source's aliases and source are added to it
    (a source with a verifier upgrades one that had none). A different verifier under the same key opens its own
    challenge: nobody can squat a problem with a rigged test.
  * A baseline is only as good as its reference: a post's `reference` solution is scored by the node and becomes the
    baseline. A merged post with a better reference raises the best (unpaid) and rebases every pledge to start there,
    so no pledge pays for a result that was already known.
  * Posting is free (the fee only; operator imports and escalations pay none). A challenge nobody pledges to within
    `unbacked_epochs` expires, like a v0.6 bounty, and so does one past its deadline: every backer gets back what its
    pledge still holds.

How it pays (whoever pays judges: the verifier the backers pledged under)
  * Each pledge is its own payment in its own escrow. It is released along a curve of the best verified score alone
    (traceex.challenges.released): with a target, half streams out in proportion to progress toward it and half on
    reaching it; without one, every `scale` of improvement releases half of what is left. A pledge made when the best
    was b (or at the `from_score` its backer names) releases floor(amount x (P(best) - P(b)) / (1 - P(b))). Because P
    depends only on the best score, payouts telescope: many small improvements pay exactly what one improvement to the
    same score pays, so splitting work earns only extra fees.
  * A submission counts as an improvement only if it beats the best by the minimum step (max(min_step, min_step_rel x
    |best|)), beyond twice the validators' standard error where validators measure it. Each improvement's tranche,
    from every pledge, is split solver 70 / traces 20 / checkers 5 / validators 5 (the bounty split) down the solution's
    family tree, vesting vest_epochs, inside the existing _split/_disburse: nothing is paid out that a payer didn't pay
    in, and audit() stays balanced. Reaching the target releases everything left and closes the challenge as solved.
  * Every paid improvement is filed as a trace (the solution, privacy open, its verifier as checker) and a learning
    (kind challenge_solution, accepted), its parents the solution trace and the traces of the leaderboard entries it
    builds on: the ones it names, and any earlier paid solution it mostly repeats (traceex.verifiers.similarity >= 0.5).
    Copying the best with a tweak pays the copied solution's producer part of the traces share, and a learning built on
    the solutions keeps paying them through ordinary usage.

How a submission is verified
  * A `python` verifier the node runs itself (built-ins: traceex.verifiers.BUILTIN; others registered in Python by the
    operator, never over HTTP): run twice, the two scores must agree (the determinism gate), None is invalid. The
    submission's bond comes back at once unless it is paid (then it waits out the vesting window, for prior art).
  * Everything else (hidden instances, Lean proofs without a node-side Lean runner, command benchmarks, an escalated
    failure's pass rate): validators drawn by stake-weighted rendezvous hashing over the beacon after the submission
    (never the submitter) run the verifier in their own sandbox on their own instances, commit sha256(measurement +
    salt), then reveal {"score", "se"?, "n"?} (operator-relayed until signed). The median counts. The submission holds
    a bond (1,000 sats): destroyed if the median says invalid, or if the submission overfits (its public score beats
    the hidden median by more than the overfit tolerance); returned otherwise.

Prior art
  * A result that was already known pays nobody. During the vesting window anyone may stake 2,000 sats on a prior-art
    claim: a reference solution and its provenance (an earlier traceX record, checked by the node, or a dated public
    record, checked by drawn validators). Upheld: tranches still vesting go back into the pledges' escrow, the best and
    the pledges are rebased to the known result, submissions that added nothing beyond it lose their bond (500 sats of
    it to the challenger, the rest destroyed), and the ones beyond it are paid again for the new part. Rejected: the
    stake is destroyed. Paid submissions hold their bond through the window.

Auto-posting
  * Importers (traceex.challenges): AlphaEvolve notebooks (python verifiers), formal-conjectures (Lean), the Erdős
    problems database (status only; merges with the Lean statement), Yukon-style benchmark.json.
  * Escalation: a registry failure open (or regressed) for `escalate_after` epochs whose growth stayed positive for
    `escalate_streak` settlements in a row becomes a challenge: maximize its pass rate on validators' hidden cases,
    target the registry's fixed rate. Unfunded, it expires unless someone backs it.
  * Agents: traceex.autopilot posts one when a failure keeps coming back after its bounty.
"""
import json
import math
import re
import statistics

from traceex import canonical, object_id
from traceex.challenges import (VERSION, direction, improves, min_step, normalize, owed, reached, released,
                                from_yukon, to_yukon)
from traceex.skeleton import find_secrets
from traceex.verifiers import BUILTIN, similarity

CHAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS challenges (id INTEGER PRIMARY KEY, key TEXT, title TEXT, path TEXT, body TEXT, best REAL,
                                       best_sub INT, poster TEXT, origin TEXT, status TEXT, epoch INT, deadline INT,
                                       pledged INT DEFAULT 0, released INT DEFAULT 0, note TEXT, failure_id TEXT,
                                       seq INT, posted_at TEXT);
CREATE TABLE IF NOT EXISTS challenge_keys (key TEXT PRIMARY KEY, challenge INT);
CREATE TABLE IF NOT EXISTS challenge_pledges (id INTEGER PRIMARY KEY, challenge INT, backer TEXT, amount INT,
                                              from_score REAL, epoch INT, payment INT, released INT DEFAULT 0,
                                              judge TEXT, level REAL, base INT, base_released INT DEFAULT 0);
CREATE INDEX IF NOT EXISTS challenge_pledges_c ON challenge_pledges(challenge);
CREATE TABLE IF NOT EXISTS challenge_subs (id INTEGER PRIMARY KEY, challenge INT, submitter TEXT, digest TEXT, body TEXT,
                                           public_score REAL, score REAL, se REAL, status TEXT, epoch INT, parents TEXT,
                                           trace TEXT, learning TEXT, bond INT DEFAULT 0, paid INT DEFAULT 0,
                                           prev_best REAL, note TEXT, bond_release INT);
CREATE UNIQUE INDEX IF NOT EXISTS challenge_subs_digest ON challenge_subs(challenge, digest);
CREATE TABLE IF NOT EXISTS challenge_assign (sub INT, validator TEXT, epoch INT, PRIMARY KEY (sub, validator));
CREATE TABLE IF NOT EXISTS challenge_commits (sub INT, validator TEXT, digest TEXT, epoch INT, PRIMARY KEY (sub, validator));
CREATE TABLE IF NOT EXISTS challenge_reveals (sub INT, validator TEXT, score REAL, se REAL, n INT, epoch INT,
                                              PRIMARY KEY (sub, validator));
CREATE TABLE IF NOT EXISTS prior_claims (id INTEGER PRIMARY KEY, challenge INT, challenger TEXT, reference TEXT,
                                         provenance TEXT, score REAL, status TEXT, epoch INT, prior_row INT, note TEXT);
CREATE TABLE IF NOT EXISTS prior_assign (claim INT, validator TEXT, epoch INT, PRIMARY KEY (claim, validator));
CREATE TABLE IF NOT EXISTS prior_commits (claim INT, validator TEXT, digest TEXT, epoch INT, PRIMARY KEY (claim, validator));
CREATE TABLE IF NOT EXISTS prior_reveals (claim INT, validator TEXT, prior INT, score REAL, epoch INT,
                                          PRIMARY KEY (claim, validator));
CREATE TABLE IF NOT EXISTS escalations (failure TEXT PRIMARY KEY, streak INT, last_epoch INT, challenge INT);
"""
LIMITS = {"solution_bytes": 48 * 1024, "open_per_poster": 20, "epochs": 520, "parents": 8}
SPLIT = {"trainer": 0.70, "traces": 0.20, "checkers": 0.05, "validators": 0.05}   # the bounty split (exchange.BOUNTY_SPLIT)
SIMILAR = 0.5                      # repeating half an earlier paid solution's numbers (or words) makes it a parent
STATUSES = ("open", "solved", "expired", "removed")


class Challenges:
    """Mixed into the sats node (node/sats.py)."""

    def _open_challenges(self):
        self.db.executescript(CHAL_SCHEMA)
        self.verifiers = getattr(self, "verifiers", {})
        for vid, fn in BUILTIN.items():
            self.verifiers.setdefault(vid, fn)

    # --- operator registrations (Python only, never over HTTP: they run code) ------------------------------------------
    def register_verifier(self, verifier_id, fn, author=None):
        """A verifier the node runs itself: fn(solution, instance) -> score, or None (invalid). `author`, if given, is
        paid the checkers' 5% of every tranche a solution it verified earns (the built-ins: the operator's)."""
        self.verifiers[str(verifier_id)] = fn
        if author:                                   # traces name a checker by id; the version is part of the trace
            self.register_checker(str(verifier_id).split("@")[0], author)
        return {"verifier": verifier_id}

    def _author_of(self, vid, poster):
        """Who is paid a verifier's checker share: whoever registered it, else the poster (for a verifier only it can
        supply, such as a command benchmark), else the operator."""
        cid = str(vid).split("@")[0]
        if self.db.execute("SELECT 1 FROM checkers WHERE id=?", (cid,)).fetchone():
            return
        who = poster if poster and re.fullmatch(r"0x[0-9a-fA-F]{40}", str(poster)) and vid not in BUILTIN else self.fee_to
        self.db.execute("INSERT OR IGNORE INTO checkers VALUES (?,?)", (cid, who))

    # --- posting -----------------------------------------------------------------------------------------------------
    def post_challenge(self, b, origin="posted"):
        """Post a challenge, free (the fee only). `b` is a challenge/0.1 file with `poster` (and optional
        `seed_msats` / `seed_sats`, the poster's first pledge), or {"format": "yukon", "benchmark": {...}, "baseline"}.
        A post whose key or any alias matches an open challenge backs that one (`merged`)."""
        b = dict(b or {})
        poster, seed = b.pop("poster", None), b.pop("seed_msats", None) or 1000 * int(b.pop("seed_sats", 0) or 0)
        if origin == "posted":
            from exchange import need_address
            need_address(poster, "poster")
        if b.get("format") == "yukon":
            files = from_yukon(b["benchmark"], baseline=b.get("baseline"))
            if len(files) != 1:
                return {"posted": [self.post_challenge(dict(f, poster=poster), origin) for f in files]}
            b = files[0]
        ref = b.pop("reference", None)
        c = normalize(b)
        ref_score = None
        if ref is not None:                # a baseline is only as good as its reference: the node scores it itself
            if c["verifier"].get("id") not in self.verifiers or (c.get("instances") or {}).get("hidden"):
                raise ValueError("a reference solution needs a verifier this node runs (and no hidden instances)")
            ref_score = self._run_verifier(c["verifier"], ref)
            if ref_score is None:
                raise ValueError("the reference solution fails the verifier")
            c = normalize(dict(b, metric=dict(b.get("metric") or {}, baseline=ref_score)))
        epochs = max(1, min(int(c.pop("epochs", 26) or 26), LIMITS["epochs"]))
        keys = [c["key"], *c["aliases"]]
        self._room()
        with self.lock:
            if seed:
                self._need_funds(poster, int(seed))
            hit = None
            for (cand,) in self.db.execute("SELECT DISTINCT k.challenge FROM challenge_keys k JOIN challenges c ON c.id = "
                                           f"k.challenge WHERE c.status='open' AND k.key IN ({','.join('?' * len(keys))}) "
                                           "ORDER BY k.challenge", keys).fetchall():
                if self._compatible(self._body(cand), c):
                    hit = (cand,)
                    break
            if hit:
                cid = hit[0]
                if origin == "posted":
                    self._tx_fee(poster)
                self._merge(cid, c)
                if ref_score is not None:
                    self._raise_best(cid, ref, ref_score, poster, c.get("source"))
                self.db.commit()
                out = {"id": cid, "status": "open", "merged": True,
                       "note": f"challenge #{cid} is already open for this problem: this post backs it (and adds its "
                               "source and aliases) instead of opening a duplicate"}
            else:
                if origin == "posted" and self.test_credits:
                    n = self.db.execute("SELECT COUNT(*) FROM challenges WHERE poster=? AND status='open'",
                                        (poster,)).fetchone()[0]
                    if n >= LIMITS["open_per_poster"]:
                        raise ValueError(f"{LIMITS['open_per_poster']} open challenges per poster")
                if origin == "posted":
                    self._tx_fee(poster)
                c["sources"] = [c.pop("source")] if c.get("source") else []
                cid = self.db.execute(
                    "INSERT INTO challenges (key, title, path, body, best, poster, origin, status, epoch, deadline, "
                    "failure_id, seq, posted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (c["key"], c["title"], c["path"], json.dumps(c, ensure_ascii=False), c["metric"]["baseline"],
                     poster, origin, "open", self.epoch, self.epoch + epochs, b.get("failure_id"),
                     self.db.execute("SELECT COALESCE(MAX(id), 0) FROM challenge_subs").fetchone()[0],
                     self._now())).lastrowid
                for k in keys:
                    self.db.execute("INSERT OR REPLACE INTO challenge_keys VALUES (?,?)", (k, cid))
                if c["verifier"].get("id"):
                    self._author_of(c["verifier"]["id"], c["verifier"].get("author") or poster)
                if ref_score is not None:
                    self._reference_row(cid, ref, ref_score, poster, c["sources"][0] if c["sources"] else None)
                self._event(f"challenge #{cid} posted free ({origin}): {c['title'][:120]}")
                self.db.commit()
                out = {"id": cid, "status": "open", "deadline_epoch": self.epoch + epochs,
                       "unbacked_expires_epoch": self.epoch + self.unbacked_epochs, "key": c["key"]}
        if seed:
            out["pledge"] = self.pledge_challenge(out["id"], poster, int(seed))
        return out

    @staticmethod
    def _compatible(body, c):
        """Two posts are the same challenge only if they are judged the same way: the same verifier and instance and
        direction, or one of them has no verifier yet. A post that only borrows another problem's key or alias with a
        verifier of its own opens its own challenge (so nobody can squat a problem's key with a different test)."""
        a, b = body["verifier"], c["verifier"]
        if a.get("kind") == "none" or b.get("kind") == "none":
            return True
        public = a.get("kind") in ("python", "lean4")        # registry / command instances are each holder's own
        return (a.get("id"), a.get("instance") if public else None, body["metric"]["direction"]) == \
            (b.get("id"), b.get("instance") if public else None, c["metric"]["direction"])

    def _reference_row(self, cid, ref, score, poster, source):
        """A verified reference solution on the board: the record a challenge starts from (never paid)."""
        dig = object_id({"solution": ref})
        self.db.execute("INSERT OR IGNORE INTO challenge_subs (challenge, submitter, digest, body, public_score, score, se, "
                        "status, epoch, parents, prev_best, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (int(cid), poster or "", dig, json.dumps({"solution": ref, "meta": {}}), score, score, 0.0,
                         "reference", self.epoch, "[]", score,
                         f"reference from {(source or {}).get('name') or 'the poster'}"))

    def _raise_best(self, cid, ref, score, poster, source):
        """A merged post brought a verified reference better than the board's best by the minimum step: the best
        rises to it, unpaid, and every pledge is rebased so it never pays for progress that was already known."""
        c, body = self._row(cid), self._body(cid)
        m = body["metric"]
        if not improves(m, c["best"], score):
            return
        self._reference_row(cid, ref, score, poster, source)
        d = direction(m)
        for pk, start, amount, done in self.db.execute(
                "SELECT id, from_score, amount, released FROM challenge_pledges WHERE challenge=?", (int(cid),)).fetchall():
            if d * (score - start) > 0:
                self.db.execute("UPDATE challenge_pledges SET from_score=?, level=?, base=?, base_released=? WHERE id=?",
                                (score, score, amount - done, done, pk))
        self.db.execute("UPDATE challenges SET best=? WHERE id=?", (score, int(cid)))
        if reached(m, score):
            self._refund_challenge(int(cid), "already solved by a known result")
            self.db.execute("UPDATE challenges SET status='solved', note='a known result reaches the target' WHERE id=?",
                            (int(cid),))
        self._event(f"challenge #{cid}: a verified reference raised the best to {score:.6g} (unpaid; pledges rebased)")

    def _merge(self, cid, c):
        """The same problem from another source: add its aliases and source; a verifier upgrades none."""
        body = self._body(cid)
        have = {s.get("url") or s.get("name") for s in body.get("sources", [])}
        if c.get("source") and (c["source"].get("url") or c["source"].get("name")) not in have:
            body.setdefault("sources", []).append(c["source"])
        body["aliases"] = sorted(set(body.get("aliases", [])) | set(c["aliases"]) | ({c["key"]} - {body["key"]}))
        if body["verifier"].get("kind") == "none" and c["verifier"].get("kind") != "none":
            body["verifier"], body["metric"], body["statement"] = c["verifier"], c["metric"], c["statement"]
            if c.get("instances"):
                body["instances"] = c["instances"]
            self._author_of(c["verifier"]["id"], None)
        for k in [c["key"], *c["aliases"]]:
            self.db.execute("INSERT OR IGNORE INTO challenge_keys VALUES (?,?)", (k, cid))
        self.db.execute("UPDATE challenges SET body=? WHERE id=?", (json.dumps(body, ensure_ascii=False), cid))
        self._event(f"challenge #{cid}: the same problem from another source merged in")

    # --- reads -------------------------------------------------------------------------------------------------------
    def _body(self, cid):
        r = self.db.execute("SELECT body FROM challenges WHERE id=?", (int(cid),)).fetchone()
        if not r:
            raise KeyError(f"challenge {cid}")
        return json.loads(r[0])

    def _row(self, cid):
        r = self.db.execute("SELECT id, key, title, path, best, best_sub, poster, origin, status, epoch, deadline, pledged, "
                            "released, note, failure_id, seq, posted_at FROM challenges WHERE id=?", (int(cid),)).fetchone()
        if not r:
            raise KeyError(f"challenge {cid}")
        return dict(zip(("id", "key", "title", "path", "best", "best_sub", "poster", "origin", "status", "posted_epoch",
                         "deadline_epoch", "pledged_msats", "released_msats", "note", "failure_id", "seq", "posted_at"), r))

    def _card(self, cid):
        row, body = self._row(cid), self._body(cid)
        held = sum(self._held(p) for (p,) in self.db.execute("SELECT payment FROM challenge_pledges WHERE challenge=?",
                                                               (int(cid),)))
        m = body["metric"]
        return dict(row, metric=m["name"], direction=m["direction"], baseline=m["baseline"], target=m.get("target"),
                    min_step=min_step(m, row["best"]), verifier=body["verifier"].get("id") or None,
                    verifier_kind=body["verifier"]["kind"], sources=[s.get("name") for s in body.get("sources", [])],
                    aliases=body.get("aliases", []), escrow_msats=held,
                    backers=self.db.execute("SELECT COUNT(DISTINCT backer) FROM challenge_pledges WHERE challenge=?",
                                            (int(cid),)).fetchone()[0],
                    submissions=self.db.execute("SELECT COUNT(*) FROM challenge_subs WHERE challenge=?",
                                                (int(cid),)).fetchone()[0],
                    released_share=round(released(m, row["best"]), 6))

    def challenges(self, status="open", path="", q="", origin="", limit=50):
        """Challenges, newest first, filtered by status, branch (its subtree), words in the title and origin."""
        if status and status not in STATUSES:
            raise ValueError(f"status is one of {STATUSES}")
        p, words = (path or "").strip("/"), [w for w in re.findall(r"\w+", (q or "").lower())]
        out = []
        for (cid,) in self.db.execute("SELECT id FROM challenges ORDER BY id DESC").fetchall():
            c = self._card(cid)
            if status and c["status"] != status:
                continue
            if p and not (c["path"] == p or c["path"].startswith(p + "/")):
                continue
            if origin and not c["origin"].startswith(origin):
                continue
            if words and not all(w in (c["title"] + " " + " ".join(c["aliases"]) + " " + c["key"]).lower() for w in words):
                continue
            out.append(c)
            if len(out) >= max(1, min(int(limit), 500)):
                break
        return {"challenges": out, "count": len(out)}

    def get_challenge(self, cid):
        body = self._body(cid)
        return dict(self._card(cid), challenge=body, leaderboard=self.leaderboard(cid, 10)["leaderboard"])

    def leaderboard(self, cid, limit=50):
        """Scored submissions, best first (by the score that counts: the validators' median where they measure)."""
        c, body = self._row(cid), self._body(cid)
        d = direction(body["metric"])
        rows = self.db.execute("SELECT id, submitter, score, public_score, se, status, epoch, paid, learning, trace, "
                               "prev_best FROM challenge_subs WHERE challenge=?", (int(cid),)).fetchall()
        keys = ("id", "submitter", "score", "public_score", "se", "status", "epoch", "paid_msats", "learning", "trace",
                "prev_best")
        subs = [dict(zip(keys, r)) for r in rows]
        scored = sorted((s for s in subs if s["score"] is not None),
                        key=lambda s: (-d * s["score"], s["epoch"], s["id"]))
        pending = [s for s in subs if s["status"] == "pending"]
        return {"challenge": int(cid), "best": c["best"], "baseline": body["metric"]["baseline"],
                "direction": body["metric"]["direction"], "leaderboard": scored[:max(1, int(limit))],
                "pending": pending, "count": len(scored)}

    def challenge_backers(self, cid):
        self._row(cid)
        rows = self.db.execute("SELECT backer, amount, from_score, released, payment, judge, level FROM "
                               "challenge_pledges WHERE challenge=? ORDER BY id", (int(cid),)).fetchall()
        return {"challenge": int(cid), "pledges": [{"backer": b, "pledged_msats": a, "from_score": f,
                                                    "released_msats": r, "held_msats": self._held(p), "judge": j,
                                                    "paid_up_to": lv}
                                                   for b, a, f, r, p, j, lv in rows]}

    def export_challenge(self, cid, fmt="yukon"):
        if fmt != "yukon":
            raise ValueError("format: yukon (benchmark.json and, for a python verifier, verify.py)")
        body = self._body(cid)
        c = {k: v for k, v in body.items() if k not in ("sources",)}
        c["source"] = (body.get("sources") or [{}])[0]
        return {"challenge": int(cid), "format": "yukon", "files": to_yukon(c)}

    def _held(self, pid):
        live = self.db.execute("SELECT COALESCE(SUM(msats), 0) FROM vesting WHERE payment=? AND status='vesting'",
                               (pid,)).fetchone()[0]
        return self._bal(f"pay:{pid}") - live

    # --- backing -----------------------------------------------------------------------------------------------------
    def pledge_challenge(self, cid, backer, msats, from_score=None, judge=None):
        """Pledge sats into a challenge's escrow, its own payment. It is released to verified improvements along the
        challenge's curve, counted from `from_score` (default: the best now, never anything worse), and what is left
        comes back at expiry. Where the node scores submissions itself (a deterministic verifier anyone can re-run),
        improvements pay at once; where validators measure them, this pledge pays only once its `judge` (default: the
        backer; or whoever it names, such as the poster who holds the hidden instances) has confirmed the improvement
        with its own measurement: whoever pays judges, so no validator verdict moves a backer's money by itself. A
        pledge buys nothing: no token, no share, nothing to trade."""
        from exchange import need_address
        need_address(backer, "backer")
        judge = need_address(judge, "judge") if judge else backer
        msats = int(msats)
        if msats <= 0:
            raise ValueError("a pledge must be more than 0 msats")
        with self.lock:
            c, body = self._row(cid), self._body(cid)
            if c["status"] != "open":
                raise ValueError(f"challenge {cid} is {c['status']}")
            m, d = body["metric"], direction(body["metric"])
            start = c["best"] if from_score is None else float(from_score)
            if d * (start - c["best"]) < 0:
                start = c["best"]                         # a pledge never pays for progress made before it
            if released(m, start) >= 1:
                raise ValueError("the target is already reached")
            pid = self._pay_in(backer, "cpledge", f"challenge:{int(cid)}", msats)
            self.db.execute("INSERT INTO challenge_pledges (challenge, backer, amount, from_score, epoch, payment, judge, "
                            "level, base) VALUES (?,?,?,?,?,?,?,?,?)", (int(cid), backer, msats, start, self.epoch, pid,
                                                                       judge, start, msats))
            self.db.execute("UPDATE challenges SET pledged=pledged+? WHERE id=?", (msats, int(cid)))
            self._event(f"challenge #{cid} backed with {self._fmt(msats)}")
            self.db.commit()
        return {"challenge": int(cid), "pledged_msats": msats, "from_score": start, "payment": pid, "judge": judge,
                "refund": "what is not released to verified improvements comes back when the challenge ends"}

    # --- submissions -------------------------------------------------------------------------------------------------
    def submit_solution(self, cid, b):
        """Submit a solution: {submitter, solution (JSON) or artifact {uri, hash}, outputs? (on public instances),
        public_score?, parents? (submission ids it builds on), model?, per_call_msats?}. A verifier the node runs scores
        it at once; otherwise drawn validators measure it (a 1,000-sat bond is held). Duplicates are refused."""
        from exchange import need_address
        submitter = need_address(b.get("submitter"), "submitter")
        with self.lock:
            c, body = self._row(cid), self._body(cid)
            if c["status"] != "open":
                raise ValueError(f"challenge {cid} is {c['status']}")
            v = body["verifier"]
            if v["kind"] == "none":
                raise ValueError("this challenge lists a problem with no verifier yet (a source with one merges in "
                                 "later); nothing can be scored")
            sol = {k: b[k] for k in ("solution", "artifact", "outputs") if b.get(k) is not None}
            if not sol:
                raise ValueError("send a solution (JSON) or an artifact {uri, hash}")
            raw = canonical(sol)
            if len(raw) > LIMITS["solution_bytes"]:
                raise ValueError(f"a submission is at most {LIMITS['solution_bytes'] // 1024} KB (larger artifacts: "
                                 "send {uri, hash})")
            if find_secrets(raw.decode("utf-8", "replace")):
                raise ValueError("rejected: the submission contains something that looks like a secret")
            dig = object_id(sol)
            dup = self.db.execute("SELECT id, submitter FROM challenge_subs WHERE challenge=? AND digest=?",
                                  (int(cid), dig)).fetchone()
            if dup:
                raise ValueError(f"already submitted as #{dup[0]}: the first submitter holds it")
            parents = sorted({int(x) for x in (b.get("parents") or [])[:LIMITS["parents"]]})
            local = v["id"] in self.verifiers and not body.get("instances", {}).get("hidden")
            bond = self.p.challenge_sub_bond_msats           # held through the vesting window if it is paid
            self._need_funds(submitter, bond)
            self._tx_fee(submitter)
            if bond:
                self._move(submitter, f"csubbond:{cid}:{dig[7:23]}", bond, "challenge submission bond")
            public = None
            if v["id"] in self.verifiers and "solution" in sol:
                public = self._run_verifier(v, sol["solution"])
            elif b.get("public_score") is not None:
                public = float(b["public_score"])
            meta = {k: b[k] for k in ("model", "per_call_msats", "note") if b.get(k) is not None}
            sid = self.db.execute(
                "INSERT INTO challenge_subs (challenge, submitter, digest, body, public_score, status, epoch, parents, "
                "bond, prev_best) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (int(cid), submitter, dig, json.dumps(dict(sol, meta=meta), ensure_ascii=False), public,
                 "pending", self.epoch, json.dumps(parents), bond, c["best"])).lastrowid
            if local:
                self._record_score(sid, public, [])
            elif self.beacon_delay == 0:
                self._draw_sub(sid)
            self.db.commit()
        return self.submission(sid)

    def _run_verifier(self, v, solution):
        fn = self.verifiers[v["id"]]
        try:
            a, b = fn(solution, v.get("instance") or {}), fn(solution, v.get("instance") or {})
        except Exception:                                  # a verifier that throws has said: invalid
            return None
        if a is None or b is None or not math.isfinite(float(a)) or float(a) != float(b):
            return None
        return float(a)

    def submission(self, sid):
        r = self.db.execute("SELECT id, challenge, submitter, public_score, score, se, status, epoch, parents, trace, "
                            "learning, bond, paid, prev_best, note FROM challenge_subs WHERE id=?", (int(sid),)).fetchone()
        if not r:
            raise KeyError(f"submission {sid}")
        keys = ("id", "challenge", "submitter", "public_score", "score", "se", "status", "epoch", "parents", "trace",
                "learning", "bond_msats", "paid_msats", "prev_best", "note")
        out = dict(zip(keys, r))
        out["parents"] = json.loads(out["parents"] or "[]")
        out["validators"] = [a for (a,) in self.db.execute("SELECT validator FROM challenge_assign WHERE sub=?",
                                                          (int(sid),))]
        return out

    # --- validators ----------------------------------------------------------------------------------------------------
    def _draw_sub(self, sid):
        if self.db.execute("SELECT 1 FROM challenge_assign WHERE sub=?", (int(sid),)).fetchone():
            return
        submitter = self.db.execute("SELECT submitter FROM challenge_subs WHERE id=?", (int(sid),)).fetchone()[0]
        vals = self._eligible(exclude=(submitter,))
        if len(vals) < max(self.p.quorum, 1):
            return
        for a, _ in sorted(vals, key=lambda v: self._score(f"csub:{sid}", 0, *v))[:max(self.p.quorum, 1)]:
            self.db.execute("INSERT INTO challenge_assign VALUES (?,?,?)", (int(sid), a, self.epoch))

    def commit_solution(self, sid, validator, digest):
        """Operator-relayed validator message: sha256(measurement + salt) before any reveal opens."""
        with self.lock:
            s = self.submission(sid)
            if s["status"] != "pending":
                raise ValueError(f"submission {sid} is {s['status']}")
            if validator not in s["validators"]:
                raise PermissionError("this validator was not drawn for that submission (the draw is random)")
            if self.db.execute("SELECT 1 FROM challenge_commits WHERE sub=? AND validator=?", (int(sid), validator)).fetchone():
                raise ValueError("already committed")
            self._tx_fee(validator)
            self.db.execute("INSERT INTO challenge_commits VALUES (?,?,?,?)", (int(sid), validator, str(digest), self.epoch))
            self.db.commit()
        return {"submission": int(sid), "committed": validator}

    def reveal_solution(self, sid, validator, measurement, salt=""):
        """Operator-relayed validator message: {"score": number or null (invalid), "se"?: standard error, "n"?: hidden
        instances}, measured with the challenge's verifier on the validator's own instances. Must match the commitment;
        reveals open once every drawn validator has committed (or the next epoch)."""
        from registry import measurement_digest
        with self.lock:
            s = self.submission(sid)
            if s["status"] != "pending":
                raise ValueError(f"submission {sid} is {s['status']}")
            if validator not in s["validators"]:
                raise PermissionError("this validator was not drawn for that submission (the draw is random)")
            c = self.db.execute("SELECT digest, epoch FROM challenge_commits WHERE sub=? AND validator=?",
                                (int(sid), validator)).fetchone()
            if not c:
                raise ValueError("commit first")
            committed = {v for (v,) in self.db.execute("SELECT validator FROM challenge_commits WHERE sub=?", (int(sid),))}
            if committed != set(s["validators"]) and self.epoch <= c[1]:
                raise ValueError("reveals open once every drawn validator has committed, or next epoch")
            if measurement_digest(measurement, salt) != c[0]:
                raise ValueError("this measurement does not match the commitment")
            if self.db.execute("SELECT 1 FROM challenge_reveals WHERE sub=? AND validator=?", (int(sid), validator)).fetchone():
                raise ValueError("already revealed")
            sc = (measurement or {}).get("score")
            if sc is not None and not math.isfinite(float(sc)):
                raise ValueError("score is a finite number, or null for invalid")
            self._tx_fee(validator)
            self.db.execute("INSERT INTO challenge_reveals VALUES (?,?,?,?,?,?)",
                            (int(sid), validator, None if sc is None else float(sc),
                             float(measurement.get("se") or 0), int(measurement.get("n") or 0), self.epoch))
            done = self.db.execute("SELECT COUNT(*) FROM challenge_reveals WHERE sub=?", (int(sid),)).fetchone()[0]
            if done >= len(s["validators"]):
                self._finalize_sub(sid)
            self.db.commit()
        return self.submission(sid)

    def _finalize_sub(self, sid):
        """The validators' median decides: invalid (most said so) or a score; the bond comes back unless the
        submission was invalid or overfit its public instances."""
        rows = self.db.execute("SELECT validator, score, se FROM challenge_reveals WHERE sub=?", (int(sid),)).fetchall()
        if not rows:
            return
        valid = [(v, s, e) for v, s, e in rows if s is not None]
        if len(valid) * 2 <= len(rows):
            return self._record_score(sid, None, [])
        med = statistics.median(s for _, s, _ in valid)
        se = statistics.median(e for _, _, e in valid)
        agreed = [v for v, s, e in valid if abs(s - med) <= max(3 * max(e, se), 1e-9)]
        self._record_score(sid, med, agreed, se)

    def _record_score(self, sid, score, validators, se=0.0):
        """Record a submission's verified score; pay it if it improves on the best by the minimum step."""
        sub = self.submission(sid)
        cid = sub["challenge"]
        c, body = self._row(cid), self._body(cid)
        m, d = body["metric"], direction(body["metric"])
        bond = sub["bond_msats"]
        acct = f"csubbond:{cid}:{self.db.execute('SELECT digest FROM challenge_subs WHERE id=?', (int(sid),)).fetchone()[0][7:23]}"
        status, note = "scored", ""
        if score is None:
            status, note = "invalid", "the verifier's gate refused it"
        elif sub["public_score"] is not None and validators and \
                d * (sub["public_score"] - score) > max(self.p.challenge_overfit_steps * min_step(m, c["best"]), 3 * se):
            status, note = "overfit", (f"public {sub['public_score']:.6g} against {score:.6g} on the validators' hidden "
                                       "instances")
        elif c["status"] == "open" and improves(m, c["best"], score, se):
            status = "improved"
        else:
            note = "not better than the best by the minimum step" if score is not None else ""
        measured = bool(self.db.execute("SELECT 1 FROM challenge_assign WHERE sub=?", (int(sid),)).fetchone())
        if bond:
            if status in ("invalid", "overfit") and measured:
                self._forfeit(acct, self._bal(acct), "challenge submission bond", f"{cid}:{sid}")
            elif status == "improved":                      # held through the vesting window: prior art can take it
                self.db.execute("UPDATE challenge_subs SET bond_release=? WHERE id=?",
                                (self.epoch + self.p.vest_epochs, int(sid)))
            else:
                self._move(acct, sub["submitter"], self._bal(acct), "challenge submission bond returned", payout=True)
        self.db.execute("UPDATE challenge_subs SET score=?, se=?, status=?, note=? WHERE id=?",
                        (score, se, status, note, int(sid)))
        if status == "improved":
            self._pay_improvement(cid, sid, c["best"], score, validators, pay=not measured)

    # --- paying an improvement -------------------------------------------------------------------------------------------
    def _pay_pledges(self, cid, sid, level, pledges):
        """Release each pledge's tranche for progress up to `level` through submission `sid`'s learning: owed along
        the curve from the pledge's from_score, less what it already released (so payouts telescope)."""
        m = self._body(cid)["metric"]
        lid, submitter = self.db.execute("SELECT learning, submitter FROM challenge_subs WHERE id=?",
                                         (int(sid),)).fetchone()
        tree, paid, d = self._split_tree(), 0, direction(m)
        for pk, pid, amount, start, done, lvl in pledges:
            if not improves(m, start if lvl is None else lvl, level):
                continue
            tranche = min(owed(amount, m, start, level) - done, self._held(pid))
            self.db.execute("UPDATE challenge_pledges SET level=? WHERE id=?", (level, pk))
            if tranche <= 0:
                continue
            got = self._split(pid, lid, SPLIT, tree, vest=("all",), amount=tranche)
            paid += got.get(submitter, 0)
            self.db.execute("UPDATE challenge_pledges SET released=released+? WHERE id=?", (tranche, pk))
            self.db.execute("UPDATE challenges SET released=released+? WHERE id=?", (tranche, int(cid)))
        self.db.execute("UPDATE challenge_subs SET paid=paid+? WHERE id=?", (paid, int(sid)))
        return paid

    def _pledge_rows(self, cid, judge=None):
        sql = ("SELECT id, payment, base, from_score, released - base_released, level FROM challenge_pledges WHERE "
               "challenge=?"
               + (" AND judge=?" if judge else "") + " ORDER BY id")
        return self.db.execute(sql, (int(cid), judge) if judge else (int(cid),)).fetchall()

    def _pay_improvement(self, cid, sid, old, new, validators, pay=True):
        """A verified improvement: file the solution, move the best, and (scored by the node itself) pay every
        pledge's tranche now. Measured by validators, each pledge waits for its judge's confirmation."""
        body = self._body(cid)
        m = body["metric"]
        self._file_solution(cid, sid, old, new, validators)
        self.db.execute("UPDATE challenge_subs SET status='best' WHERE id=?", (int(sid),))
        if pay:
            self._pay_pledges(cid, sid, new, self._pledge_rows(cid))
        solved = reached(m, new) and pay
        self.db.execute("UPDATE challenges SET best=?, best_sub=?, status=? WHERE id=?",
                        (new, int(sid), "solved" if solved else "open", int(cid)))
        self.db.execute("UPDATE challenge_subs SET status='improved' WHERE challenge=? AND status='best' AND id != ?",
                        (int(cid), int(sid)))
        self._event(f"challenge #{cid}: {m['name']} {old:.6g} → {new:.6g}" + (" (target reached: solved)" if solved else ""))

    def confirm_solution(self, sid, judge, measurement):
        """A pledge's judge confirms a validator-measured submission with its own measurement ({"score"}), operator-
        relayed until signed, as a bounty poster's is. Every pledge this judge holds pays its tranche up to the lower of
        the judge's score and the validators' (the better side of neither), through that submission's learning, if
        that beats what the pledge has already paid for by the minimum step. The board's best doesn't matter here, so a
        fake best that a captured validator majority put on the board can't block honest work from being paid."""
        with self.lock:
            s = self.submission(sid)
            cid = s["challenge"]
            c, body = self._row(cid), self._body(cid)
            m, d = body["metric"], direction(body["metric"])
            if s["status"] not in ("best", "improved", "scored", "rebased") or s["score"] is None:
                raise ValueError(f"submission {sid} is {s['status']}: only a scored submission can be confirmed")
            mine = self._pledge_rows(cid, judge)
            if not mine:
                raise PermissionError("this address judges no pledge on that challenge")
            js = float((measurement or {}).get("score"))
            level = min(js, s["score"]) if d > 0 else max(js, s["score"])
            self._tx_fee(judge)
            if not s["learning"]:              # below a (perhaps captured) board's best, but better than this judge paid
                vals = [v for (v,) in self.db.execute("SELECT validator FROM challenge_reveals WHERE sub=?", (int(sid),))]
                self._file_solution(cid, sid, s["prev_best"], s["score"], vals)
            paid = self._pay_pledges(cid, sid, level, mine)
            if reached(m, c["best"]) and not any(self._held(p) for (_, p, *_r) in self._pledge_rows(cid)):
                self.db.execute("UPDATE challenges SET status='solved' WHERE id=?", (int(cid),))
            self._event(f"challenge #{cid}: a backer's judge confirmed submission #{sid}")
            self.db.commit()
        return dict(self.submission(sid), confirmed_level=level, paid_now_msats=paid)

    def _file_solution(self, cid, sid, old, new, validators):
        """The improvement as a trace (the solution, open) and an accepted learning (kind challenge_solution) whose
        parents are its own trace and the paid solutions it builds on, so reuse keeps paying their producers."""
        sub = self.submission(sid)
        body, sb = self._body(cid), json.loads(self.db.execute("SELECT body FROM challenge_subs WHERE id=?",
                                                               (int(sid),)).fetchone()[0])
        v, m = body["verifier"], body["metric"]
        meta = sb.pop("meta", {})
        ck_id, _, ck_ver = v["id"].partition("@")
        shown = canonical(sb).decode()
        t = {"v": "trace/0.1", "task": f"challenge.{cid}", "base_model": {"name": str(meta.get("model") or "unspecified")[:120],
                                                                        "hash": None},
             "input": f"{body['title']}\n{body['statement'][:4000]}", "model_output": {"solution": f"best {old:.12g}"},
             "verified_output": {"solution": shown[:16_000]}, "fixed_fields": ["solution"],
             "fixed_by": {"solution": "unknown"}, "slots": {}, "checker": {"id": ck_id, "version": ck_ver or "1", "hash": None},
             "privacy": "open", "license": {"kind": "shared", "max_licensees": 10, "exclusive_days": 0},
             "producer": sub["submitter"], "created": self._now(), "failure_modes": {"solution": "improvement"},
             "feedback": [f"{m['name']} {old:.12g} -> {new:.12g} ({m['direction']}), challenge #{cid}, submission #{sid}"]}
        tid = object_id(t)
        if not self.db.execute("SELECT 1 FROM traces WHERE id=?", (tid,)).fetchone():
            self.db.execute("INSERT INTO traces VALUES (?,?,?,?,?,?)",
                            (tid, f"challenge.{cid}|{t['base_model']['name']}|{v['id']}", sub["submitter"], ck_id,
                             canonical(t).decode(), self.epoch))
            lab = {"path_str": body["path"], "confidence": 1.0, "engine": "challenge", "signature": "solution:improvement",
                   "failure_modes": {"solution": "improvement"}}
            self.db.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?,?)",
                            (tid, body["path"], 1.0, "challenge", lab["signature"], json.dumps(lab["failure_modes"]),
                             t["base_model"]["name"], t["task"]))
            self._index_trace(tid, t, lab, f"challenge:{cid}")
            self.db.execute("INSERT OR IGNORE INTO dups (trace, canonical, key, fix) VALUES (?,?,?,?)",
                            (tid, tid, tid, tid))
        parents = [tid] + [x for x in self._lineage(cid, sid, sb, sub["parents"]) if x != tid]
        art = sb.get("artifact") or {}
        L = {"v": "learning/0.1", "kind": "challenge_solution", "task": t["task"], "base_model": t["base_model"],
             "artifact": {"uri": art.get("uri") or f"challenge:{cid}/submission:{sid}",
                          "hash": art.get("hash") or self.db.execute("SELECT digest FROM challenge_subs WHERE id=?",
                                                                     (int(sid),)).fetchone()[0]},
             "parents": [{"trace": p, "weight": round(1 / len(parents), 9)} for p in parents],
             "trainer": sub["submitter"],
             "attestation": {"validator": "node" if not validators else "validators", "eval_set": f"challenge:{cid}",
                             "metric": m["name"], "before": old, "after": new},
             "royalty": {"per_call_msats": max(0, int(meta.get("per_call_msats") or 0)), "split": dict(SPLIT)}}
        lid = object_id(L)
        self.db.execute("INSERT OR IGNORE INTO learnings VALUES (?,?,?)", (lid, canonical(L).decode(), self.epoch))
        self.db.execute("INSERT OR IGNORE INTO verdicts (learning, status, round, gain, bond, trainer, registered, "
                        "accepted, note) VALUES (?,?,?,?,?,?,?,?,?)",
                        (lid, "accepted", 0, abs(new - old), 0, sub["submitter"], self.epoch, self.epoch,
                         f"challenge #{cid}"))
        for val in validators:
            self.db.execute("INSERT OR IGNORE INTO reveals VALUES (?,?,?,?,?,?,?)",
                            (lid, 0, val, "{}", abs(new - old), self.epoch, 1))
        self.db.execute("UPDATE challenge_subs SET trace=?, learning=? WHERE id=?", (tid, lid, int(sid)))
        return lid, tid

    def _lineage(self, cid, sid, sb, named):
        """The paid solutions a submission builds on: the ones it names, and any it mostly repeats."""
        out = []
        for pid, ptrace, pbody in self.db.execute(
                "SELECT id, trace, body FROM challenge_subs WHERE challenge=? AND trace IS NOT NULL AND id != ? "
                "ORDER BY id", (int(cid), int(sid))).fetchall():
            pb = json.loads(pbody)
            what = lambda x: x.get("solution", x.get("artifact"))
            if pid in named or similarity(what(sb), what(pb)) >= SIMILAR:
                out.append(ptrace)
        return out[:LIMITS["parents"]]

    # --- prior-art challenges: a result that was already known pays nobody ------------------------------------------------
    def _paid_live(self, cid, below):
        """Paid submissions on challenge `cid` whose tranches still vest and whose progress started below `below`."""
        d = direction(self._body(cid)["metric"])
        out = []
        for sid, prev, lid in self.db.execute(
                "SELECT id, prev_best, learning FROM challenge_subs WHERE challenge=? AND learning IS NOT NULL AND "
                "status IN ('best', 'improved', 'rebased') ORDER BY id", (int(cid),)).fetchall():
            live = self.db.execute("SELECT 1 FROM vesting WHERE learning=? AND status='vesting' LIMIT 1", (lid,)).fetchone()
            if live and d * (below - prev) > 0:
                out.append(sid)
        return out

    def _prior_row(self, cid, digest, v):
        """An earlier traceX record of this solution under the same verifier and instance: a submission or a reference
        on the board before challenge `cid` was posted."""
        seq = self._row(cid)["seq"] or 0
        for rid, rc, score in self.db.execute("SELECT id, challenge, score FROM challenge_subs WHERE digest=? AND id <= ? "
                                              "AND score IS NOT NULL ORDER BY id", (digest, seq)).fetchall():
            ov = self._body(rc)["verifier"]
            if (ov.get("id"), ov.get("instance")) == (v.get("id"), v.get("instance")):
                return rid
        return None

    def file_prior_art(self, cid, b):
        """A prior-art challenge, during the vesting window of what it disputes: {challenger, reference (a solution),
        provenance: {kind: "tracex"} (the same solution was on traceX's boards before the challenge was posted: an
        import's reference or any earlier submission) or {kind: "external", url, date, commit?} (a dated public record
        that drawn validators check), score? (for a verifier the node does not run)}. It stakes 2,000 sats. Upheld
        (the reference scores R with the challenge's verifier, R beats where some still-vesting paid progress started,
        and the record is older than the challenge): every tranche still vesting from progress at or below R goes back
        into the backers' escrow, the best and every pledge are rebased to R, submissions that added nothing beyond R
        lose their bond (the challenger's reward, a fixed 500 sats, comes out of it; the rest is destroyed), and the
        ones that went beyond R are paid again for that part only. Rejected: the stake is destroyed."""
        from exchange import need_address
        challenger = need_address(b.get("challenger"), "challenger")
        ref, prov = b.get("reference"), dict(b.get("provenance") or {})
        if ref is None or prov.get("kind") not in ("tracex", "external"):
            raise ValueError("send reference (a solution) and provenance {kind: tracex} or {kind: external, url, date}")
        if prov["kind"] == "external":
            for k in ("url", "date"):
                if not isinstance(prov.get(k), str) or not prov[k] or len(prov[k]) > 500:
                    raise ValueError("an external record names its url and date")
        with self.lock:
            c, body = self._row(cid), self._body(cid)
            v = body["verifier"]
            if v["kind"] == "none":
                raise ValueError("this challenge has no verifier")
            node_scores = v["id"] in self.verifiers and not (body.get("instances") or {}).get("hidden")
            if node_scores:
                score = self._run_verifier(v, ref)
                if score is None:
                    raise ValueError("the reference fails the challenge's verifier")
            elif b.get("score") is None:
                raise ValueError("this verifier runs on validators: send the score the reference reaches")
            else:
                score = float(b["score"])
            if not self._paid_live(cid, score):
                raise ValueError("no paid progress still vesting on this challenge starts below that result")
            digest = object_id({"solution": ref})
            prior = None
            if prov["kind"] == "tracex":
                prior = self._prior_row(cid, digest, v)
                if prior is None:
                    raise ValueError("no earlier record of this solution under this verifier on traceX's boards "
                                     "(before the challenge was posted)")
            stake = self.p.challenge_stake_msats
            self._need_funds(challenger, stake)
            self._tx_fee(challenger)
            pa = self.db.execute("INSERT INTO prior_claims (challenge, challenger, reference, provenance, score, status, "
                                 "epoch, prior_row) VALUES (?,?,?,?,?,?,?,?)",
                                 (int(cid), challenger, json.dumps(ref), json.dumps(prov), score, "pending", self.epoch,
                                  prior)).lastrowid
            self._move(challenger, f"priorart:{pa}", stake, "prior-art stake")
            self._event(f"challenge #{cid}: a prior-art claim says {score:.6g} was already known")
            if node_scores and prior is not None:
                self._settle_prior(pa, True, score)
            elif self.beacon_delay == 0:
                self._draw_prior(pa)
            self.db.commit()
        return self.prior_claim(pa)

    def prior_claim(self, pa):
        r = self.db.execute("SELECT id, challenge, challenger, provenance, score, status, epoch, prior_row, note FROM "
                            "prior_claims WHERE id=?", (int(pa),)).fetchone()
        if not r:
            raise KeyError(f"prior-art claim {pa}")
        out = dict(zip(("id", "challenge", "challenger", "provenance", "score", "status", "epoch", "prior_row", "note"), r))
        out["provenance"] = json.loads(out["provenance"])
        out["validators"] = [a for (a,) in self.db.execute("SELECT validator FROM prior_assign WHERE claim=?", (int(pa),))]
        return out

    def _draw_prior(self, pa):
        if self.db.execute("SELECT 1 FROM prior_assign WHERE claim=?", (int(pa),)).fetchone():
            return
        cid, who = self.db.execute("SELECT challenge, challenger FROM prior_claims WHERE id=?", (int(pa),)).fetchone()
        involved = {who} | {x for (x,) in self.db.execute("SELECT submitter FROM challenge_subs WHERE challenge=?", (cid,))}
        vals = self._eligible(exclude=tuple(involved))
        for a, _ in sorted(vals, key=lambda v: self._score(f"prior:{pa}", 0, *v))[:max(self.p.quorum, 1)]:
            self.db.execute("INSERT INTO prior_assign VALUES (?,?,?)", (int(pa), a, self.epoch))

    def commit_prior(self, pa, validator, digest):
        with self.lock:
            c = self.prior_claim(pa)
            if c["status"] != "pending":
                raise ValueError(f"claim {pa} is {c['status']}")
            if validator not in c["validators"]:
                raise PermissionError("this validator was not drawn for that claim (the draw is random)")
            if self.db.execute("SELECT 1 FROM prior_commits WHERE claim=? AND validator=?", (int(pa), validator)).fetchone():
                raise ValueError("already committed")
            self._tx_fee(validator)
            self.db.execute("INSERT INTO prior_commits VALUES (?,?,?,?)", (int(pa), validator, str(digest), self.epoch))
            self.db.commit()
        return {"claim": int(pa), "committed": validator}

    def reveal_prior(self, pa, validator, measurement, salt=""):
        """{"prior": was the record public before the challenge was posted (its date, its commit), "score": what the
        reference scores on the validator's own run of the verifier (for a verifier the node does not run)}."""
        from registry import measurement_digest
        with self.lock:
            c = self.prior_claim(pa)
            if c["status"] != "pending":
                raise ValueError(f"claim {pa} is {c['status']}")
            if validator not in c["validators"]:
                raise PermissionError("this validator was not drawn for that claim (the draw is random)")
            cm = self.db.execute("SELECT digest, epoch FROM prior_commits WHERE claim=? AND validator=?",
                                 (int(pa), validator)).fetchone()
            if not cm:
                raise ValueError("commit first")
            committed = {v for (v,) in self.db.execute("SELECT validator FROM prior_commits WHERE claim=?", (int(pa),))}
            if committed != set(c["validators"]) and self.epoch <= cm[1]:
                raise ValueError("reveals open once every drawn validator has committed, or next epoch")
            if measurement_digest(measurement, salt) != cm[0]:
                raise ValueError("this measurement does not match the commitment")
            if self.db.execute("SELECT 1 FROM prior_reveals WHERE claim=? AND validator=?", (int(pa), validator)).fetchone():
                raise ValueError("already revealed")
            sc = measurement.get("score")
            self._tx_fee(validator)
            self.db.execute("INSERT INTO prior_reveals VALUES (?,?,?,?,?)", (int(pa), validator,
                                                                             int(bool(measurement.get("prior"))),
                                                                             None if sc is None else float(sc), self.epoch))
            if self.db.execute("SELECT COUNT(*) FROM prior_reveals WHERE claim=?", (int(pa),)).fetchone()[0] \
                    >= len(c["validators"]):
                self._finalize_prior(pa)
            self.db.commit()
        return self.prior_claim(pa)

    def _finalize_prior(self, pa):
        c = self.prior_claim(pa)
        body = self._body(c["challenge"])
        rows = self.db.execute("SELECT prior, score FROM prior_reveals WHERE claim=?", (int(pa),)).fetchall()
        yes = [sc for pr, sc in rows if pr]
        ok = len(yes) * 2 > len(rows)
        score = c["score"]
        if ok and not (body["verifier"]["id"] in self.verifiers and not (body.get("instances") or {}).get("hidden")):
            scored = [sc for sc in yes if sc is not None]
            if len(scored) * 2 <= len(rows):
                ok = False
            else:
                med = statistics.median(scored)
                d = direction(body["metric"])
                score = min(med, score) if d > 0 else max(med, score)     # never more than the claim said
        self._settle_prior(pa, ok, score)

    def _settle_prior(self, pa, upheld, score):
        c = self.prior_claim(pa)
        cid, who = c["challenge"], c["challenger"]
        stake_acct = f"priorart:{pa}"
        live = self._paid_live(cid, score) if upheld else []
        if not live:
            self._forfeit(stake_acct, self._bal(stake_acct), "prior-art stake", f"{cid}:{pa}")
            self.db.execute("UPDATE prior_claims SET status='rejected', note=? WHERE id=?",
                            ("not upheld: the stake is destroyed" if not upheld else "nothing left to claw back", int(pa)))
            self._event(f"challenge #{cid}: a prior-art claim was rejected; its stake is destroyed")
            return
        self._move(stake_acct, who, self._bal(stake_acct), "prior-art stake back", payout=True)
        reward = self._rebase_on_prior(cid, score, json.loads(self.db.execute(
            "SELECT reference FROM prior_claims WHERE id=?", (int(pa),)).fetchone()[0]), live, who, pa)
        self.db.execute("UPDATE prior_claims SET status='upheld', note=? WHERE id=?",
                        (f"{len(live)} paid submission(s) rebased to {score:.6g}; reward {reward} msats", int(pa)))

    def _rebase_on_prior(self, cid, R, ref, live, challenger, pa):
        """Claw every tranche still vesting from the first disputed submission on back into the pledges' escrow,
        rebase the best and every pledge to the known result R, and pay again, from R, the submissions that beat it."""
        body = self._body(cid)
        m, d = body["metric"], direction(body["metric"])
        first = min(live)
        pledges = {pid: pk for pk, pid in self.db.execute(
            "SELECT id, payment FROM challenge_pledges WHERE challenge=?", (int(cid),)).fetchall()}
        subs = self.db.execute("SELECT id, learning, score, submitter, digest, bond FROM challenge_subs WHERE challenge=? "
                               "AND id >= ? AND learning IS NOT NULL AND status IN ('best', 'improved', 'rebased') "
                               "ORDER BY id", (int(cid), first)).fetchall()
        reward_left, replay = self.p.prior_art_reward_msats, []
        for sid, lid, sc, submitter, dig, bond in subs:
            back = {}
            for vid, pid, msats in self.db.execute("SELECT id, payment, msats FROM vesting WHERE learning=? AND "
                                                   "status='vesting'", (lid,)).fetchall():
                if pid in pledges:                         # the money never left the pledge's escrow: it is free again
                    self.db.execute("UPDATE vesting SET status='prior_art' WHERE id=?", (vid,))
                    back[pid] = back.get(pid, 0) + msats
            for pid, msats in back.items():
                self.db.execute("UPDATE challenge_pledges SET released=released-? WHERE id=?", (msats, pledges[pid]))
                self.db.execute("UPDATE challenges SET released=released-? WHERE id=?", (msats, int(cid)))
            self.db.execute("UPDATE challenge_subs SET paid=0 WHERE id=?", (sid,))
            if d * (sc - R) >= max(min_step(m, R), 1e-12):
                self.db.execute("UPDATE challenge_subs SET status='rebased' WHERE id=?", (sid,))
                replay.append((sid, sc))
                continue
            self.db.execute("UPDATE challenge_subs SET status='prior_art', note=? WHERE id=?",
                            (f"already known: {R:.12g} (prior-art claim #{pa})", sid))
            acct = f"csubbond:{cid}:{dig[7:23]}"
            held = self._bal(acct)
            if held > 0:
                cut = min(reward_left, held)
                if cut:
                    self._move(acct, challenger, cut, "prior-art reward, from the bond", payout=True)
                    reward_left -= cut
                self._forfeit(acct, held - cut, "challenge submission bond", f"{cid}:{sid}")
            self.db.execute("UPDATE challenge_subs SET bond_release=NULL WHERE id=?", (sid,))
        for pk, start, base, done, based in self.db.execute(
                "SELECT id, from_score, base, released, base_released FROM challenge_pledges WHERE challenge=?",
                (int(cid),)).fetchall():
            if d * (R - start) > 0:
                self.db.execute("UPDATE challenge_pledges SET from_score=?, level=?, base=?, base_released=? WHERE id=?",
                                (R, R, base - (done - based), done, pk))
            else:
                self.db.execute("UPDATE challenge_pledges SET level=from_score WHERE id=?", (pk,))
        self.db.execute("INSERT OR IGNORE INTO challenge_subs (challenge, submitter, digest, body, public_score, score, se, "
                        "status, epoch, parents, prev_best, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (int(cid), challenger, object_id({"solution": ref, "prior_art": int(pa)}),
                         json.dumps({"solution": ref, "meta": {}}), R, R, 0.0, "reference", self.epoch, "[]", R,
                         f"prior art (claim #{pa})"))
        best, best_sub = R, None
        for sid, sc in replay:
            measured = self.db.execute("SELECT 1 FROM challenge_assign WHERE sub=?", (sid,)).fetchone()
            if improves(m, best, sc):
                if not measured:                           # judged pledges wait for their judge again, from R
                    self._pay_pledges(cid, sid, sc, self._pledge_rows(cid))
                if best_sub:
                    self.db.execute("UPDATE challenge_subs SET status='improved' WHERE id=?", (best_sub,))
                self.db.execute("UPDATE challenge_subs SET status='best' WHERE id=?", (sid,))
                best, best_sub = sc, sid
        st = self._row(cid)["status"]
        if st == "solved" and not reached(m, best):
            st = "open"
        self.db.execute("UPDATE challenges SET best=?, best_sub=?, status=? WHERE id=?", (best, best_sub, st, int(cid)))
        if st in ("expired", "removed"):
            self._refund_challenge(int(cid), "prior art")
        self._event(f"challenge #{cid}: prior art upheld; tranches up to {R:.6g} went back to the backers' escrow")
        return self.p.prior_art_reward_msats - reward_left

    def _challenge_paused(self):
        """Learnings whose tranches wait: every paid solution on a challenge with a prior-art claim pending."""
        return {lid for (lid,) in self.db.execute(
            "SELECT s.learning FROM challenge_subs s WHERE s.learning IS NOT NULL AND s.challenge IN "
            "(SELECT challenge FROM prior_claims WHERE status='pending')").fetchall()}

    # --- operator takedown ---------------------------------------------------------------------------------------------
    def remove(self, kind, oid):
        """A challenge taken down: its solutions' vesting tranches go back to the backers, every pledge's remainder is
        refunded, and it leaves the board (its leaderboard stays readable)."""
        if kind != "challenge":
            return super().remove(kind, oid)
        with self.lock:
            c = self._row(oid)
            for (lid,) in self.db.execute("SELECT learning FROM challenge_subs WHERE challenge=? AND learning IS NOT NULL",
                                          (int(oid),)).fetchall():
                self._claw(lid)
            self._refund_challenge(int(oid), "taken down")
            self.db.execute("UPDATE challenges SET status='removed' WHERE id=?", (int(oid),))
            self.db.commit()
        return {"removed": "challenge", "id": c["id"]}

    def _refund_challenge(self, cid, why):
        for (pid,) in self.db.execute("SELECT payment FROM challenge_pledges WHERE challenge=?", (cid,)).fetchall():
            self._refund_payment(pid, f"challenge {cid} {why}")

    # --- settlement ------------------------------------------------------------------------------------------------------
    def _challenge_settle(self):
        """Expire challenges past their deadline or never backed (refunding what each pledge holds), settle validator
        rounds that ran a whole epoch with a majority revealed, and escalate failures that keep growing."""
        e = self.epoch
        for cid, deadline, posted, pledged in self.db.execute(
                "SELECT id, deadline, epoch, pledged FROM challenges WHERE status='open'").fetchall():
            unbacked = not pledged and e - posted >= self.unbacked_epochs
            if deadline >= e and not unbacked:
                continue
            self._refund_challenge(cid, "ended")
            for sid, who, dig in self.db.execute("SELECT id, submitter, digest FROM challenge_subs WHERE challenge=? AND "
                                                 "status='pending'", (cid,)).fetchall():
                acct = f"csubbond:{cid}:{dig[7:23]}"          # validators never finished: the bond goes home, unjudged
                self._move(acct, who, self._bal(acct), "challenge submission bond returned", payout=True)
                self.db.execute("UPDATE challenge_subs SET status='unjudged' WHERE id=?", (sid,))
            self.db.execute("UPDATE challenges SET status='expired', note=? WHERE id=?",
                            ("unbacked" if unbacked and deadline >= e else "deadline", cid))
            self._event(f"challenge #{cid} expired: what its pledges still held went back to its backers")
        for (sid,) in self.db.execute(
                "SELECT s.id FROM challenge_subs s WHERE s.status='pending' AND EXISTS (SELECT 1 FROM challenge_assign a "
                "WHERE a.sub=s.id AND a.epoch < ?)", (e,)).fetchall():
            n = self.db.execute("SELECT COUNT(*) FROM challenge_reveals WHERE sub=?", (sid,)).fetchone()[0]
            if n >= self.p.quorum // 2 + 1:
                self._finalize_sub(sid)
        for (pa,) in self.db.execute("SELECT c.id FROM prior_claims c WHERE c.status='pending' AND EXISTS (SELECT 1 FROM "
                                     "prior_assign a WHERE a.claim=c.id AND a.epoch < ?)", (e,)).fetchall():
            n = self.db.execute("SELECT COUNT(*) FROM prior_reveals WHERE claim=?", (pa,)).fetchone()[0]
            if n >= self.p.quorum // 2 + 1:
                self._finalize_prior(pa)
            elif self.db.execute("SELECT epoch FROM prior_claims WHERE id=?", (pa,)).fetchone()[0] + 2 < e:
                who = self.prior_claim(pa)["challenger"]          # validators never answered: nobody is at fault
                self._move(f"priorart:{pa}", who, self._bal(f"priorart:{pa}"), "prior-art stake back", payout=True)
                self.db.execute("UPDATE prior_claims SET status='lapsed' WHERE id=?", (pa,))
        paused = {c for (c,) in self.db.execute("SELECT challenge FROM prior_claims WHERE status='pending'")}
        for sid, cid, who, dig in self.db.execute(
                "SELECT id, challenge, submitter, digest FROM challenge_subs WHERE bond_release IS NOT NULL AND "
                "bond_release <= ?", (e,)).fetchall():
            if cid in paused:
                continue
            acct = f"csubbond:{cid}:{dig[7:23]}"            # the window closed with no prior art: the bond goes home
            self._move(acct, who, self._bal(acct), "challenge submission bond returned", payout=True)
            self.db.execute("UPDATE challenge_subs SET bond_release=NULL WHERE id=?", (sid,))
        self._escalate_failures()

    def _challenge_assign(self):
        for (sid,) in self.db.execute("SELECT id FROM challenge_subs WHERE status='pending'").fetchall():
            self._draw_sub(sid)
        for (pa,) in self.db.execute("SELECT id FROM prior_claims WHERE status='pending'").fetchall():
            self._draw_prior(pa)

    def _escalate_failures(self):
        """A failure that stays open (or regressed) for escalate_after epochs while its growth stays positive for
        escalate_streak settlements in a row becomes a challenge: maximize its pass rate on validators' hidden cases."""
        if not self.p.escalate_after:
            return
        e = self.epoch
        from registry import FIXED_RATE
        for (fid,) in self.db.execute("SELECT id FROM failures WHERE status IN ('open', 'regressed') AND "
                                      "first_epoch <= ?", (e - self.p.escalate_after,)).fetchall():
            f = self._failure_row(fid)
            s = self._summary(f)
            r = self.db.execute("SELECT streak, last_epoch, challenge FROM escalations WHERE failure=?", (fid,)).fetchone()
            streak = (r[0] if r and r[1] == e - 1 else 0) + 1 if s["growth"] > 0 and s["occurrences"] else 0
            self.db.execute("INSERT OR REPLACE INTO escalations VALUES (?,?,?,?)", (fid, streak, e, r[2] if r else None))
            if streak < self.p.escalate_streak or (r and r[2] and self._row(r[2])["status"] == "open"):
                continue
            base = float(f["pass_rate"] or 0.0)
            out = self.post_challenge({
                "v": VERSION, "title": f"Open problem: {f['title']}"[:200],
                "statement": (f"Failure {fid} keeps growing and has stayed {f['status']} for {e - f['first_epoch']} epochs "
                              f"({s['occurrences']} distinct cases, {s['reporters']} verified reporters). Raise its pass "
                              "rate on validators' own hidden cases of the failure: submit a fix (a learning, prompt patch, "
                              "tool or weights) as an artifact; validators run it and reveal the pass rate."),
                "path": f["path"] or "uncategorised", "key": f"failure|{fid}",
                "metric": {"name": "pass_rate", "direction": "maximize", "baseline": base,
                           "target": max(FIXED_RATE, base + 0.01), "min_step": 0.02},
                "verifier": {"id": "registry-pass-rate@1", "kind": "registry", "instance": {"failure": fid}},
                "instances": {"hidden": {"held_by": "validators", "what": f"their own cases of {fid}"}},
                "source": {"name": "traceX failure registry (escalation)", "ref": fid},
                "failure_id": fid}, origin="escalation")
            self.db.execute("UPDATE escalations SET challenge=? WHERE failure=?", (out["id"], fid))

    def challenge_stats(self):
        rows = dict(self.db.execute("SELECT status, COUNT(*) FROM challenges GROUP BY status").fetchall())
        return {"by_status": rows, "submissions": self.db.execute("SELECT COUNT(*) FROM challenge_subs").fetchone()[0],
                "improvements": self.db.execute("SELECT COUNT(*) FROM challenge_subs WHERE learning IS NOT NULL").fetchone()[0],
                "escrow_msats": int(self.db.execute(
                    "SELECT COALESCE(SUM(l.msats), 0) FROM sledger l JOIN challenge_pledges p ON l.account = 'pay:' || "
                    "p.payment").fetchone()[0])}
