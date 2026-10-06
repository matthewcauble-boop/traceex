"""traceX v0.7: the failure registry and fix tracking (SPEC 4g, 4h).

The durable asset is the record of what breaks in the field and what provably fixes it: a CVE-like registry for model
failures.

Failures
  * Every trace is filed under a canonical failure keyed by the classifier's branch, the failure signature (each fixed
    field with its failure mode; for a runtime error, the exception class the checker reported first) and the base
    model family. The failure gets a stable public id, TXF-<year>-<sequence>, the first time its key is seen; the id
    never changes and is never reused.
  * A failure holds its reproduction set (the skeletonized cases of its traces), the checkers that judged them, and live
    counters: distinct verified reporters, occurrences (distinct cases: a copy counts as its original), first and last
    seen, growth (occurrences from verified reporters in the last 3 epochs against the 3 before), and the model
    versions that hit it. Frequency and growth are the demand signal; bounties attach to failure ids.
  * Who counts as a reporter: the producer of each case's original (a near-duplicate counts for whoever filed the first
    copy). A verified reporter is one the node can hold to account: on a sats node, an address with a reporter bond
    (1,000 sats) in escrow; on the dollar node with wallets, an address that opened one (the faucet is rate limited).
    Validators who re-run a case and find it does not reproduce drop it from the counters, and its reporter's bond is
    destroyed.

Fixes
  * A fix (a learning, prompt patch or tool, or a new model version) claims one or more failure ids. The node runs the
    claimed failures' public repro sets through the checkers it can run (by default, the verified output must match),
    on outputs the claimant sends; validators drawn at random measure the hidden part (their own private cases of the
    same failure) and commit, then reveal, a pass count per failure. The median of their pass rates sets the status of
    each (failure, model version): open (under 10%), partly_fixed (with its pass rate), fixed (90% or more), or
    regressed (it was fixed, and a later model version fails it again). The public repro alone never fixes anything:
    its answers are public, so anyone can copy them.
  * Registering a model version re-checks every tracked failure of its family and reports which it fixed and which
    regressed.
  * A bounty attached to a failure pays automatically when that failure's status flips to fixed by a validated fix
    carrying a learning the federation accepted, once the bounty's poster has measured that fix on its own hidden eval
    at the target: every v0.6 payment rule still applies (the pledges split 70 / 20 / 5 / 5 down the learning's tree,
    vesting, refundable), and every payout is still a split of a real payment.
"""
import datetime as dt
import hashlib
import json
import re
import sqlite3
import statistics
import time

from traceex import canonical
from traceex.classify import nodes
from leviathan_search import SearchIndex, record, render, best_snippet, truncate, one_line

FIXED_RATE = 0.90                 # a measured pass rate at or above this: fixed
PARTLY_RATE = 0.10                # at or above this (and below FIXED_RATE): partly_fixed, with its pass rate
GROWTH_EPOCHS = 3
REPRO_CASES = 50                  # a failure's public repro set: its most recent distinct cases, at most this many
MIN_CASES = 10                    # fewer hidden cases than this across the validators: inconclusive, nothing changes
MAX_CLAIMS = 50
FIX_KINDS = ("learning", "prompt_patch", "tool")
STATUS_RANK = {"open": 0, "regressed": 0, "partly_fixed": 1, "fixed": 2}
FAILURE_ID = re.compile(r"TXF-\d{4}-\d{6}")
FIX_ID = re.compile(r"TXFIX-\d{4}-\d{6}")
EXC = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception|Exit|Interrupt|Timeout))\b")
PREAMBLE = re.compile(r"^.{0,160}?\bhere is your task:\s*", re.I | re.S)
SORTS = ("frequency", "growth", "bounty", "new")
UNSAFE = re.compile(r"[<>\"\x00-\x1f]")

REG_SCHEMA = """
CREATE TABLE IF NOT EXISTS failures (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, key TEXT UNIQUE, path TEXT,
                                     signature TEXT, modes TEXT, family TEXT, title TEXT, first_at TEXT, last_at TEXT,
                                     first_epoch INT, last_epoch INT, status TEXT DEFAULT 'open', pass_rate REAL,
                                     status_model TEXT, status_fix TEXT, status_epoch INT);
CREATE TABLE IF NOT EXISTS occurrences (trace TEXT PRIMARY KEY, failure TEXT, reporter TEXT, model TEXT, checker TEXT,
                                        epoch INT, at TEXT, canonical TEXT, reproduced INT, rejected INT DEFAULT 0);
CREATE INDEX IF NOT EXISTS occurrences_failure ON occurrences(failure);
CREATE INDEX IF NOT EXISTS occurrences_canonical ON occurrences(canonical);
CREATE TABLE IF NOT EXISTS repro_checks (trace TEXT, validator TEXT, reproduced INT, epoch INT,
                                         PRIMARY KEY (trace, validator));
CREATE TABLE IF NOT EXISTS models (version TEXT PRIMARY KEY, family TEXT, parent TEXT, epoch INT, at TEXT, fix TEXT);
CREATE TABLE IF NOT EXISTS fixes (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, kind TEXT, claimant TEXT,
                                  model TEXT, learning TEXT, artifact TEXT, claims TEXT, prior TEXT, epoch INT, at TEXT,
                                  status TEXT, bond INT DEFAULT 0, note TEXT);
CREATE TABLE IF NOT EXISTS fix_assign (fix TEXT, validator TEXT, epoch INT, PRIMARY KEY (fix, validator));
CREATE TABLE IF NOT EXISTS fix_commits (fix TEXT, validator TEXT, digest TEXT, epoch INT, PRIMARY KEY (fix, validator));
CREATE TABLE IF NOT EXISTS fix_reveals (fix TEXT, validator TEXT, epoch INT, PRIMARY KEY (fix, validator));
CREATE TABLE IF NOT EXISTS fix_results (fix TEXT, failure TEXT, source TEXT, who TEXT, passed INT, n INT, epoch INT,
                                        PRIMARY KEY (fix, failure, source, who));
CREATE TABLE IF NOT EXISTS fstatus (failure TEXT, model TEXT, fix TEXT, status TEXT, pass_rate REAL, n INT,
                                    validated INT, epoch INT, PRIMARY KEY (failure, model, fix));
CREATE TABLE IF NOT EXISTS fhistory (id INTEGER PRIMARY KEY, failure TEXT, model TEXT, fix TEXT, kind TEXT, status TEXT,
                                     pass_rate REAL, n INT, validated INT, epoch INT, at TEXT, note TEXT);
CREATE INDEX IF NOT EXISTS fhistory_failure ON fhistory(failure);
CREATE TABLE IF NOT EXISTS poster_marks (bounty INT, fix TEXT, body TEXT, epoch INT, PRIMARY KEY (bounty, fix));
CREATE TABLE IF NOT EXISTS fix_payouts (failure TEXT, learning TEXT, bounty INT, fix TEXT, epoch INT,
                                        PRIMARY KEY (failure, learning, bounty));
CREATE TABLE IF NOT EXISTS reporters (address TEXT PRIMARY KEY, bond INT, joined INT, leaving INT);
"""


def model_family(model):
    """The family a model version belongs to: its organisation dropped, then its leading name and version number
    ('Qwen/Qwen2.5-0.5B-Instruct' -> 'qwen2.5', 'llama-3.1-8b' -> 'llama3.1', 'needle3' -> 'needle3'). A trace or a
    model registration can name its family outright."""
    if isinstance(model, dict):
        if model.get("family"):
            return str(model["family"]).strip().lower()[:60]
        model = model.get("name") or ""
    name = str(model).strip().lower().rsplit("/", 1)[-1]
    m = re.match(r"([a-z]+)[-_ ]?(\d+(?:\.\d+)?)?", name)
    return (m[1] + (m[2] or "")) if m else (name[:60] or "unknown")


def failure_signature(trace, modes):
    """What failed, exactly: each fixed field with its failure mode, sorted; for a runtime error, the exception class
    the checker reported first ('code:runtime_error/TypeError'). Two traces with the same branch, signature and model
    family are the same failure."""
    fb = trace.get("feedback") or []
    exc = EXC.search(str(fb[0])) if fb else None
    parts = []
    for field in sorted(modes):
        mode = modes[field]
        if mode == "runtime_error" and exc:
            mode += "/" + exc[1]
        parts.append(f"{field}:{mode}")
    return " ".join(parts) or "unspecified"


def exception_words(*texts):
    """Exception class names spelled out, so 'type error' finds TypeError: ['type error', 'index error']."""
    found = []
    for t in texts:
        for m in EXC.finditer(str(t or '')):
            w = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', m[1]).lower()
            if w not in found:
                found.append(w)
    return found


def describe_signature(sig):
    """'code:runtime_error/TypeError' -> 'runtime error (TypeError) in code'."""
    out = []
    for part in sig.split():
        field, _, mode = part.partition(":")
        mode, _, exc = mode.partition("/")
        out.append(f"{mode.replace('_', ' ')}{f' ({exc})' if exc else ''} in {field}")
    return ", ".join(out)


def exact_runner(trace, output):
    """The checker every node can run: the output's fixed fields must equal the verified output (whitespace and case
    aside). Public answers can be copied, so this is a sanity check on a claim, never proof of a fix."""
    norm = lambda v: re.sub(r"\s+", " ", str(v)).strip().lower()
    if not isinstance(output, dict):
        return False
    v = trace.get("verified_output") or {}
    return all(norm(output.get(f, "")) == norm(v.get(f, "")) for f in trace.get("fixed_fields") or [])


def measurement_digest(measurement, salt):
    """What a validator commits before revealing its measurement of a fix: sha256(canonical measurement + salt)."""
    return hashlib.sha256(canonical(measurement) + str(salt).encode()).hexdigest()


def status_for(rate):
    if rate is None:
        return "open"
    return "fixed" if rate >= FIXED_RATE else ("partly_fixed" if rate >= PARTLY_RATE else "open")


class Registry:
    """Mixed into the exchange node (node/exchange.py) and so into the sats node (node/sats.py)."""

    # --- opening ---------------------------------------------------------------------------------------------------
    def _open_registry(self):
        """Create the registry and search tables (an older database gains them), and file every trace an older node
        stored. Runs once the node knows the database is its own."""
        self.db.executescript(REG_SCHEMA)
        try:
            self.db.execute("ALTER TABLE bounties ADD COLUMN failure_id TEXT")      # a v0.6 database
        except sqlite3.OperationalError:
            pass
        self.db.execute("DROP TABLE IF EXISTS trace_fts")                           # v0.6's search table
        self.index = SearchIndex(self.db)
        self.fts = self.index.fts
        self._branch_names = {"/".join(p): d for p, d in nodes()}
        self.index.set_branches(sorted(self._branch_names.items()))
        self.checker_runners = getattr(self, "checker_runners", {})
        if self.index.count("trace") == 0 and self.db.execute("SELECT 1 FROM labels LIMIT 1").fetchone():
            self._rebuild_registry()
        self.db.commit()

    def _rebuild_registry(self):
        """File and index every trace an older node stored, from its saved label (no reclassification)."""
        with self.lock:
            for tid, body, path, conf, engine, sig, modes in self.db.execute(
                    "SELECT t.id, t.body, l.path, l.confidence, l.engine, l.signature, l.modes FROM traces t "
                    "JOIN labels l ON l.id = t.id ORDER BY t.rowid").fetchall():
                c = {"path_str": path, "confidence": conf, "engine": engine, "signature": sig,
                     "failure_modes": json.loads(modes or "{}")}
                self._index(tid, json.loads(body), c)

    def _now(self):
        clock = getattr(self, "clock", None) or time.time
        return dt.datetime.fromtimestamp(clock(), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def register_runner(self, checker, fn):
        """Operator, in Python only (never over HTTP: it runs code): a checker the node can run on a fix's public
        outputs, fn(trace, output) -> bool, for traces judged by `checker` ('id' or 'id@version')."""
        self.checker_runners[str(checker)] = fn

    def _runner(self, checker):
        cid = checker.split("@")[0]
        return self.checker_runners.get(checker) or self.checker_runners.get(cid) or exact_runner

    # --- filing: every trace into the index and the registry, in the caller's transaction ---------------------------------
    def _index(self, tid, t, c):
        """Write (or rewrite) a trace's label, its failure and its search records. Callers hold the lock and commit, so
        the index changes in the same transaction as the write it indexes. Returns the failure id."""
        self.db.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?,?)",
                        (tid, c["path_str"], c["confidence"], c["engine"], c["signature"], json.dumps(c["failure_modes"]),
                         t["base_model"]["name"], t["task"]))
        fid, moved_from = self._file_failure(tid, t, c)
        self._index_trace(tid, t, c, fid)
        self._index_failure(fid)
        if moved_from:
            self._index_failure(moved_from)
        return fid

    def _file_failure(self, tid, t, c):
        sig = failure_signature(t, c["failure_modes"])
        family = model_family(t["base_model"])
        key = "|".join((c["path_str"], sig, family))
        now = self._now()
        row = self.db.execute("SELECT id FROM failures WHERE key=?", (key,)).fetchone()
        if row:
            fid = row[0]
        else:
            modes = sorted(set(c["failure_modes"].values()))
            title = f"{describe_signature(sig)} · {c['path_str'] or 'uncategorised'} · {family}"
            seq = self.db.execute("INSERT INTO failures (key, path, signature, modes, family, title, first_at, last_at, "
                                  "first_epoch, last_epoch, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                  (key, c["path_str"], sig, json.dumps(modes), family, title[:240], now, now, self.epoch,
                                   self.epoch, "open")).lastrowid
            fid = f"TXF-{now[:4]}-{seq:06d}"
            self.db.execute("UPDATE failures SET id=? WHERE seq=?", (fid, seq))
            self._event(f"new failure {fid}: {describe_signature(sig)[:80]} on {family}")
        old = self.db.execute("SELECT failure FROM occurrences WHERE trace=?", (tid,)).fetchone()
        if old:
            if old[0] != fid:                     # re-filed under another branch: the case moves, both ids stay
                self.db.execute("UPDATE occurrences SET failure=? WHERE trace=?", (fid, tid))
                return fid, old[0]
            return fid, None
        ck = t["checker"]
        self.db.execute("INSERT INTO occurrences (trace, failure, reporter, model, checker, epoch, at, canonical) "
                        "VALUES (?,?,?,?,?,?,?,?)", (tid, fid, t["producer"], t["base_model"]["name"],
                                                     f"{ck['id']}@{ck['version']}", self.epoch, now, tid))
        self.db.execute("UPDATE failures SET last_at=?, last_epoch=? WHERE id=?", (now, self.epoch, fid))
        return fid, None

    def _link_copy(self, tid, original):
        """A near-duplicate counts as its original's case, and for its original's reporter."""
        with self.lock:
            r = self.db.execute("SELECT failure FROM occurrences WHERE trace=?", (tid,)).fetchone()
            if r:
                self.db.execute("UPDATE occurrences SET canonical=? WHERE trace=?", (original, tid))
                self._index_failure(r[0])

    def _index_trace(self, tid, t, c, fid):
        modes = sorted(set(c["failure_modes"].values()))
        ck = f"{t['checker']['id']}@{t['checker']['version']}"
        body = PREAMBLE.sub("", t["input"])
        texts = [*map(str, t.get("feedback") or []), body, " ".join(f.replace("_", " ") for f in t["fixed_fields"])]
        from exchange import snippet                  # the node's own one-line card title
        title = snippet(t["input"])
        doc = {"id": tid, "kind": "trace", "path": c["path_str"], "signature": c["signature"],
               "model": t["base_model"]["name"], "task": t["task"],
               "lot": f"{t['task']}|{t['base_model']['name']}|{ck}", "fixed_fields": t["fixed_fields"],
               "snippet": title, "producer": t["producer"], "privacy": t["privacy"], "created": t.get("created"),
               "failure_id": fid, "modes": modes, "family": model_family(t["base_model"]), "checker": ck,
               "texts": [truncate(x, 1500) for x in texts if x]}
        cols = {"title": title, "body": "\n".join(texts),
                "code": "\n".join(str(v) for v in (t.get("verified_output") or {}).values())[:8000],
                "names": "\n".join(x for x in (c["path_str"], self._branch_names.get(c["path_str"], ""), t["task"]) if x),
                "labels": " ".join(modes + [c["signature"], t["base_model"]["name"], ck, fid] +
                                   exception_words(*(t.get("feedback") or [])[:1])).replace("_", " ")}
        facets = [("kind", "trace"), *(("failure", m) for m in modes), ("model", t["base_model"]["name"]),
                  ("family", doc["family"]), ("checker", ck), ("task", t["task"]), ("privacy", t["privacy"]),
                  ("failure_id", fid)]
        self.index.upsert(record(tid, "trace", c["path_str"], self._branch_names.get(c["path_str"]),
                                 t.get("created") or self._now(), cols, facets, doc, t["input"].splitlines()))

    def _index_failure(self, fid):
        f = self._failure_row(fid)
        if not f:
            return
        s = self._summary(f)
        if not s["occurrences"]:                      # every case taken down or refuted: out of the index
            self.index.delete(fid)
            return
        cases = self.db.execute("SELECT t.body FROM occurrences o JOIN traces t ON t.id = o.trace WHERE o.failure=? AND "
                                "o.rejected=0 ORDER BY o.rowid DESC LIMIT 5", (fid,)).fetchall()
        from exchange import snippet
        texts = []
        for (b,) in cases:
            t = json.loads(b)
            texts.append(snippet(t["input"]))
            texts += [one_line(x)[:200] for x in (t.get("feedback") or [])[:1]]
        texts = list(dict.fromkeys(texts))
        doc = {"id": fid, "kind": "failure", "path": f["path"], "title": f["title"], "signature": f["signature"],
               "modes": f["modes"], "family": f["family"], "status": s["status"], "pass_rate": s["pass_rate"],
               "reporters": s["reporters"], "occurrences": s["occurrences"], "last_seen": f["last_at"],
               "models": sorted(s["models"]), "texts": texts}
        cols = {"title": f["title"], "body": "\n".join(texts),
                "names": "\n".join(x for x in (f["path"], self._branch_names.get(f["path"], "")) if x),
                "labels": " ".join(f["modes"] + [f["signature"], f["family"], s["status"], fid] +
                                   exception_words(f["signature"]) +
                                   sorted(s["models"]) + sorted(s["checkers"])).replace("_", " ")}
        facets = [("kind", "failure"), *(("failure", m) for m in f["modes"]), ("family", f["family"]),
                  *(("model", m) for m in s["models"]), ("status", s["status"]), *(("checker", k) for k in s["checkers"]),
                  ("failure_id", fid)]
        self.index.upsert(record(fid, "failure", f["path"], self._branch_names.get(f["path"]), f["last_at"], cols,
                                 facets, doc))

    # --- counters ----------------------------------------------------------------------------------------------------
    def _verified_set(self):
        """Reporters the node can hold to account; None means every reporter counts (a node with no wallets). The
        dollar node with wallets counts addresses that opened one; the sats node overrides this with reporter bonds."""
        if not self.test_credits:
            return None
        return {a for (a,) in self.db.execute("SELECT account FROM grants").fetchall()}

    def _failure_row(self, fid):
        r = self.db.execute("SELECT id, path, signature, modes, family, title, first_at, last_at, first_epoch, last_epoch, "
                            "status, pass_rate, status_model, status_fix, status_epoch FROM failures WHERE id=?",
                            (fid,)).fetchone()
        if not r:
            return None
        keys = ("id", "path", "signature", "modes", "family", "title", "first_at", "last_at", "first_epoch",
                "last_epoch", "status", "pass_rate", "status_model", "status_fix", "status_epoch")
        f = dict(zip(keys, r))
        f["modes"] = json.loads(f["modes"] or "[]")
        return f

    def _summary(self, f, verified=None):
        """A failure's live counters and status, as listed."""
        fid = f["id"]
        verified = self._verified_set() if verified is None else verified
        rows = self.db.execute(
            "SELECT o.trace, o.canonical, COALESCE(t.producer, o.reporter), o.model, o.epoch, o.checker FROM occurrences o "
            "LEFT JOIN traces t ON t.id = o.canonical WHERE o.failure=? AND o.rejected=0", (fid,)).fetchall()
        cases, reporters, models, checkers = {}, set(), {}, set()
        for tid, canon, who, model, epoch, ck in rows:
            cases.setdefault(canon, (who, epoch))
            reporters.add(who)
            models[model] = models.get(model, 0) + 1
            checkers.add(ck)
        ok = (lambda a: True) if verified is False or verified is None else (lambda a: a in verified)
        vr = {r for r in reporters if ok(r)}
        e = self.epoch
        recent = sum(1 for who, ep in cases.values() if ok(who) and ep > e - GROWTH_EPOCHS)
        prior = sum(1 for who, ep in cases.values() if ok(who) and e - 2 * GROWTH_EPOCHS < ep <= e - GROWTH_EPOCHS)
        pool, bounties = 0, []
        for bid, p in self.db.execute("SELECT id, pool FROM bounties WHERE failure_id=? AND status='open'", (fid,)):
            pool += p
            bounties.append(bid)
        return {"id": fid, "path": f["path"], "title": f["title"], "signature": f["signature"], "modes": f["modes"],
                "family": f["family"], "status": f["status"], "pass_rate": f["pass_rate"],
                "status_model": f["status_model"], "status_fix": f["status_fix"],
                "reporters": len(vr), "reporters_unverified": len(reporters) - len(vr), "occurrences": len(cases),
                "reports": len(rows), "first_seen": f["first_at"], "last_seen": f["last_at"],
                "first_epoch": f["first_epoch"], "last_epoch": f["last_epoch"], "recent": recent, "growth": recent - prior,
                "models": models, "checkers": sorted(checkers), "bounties": bounties,
                f"open_bounty_{self.money}": pool}

    # --- reads --------------------------------------------------------------------------------------------------------
    def failures(self, path="", failure="", model="", status="", sort="frequency", limit=20, offset=0):
        """The registry, filtered by branch (its subtree), failure mode, model (a version, or a family) and status;
        sorted by frequency (distinct verified reporters, then occurrences), growth, open bounty or newest."""
        sort = sort or "frequency"
        if sort not in SORTS:
            raise ValueError(f"sort is one of {SORTS}")
        if status and status not in STATUS_RANK:
            raise ValueError(f"status is one of {tuple(STATUS_RANK)}")
        limit, offset = max(1, min(int(limit), 500)), max(0, int(offset))
        path = (path or "").strip().strip("/")
        verified = self._verified_set()
        out = []
        for (fid,) in self.db.execute("SELECT id FROM failures ORDER BY seq").fetchall():
            f = self._failure_row(fid)
            if path and not (f["path"] == path or f["path"].startswith(path + "/")):
                continue
            if failure and failure not in f["modes"]:
                continue
            if status and f["status"] != status:
                continue
            s = self._summary(f, verified if verified is not None else False)
            if not s["occurrences"]:
                continue
            if model and model.lower() != f["family"] and model not in s["models"]:
                continue
            out.append(s)
        money = f"open_bounty_{self.money}"
        keys = {"frequency": lambda s: (-s["reporters"], -s["occurrences"], s["id"]),
                "growth": lambda s: (-s["growth"], -s["recent"], -s["reporters"], s["id"]),
                "bounty": lambda s: (-s[money], -s["reporters"], -s["occurrences"], s["id"]),
                "new": lambda s: (_desc(s["last_seen"]), s["id"])}
        out.sort(key=keys[sort])
        return {"failures": out[offset:offset + limit], "count": min(limit, max(len(out) - offset, 0)),
                "total": len(out), "sort": sort, "offset": offset,
                "statuses": {k: sum(1 for s in out if s["status"] == k) for k in STATUS_RANK}}

    def get_failure(self, fid):
        f = self._failure_row(str(fid).upper())
        if not f:
            raise KeyError(f"failure {fid}")
        s = self._summary(f)
        cases = self.db.execute(
            "SELECT o.trace, o.reproduced, t.body FROM occurrences o JOIN traces t ON t.id = o.trace WHERE o.failure=? "
            "AND o.rejected=0 AND o.canonical = o.trace ORDER BY o.rowid DESC LIMIT ?", (f["id"], REPRO_CASES)).fetchall()
        from exchange import snippet
        repro = [{"trace": tid, "snippet": snippet(json.loads(b)["input"]), "reproduced": None if r is None else bool(r)}
                 for tid, r, b in cases]
        by_model = [{"model": m, "fix": x, "kind": self._fix_kind(x), "status": st, "pass_rate": pr, "n": n,
                     "validated": bool(v), "epoch": e}
                    for m, x, st, pr, n, v, e in self.db.execute(
                        "SELECT model, fix, status, pass_rate, n, validated, epoch FROM fstatus WHERE failure=? "
                        "ORDER BY epoch, model", (f["id"],)).fetchall()]
        fixes = [x for (x, claims) in self.db.execute("SELECT id, claims FROM fixes ORDER BY seq").fetchall()
                 if f["id"] in json.loads(claims)]
        return dict(s, repro={"cases": s["occurrences"], "public": repro,
                              "hidden": "each validator's own private cases of this failure; only their pass counts are "
                                        "published"},
                    by_model=by_model, fixes=fixes,
                    history=self.db.execute("SELECT COUNT(*) FROM fhistory WHERE failure=?", (f["id"],)).fetchone()[0])

    def failure_history(self, fid):
        f = self._failure_row(str(fid).upper())
        if not f:
            raise KeyError(f"failure {fid}")
        rows = self.db.execute("SELECT epoch, at, model, fix, kind, status, pass_rate, n, validated, note FROM fhistory "
                               "WHERE failure=? ORDER BY id", (f["id"],)).fetchall()
        keys = ("epoch", "at", "model", "fix", "kind", "status", "pass_rate", "n", "validated", "note")
        return {"failure": f["id"], "title": f["title"], "status": f["status"], "first_seen": f["first_at"],
                "history": [dict(zip(keys, r), validated=bool(r[8])) for r in rows]}

    def get_trace(self, tid):
        tid = str(tid)
        rows = self.db.execute("SELECT id, body FROM traces WHERE id=? OR (length(?) >= 15 AND id LIKE ? || '%') LIMIT 2",
                               (tid, tid, tid.replace("%", ""))).fetchall()
        if len(rows) > 1:
            raise ValueError(f"{tid} is ambiguous; give more of the id")
        if not rows or not self.db.execute("SELECT 1 FROM labels WHERE id=?", (rows[0][0],)).fetchone():
            raise KeyError(f"trace {tid}")
        tid, body = rows[0]
        lab = self.db.execute("SELECT path, signature, engine FROM labels WHERE id=?", (tid,)).fetchone()
        occ = self.db.execute("SELECT failure, canonical, rejected FROM occurrences WHERE trace=?", (tid,)).fetchone()
        return {"id": tid, "trace": json.loads(body), "path": lab[0], "signature": lab[1], "classified_by": lab[2],
                "failure_id": occ[0] if occ else None, "copy_of": occ[1] if occ and occ[1] != tid else None,
                "refuted": bool(occ[2]) if occ else False}

    # --- reporters: refuting fabricated cases ----------------------------------------------------------------------------
    def _validator_ok(self, validator):
        return validator in self._validator_list()

    def _validator_list(self):
        return list(self.validators)

    def _repro_quorum(self):
        return 1

    def repro_check(self, fid, validator, results):
        """Operator-relayed validator message: validators re-run a failure's public cases on the base model and say
        which reproduce. A case a majority of at least `_repro_quorum()` validators finds does not reproduce leaves the
        counters (with every copy of it), and its reporter's bond is destroyed (sats node)."""
        f = self._failure_row(str(fid).upper())
        if not f:
            raise KeyError(f"failure {fid}")
        if not self._validator_ok(validator):
            raise PermissionError("only a validator can re-check a failure's cases")
        if not isinstance(results, dict) or not results:
            raise ValueError("results: {trace id: true if it reproduced}")
        refuted, confirmed = [], []
        with self.lock:
            self._tx_fee(validator)
            for tid, rep in list(results.items())[:REPRO_CASES]:
                if not self.db.execute("SELECT 1 FROM occurrences WHERE trace=? AND failure=?", (tid, f["id"])).fetchone():
                    raise ValueError(f"{tid} is not a case of {f['id']}")
                self.db.execute("INSERT OR REPLACE INTO repro_checks VALUES (?,?,?,?)", (tid, validator, int(bool(rep)),
                                                                                       self.epoch))
                votes = [v for (v,) in self.db.execute("SELECT reproduced FROM repro_checks WHERE trace=?", (tid,))]
                if len(votes) < self._repro_quorum():
                    continue
                if sum(votes) * 2 < len(votes):                    # a majority says it does not reproduce
                    who = self.db.execute("SELECT COALESCE(t.producer, o.reporter) FROM occurrences o LEFT JOIN traces t "
                                          "ON t.id = o.trace WHERE o.trace=?", (tid,)).fetchone()[0]
                    self.db.execute("UPDATE occurrences SET rejected=1 WHERE trace=? OR canonical=?", (tid, tid))
                    self._reporter_fabricated(who, tid)
                    refuted.append(tid)
                elif sum(votes) * 2 > len(votes):
                    self.db.execute("UPDATE occurrences SET reproduced=1 WHERE trace=?", (tid,))
                    confirmed.append(tid)
            self._index_failure(f["id"])
            if refuted:
                self._event(f"{f['id']}: validators could not reproduce {len(refuted)} reported case"
                            f"{'s' if len(refuted) != 1 else ''}; they leave the counters")
            self.db.commit()
        return {"failure": f["id"], "refuted": refuted, "reproduced": confirmed,
                "counters": {k: v for k, v in self._summary(self._failure_row(f["id"])).items()
                             if k in ("reporters", "reporters_unverified", "occurrences")}}

    def _reporter_fabricated(self, address, tid):
        """A reporter whose case did not reproduce. The dollar node only drops the case; the sats node destroys its bond."""
        return 0

    # --- fixes -----------------------------------------------------------------------------------------------------------
    def _fix_kind(self, fix_id):
        r = self.db.execute("SELECT kind FROM fixes WHERE id=?", (fix_id,)).fetchone()
        return r[0] if r else None

    def _fix_can_pay(self, claimant):
        """Before anything is written: can the claimant cover the fix's fee (and bond, on a sats node)?"""

    def _fix_charge(self, claimant, fix_id):
        """The claim's fee (and, on a sats node, its bond). Returns the bond held."""
        self._tx_fee(claimant)
        return 0

    def _fix_bond_settle(self, fix_id, keep):
        return 0

    def _fix_quorum(self):
        """How many validator reveals settle a fix: the dollar node takes one of its known validators."""
        return 1 if self._validator_list() else 0

    def _fix_assign(self, fix_id, claimant):
        """Who measures the hidden part. The dollar node: any of its validators (operator-relayed)."""
        for v in self._validator_list():
            if v != claimant:
                self.db.execute("INSERT OR IGNORE INTO fix_assign VALUES (?,?,?)", (fix_id, v, self.epoch))

    def _commit_required(self):
        return False

    def _ensure_model(self, version, family=None, parent=None, fix=None):
        if not self.db.execute("SELECT 1 FROM models WHERE version=?", (version,)).fetchone():
            self.db.execute("INSERT INTO models VALUES (?,?,?,?,?,?)", (version, family or model_family(version), parent,
                                                                        self.epoch, self._now(), fix))

    def _new_fix_id(self, kind, claimant, model, learning, artifact, claims, bond_note=""):
        prior = {}
        for fid in claims:
            st, pr = self.db.execute("SELECT status, pass_rate FROM failures WHERE id=?", (fid,)).fetchone()
            prior[fid] = [st, pr]
        now = self._now()
        seq = self.db.execute("INSERT INTO fixes (kind, claimant, model, learning, artifact, claims, prior, epoch, at, "
                              "status, note) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                              (kind, claimant, model, learning, json.dumps(artifact or {}), json.dumps(claims),
                               json.dumps(prior), self.epoch, now, "pending", bond_note or None)).lastrowid
        fix_id = f"TXFIX-{now[:4]}-{seq:06d}"
        self.db.execute("UPDATE fixes SET id=? WHERE seq=?", (fix_id, seq))
        return fix_id

    def _claims(self, raw):
        claims = list(dict.fromkeys(str(c).strip().upper() for c in (raw or [])))
        if not claims:
            raise ValueError("a fix claims at least one failure id (TXF-…)")
        if len(claims) > MAX_CLAIMS:
            raise ValueError(f"a fix claims at most {MAX_CLAIMS} failures")
        for c in claims:
            if not FAILURE_ID.fullmatch(c) or not self.db.execute("SELECT 1 FROM failures WHERE id=?", (c,)).fetchone():
                raise ValueError(f"unknown failure {c!r}")
        return claims

    def claim_fix(self, b):
        """A fix claims failure ids. Body: {claimant, kind: learning | prompt_patch | tool, claims: [TXF-…], model (the
        model version it applies to), learning? (the learning that carries it: needed for a bounty to pay), artifact?,
        outputs? ({trace id: the fixed model's output} on the claimed failures' public cases)}. The node runs the public
        repro checks at once; validators drawn at random measure the hidden part. New model versions are registered with
        POST /v0/models instead."""
        claimant = str(b.get("claimant") or "")
        from exchange import need_address, LIMITS
        need_address(claimant, "claimant")
        kind = b.get("kind") or "learning"
        if kind == "model":
            raise ValueError("register a new model version with POST /v0/models (operator): it re-checks every failure "
                             "of its family")
        if kind not in FIX_KINDS:
            raise ValueError(f"kind is one of {FIX_KINDS}")
        model = str(b.get("model") or "").strip()
        if not model or len(model) > LIMITS["model"] or UNSAFE.search(model):
            raise ValueError(f"model: the model version the fix applies to, plain text up to {LIMITS['model']} characters")
        claims = self._claims(b.get("claims"))
        lid = b.get("learning") or None
        if lid:
            r = self.db.execute("SELECT body FROM learnings WHERE id=?", (lid,)).fetchone()
            if not r:
                raise ValueError("register the learning first (POST /v0/learnings)")
            if json.loads(r[0])["trainer"] != claimant:
                raise ValueError("only the learning's trainer can claim fixes with it")
        artifact = b.get("artifact") or {}
        if not isinstance(artifact, dict) or len(canonical(artifact)) > 4096:
            raise ValueError("artifact: an object up to 4 KB (name, uri, hash)")
        outputs = b.get("outputs") or {}
        if not isinstance(outputs, dict):
            raise ValueError("outputs: {trace id: the fixed model's output on that case}")
        self._room()
        with self.lock:
            self._fix_can_pay(claimant)
            fix_id = self._new_fix_id(kind, claimant, model, lid, artifact, claims)
            bond = self._fix_charge(claimant, fix_id)
            self.db.execute("UPDATE fixes SET bond=? WHERE id=?", (bond, fix_id))
            self._ensure_model(model)
            self._run_repro(fix_id, claims, outputs)
            self._fix_assign(fix_id, claimant)
            self._event(f"fix {fix_id} claims {len(claims)} failure{'s' if len(claims) != 1 else ''} on {model[:60]}: "
                        "validators measure it on their own cases")
            if self._fix_quorum() == 0:
                self._finalize_fix(fix_id)
            self.db.commit()
        return self.get_fix(fix_id)

    def _run_repro(self, fix_id, claims, outputs):
        """The public part: each claimed failure's public cases through the checkers the node can run, on the outputs
        the claimant sent (a case without an output fails)."""
        if not outputs:
            return
        for fid in claims:
            passed = n = 0
            for tid, body, ck in self.db.execute(
                    "SELECT o.trace, t.body, o.checker FROM occurrences o JOIN traces t ON t.id = o.trace WHERE o.failure=? "
                    "AND o.rejected=0 AND o.canonical = o.trace ORDER BY o.rowid DESC LIMIT ?", (fid, REPRO_CASES)).fetchall():
                n += 1
                if tid in outputs:
                    try:
                        passed += bool(self._runner(ck)(json.loads(body), outputs[tid]))
                    except Exception:                          # a runner that breaks on an output: that case fails
                        pass
            self.db.execute("INSERT OR REPLACE INTO fix_results VALUES (?,?,?,?,?,?,?)",
                            (fix_id, fid, "repro", "node", passed, n, self.epoch))

    def _fix_row(self, fix_id):
        r = self.db.execute("SELECT id, kind, claimant, model, learning, artifact, claims, prior, epoch, at, status, bond, "
                            "note FROM fixes WHERE id=?", (str(fix_id).upper(),)).fetchone()
        if not r:
            raise KeyError(f"fix {fix_id}")
        keys = ("id", "kind", "claimant", "model", "learning", "artifact", "claims", "prior", "epoch", "at", "status",
                "bond", "note")
        f = dict(zip(keys, r))
        for k in ("artifact", "claims", "prior"):
            f[k] = json.loads(f[k] or "null")
        return f

    def get_fix(self, fix_id):
        f = self._fix_row(fix_id)
        fid = f["id"]
        assigned = [v for (v,) in self.db.execute("SELECT validator FROM fix_assign WHERE fix=? ORDER BY validator", (fid,))]
        committed = [v for (v,) in self.db.execute("SELECT validator FROM fix_commits WHERE fix=?", (fid,))]
        revealed = [v for (v,) in self.db.execute("SELECT validator FROM fix_reveals WHERE fix=? ORDER BY validator",
                                                  (fid,))]
        claims = []
        for x in f["claims"]:
            row = self.db.execute("SELECT status, pass_rate, n, validated FROM fstatus WHERE failure=? AND fix=?",
                                  (x, fid)).fetchone()
            pub = self.db.execute("SELECT passed, n FROM fix_results WHERE fix=? AND failure=? AND source='repro'",
                                  (fid, x)).fetchone()
            hidden = [{"validator": w, "passed": p, "n": n} for w, p, n in self.db.execute(
                "SELECT who, passed, n FROM fix_results WHERE fix=? AND failure=? AND source='validator' ORDER BY who",
                (fid, x))] if f["status"] != "pending" else []
            claims.append({"failure": x, "before": dict(zip(("status", "pass_rate"), (f["prior"] or {}).get(x, [None, None]))),
                           "status": row[0] if row else "pending", "pass_rate": row[1] if row else None,
                           "n": row[2] if row else None, "validated": bool(row[3]) if row else False,
                           "public_repro": {"passed": pub[0], "n": pub[1]} if pub else None, "hidden": hidden})
        bounties = [{"bounty": b, "epoch": e} for b, e in self.db.execute(
            "SELECT bounty, epoch FROM fix_payouts WHERE fix=?", (fid,)).fetchall()]
        return {"id": fid, "kind": f["kind"], "claimant": f["claimant"], "model": f["model"], "learning": f["learning"],
                "artifact": f["artifact"], "status": f["status"], "epoch": f["epoch"], "at": f["at"],
                f"bond_{self.money}": f["bond"], "note": f["note"], "assigned": assigned, "committed": committed,
                "revealed": revealed, "claims": claims, "bounties_paid": bounties}

    def fixes(self, failure="", status="", limit=50):
        out = []
        for (fix_id,) in self.db.execute("SELECT id FROM fixes ORDER BY seq DESC").fetchall():
            f = self._fix_row(fix_id)
            if (failure and failure.upper() not in f["claims"]) or (status and f["status"] != status):
                continue
            out.append({k: f[k] for k in ("id", "kind", "claimant", "model", "learning", "status", "epoch", "claims")})
            if len(out) >= int(limit):
                break
        return {"fixes": out, "count": len(out)}

    def commit_fix(self, fix_id, validator, digest):
        """Operator-relayed validator message: commit sha256(measurement + salt) before any reveal opens."""
        f = self._fix_row(fix_id)
        with self.lock:
            if f["status"] != "pending":
                raise ValueError(f"fix {f['id']} is {f['status']}")
            if not self.db.execute("SELECT 1 FROM fix_assign WHERE fix=? AND validator=?", (f["id"], validator)).fetchone():
                raise PermissionError("this validator was not drawn for that fix (the draw is random)")
            if self.db.execute("SELECT 1 FROM fix_commits WHERE fix=? AND validator=?", (f["id"], validator)).fetchone():
                raise ValueError("already committed")
            self._tx_fee(validator)
            self.db.execute("INSERT INTO fix_commits VALUES (?,?,?,?)", (f["id"], validator, str(digest), self.epoch))
            self.db.commit()
        return {"fix": f["id"], "committed": validator}

    def reveal_fix(self, fix_id, validator, measurement, salt=""):
        """Operator-relayed validator message: the measurement on the validator's own private cases of each claimed
        failure, {"results": {TXF-…: {"passed": k, "n": n}}, "eval_set"?: hash}. On a sats node it must match the
        commitment, and reveals open once every drawn validator has committed (or the next epoch)."""
        f = self._fix_row(fix_id)
        fid = f["id"]
        with self.lock:
            if f["status"] != "pending":
                raise ValueError(f"fix {fid} is {f['status']}")
            if not self.db.execute("SELECT 1 FROM fix_assign WHERE fix=? AND validator=?", (fid, validator)).fetchone():
                raise PermissionError("this validator was not drawn for that fix (the draw is random)")
            if self._commit_required():
                c = self.db.execute("SELECT digest, epoch FROM fix_commits WHERE fix=? AND validator=?",
                                    (fid, validator)).fetchone()
                if not c:
                    raise ValueError("commit first")
                assigned = {v for (v,) in self.db.execute("SELECT validator FROM fix_assign WHERE fix=?", (fid,))}
                committed = {v for (v,) in self.db.execute("SELECT validator FROM fix_commits WHERE fix=?", (fid,))}
                if committed != assigned and self.epoch <= c[1]:
                    raise ValueError("reveals open once every drawn validator has committed, or next epoch")
                if measurement_digest(measurement, salt) != c[0]:
                    raise ValueError("this measurement does not match the commitment")
            res = (measurement or {}).get("results")
            if not isinstance(res, dict) or set(res) - set(f["claims"]):
                raise ValueError("results: {failure id: {passed, n}} for the failures this fix claims")
            rows = []
            for x, r in res.items():
                passed, n = int(r.get("passed", -1)), int(r.get("n", 0))
                if not (0 < n <= 100_000 and 0 <= passed <= n):
                    raise ValueError(f"{x}: passed and n are whole numbers, 0 <= passed <= n, n > 0")
                rows.append((x, passed, n))
            if self.db.execute("SELECT 1 FROM fix_reveals WHERE fix=? AND validator=?", (fid, validator)).fetchone():
                raise ValueError("already revealed")
            self._tx_fee(validator)
            for x, passed, n in rows:          # a failure it holds no cases of is simply left out
                self.db.execute("INSERT OR REPLACE INTO fix_results VALUES (?,?,?,?,?,?,?)",
                                (fid, x, "validator", validator, passed, n, self.epoch))
            self.db.execute("INSERT INTO fix_reveals VALUES (?,?,?)", (fid, validator, self.epoch))
            done = self.db.execute("SELECT COUNT(*) FROM fix_reveals WHERE fix=?", (fid,)).fetchone()[0]
            need = max(self._fix_quorum(), 1)
            drawn = self.db.execute("SELECT COUNT(*) FROM fix_assign WHERE fix=?", (fid,)).fetchone()[0]
            if done >= (need if not self._commit_required() else max(need, drawn)):
                self._finalize_fix(fid)
            self.db.commit()
        return self.get_fix(fid)

    def _finalize_fix(self, fix_id, partial=False):
        """Set each claimed failure's status for this fix from the validators' median pass rate on their own cases (the
        public repro can only hold a fix back), record it in the failure's history, settle the fix's bond, and pay any
        bounty whose failure is now fixed."""
        f = self._fix_row(fix_id)
        if f["status"] != "pending":
            return
        any_moved, refuted, notes = False, True, []
        for x in f["claims"]:
            hidden = [(p, n) for p, n in self.db.execute(
                "SELECT passed, n FROM fix_results WHERE fix=? AND failure=? AND source='validator'", (f["id"], x))]
            pub = self.db.execute("SELECT passed, n FROM fix_results WHERE fix=? AND failure=? AND source='repro'",
                                  (f["id"], x)).fetchone()
            pub_rate = pub[0] / pub[1] if pub and pub[1] else None
            if hidden:
                rate, n, validated = statistics.median(p / n for p, n in hidden), sum(n for _, n in hidden), True
            elif self._fix_quorum() == 0:             # a node that runs no validator round trusts its own checkers
                rate, n, validated = pub_rate, pub[1] if pub else 0, True
            else:                                     # a validator measured other failures, not this one
                rate, n, validated = None, 0, False
            st, note = status_for(rate), None
            if validated and hidden and n < MIN_CASES:      # too few cases to call either way: recorded, not counted
                st, note, validated = "open", f"inconclusive: {n} hidden case{'s' if n != 1 else ''} (needs {MIN_CASES})", False
            if st == "fixed" and pub_rate is not None and pub_rate < FIXED_RATE:
                st, note = "partly_fixed", "fails its own public repro"
            prior = (f["prior"] or {}).get(x, [None, None])
            if f["kind"] == "model" and st != "fixed" and validated and (prior[0] == "fixed" or self._fixed_in_earlier(x, f)):
                st, note = "regressed", f"was fixed before {f['model']}"
            any_moved |= validated and st in ("fixed", "partly_fixed")
            refuted &= validated and st not in ("fixed", "partly_fixed")
            self.db.execute("INSERT OR REPLACE INTO fstatus VALUES (?,?,?,?,?,?,?,?)",
                            (x, f["model"], f["id"], st, rate, n, int(validated), self.epoch))
            self.db.execute("INSERT INTO fhistory (failure, model, fix, kind, status, pass_rate, n, validated, epoch, at, "
                            "note) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (x, f["model"], f["id"], f["kind"], st, rate, n,
                                                                   int(validated), self.epoch, self._now(), note))
            if validated:
                self._set_failure_status(x, st, rate, f)
            notes.append(st)
        if f["kind"] == "model":
            status = "checked"
        else:                      # a claim validators found fixes none of its failures loses its bond; too few cases don't
            status = "validated" if any_moved else ("rejected" if refuted else "inconclusive")
        self.db.execute("UPDATE fixes SET status=? WHERE id=?", (status, f["id"]))
        self._fix_bond_settle(f["id"], keep=not refuted)
        for x in f["claims"]:
            self._index_failure(x)
        moved = {s: notes.count(s) for s in sorted(set(notes))}
        self._event(f"{'model ' + f['model'][:40] if f['kind'] == 'model' else 'fix ' + f['id']} measured: "
                    + ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in moved.items()))
        self._pay_fixed_bounties(f["claims"])

    def _fixed_in_earlier(self, failure, fix):
        """Was this failure fixed (validated) by an earlier version of the same family?"""
        return bool(self.db.execute(
            "SELECT 1 FROM fstatus s JOIN fixes x ON x.id = s.fix WHERE s.failure=? AND s.status='fixed' AND s.validated=1 "
            "AND x.kind='model' AND s.model != ? AND x.seq < (SELECT seq FROM fixes WHERE id=?)",
            (failure, fix["model"], fix["id"])).fetchone())

    def _set_failure_status(self, fid, st, rate, fix):
        """A failure's own status is the newest verified state of the model: a model version's re-check sets it outright
        (fixed, regressed, or still open); a fix can only improve it, so a weaker claim never hides a stronger one."""
        cur = self.db.execute("SELECT status, pass_rate FROM failures WHERE id=?", (fid,)).fetchone()
        better = STATUS_RANK[st] > STATUS_RANK[cur[0]] or (st == cur[0] == "partly_fixed" and (rate or 0) > (cur[1] or 0))
        if fix["kind"] == "model" or better:
            if st != cur[0]:
                self._event(f"{fid} is now {st.replace('_', ' ')}"
                            + (f" ({rate:.0%})" if rate is not None and st == "partly_fixed" else "")
                            + f" on {fix['model'][:50]}")
            self.db.execute("UPDATE failures SET status=?, pass_rate=?, status_model=?, status_fix=?, status_epoch=? "
                            "WHERE id=?", (st, rate, fix["model"], fix["id"], self.epoch, fid))

    # --- model versions --------------------------------------------------------------------------------------------------
    def register_model(self, b):
        """Operator: a new model version. Every tracked failure of its family is re-checked: validators measure the
        version on their own cases of each (and the node runs the public cases on any outputs sent); GET
        /v0/models/{version}/report then says which it fixed and which regressed."""
        version = str(b.get("version") or "").strip()
        from exchange import LIMITS
        if not version or len(version) > LIMITS["model"] or UNSAFE.search(version):
            raise ValueError(f"version: the model's name, plain text up to {LIMITS['model']} characters")
        family = str(b.get("family") or model_family(version)).strip().lower()[:60]
        parent = b.get("parent") or None
        with self.lock:
            if self.db.execute("SELECT fix FROM models WHERE version=? AND fix IS NOT NULL", (version,)).fetchone():
                raise ValueError(f"{version} is already registered: GET /v0/models/{version}/report")
            claims = [x for (x,) in self.db.execute(
                "SELECT f.id FROM failures f WHERE f.family=? AND EXISTS (SELECT 1 FROM occurrences o WHERE o.failure=f.id "
                "AND o.rejected=0) ORDER BY f.seq", (family,)).fetchall()][:500]
            fix_id = None
            if claims:
                fix_id = self._new_fix_id("model", "operator", version, None, {"name": version, "parent": parent}, claims)
            if self.db.execute("SELECT 1 FROM models WHERE version=?", (version,)).fetchone():
                self.db.execute("UPDATE models SET family=?, parent=?, fix=? WHERE version=?", (family, parent, fix_id,
                                                                                            version))
            else:
                self.db.execute("INSERT INTO models VALUES (?,?,?,?,?,?)", (version, family, parent, self.epoch,
                                                                            self._now(), fix_id))
            if fix_id:
                self._run_repro(fix_id, claims, b.get("outputs") or {})
                self._fix_assign(fix_id, None)
                if self._fix_quorum() == 0:
                    self._finalize_fix(fix_id)
            self._event(f"model {version[:60]} registered: re-checking {len(claims)} failure"
                        f"{'s' if len(claims) != 1 else ''} of {family}")
            self.db.commit()
        return self.model_report(version)

    def models(self):
        rows = self.db.execute("SELECT version, family, parent, epoch, at, fix FROM models ORDER BY rowid").fetchall()
        return {"models": [{"version": v, "family": fam, "parent": p, "epoch": e, "registered": a, "recheck": x}
                           for v, fam, p, e, a, x in rows]}

    def model_report(self, version):
        r = self.db.execute("SELECT family, parent, epoch, fix FROM models WHERE version=?", (version,)).fetchone()
        if not r:
            raise KeyError(f"model {version}")
        family, parent, epoch, fix_id = r
        out = {"version": version, "family": family, "parent": parent, "registered_epoch": epoch, "recheck": fix_id,
               "fixed": [], "still_fixed": [], "regressed": [], "partly_fixed": [], "still_open": [], "inconclusive": [],
               "pending": [], "worse": []}
        if not fix_id:
            out["note"] = "registered as the model a fix applies to, or with no tracked failures to re-check"
            return out
        f = self._fix_row(fix_id)
        out.update(status=f["status"], assigned=[v for (v,) in self.db.execute(
            "SELECT validator FROM fix_assign WHERE fix=?", (fix_id,))])
        for x in f["claims"]:
            title = self.db.execute("SELECT title FROM failures WHERE id=?", (x,)).fetchone()[0]
            before = dict(zip(("status", "pass_rate"), (f["prior"] or {}).get(x, [None, None])))
            row = self.db.execute("SELECT status, pass_rate, n, validated FROM fstatus WHERE failure=? AND fix=?",
                                  (x, fix_id)).fetchone()
            item = {"failure": x, "title": title, "before": before}
            if not row:
                out["pending"].append(item)
                continue
            item.update(status=row[0], pass_rate=row[1], n=row[2])
            if not row[3]:                                 # too few hidden cases to call
                out["inconclusive"].append(item)
            elif row[0] == "fixed":
                out["still_fixed" if before["status"] == "fixed" else "fixed"].append(item)
            elif row[0] == "regressed":
                out["regressed"].append(item)
            elif row[0] == "partly_fixed":
                out["partly_fixed"].append(item)
            else:
                out["still_open"].append(item)
            if before["pass_rate"] is not None and row[1] is not None and row[1] < before["pass_rate"] - 0.10:
                out["worse"].append(item)
        out["summary"] = {k: len(out[k]) for k in ("fixed", "still_fixed", "regressed", "partly_fixed", "still_open",
                                                   "inconclusive", "pending", "worse")}
        return out

    # --- bounties on failures --------------------------------------------------------------------------------------------
    def poster_measure(self, bounty_id, fix_id, attestation):
        """Operator-relayed: a failure bounty's poster measured a fix on its own hidden eval. With the failure fixed by
        that fix (validators) and the fix's learning accepted, the bounty pays at once (v0.6: whoever pays judges)."""
        r = self.db.execute("SELECT poster, eval_set, target, failure_id, status FROM bounties WHERE id=?",
                            (int(bounty_id),)).fetchone()
        if not r:
            raise KeyError(f"bounty {bounty_id}")
        poster, eval_set, target, fid, status = r
        if not fid:
            raise ValueError("this bounty is not attached to a failure: claim it with POST /v0/bounties/{id}/claims")
        if status != "open":
            raise ValueError(f"bounty {bounty_id} is {status}")
        f = self._fix_row(fix_id)
        if fid not in f["claims"]:
            raise ValueError(f"{f['id']} does not claim {fid}")
        a = attestation or {}
        if a.get("validator") != poster or a.get("eval_set") != eval_set:
            raise ValueError("the poster's own measurement on the bounty's hidden eval set (validator = poster)")
        float(a.get("after"))
        with self.lock:
            self._tx_fee(poster)
            self.db.execute("INSERT OR REPLACE INTO poster_marks VALUES (?,?,?,?)", (int(bounty_id), f["id"],
                                                                                   canonical(a).decode(), self.epoch))
            paid = self._pay_fixed_bounties([fid])
            self.db.commit()
        return {"bounty": int(bounty_id), "fix": f["id"], "failure": fid, "measured": float(a["after"]),
                "target": target, "paid": paid}

    def _pay_fixed_bounties(self, fids):
        """Every open bounty on these failures whose failure is fixed by a validated fix with an accepted learning, and
        whose poster measured that fix at the target, pays now through the v0.6 claim. One payout per (failure,
        learning): a fix that is paid stays paid, and can't be paid twice."""
        paid = []
        for x in fids:
            for bid, eval_set, target in self.db.execute(
                    "SELECT id, eval_set, target FROM bounties WHERE failure_id=? AND status='open' ORDER BY id",
                    (x,)).fetchall():
                for fix_id, lid in self.db.execute(
                        "SELECT s.fix, f.learning FROM fstatus s JOIN fixes f ON f.id = s.fix WHERE s.failure=? AND "
                        "s.status='fixed' AND s.validated=1 AND f.learning IS NOT NULL AND f.kind != 'model' "
                        "ORDER BY s.epoch, f.seq", (x,)).fetchall():
                    if self.db.execute("SELECT 1 FROM fix_payouts WHERE failure=? AND learning=?", (x, lid)).fetchone():
                        continue
                    mark = self.db.execute("SELECT body FROM poster_marks WHERE bounty=? AND fix=?", (bid, fix_id)).fetchone()
                    if not mark:
                        continue
                    att = json.loads(mark[0])
                    if att.get("eval_set") != eval_set or float(att.get("after", 0)) < target:
                        continue
                    try:
                        self.claim_bounty(bid, lid, att)
                    except ValueError:                 # e.g. the learning is not accepted yet: tried again at settlement
                        continue
                    self.db.execute("INSERT OR IGNORE INTO fix_payouts VALUES (?,?,?,?,?)", (x, lid, bid, fix_id, self.epoch))
                    self._event(f"bounty #{bid} paid automatically: {x} is fixed by {fix_id}")
                    paid.append(bid)
                    break
        return paid

    def _registry_settle(self):
        """At settlement: pay bounties whose fix became payable (a learning accepted since, say)."""
        fids = [x for (x,) in self.db.execute("SELECT DISTINCT failure_id FROM bounties WHERE failure_id IS NOT NULL AND "
                                              "status='open'").fetchall()]
        if fids:
            self._pay_fixed_bounties(fids)

    def _registry_assign(self):
        """After the new beacon: draw validators for fixes waiting for them (sats node)."""

    # --- search ---------------------------------------------------------------------------------------------------------
    def registry_stats(self):
        rows = dict(self.db.execute("SELECT status, COUNT(*) FROM failures f WHERE EXISTS (SELECT 1 FROM occurrences o "
                                    "WHERE o.failure = f.id AND o.rejected=0) GROUP BY status").fetchall())
        return {"failures": sum(rows.values()), "by_status": rows,
                "fixes": self.db.execute("SELECT COUNT(*) FROM fixes WHERE kind != 'model'").fetchone()[0],
                "models": self.db.execute("SELECT COUNT(*) FROM models").fetchone()[0]}


def _desc(iso):
    return tuple(-ord(c) for c in (iso or ""))


__all__ = ["Registry", "model_family", "failure_signature", "exact_runner", "measurement_digest", "status_for",
           "FIXED_RATE", "PARTLY_RATE", "render", "best_snippet"]
