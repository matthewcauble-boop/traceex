"""Reference exchange node (spec section 7). Standard library only: http.server + sqlite3.

    python node/exchange.py --port 8787 --db exchange.db                 # local
    python node/exchange.py --host 0.0.0.0 --public --seed --test-credits 25000000 --epoch-hours 24   # hosted

One process plays the off-chain half of the protocol: it accepts skeleton traces, groups them into lots, takes sealed
bids, clears each epoch, registers learnings with validator attestations, meters usage, and at settlement computes
every address's earnings and the epoch's Merkle payout root (the one value that goes on-chain). It also serves the
website (site/) and the MCP endpoint, so one process is the whole public exchange.
"""
import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk", "python"))
from traceex import canonical, object_id  # noqa: E402
from traceex.client import privacy_leaks  # noqa: E402
from traceex.auction import Bid, clear_shared  # noqa: E402
from traceex.royalty import split_trace_sale, split_usage, MAX_DEPTH  # noqa: E402
from traceex.merkle import leaf, build_tree, proof  # noqa: E402
from traceex.classify import classify, default_engine, nodes, RulesEngine, TAXONOMY_VERSION  # noqa: E402
from traceex import bountycoin as coin  # noqa: E402

# A bounty's pool, when a learning claims it: the solver is paid most, the traces it was built from still earn.
BOUNTY_SPLIT = {"trainer": 0.70, "traces": 0.20, "checkers": 0.05, "validators": 0.05}
# The standard transaction fee, in nano-dollars: $0.00005, about 125 times the electricity of the dearest transaction
# examples/fees/measure.py finds (registering a learning, its 5.5 KB kept in three copies for ten years: about
# $0.0000004), and far more than any other. The margin is the point: the fee funds the network (on a coin node it is
# paid in credits and burned, and it is the operator's claim on 10% of each epoch's emission) and prices out spam at
# machine scale, where a billion junk transactions cost $50,000 instead of $400. This is the one setting; 100x to 200x
# the measured electricity (40,000 to 80,000) keeps both jobs. Fees accrue per account and are billed each epoch in
# whole micro-dollars (credits), the smallest amount USDC can move; the fraction carries over.
TX_FEE_NANOS = 50_000
SAFE_INT = 2 ** 53                 # JavaScript loses integers above this

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces   (id TEXT PRIMARY KEY, lot TEXT, producer TEXT, checker TEXT, body TEXT, epoch INT);
CREATE TABLE IF NOT EXISTS checkers (id TEXT PRIMARY KEY, author TEXT);
CREATE TABLE IF NOT EXISTS bids     (lot TEXT, bidder TEXT, price INT, license TEXT, epoch INT);
CREATE TABLE IF NOT EXISTS licences (lot TEXT, buyer TEXT, price INT, epoch INT, traces TEXT);
CREATE TABLE IF NOT EXISTS learnings(id TEXT PRIMARY KEY, body TEXT, epoch INT);
CREATE TABLE IF NOT EXISTS usage    (learning TEXT, consumer TEXT, calls INT, epoch INT);
CREATE TABLE IF NOT EXISTS ledger   (epoch INT, account TEXT, micros INT, memo TEXT);
CREATE TABLE IF NOT EXISTS roots    (epoch INT PRIMARY KEY, root TEXT, total INT, leaves TEXT);
CREATE TABLE IF NOT EXISTS meta     (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS labels   (id TEXT PRIMARY KEY, path TEXT, confidence REAL, engine TEXT, signature TEXT,
                                     modes TEXT, model TEXT, task TEXT);
CREATE TABLE IF NOT EXISTS bounties (id INTEGER PRIMARY KEY, poster TEXT, title TEXT, path TEXT, failure TEXT,
                                     base_model TEXT, eval_set TEXT, target REAL, pool INT DEFAULT 0,
                                     supply REAL DEFAULT 0, deadline INT, status TEXT, winner TEXT, learning TEXT,
                                     epoch INT);
CREATE TABLE IF NOT EXISTS holdings (bounty INT, holder TEXT, coins REAL, PRIMARY KEY (bounty, holder));
CREATE TABLE IF NOT EXISTS bases    (bounty INT, holder TEXT, amount INT, PRIMARY KEY (bounty, holder));
CREATE TABLE IF NOT EXISTS fees     (account TEXT PRIMARY KEY, nanos INT);
CREATE TABLE IF NOT EXISTS events   (id INTEGER PRIMARY KEY, at TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS grants   (account TEXT PRIMARY KEY, micros INT, at TEXT, source TEXT);
"""
ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}")
PATH = re.compile(r"[a-z_]+(/[a-z_]+){0,5}")
SORTS = ("relevant", "new", "bounty")
FTS_WEIGHTS = "0, 4.0, 3.0, 2.0, 2.0, 1.0"        # id, path, signature, model, task, body: a hit in the branch name counts most
LIMITS = {"title": 200, "failure": 200, "base_model": 120, "open_bounties_per_poster": 20, "wallets_per_source_day": 3,
          "trace_bytes": 32 * 1024, "task": 80, "model": 120, "checker": 80}
MODE = re.compile(r"[a-z0-9_:.,-]{1,80}")         # a failure mode label: wrong_answer, role_swap, unresolved:date…
UNSAFE = re.compile(r"[<>\"\x00-\x1f]")


PREAMBLE = re.compile(r"^.{0,160}?\bhere is your task:\s*", re.I)    # prompt templates shared by every trace


def snippet(text, n=160):
    """The line a person scans a result by: the first non-empty line, minus a prompt preamble every trace in the lot
    repeats (MBPP's "You are an expert Python programmer, and here is your task: ...")."""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = PREAMBLE.sub("", line) or line
    return line if len(line) <= n else line[:n - 1].rstrip() + "…"


class Full(Exception):
    """The node's disk budget is used up: reads keep working, new writes wait for the operator."""


def need_address(a, what):
    """Anything that can be paid must be a real address: one malformed account would make the epoch's Merkle root
    impossible to build, and settlement would stall for everyone."""
    if not ADDRESS.fullmatch(str(a or "")):
        raise ValueError(f"{what} must be a 0x address (40 hex characters)")
    return a


def wire(obj, key=""):
    """What goes into JSON for any client: TXC amounts (keys ending in _units, 18 decimals) travel as decimal strings,
    and so does any other integer JavaScript would round (above 2**53)."""
    if isinstance(obj, dict):
        return {k: wire(v, k if str(k).endswith("_units") else key) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [wire(v, key) for v in obj]
    if isinstance(obj, int) and not isinstance(obj, bool) and (key.endswith("_units") or abs(obj) >= SAFE_INT):
        return str(obj)
    return obj
FTS = "CREATE VIRTUAL TABLE IF NOT EXISTS trace_fts USING fts5(id UNINDEXED, path, signature, model, task, body)"


class _Rows:
    """A statement's result, fetched: what a sqlite3 cursor offers, without holding the connection."""

    def __init__(self, cur):
        self.rows = cur.fetchall() if cur.description else []
        self.rowcount, self.lastrowid, self.description = cur.rowcount, cur.lastrowid, cur.description
        self._i = 0

    def fetchone(self):
        if self._i >= len(self.rows):
            return None
        self._i += 1
        return self.rows[self._i - 1]

    def fetchall(self):
        rest, self._i = self.rows[self._i:], len(self.rows)
        return rest

    def __iter__(self):
        return iter(self.fetchall())


class SafeDB:
    """One SQLite connection shared by the server's threads. Every statement runs under one re-entrant lock (the
    exchange's own) and comes back fetched, so no cursor outlives the lock and concurrent requests can't collide."""

    def __init__(self, conn, lock):
        self.conn, self.lock = conn, lock

    def execute(self, sql, args=()):
        with self.lock:
            return _Rows(self.conn.execute(sql, args))

    def executescript(self, sql):
        with self.lock:
            return self.conn.executescript(sql)

    def commit(self):
        with self.lock:
            self.conn.commit()

    def close(self):
        self.conn.close()


class Exchange:
    def __init__(self, path=":memory:", *, k=10, reserve_micros=1_000, validators=("0x" + "5" * 40,),
                 tx_fee_nanos=None, fee_to=None, engine=None, test_credits=0, max_db_bytes=0):
        """test_credits: run as a testnet. Each new wallet can take this many micros of test credits once, and every
        spend (coins, bids, metered usage, the transaction fee) must be covered by the wallet's balance. 0 = settlement
        is external (x402 / USDC), the reference behaviour. A node that keeps wallets charges every transaction the
        standard fee (TX_FEE_NANOS unless `tx_fee_nanos` says otherwise) and pays it to `fee_to`, whoever runs it and
        so pays its electricity ("network" until the operator names an address)."""
        self.path, self.max_db_bytes = path, int(max_db_bytes)
        self.tx_fee_nanos = int(TX_FEE_NANOS if tx_fee_nanos is None and test_credits else tx_fee_nanos or 0)
        self.fee_to = fee_to or "network"
        self.lock = threading.RLock()
        self.db = SafeDB(sqlite3.connect(path, check_same_thread=False), self.lock)
        self.db.executescript(SCHEMA)
        self.test_credits = int(test_credits)
        self.quiet = False                        # bulk loads (the seed) write one summary event instead of one per row
        self.refile = None                        # status of the last re-classification run
        try:
            self.db.execute(FTS)
            self.fts = True
        except sqlite3.OperationalError:          # SQLite built without FTS5: search falls back to LIKE
            self.fts = False
        self.engine = engine or default_engine()
        self.k, self.reserve, self.validators = k, reserve_micros, list(validators)
        if self._meta("epoch") is None:
            self._set_meta("epoch", "1")

    # --- helpers -----------------------------------------------------------------------------------------------
    def _meta(self, k):
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else None

    def _set_meta(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, str(v)))

    @property
    def epoch(self):
        return int(self._meta("epoch"))

    def _credit(self, account, micros, memo):
        if micros:
            self.db.execute("INSERT INTO ledger VALUES (?,?,?,?)", (self.epoch, account, micros, memo))

    def _room(self):
        if self.max_db_bytes and self.path != ":memory:" and os.path.exists(self.path) \
                and os.path.getsize(self.path) > self.max_db_bytes:
            raise Full("this exchange is full for now; the operator has been asked to add room")

    def _event(self, text, force=False):
        """The public activity feed. Never personal data: paths, failure signatures, amounts, bounty titles."""
        if self.quiet and not force:
            return
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.db.execute("INSERT INTO events (at, text) VALUES (?,?)", (now, str(text)[:200]))
        self.db.execute("DELETE FROM events WHERE id <= (SELECT MAX(id) FROM events) - 500")

    def _funds(self, account):
        g = self.db.execute("SELECT micros FROM grants WHERE account=?", (account,)).fetchone()
        led = self.db.execute("SELECT COALESCE(SUM(micros), 0) FROM ledger WHERE account=?", (account,)).fetchone()[0]
        return (g[0] if g else 0) + led

    def _need_funds(self, account, micros, pending=0):
        if self.test_credits and self._funds(account) - pending < micros:
            raise ValueError(f"not enough test credits: {account[:10]}… has ${(self._funds(account) - pending) / 1e6:,.2f}"
                             f", this needs ${micros / 1e6:,.2f} (POST /v0/faucet opens a wallet)")

    def _tx_fee(self, account):
        """Charge one transaction its standard fee: accrued in nano-dollars, billed at settlement (see TX_FEE_NANOS).
        The account must be able to cover what it owes, rounded up to the micro-dollar."""
        if not self.tx_fee_nanos or not account:
            return
        r = self.db.execute("SELECT nanos FROM fees WHERE account=?", (account,)).fetchone()
        owed = (r[0] if r else 0) + self.tx_fee_nanos
        self._need_funds(account, -(-owed // 1000))
        self.db.execute("INSERT OR REPLACE INTO fees VALUES (?,?)", (account, owed))

    def _bill_fees(self):
        """At settlement: every account pays its whole micro-dollars of transaction fees to whoever runs the node; the
        fraction of a micro-dollar carries over to the next epoch."""
        for account, nanos in self.db.execute("SELECT account, nanos FROM fees WHERE nanos >= 1000").fetchall():
            due = nanos // 1000
            self._credit(account, -due, "transaction fees")
            self._credit(self.fee_to, due, "transaction fees")
            self.db.execute("UPDATE fees SET nanos = nanos - ? WHERE account=?", (due * 1000, account))

    def fees(self):
        """The standard fee, and what it has collected."""
        billed = self.db.execute("SELECT COALESCE(SUM(micros), 0) FROM ledger WHERE account=? AND memo='transaction fees'",
                                 (self.fee_to,)).fetchone()[0]
        accrued = self.db.execute("SELECT COALESCE(SUM(nanos), 0) FROM fees").fetchone()[0]
        return {"per_transaction_nanos": self.tx_fee_nanos, "per_transaction_usd": f"{self.tx_fee_nanos / 1e9:.7f}",
                "billed": "each epoch, in whole micro-dollars", "to": self.fee_to, "billed_micros": billed,
                "accrued_nanos": accrued, "how_it_was_set": "examples/fees/measure.py"}

    # --- testnet wallets, feed, stats -------------------------------------------------------------------------------
    def faucet(self, account, source=""):
        """Open a testnet wallet: test credits once per address, a few addresses per source (hashed IP) per day."""
        if not self.test_credits:
            raise ValueError("this node settles in USDC; it has no test credits")
        if not ADDRESS.fullmatch(str(account or "")):
            raise ValueError("address must be 0x followed by 40 hex characters")
        src = hashlib.sha256(f"tracex:{source}".encode()).hexdigest()[:16] if source else ""
        with self.lock:
            if self.db.execute("SELECT 1 FROM grants WHERE account=?", (account,)).fetchone():
                return dict(self.wallet(account), already=True)
            if src:
                day = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                n = self.db.execute("SELECT COUNT(*) FROM grants WHERE source=? AND at > ?", (src, day)).fetchone()[0]
                if n >= LIMITS["wallets_per_source_day"]:
                    raise PermissionError("this network already opened its test wallets for today")
            now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.db.execute("INSERT INTO grants VALUES (?,?,?,?)", (account, self.test_credits, now, src))
            self._event(f"a new wallet joined with ${self.test_credits / 1e6:,.0f} of test credits")
            self.db.commit()
        return self.wallet(account)

    def wallet(self, account):
        g = self.db.execute("SELECT micros FROM grants WHERE account=?", (account,)).fetchone()
        coins = {str(b): round(c, 6) for b, c in self.db.execute(
            "SELECT bounty, coins FROM holdings WHERE holder=? AND coins > 1e-9", (account,))}
        return {"account": account, "opened": bool(g), "grant_micros": g[0] if g else 0,
                "balance_micros": self._funds(account), "coins": coins, "testnet": bool(self.test_credits)}

    def events(self, limit=30):
        rows = self.db.execute("SELECT at, text FROM events ORDER BY id DESC LIMIT ?", (min(int(limit), 200),)).fetchall()
        return {"events": [{"at": a, "text": t} for a, t in rows]}

    def stats(self):
        one = lambda sql: self.db.execute(sql).fetchone()[0]
        r = self.db.execute("SELECT epoch, root, total FROM roots ORDER BY epoch DESC LIMIT 1").fetchone()
        return {"epoch": self.epoch, "traces": one("SELECT COUNT(*) FROM traces"),
                "producers": one("SELECT COUNT(DISTINCT producer) FROM traces"),
                "learnings": one("SELECT COUNT(*) FROM learnings"),
                "bounties_open": one("SELECT COUNT(*) FROM bounties WHERE status='open'"),
                "bounties_solved": one("SELECT COUNT(*) FROM bounties WHERE status='solved'"),
                "pools_open_micros": one("SELECT COALESCE(SUM(pool), 0) FROM bounties WHERE status='open'"),
                "pools_paid_micros": one("SELECT COALESCE(SUM(pool), 0) FROM bounties WHERE status='solved'"),
                "last_root": {"epoch": r[0], "root": r[1], "total_micros": r[2]} if r else None,
                "testnet": bool(self.test_credits), "epoch_hours": getattr(self, "epoch_hours", 0),
                "fees": self.fees(),
                "classifier": {"engine": self.engine.name,
                               "filed_by": dict(self.db.execute("SELECT engine, COUNT(*) FROM labels GROUP BY engine")),
                               "usage": getattr(self.engine, "usage", None), "refile": self.refile}}

    # --- API -----------------------------------------------------------------------------------------------------
    def register_checker(self, checker_id, author):
        need_address(author, "author")
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO checkers VALUES (?,?)", (checker_id, author))
            self.db.commit()
        return {"checker": checker_id, "author": author}

    def submit_trace(self, t):
        if t.get("v") != "trace/0.1":
            raise ValueError("unknown trace version")
        if t.get("privacy") not in ("skeleton", "open"):
            raise ValueError("rejected: privacy must be 'skeleton' or 'open'")
        leaks = privacy_leaks(t)
        if leaks:
            what = "secrets or contact details" if t["privacy"] == "open" else "personal data outside placeholders"
            raise ValueError(f"rejected: {what}: {leaks[:3]}")
        if not t.get("fixed_fields"):
            raise ValueError("rejected: a trace must record at least one fixed field")
        need_address(t.get("producer"), "producer")
        for what, val, cap in (("task", t.get("task"), LIMITS["task"]),
                               ("base_model.name", (t.get("base_model") or {}).get("name"), LIMITS["model"]),
                               ("checker.id", (t.get("checker") or {}).get("id"), LIMITS["checker"])):
            if not isinstance(val, str) or not val or len(val) > cap or UNSAFE.search(val):
                raise ValueError(f"rejected: {what} must be plain text up to {cap} characters")
        modes = t.get("failure_modes") or {}
        if not isinstance(modes, dict) or not all(isinstance(m, str) and MODE.fullmatch(m) for m in modes.values()):
            raise ValueError("rejected: failure modes are short labels like wrong_answer or role_swap")
        if len(canonical(t)) > LIMITS["trace_bytes"]:
            raise ValueError(f"rejected: a trace is limited to {LIMITS['trace_bytes'] // 1024} KB")
        self._room()
        tid = object_id(t)
        ck = f"{t['checker']['id']}@{t['checker']['version']}"
        lot = f"{t['task']}|{t['base_model']['name']}|{ck}"
        if self.db.execute("SELECT 1 FROM traces WHERE id=?", (tid,)).fetchone():   # before paying to classify it
            return {"id": tid, "lot": lot, "duplicate": True}
        c = classify(t, self.engine)            # may call a hosted engine: never while holding the write lock
        with self.lock:
            if self.db.execute("SELECT 1 FROM traces WHERE id=?", (tid,)).fetchone():
                return {"id": tid, "lot": lot, "duplicate": True}
            self._tx_fee(t["producer"])
            self.db.execute("INSERT INTO traces VALUES (?,?,?,?,?,?)",
                            (tid, lot, t["producer"], t["checker"]["id"], canonical(t).decode(), self.epoch))
            self._index(tid, t, c)
            self._event(f"trace filed under {c['path_str'] or 'uncategorised'}" + (f" · {c['signature']}" if c["signature"] else ""))
            self.db.commit()
        return {"id": tid, "lot": lot, "epoch": self.epoch, "classified": c, "bounties": self._matching_bounties(c, t)}

    def _index(self, tid, t, c):
        """Write (or rewrite) a trace's label and full-text row. Callers hold the lock."""
        self.db.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?,?)",
                        (tid, c["path_str"], c["confidence"], c["engine"], c["signature"], json.dumps(c["failure_modes"]),
                         t["base_model"]["name"], t["task"]))
        if self.fts:
            self.db.execute("DELETE FROM trace_fts WHERE id=?", (tid,))
            self.db.execute("INSERT INTO trace_fts VALUES (?,?,?,?,?,?)",
                            (tid, c["path_str"].replace("/", " "), c["signature"], t["base_model"]["name"], t["task"],
                             t["input"] + " " + " ".join(t["fixed_fields"]).replace("_", " ")))

    def reclassify(self, engine=None, only="rules", limit=0, workers=4):
        """Re-file traces with `engine` (the node's own by default). only="rules" re-files what the keyword engine
        filed, which is what to run once a Jev key is added; "all" re-files everything (after a taxonomy change). The
        engine is called in parallel outside the lock and each result is written as it lands. A run stops early if
        the engine falls back (its daily budget is spent), so nothing is relabelled worse than it was."""
        engine = engine or self.engine
        if only not in ("rules", "all"):
            raise ValueError("only is 'rules' or 'all'")
        if only == "rules" and engine.name == "rules":
            return {"skipped": "the node's classifier is the keyword engine; add TYPESAFE_API_KEY to re-file with Jev"}
        if self.refile and self.refile.get("running"):
            raise ValueError("a re-file is already running")
        where = "WHERE l.engine = 'rules'" if only == "rules" else ""
        rows = self.db.execute(f"SELECT l.id, t.body, l.path FROM labels l JOIN traces t ON t.id = l.id {where} "
                               "ORDER BY t.rowid").fetchall()
        rows = rows[:int(limit)] if limit else rows
        st = self.refile = {"running": True, "engine": engine.name, "only": only, "total": len(rows), "done": 0,
                            "moved": 0, "stopped": None,
                            "started": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

        def one(row):
            tid, body, old = row
            if st["stopped"]:
                return
            t = json.loads(body)
            c = classify(t, engine)
            if c.get("fallback"):                 # engine unavailable or out of budget: keep the old label
                st["stopped"] = c["fallback"]
                return
            with self.lock:
                if self.db.execute("SELECT 1 FROM labels WHERE id=?", (tid,)).fetchone():   # not taken down meanwhile
                    self._index(tid, t, c)
                    self.db.commit()
                st["done"] += 1
                st["moved"] += c["path_str"] != old

        try:
            with ThreadPoolExecutor(max(1, int(workers))) as pool:
                list(pool.map(one, rows))
        finally:
            st["running"] = False
        if st["done"]:
            with self.lock:
                self._event(f"re-filed {st['done']} traces with the {engine.name} classifier: {st['moved']} moved to a "
                            "better branch", force=True)
                self.db.commit()
        return st

    # --- classifier, search, bounties -----------------------------------------------------------------------------
    def taxonomy(self):
        counts = dict(self.db.execute("SELECT path, COUNT(*) FROM labels GROUP BY path").fetchall())
        out = []
        for path, desc in nodes():
            p = "/".join(path)
            n = sum(v for k, v in counts.items() if k == p or k.startswith(p + "/"))
            out.append({"path": p, "description": desc, "traces": n})
        return {"version": TAXONOMY_VERSION, "engine": self.engine.name, "nodes": out}

    WORD = re.compile(r"[\w.:/@+-]*\w[\w.:/@+-]*")

    @classmethod
    def _fts_query(cls, q, op=" "):
        return op.join('"' + w + '"*' for w in cls.WORD.findall(q)[:8])

    def search(self, q="", path="", failure="", model="", limit=20, sort="", offset=0, facets=False):
        """Find traces by words, taxonomy branch, failure mode and base model.

        sort: "relevant" (default with words: full-text rank, a match in the branch name or failure label counts most),
        "new" (default without words) or "bounty" (traces that feed the richest open bounty first). Open bounties on the
        same branch come back with the results, so a trainer sees supply (traces) and demand (bounties) together.
        facets=True adds counts per branch and per failure mode, for building a browse tree."""
        q, path = (q or "").strip(), (path or "").strip("/")
        sort = sort or ("relevant" if q else "new")
        if sort not in SORTS:
            raise ValueError(f"sort is one of {SORTS}")
        limit, offset = max(1, min(int(limit), 1000)), max(0, int(offset))
        modes_of = lambda sig: [m.split(":")[0] for m in sig.split()]
        every = self._candidates(q, model)       # (id, path, model, signature, rowid, rank, task)
        rows = [r for r in every if not path or r[1] == path or r[1].startswith(path + "/")]
        mode_counts = {}
        for r in rows:
            for m in modes_of(r[3]):
                mode_counts[m] = mode_counts.get(m, 0) + 1
        if failure:
            rows = [r for r in rows if failure in modes_of(r[3])]
        open_all = self.bounties(status="open")["bounties"]
        pool = lambda r: max([b.get("pool_micros", b.get("pool_units", 0)) for b in open_all
                              if self._feeds(b, r[1], r[2], modes_of(r[3]))] or [0])
        if sort == "relevant":
            rows.sort(key=lambda r: (r[5], -r[4]))
        elif sort == "new":
            rows.sort(key=lambda r: -r[4])
        else:
            pools = {r[0]: pool(r) for r in rows} if open_all else {}
            rows.sort(key=lambda r: (-pools.get(r[0], 0), r[5], -r[4]))
        page = rows[offset:offset + limit]
        stored = {i: (lot, body) for i, lot, body in self.db.execute(
            f"SELECT id, lot, body FROM traces WHERE id IN ({','.join('?' * len(page))})", [r[0] for r in page])} if page else {}
        hits = []
        for r in page:
            tid, p, m, sig, _, rank, task = r
            lot, body = stored[tid]
            b = json.loads(body)
            hit = {"id": tid, "path": p, "signature": sig, "model": m, "task": task, "lot": lot,
                   "fixed_fields": b["fixed_fields"], "snippet": snippet(b["input"]),
                   "producer": b["producer"], "privacy": b["privacy"], "created": b.get("created")}
            if q and self.fts:
                hit["score"] = round(-rank, 3)
            feeds = pool(r) if open_all else 0
            if feeds:
                hit["bounty_pool_micros"] = feeds
            hits.append(hit)
        out = {"results": hits, "count": len(hits), "total": len(rows), "sort": sort, "offset": offset,
               "bounties": [b for b in open_all if not path or path == b["path"] or path.startswith(b["path"] + "/")
                            or b["path"].startswith(path + "/")]}
        if facets:
            paths = {}
            for r in every:
                paths[r[1]] = paths.get(r[1], 0) + 1
            out["facets"] = {"paths": paths, "modes": mode_counts}
        return out

    def _candidates(self, q, model, cap=50_000):
        """(id, path, model, signature, rowid, rank, task) for every trace matching the words and model. Rank is FTS5's
        bm25 (lower is better), 0 without words. When all the words together find nothing, any of them will do."""
        mwhere, margs = ("AND l.model = ?", [model]) if model else ("", [])
        if q and self.fts:
            words = self.WORD.findall(q)
            for op in ((" ", " OR ") if len(words) > 1 else (" ",)):
                expr = self._fts_query(q, op)
                if not expr:
                    return []
                try:
                    rows = self.db.execute(
                        f"SELECT l.id, l.path, l.model, l.signature, t.rowid, bm25(trace_fts, {FTS_WEIGHTS}), l.task "
                        f"FROM trace_fts JOIN labels l ON l.id = trace_fts.id JOIN traces t ON t.id = l.id "
                        f"WHERE trace_fts MATCH ? {mwhere} LIMIT ?", [expr, *margs, cap]).fetchall()
                except sqlite3.OperationalError:     # a query FTS5 can't parse matches nothing
                    return []
                if rows:
                    return rows
            return []
        base = ("SELECT l.id, l.path, l.model, l.signature, t.rowid, 0, l.task FROM labels l "
                "JOIN traces t ON t.id = l.id WHERE ")
        if q:                                    # SQLite without FTS5: substring match
            like = f"%{q}%"
            return self.db.execute(base + f"(t.body LIKE ? OR l.path LIKE ?) {mwhere} LIMIT ?",
                                   [like, like, *margs, cap]).fetchall()
        return self.db.execute(base + f"1=1 {mwhere} LIMIT ?", [*margs, cap]).fetchall()

    @staticmethod
    def _feeds(b, path, model, modes):
        """Would a trace on `path` from `model` with these failure modes count toward bounty `b`?"""
        on_branch = path == b["path"] or path.startswith(b["path"] + "/")
        return on_branch and (not b["base_model"] or b["base_model"] == model) and (not b["failure"] or b["failure"] in modes)

    def post_bounty(self, b):
        """Posting is free. A bounty names a taxonomy branch (optionally a failure mode and base model), a hidden eval
        set by hash and the score a solution must reach, and mints a coin on the bonding curve. `seed_micros` lets the
        poster buy the first coins in the same call."""
        if not b.get("eval_set") or not b.get("path"):
            raise ValueError("a bounty needs a path and an eval_set hash")
        need_address(b.get("poster"), "poster")
        self._room()
        path = str(b["path"]).strip("/")
        if not PATH.fullmatch(path):
            raise ValueError("path must look like a taxonomy branch, e.g. code/generate")
        for k in ("title", "failure", "base_model"):
            if len(str(b.get(k) or "")) > LIMITS[k]:
                raise ValueError(f"{k} is limited to {LIMITS[k]} characters")
        target = float(b["target"])
        if not 0 < target <= 1:
            raise ValueError("target is a score between 0 and 1")
        epochs = max(0, min(int(b.get("epochs", 4)), 52))
        seed = int(b.get("seed_micros") or b.get("reward_micros") or 0)
        with self.lock:
            if seed > 0:
                self._need_funds(b["poster"], seed)          # before the bounty exists, so a failed seed leaves nothing
            if self.test_credits:
                n = self.db.execute("SELECT COUNT(*) FROM bounties WHERE poster=? AND status='open'",
                                    (b["poster"],)).fetchone()[0]
                if n >= LIMITS["open_bounties_per_poster"]:
                    raise ValueError(f"{LIMITS['open_bounties_per_poster']} open bounties per poster")
            self._tx_fee(b["poster"])
            cur = self.db.execute(
                "INSERT INTO bounties (poster,title,path,failure,base_model,eval_set,target,deadline,status,epoch)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (b["poster"], b.get("title", ""), path, b.get("failure", ""), b.get("base_model", ""),
                 str(b["eval_set"])[:200], target, self.epoch + epochs, "open", self.epoch))
            bid = cur.lastrowid
            self._event(f"bounty #{bid} posted free on {path}: {b.get('title') or 'untitled'}")
            self.db.commit()
        out = {"id": bid, "status": "open", "deadline_epoch": self.epoch + epochs, "price_micros": coin.price(0)}
        if seed > 0:
            out["seed"] = self.buy_coins(bid, b["poster"], seed)
        return out

    def _bounty(self, bounty_id):
        r = self.db.execute("SELECT status, pool, supply FROM bounties WHERE id=?", (bounty_id,)).fetchone()
        if not r:
            raise KeyError(f"bounty {bounty_id}")
        return r

    def _holding(self, bounty_id, holder):
        r = self.db.execute("SELECT coins FROM holdings WHERE bounty=? AND holder=?", (bounty_id, holder)).fetchone()
        return r[0] if r else 0.0

    def _set_holding(self, bounty_id, holder, coins):
        self.db.execute("INSERT OR REPLACE INTO holdings VALUES (?,?,?)", (bounty_id, holder, max(coins, 0.0)))

    def _basis(self, bounty_id, holder):
        """What a holder has put into a bounty and not yet taken out: its cost basis."""
        r = self.db.execute("SELECT amount FROM bases WHERE bounty=? AND holder=?", (bounty_id, holder)).fetchone()
        return r[0] if r else 0

    def _set_basis(self, bounty_id, holder, amount):
        self.db.execute("INSERT OR REPLACE INTO bases VALUES (?,?,?)", (bounty_id, holder, max(int(amount), 0)))

    def _refund_weights(self, bounty_id):
        """Refunds go back by what each holder put in, never by coin count: cheap early coins don't claim a share of
        later backers' money."""
        bases = dict(self.db.execute("SELECT holder, amount FROM bases WHERE bounty=? AND amount > 0", (bounty_id,)).fetchall())
        return bases or dict(self.db.execute("SELECT holder, coins FROM holdings WHERE bounty=?", (bounty_id,)).fetchall())

    def buy_coins(self, bounty_id, buyer, micros):
        """Spend `micros` on the curve. Every micro goes into the bounty's pool."""
        micros = int(micros)
        if micros <= 0:
            raise ValueError("spend must be positive")
        need_address(buyer, "buyer")
        with self.lock:
            status, pool, supply = self._bounty(bounty_id)
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}; buy its coins from a holder")
            self._need_funds(buyer, micros)
            self._tx_fee(buyer)
            n = coin.coins_for(supply, micros)
            self._set_holding(bounty_id, buyer, self._holding(bounty_id, buyer) + n)
            self._set_basis(bounty_id, buyer, self._basis(bounty_id, buyer) + micros)
            self.db.execute("UPDATE bounties SET pool=pool+?, supply=supply+? WHERE id=?", (micros, n, bounty_id))
            self._credit(buyer, -micros, f"bounty {bounty_id} coins")
            self._event(f"bounty #{bounty_id} backed with ${micros / 1e6:,.2f}: {n:,.1f} coins at "
                        f"${micros / n / 1e6:.4f}; pool ${(pool + micros) / 1e6:,.2f}")
            self.db.commit()
        return {"bounty": bounty_id, "coins": round(n, 6), "avg_price_micros": round(micros / n),
                "next_price_micros": round(coin.price(supply + n)), "pool_micros": pool + micros}

    def sell_coins(self, bounty_id, seller, coins):
        """While a bounty is open, sell coins back for what they cost (pro rata), never more: any profit could only come
        out of later backers' money. The curve decides how many coins a dollar buys, so early backers hold a bigger
        share of the solution's revenue; that is their reward."""
        need_address(seller, "seller")
        with self.lock:
            status, pool, supply = self._bounty(bounty_id)
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}; its coins now earn from the solution")
            have = self._holding(bounty_id, seller)
            coins = min(float(coins), have)
            if coins <= 0:
                raise ValueError("no coins to sell")
            self._tx_fee(seller)
            basis = self._basis(bounty_id, seller)
            cost = int(basis * coins / have)
            value = min(cost, pool)
            self._set_basis(bounty_id, seller, basis - cost)
            self._set_holding(bounty_id, seller, have - coins)
            self.db.execute("UPDATE bounties SET pool=pool-?, supply=supply-? WHERE id=?", (value, coins, bounty_id))
            self._credit(seller, value, f"bounty {bounty_id} sell")
            self._event(f"{coins:,.1f} coins of bounty #{bounty_id} sold back for what they cost, ${value / 1e6:,.2f}")
            self.db.commit()
        return {"bounty": bounty_id, "sold": round(coins, 6), "paid_micros": value,
                "next_price_micros": round(coin.price(supply - coins))}

    def transfer_coins(self, bounty_id, sender, to, coins):
        """Move coins between holders, any time. (A live network checks the sender's signature; the contract does.)"""
        need_address(sender, "from")
        need_address(to, "to")
        with self.lock:
            self._bounty(bounty_id)
            have = self._holding(bounty_id, sender)
            coins = float(coins)
            if coins <= 0 or coins > have + 1e-9:
                raise ValueError(f"{sender} holds {have:.6f} coins")
            self._tx_fee(sender)
            basis = self._basis(bounty_id, sender)                 # the coins carry what they cost
            moved = int(basis * min(coins / have, 1.0))
            self._set_basis(bounty_id, sender, basis - moved)
            self._set_basis(bounty_id, to, self._basis(bounty_id, to) + moved)
            self._set_holding(bounty_id, sender, have - coins)
            self._set_holding(bounty_id, to, self._holding(bounty_id, to) + coins)
            self.db.commit()
        return {"bounty": bounty_id, "from": sender, "to": to, "coins": coins}

    def holders(self, bounty_id):
        status, pool, supply = self._bounty(bounty_id)
        rows = self.db.execute("SELECT holder, coins FROM holdings WHERE bounty=? AND coins > 1e-9 ORDER BY coins DESC",
                               (bounty_id,)).fetchall()
        return {"bounty": bounty_id, "status": status, "pool_micros": pool, "supply": round(supply, 6),
                "price_micros": round(coin.price(supply)), "holders": {h: round(c, 6) for h, c in rows}}

    def bounties(self, path="", status=""):
        rows = self.db.execute("SELECT id,poster,title,path,failure,base_model,eval_set,target,pool,supply,deadline,"
                               "status,winner,learning FROM bounties WHERE status != 'removed' ORDER BY id").fetchall()
        keys = ["id", "poster", "title", "path", "failure", "base_model", "eval_set", "target", "pool_micros", "supply",
                "deadline_epoch", "status", "winner", "learning"]
        out = [dict(zip(keys, r)) for r in rows]
        for b in out:
            b["price_micros"] = round(coin.price(b["supply"]))
        if status:
            out = [b for b in out if b["status"] == status]
        if path:
            p = path.strip("/")
            out = [b for b in out if p == b["path"] or p.startswith(b["path"] + "/") or b["path"].startswith(p + "/")]
        return {"bounties": out}

    def _matching_bounties(self, c, t):
        return [b["id"] for b in self.bounties(path=c["path_str"], status="open")["bounties"]
                if (not b["base_model"] or b["base_model"] == t["base_model"]["name"])
                and (not b["failure"] or b["failure"] in c["failure_modes"].values())]

    def claim_bounty(self, bounty_id, learning_id):
        """A learning claims a bounty when its validator attested the bounty's own eval set and reached the target.
        The pool pays the solver and, through the learning's family tree, the traces it was built from. From then on
        the bounty's coins earn HOLDER_CUT of every metered use of that learning."""
        with self.lock:
            row = self.db.execute("SELECT status, eval_set, target, pool, base_model FROM bounties WHERE id=?",
                                  (bounty_id,)).fetchone()
            if not row:
                raise KeyError(f"bounty {bounty_id}")
            status, eval_set, target, pool, base_model = row
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}")
            r = self.db.execute("SELECT body FROM learnings WHERE id=?", (learning_id,)).fetchone()
            if not r:
                raise ValueError("register the learning first")
            L = json.loads(r[0])
            a = L["attestation"]
            if a.get("eval_set") != eval_set:
                raise ValueError("the attestation is not on this bounty's eval set")
            if float(a["after"]) < target:
                raise ValueError(f"scored {a['after']}, the bounty needs {target}")
            if base_model and L["base_model"]["name"] != base_model:
                raise ValueError(f"the bounty is for {base_model}")
            self._tx_fee(L["trainer"])
            trace_info, learnings = self._tree()
            payout = split_usage(pool, dict(L, royalty=dict(L["royalty"], split=BOUNTY_SPLIT)), trace_info,
                                 self.validators, learnings)
            for acct, m in payout.items():
                self._credit(acct, m, f"bounty {bounty_id}")
            self.db.execute("UPDATE bounties SET status='solved', winner=?, learning=? WHERE id=?",
                            (L["trainer"], learning_id, bounty_id))
            self._event(f"bounty #{bounty_id} solved: {a.get('metric') or 'score'} {float(a['after']):.1%} beat the "
                        f"{target:.1%} target; pool ${pool / 1e6:,.2f} paid; its coins now earn {coin.HOLDER_CUT:.0%} of every use")
            self.db.commit()
        return {"bounty": bounty_id, "status": "solved", "winner": L["trainer"], "pool_micros": pool, "payout": payout,
                "holders_now_earn": f"{coin.HOLDER_CUT:.0%} of every use of {learning_id[:19]}"}

    def _tree(self):
        trace_info = {tid: {"producer": p, "checker_author": self._checker_author(c)}
                      for tid, p, c in self.db.execute("SELECT id, producer, checker FROM traces")}
        learnings = {lid: json.loads(b) for lid, b in self.db.execute("SELECT id, body FROM learnings")}
        return trace_info, learnings

    def _expire_bounties(self):
        """At settlement, a bounty unsolved past its deadline returns its pool to its backers, by what each put in."""
        for bid, pool in self.db.execute(
                "SELECT id, pool FROM bounties WHERE status='open' AND deadline < ?", (self.epoch,)).fetchall():
            for h, m in coin.pro_rata(pool, self._refund_weights(bid)).items():
                self._credit(h, m, f"bounty {bid} refund")
            self.db.execute("UPDATE bounties SET status='expired', pool=0 WHERE id=?", (bid,))
            self._event(f"bounty #{bid} expired unsolved: ${pool / 1e6:,.2f} back to its backers, by what each put in")

    def remove(self, kind, oid):
        """Operator takedown. A bounty is withdrawn and its pool refunded to its backers, by what each put in; a trace
        leaves the index and search (its record stays, so royalties already owed down a learning's tree still add up)."""
        with self.lock:
            if kind == "bounty":
                r = self.db.execute("SELECT status, pool FROM bounties WHERE id=?", (int(oid),)).fetchone()
                if not r:
                    raise KeyError(f"bounty {oid}")
                if r[0] == "open":
                    for h, m in coin.pro_rata(r[1], self._refund_weights(int(oid))).items():
                        self._credit(h, m, f"bounty {oid} refund")
                self.db.execute("UPDATE bounties SET status='removed', pool=0, title='(removed)' WHERE id=?", (int(oid),))
                self.db.execute("DELETE FROM events WHERE text LIKE ?", (f"bounty #{int(oid)} %",))
            elif kind == "trace":
                if not self.db.execute("SELECT 1 FROM labels WHERE id=?", (oid,)).fetchone():
                    raise KeyError(f"trace {oid}")
                self.db.execute("DELETE FROM labels WHERE id=?", (oid,))
                if self.fts:
                    self.db.execute("DELETE FROM trace_fts WHERE id=?", (oid,))
            else:
                raise ValueError("kind is 'bounty' or 'trace'")
            self.db.commit()
        return {"removed": kind, "id": oid}

    def lots(self):
        rows = self.db.execute("SELECT lot, COUNT(*), COUNT(DISTINCT producer) FROM traces GROUP BY lot").fetchall()
        return {"epoch": self.epoch, "k": self.k, "reserve_micros": self.reserve,
                "lots": [{"lot": l, "traces": n, "producers": p} for l, n, p in rows]}

    def bid(self, b):
        if b.get("license", "shared") != "shared":
            raise ValueError("v0.1 reference node clears shared licences only")
        price = int(b["price_micros"])
        if price <= 0:
            raise ValueError("price must be positive")
        need_address(b.get("bidder"), "bidder")
        with self.lock:
            if not self.db.execute("SELECT 1 FROM traces WHERE lot=? LIMIT 1", (b["lot"],)).fetchone():
                raise ValueError(f"no traces in lot {b['lot']!r}; GET /v0/lots lists them")
            pending = self.db.execute("SELECT COALESCE(SUM(price), 0) FROM bids WHERE bidder=? AND epoch=?",
                                      (b["bidder"], self.epoch)).fetchone()[0]
            self._need_funds(b["bidder"], price, pending)
            self._tx_fee(b["bidder"])
            self.db.execute("INSERT INTO bids VALUES (?,?,?,?,?)", (b["lot"], b["bidder"], price, "shared", self.epoch))
            self.db.commit()
        return {"accepted": True, "epoch": self.epoch}

    def clear(self):
        """Batch-clear every lot with bids this epoch. Each licence buys every trace in the lot so far; the payment is
        split across those traces equally, then 85/10/5 to producer, checker author, validators."""
        out = []
        with self.lock:
            e = self.epoch
            by_lot = defaultdict(list)
            for lot, bidder, price in self.db.execute("SELECT lot, bidder, price FROM bids WHERE epoch=?", (e,)):
                by_lot[lot].append(Bid(bidder, price))
            for lot, bids in sorted(by_lot.items()):
                winners, price = clear_shared(bids, self.k, self.reserve)
                traces = self.db.execute("SELECT id, producer, checker FROM traces WHERE lot=? ORDER BY id", (lot,)).fetchall()
                if not winners or not traces:
                    continue
                for w in winners:
                    self.db.execute("INSERT INTO licences VALUES (?,?,?,?,?)", (lot, w, price, e, json.dumps([t[0] for t in traces])))
                    self._credit(w, -price, f"licence {lot}")
                    per, dust = divmod(price, len(traces))
                    for i, (tid, producer, checker) in enumerate(traces):
                        amt = per + (dust if i == 0 else 0)
                        author = self._checker_author(checker)
                        for acct, m in split_trace_sale(amt, producer, author, self.validators).items():
                            self._credit(acct, m, f"sale {tid[:19]}")
                out.append({"lot": lot, "winners": winners, "price_micros": price, "traces": len(traces)})
                self._event(f"lot {lot.split('|')[0]} cleared: {len(winners)} licence{'s' if len(winners) != 1 else ''} "
                            f"at ${price / 1e6:,.2f}, paid to {len(traces)} traces' producers")
            self.db.execute("DELETE FROM bids WHERE epoch=?", (e,))        # cleared once: a second call charges nobody
            self.db.commit()
        return {"epoch": e, "cleared": out}

    def _checker_author(self, checker_id):
        r = self.db.execute("SELECT author FROM checkers WHERE id=?", (checker_id,)).fetchone()
        return r[0] if r else self.validators[0]

    def _too_deep(self, parents):
        """Would a learning citing these parents sit more than MAX_DEPTH learnings deep? Walked level by level, so it
        stops at MAX_DEPTH however long a chain someone built."""
        frontier, depth = {p["trace"] for p in parents}, 0
        while frontier:
            bodies = [r[0] for r in (self.db.execute("SELECT body FROM learnings WHERE id=?", (pid,)).fetchone()
                                     for pid in frontier) if r]
            if not bodies:
                return False
            depth += 1
            if depth >= MAX_DEPTH:
                return True
            frontier = {p["trace"] for b in bodies for p in json.loads(b)["parents"]}
        return False

    def register_learning(self, l):
        a = l.get("attestation") or {}
        need_address(l.get("trainer"), "trainer")
        if not (a.get("validator") in self.validators and float(a.get("after", 0)) > float(a.get("before", 1))):
            raise ValueError("rejected: needs a known validator's attestation showing after > before")
        for p in l["parents"]:
            known = self.db.execute("SELECT 1 FROM traces WHERE id=? UNION SELECT 1 FROM learnings WHERE id=?",
                                    (p["trace"], p["trace"])).fetchone()
            if not known:
                raise ValueError(f"rejected: unknown parent {p['trace']}")
        if self._too_deep(l["parents"]):
            raise ValueError(f"rejected: learnings nest at most {MAX_DEPTH} deep")
        lid = object_id(l)
        with self.lock:
            if not self.db.execute("SELECT 1 FROM learnings WHERE id=?", (lid,)).fetchone():
                self._tx_fee(l["trainer"])
            new = self.db.execute("INSERT OR IGNORE INTO learnings VALUES (?,?,?)",
                                  (lid, canonical(l).decode(), self.epoch)).rowcount
            if new:
                self._event(f"learning attested: {l['kind']} for {l['base_model']['name']}, "
                            f"{a.get('metric') or 'score'} {float(a['before']):.1%} → {float(a['after']):.1%}")
            self.db.commit()
        return {"id": lid}

    def usage(self, u):
        calls = int(u["calls"])
        if calls <= 0:
            raise ValueError("calls must be positive")
        need_address(u.get("consumer"), "consumer")
        with self.lock:
            r = self.db.execute("SELECT body FROM learnings WHERE id=?", (u["learning"],)).fetchone()
            if not r:
                raise ValueError(f"unknown learning {u['learning']}")
            if self.test_credits:            # usage is charged at settlement; it must be covered when it is reported
                price = lambda lid: json.loads(self.db.execute("SELECT body FROM learnings WHERE id=?", (lid,))
                                               .fetchone()[0])["royalty"]["per_call_micros"]
                pending = sum(c * price(lid) for lid, c in self.db.execute(
                    "SELECT learning, SUM(calls) FROM usage WHERE consumer=? AND epoch=? GROUP BY learning",
                    (u["consumer"], self.epoch)).fetchall())
                self._need_funds(u["consumer"], calls * json.loads(r[0])["royalty"]["per_call_micros"], pending)
            self._tx_fee(u["consumer"])
            self.db.execute("INSERT INTO usage VALUES (?,?,?,?)", (u["learning"], u["consumer"], calls, self.epoch))
            self.db.commit()
        return {"metered": calls}

    def settle(self):
        """Charge metered usage, pay royalties down the family tree, then publish this epoch's Merkle payout root."""
        with self.lock:
            e = self.epoch
            self._expire_bounties()
            trace_info, learnings = self._tree()
            solved = {lid: bid for bid, lid in self.db.execute(
                "SELECT id, learning FROM bounties WHERE status='solved'").fetchall()}
            for lid, consumer, calls in self.db.execute(
                    "SELECT learning, consumer, SUM(calls) FROM usage WHERE epoch=? GROUP BY learning, consumer", (e,)).fetchall():
                L = learnings[lid]
                amount = calls * L["royalty"]["per_call_micros"]
                self._credit(consumer, -amount, f"usage {lid[:19]}")
                if lid in solved:      # a bounty's winning learning: its coin holders take their cut off the top
                    holds = dict(self.db.execute("SELECT holder, coins FROM holdings WHERE bounty=?",
                                                 (solved[lid],)).fetchall())
                    cut = coin.pro_rata(int(amount * coin.HOLDER_CUT), holds)
                    for h, m in cut.items():
                        self._credit(h, m, f"bounty {solved[lid]} coin")
                    amount -= sum(cut.values())
                for acct, m in split_usage(amount, L, trace_info, self.validators, learnings).items():
                    self._credit(acct, m, f"royalty {lid[:19]}")
            self._bill_fees()
            # debits were collected up front (x402 / payment channel); the root pays out every credit, gross
            payouts = {a: m for a, m in self.db.execute(
                "SELECT account, SUM(micros) FROM ledger WHERE epoch=? AND micros > 0 GROUP BY account", (e,))
                if ADDRESS.fullmatch(str(a))}
            leaves = {a: leaf(e, a, m) for a, m in payouts.items()}
            levels = build_tree(list(leaves.values()))
            root = levels[-1][0].hex()
            claims = {a: {"amount_micros": m, "proof": ["0x" + h.hex() for h in proof(levels, leaves[a])]}
                      for a, m in payouts.items()}
            self.db.execute("INSERT OR REPLACE INTO roots VALUES (?,?,?,?)",
                            (e, "0x" + root, sum(payouts.values()), json.dumps(claims)))
            self._event(f"epoch {e} settled: ${sum(payouts.values()) / 1e6:,.2f} to {len(payouts)} addresses, "
                        f"payout root 0x{root[:8]}…", force=True)
            self._set_meta("epoch", e + 1)
            self.db.commit()
        return {"epoch": e, "root": "0x" + root, "total_micros": sum(payouts.values()), "claims": claims}

    def _learning_path(self, L, seen=None):
        """A learning's taxonomy branch: the most common branch among the traces it was built from."""
        seen = seen or set()
        counts = {}
        for p in L["parents"]:
            pid = p["trace"]
            r = self.db.execute("SELECT path FROM labels WHERE id=?", (pid,)).fetchone()
            if r:
                counts[r[0]] = counts.get(r[0], 0) + p["weight"]
            elif pid not in seen:
                b = self.db.execute("SELECT body FROM learnings WHERE id=?", (pid,)).fetchone()
                if b:
                    seen.add(pid)
                    sub = self._learning_path(json.loads(b[0]), seen)
                    counts[sub] = counts.get(sub, 0) + p["weight"]
        return max(counts, key=counts.get) if counts else ""

    def find_learnings(self, path="", model="", kind="", min_gain=0.0, limit=20):
        """What an agent asks when its checker keeps failing: attested learnings for this branch and base model, the
        biggest measured gain first."""
        solved = {lid: bid for bid, lid in self.db.execute("SELECT id, learning FROM bounties WHERE status='solved'")}
        out = []
        for lid, body in self.db.execute("SELECT id, body FROM learnings").fetchall():
            L = json.loads(body)
            a = L.get("attestation") or {}
            gain = float(a.get("after", 0)) - float(a.get("before", 0))
            p = self._learning_path(L)
            if path and not (p == path.strip("/") or p.startswith(path.strip("/") + "/")):
                continue
            if model and L["base_model"]["name"] != model:
                continue
            if (kind and L["kind"] != kind) or gain < float(min_gain):
                continue
            out.append({"id": lid, "kind": L["kind"], "task": L["task"], "path": p, "base_model": L["base_model"]["name"],
                        "name": (L.get("artifact") or {}).get("name"),
                        "metric": a.get("metric"), "before": a.get("before"), "after": a.get("after"),
                        "p_value": a.get("p_value"), "n": a.get("n"), "first_try": a.get("first_try"),
                        "gain": round(gain, 4), "eval_set": a.get("eval_set"), "release": L.get("release", "licensed"),
                        "per_call_micros": L["royalty"]["per_call_micros"], "artifact": L["artifact"],
                        "traces": len(L["parents"]), "solved_bounty": solved.get(lid)})
        out.sort(key=lambda x: (-x["gain"], x["id"]))
        return {"learnings": out[:int(limit)], "count": len(out)}

    def get_learning(self, lid):
        r = self.db.execute("SELECT body FROM learnings WHERE id=?", (lid,)).fetchone()
        if not r:
            raise KeyError(lid)
        return dict(json.loads(r[0]), id=lid)

    def describe(self):
        """/.well-known/trace-exchange.json: how an agent that finds this node can use it."""
        settlement = ({"asset": "test credits", "network": "testnet", "faucet": "POST /v0/faucet",
                       "credits_micros": self.test_credits} if self.test_credits else {"asset": "USDC", "chain": "base"})
        return {"protocol": "trace-exchange/0.1", "name": "traceX", "api": "/v0", "mcp": "/mcp",
                "taxonomy": TAXONOMY_VERSION, "classifier": self.engine.name, "epoch": self.epoch,
                "settlement": settlement, "privacy": ["skeleton", "open"],
                "fee_per_transaction_nanos": self.tx_fee_nanos,
                "start_here": ["GET /v0/taxonomy", "GET /v0/search", "GET /v0/learnings", "GET /v0/bounties"]}

    def provenance(self, oid, _depth=0):
        r = self.db.execute("SELECT body FROM learnings WHERE id=?", (oid,)).fetchone()
        if r:
            L = json.loads(r[0])
            parents = ([dict(p, **self.provenance(p["trace"], _depth + 1)) for p in L["parents"]]
                       if _depth < MAX_DEPTH else [dict(p, truncated=True) for p in L["parents"]])
            return {"id": oid, "type": "learning", "trainer": L["trainer"], "attestation": L["attestation"],
                    "parents": parents}
        r = self.db.execute("SELECT producer, lot FROM traces WHERE id=?", (oid,)).fetchone()
        if r:
            return {"id": oid, "type": "trace", "producer": r[0], "lot": r[1]}
        raise KeyError(oid)

    def balance(self, account):
        rows = self.db.execute("SELECT epoch, SUM(micros) FROM ledger WHERE account=? GROUP BY epoch", (account,)).fetchall()
        return {"account": account, "by_epoch": {str(e): m for e, m in rows}, "total_micros": sum(m for _, m in rows)}


ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
STATIC = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
          ".mp4": "video/mp4", ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8"}
MAX_BODY = 64 * 1024
# On a public node these stay with the operator: attestations and settlement are not signed yet, so whoever could call
# them could mint payouts. Coin transfers wait for signed wallets for the same reason.
ADMIN_ROUTES = {"/v0/epochs/clear", "/v0/epochs/settle", "/v0/checkers", "/v0/learnings", "/v0/admin/remove",
                "/v0/admin/reclassify", "/v0/validators", "/v0/decoys", "/v0/decoys/unseal", "/v0/licences/direct"}
ADMIN_LEARNING_ACTIONS = {"commits", "reveals"}   # validator messages: operator-relayed until they are signed
ADMIN_ACTIONS = {"claims", "transfer"}


class RateLimit:
    """Per-client sliding window, so one noisy client cannot take a public node down for everyone else."""

    def __init__(self, reads=600, writes=30, window=60.0, all_writes=600):
        self.limits, self.window, self.hits, self.lock = {"read": reads, "write": writes}, window, {}, threading.Lock()
        self.all_writes, self.recent_writes = all_writes, []

    def allow(self, client, kind):
        now = time.monotonic()
        with self.lock:
            if kind == "write":                              # a ceiling for everyone together, too
                self.recent_writes = [t for t in self.recent_writes if now - t < self.window]
                if len(self.recent_writes) >= self.all_writes:
                    return False
            q = [t for t in self.hits.get((client, kind), ()) if now - t < self.window]
            ok = len(q) < self.limits[kind]
            if ok:
                q.append(now)
                if kind == "write":
                    self.recent_writes.append(now)
            self.hits[(client, kind)] = q
            if len(self.hits) > 20_000:                      # forget idle clients
                self.hits = {k: v for k, v in self.hits.items() if v and now - v[-1] < self.window}
            return ok


def make_handler(ex, public=False, admin_token=None, limiter=None):
    limiter = limiter or (RateLimit() if public else None)
    coin_mode = getattr(ex, "economy", "") == "coin"
    to_units = sys.modules[type(ex).__module__].to_units if coin_mode else None     # "12.5" TXC -> base units, exactly

    class H(BaseHTTPRequestHandler):
        server_version = "traceX/0.1"

        def handle(self):
            try:
                super().handle()
            except (ConnectionError, TimeoutError):        # the client went away mid-response: nothing to do
                pass

        def _client(self):
            if not public:
                return self.client_address[0]
            h = self.headers                              # behind the host's proxy (Render) the socket is the proxy
            fwd = h.get("CF-Connecting-IP") or h.get("True-Client-IP") or h.get("X-Forwarded-For", "").split(",")[0]
            return fwd.strip() or self.client_address[0]

        def _headers(self, ctype, n, extra=None):
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(n))
            self.send_header("Access-Control-Allow-Origin", "*")       # no cookies or sessions: any page may call it
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def _send(self, code, obj, extra=None):
            body = json.dumps(wire(obj), indent=1).encode()
            self.send_response(code)
            self._headers("application/json", len(body), dict({"Cache-Control": "no-store"}, **(extra or {})))
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError(f"request body over {MAX_BODY // 1024} KB")
            raw = self.rfile.read(n) if n else b""
            return json.loads(raw or b"{}")

        def _limited(self, kind):
            if limiter and not limiter.allow(self._client(), kind):
                self._send(429, {"error": "too many requests; slow down"}, {"Retry-After": "30"})
                return True
            return False

        def _admin(self):
            if not public:
                return True
            got = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            return bool(admin_token) and hmac.compare_digest(got.encode(), admin_token.encode())

        def _static(self, path):
            rel = "site/index.html" if path in ("/", "/index.html") else path.lstrip("/")
            if not (rel.startswith("site/") or rel.startswith("video/")):
                return False
            full = os.path.realpath(os.path.join(ROOT, rel))
            ctype = STATIC.get(os.path.splitext(full)[1].lower())
            if not full.startswith(ROOT + os.sep) or not ctype or not os.path.isfile(full):
                return False
            with open(full, "rb") as f:
                body = f.read()
            self.send_response(200)
            self._headers(ctype, len(body), {"Cache-Control": "public, max-age=300",
                                             "Referrer-Policy": "strict-origin-when-cross-origin"})
            if self.command != "HEAD":
                self.wfile.write(body)
            return True

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Mcp-Session-Id")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self):
            u = urlparse(self.path)
            if u.path == "/healthz" or not self._static(u.path):
                self.send_response(200 if u.path == "/healthz" else 404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def do_GET(self):
            if self._limited("read"):
                return
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/healthz":
                    return self._send(200, {"ok": True, "epoch": ex.epoch})
                if u.path == "/v0/lots":
                    return self._send(200, ex.lots())
                if u.path == "/v0/taxonomy":
                    return self._send(200, ex.taxonomy())
                if u.path == "/v0/search":
                    return self._send(200, ex.search(q.get("q", ""), q.get("path", ""), q.get("failure", ""),
                                                      q.get("model", ""), int(q.get("limit", 20)), q.get("sort", ""),
                                                      int(q.get("offset", 0)), q.get("facets") in ("1", "true")))
                if u.path == "/v0/bounties":
                    return self._send(200, ex.bounties(q.get("path", ""), q.get("status", "")))
                mv = re.fullmatch(r"/v0/learnings/([^/]+)/verdict", u.path)
                if mv and coin_mode:
                    return self._send(200, ex.verdict(mv[1]))
                if u.path.startswith("/v0/learnings/"):
                    return self._send(200, ex.get_learning(u.path.split("/", 3)[3]))
                if u.path == "/v0/coin" and coin_mode:
                    return self._send(200, ex.coin_stats())
                if u.path == "/v0/fees" and hasattr(ex, "fees"):
                    return self._send(200, ex.fees())
                if u.path == "/v0/validators" and coin_mode:
                    return self._send(200, ex.validators_list())
                if u.path == "/v0/quote" and coin_mode:
                    return self._send(200, ex.quote(q.get("side", "buy"), int(q.get("amount", 0))))
                if u.path == "/v0/learnings":
                    extra = {"include_pending": q.get("all") in ("1", "true")} if coin_mode else {}
                    return self._send(200, ex.find_learnings(q.get("path", ""), q.get("model", ""), q.get("kind", ""),
                                                             float(q.get("min_gain", 0)), min(int(q.get("limit", 20)), 500),
                                                             **extra))
                if u.path == "/v0/stats":
                    return self._send(200, ex.stats())
                if u.path == "/v0/events":
                    return self._send(200, ex.events(int(q.get("limit", 30))))
                if u.path == "/.well-known/trace-exchange.json":
                    return self._send(200, ex.describe())
                if u.path == "/mcp":
                    return self._send(405, {"error": "use POST /mcp (MCP streamable HTTP, JSON responses)"})
                mh = re.fullmatch(r"/v0/bounties/(\d+)/holders", u.path)
                if mh:
                    return self._send(200, ex.holders(int(mh[1])))
                if u.path.startswith("/v0/provenance/"):
                    return self._send(200, ex.provenance(u.path.split("/", 3)[3]))
                if u.path.startswith("/v0/balances/"):
                    return self._send(200, ex.balance(u.path.split("/", 3)[3]))
                if u.path.startswith("/v0/wallets/"):
                    return self._send(200, ex.wallet(u.path.split("/", 3)[3]))
                if self._static(u.path):
                    return
                self._send(404, {"error": "not found"})
            except KeyError as e:
                self._send(404, {"error": f"unknown {e}"})
            except ValueError as e:
                self._send(400, {"error": str(e)})

        def do_POST(self):
            if self._limited("write"):
                return
            self.path = urlparse(self.path).path         # a POST says everything in its body; a proxy's query is noise
            try:
                if self.path == "/mcp":                  # agents connect here as an MCP server (JSON-RPC over HTTP)
                    from traceex.mcp import handle, NodeBackend
                    msg = self._body()
                    backend = NodeBackend(ex)
                    if isinstance(msg, list):
                        out = [r for r in (handle(m, backend) for m in msg[:20]) if r is not None]
                    else:
                        out = handle(msg, backend)
                    if out is None or out == []:
                        self.send_response(202)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    return self._send(200, out)
                routes = {"/v0/traces": ex.submit_trace, "/v0/bids": ex.bid, "/v0/learnings": ex.register_learning,
                          "/v0/usage": ex.usage, "/v0/epochs/clear": lambda _: ex.clear(),
                          "/v0/epochs/settle": lambda _: ex.settle(),
                          "/v0/checkers": lambda b: ex.register_checker(b["id"], b["author"]),
                          "/v0/bounties": ex.post_bounty,
                          "/v0/admin/remove": lambda b: ex.remove(b.get("kind"), b.get("id")),
                          "/v0/admin/reclassify": lambda b: refile_in_background(ex, b.get("only", "rules"),
                                                                                  int(b.get("limit") or 0)),
                          "/v0/faucet": lambda b: ex.faucet(b.get("address"), self._client() if public else "")}
                if coin_mode:
                    routes.update({"/v0/swap": lambda b: ex.swap(b.get("account"), b.get("side"), int(b.get("amount") or 0)),
                                   "/v0/credits": lambda b: ex.buy_credits(b.get("account"), int(b.get("micros") or 0),
                                                                           int(b.get("units") or 0)),
                                   "/v0/validators": lambda b: ex.register_validator(b.get("address"),
                                                                                     int(b.get("stake_units") or 0)),
                                   "/v0/decoys": lambda b: ex.register_decoy(b["learning"], b["digest"], b["funder"]),
                                   "/v0/decoys/unseal": lambda b: ex.unseal_decoy(b["learning"], b["gain"], b["salt"]),
                                   "/v0/licences/direct": lambda b: ex.direct_licence(b["lot"], b["buyer"], b["traces"])})
                ml = re.fullmatch(r"/v0/learnings/([^/]+)/(commits|reveals|challenges)", self.path)
                if ml and coin_mode:
                    lid, act = ml[1], ml[2]
                    admin = act in ADMIN_LEARNING_ACTIONS
                    fn = {"commits": lambda b: ex.commit(lid, b["validator"], b["digest"], b.get("round")),
                          "reveals": lambda b: ex.reveal(lid, b["validator"], b["attestation"], b["salt"], b.get("round")),
                          "challenges": lambda b: ex.challenge(lid, b["challenger"])}[act]
                    if admin and not self._admin():
                        return self._send(403, {"error": "validator messages are relayed by the operator until they are signed"})
                    return self._send(200, fn(self._body()))
                fn, admin = routes.get(self.path), self.path in ADMIN_ROUTES
                m = re.fullmatch(r"/v0/bounties/(\d+)/(claims|buy|sell|transfer)", self.path)
                if m:
                    i, act = int(m[1]), m[2]
                    admin = act in ADMIN_ACTIONS
                    fn = {"claims": (lambda b: ex.claim_bounty(i, b["learning"], b.get("attestation"))) if coin_mode
                          else (lambda b: ex.claim_bounty(i, b["learning"])),
                          "buy": (lambda b: ex.buy_coins(i, b["buyer"], int(b.get("micros") or 0),
                                                         units=int(b.get("units") or 0) or to_units(b.get("coins") or 0)))
                          if coin_mode else (lambda b: ex.buy_coins(i, b["buyer"], b["micros"])),
                          "sell": lambda b: ex.sell_coins(i, b["seller"], b["coins"]),
                          "transfer": lambda b: ex.transfer_coins(i, b["from"], b["to"], b["coins"])}[act]
                if not fn:
                    return self._send(404, {"error": "not found"})
                if admin and not self._admin():
                    return self._send(403, {"error": "on this public node that call is kept to the operator "
                                                     "(attestations and transfers are not signed yet)"})
                self._send(200, fn(self._body()))
            except PermissionError as e:
                self._send(403, {"error": str(e)})
            except Full as e:
                self._send(503, {"error": str(e)}, {"Retry-After": "3600"})
            except KeyError as e:
                self._send(400, {"error": f"missing or unknown {e}"})
            except (ValueError, TypeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, fmt, *a):
            if public:                                   # one line per request in the host's log, no bodies
                sys.stderr.write(f"{self.command} {urlparse(self.path).path} {a[1] if len(a) > 1 else ''}\n")
    return H


def serve(port=8787, db="exchange.db", host="127.0.0.1", public=False, admin_token=None, economy="usdc", **kw):
    if economy == "coin":
        from coin import CoinExchange                  # the coin economy: node/coin.py
        ex = CoinExchange(db, **kw)
    else:
        ex = Exchange(db, **kw)
    srv = ThreadingHTTPServer((host, port), make_handler(ex, public=public, admin_token=admin_token))
    srv.daemon_threads = True
    return ex, srv


def refile_in_background(ex, only="rules", limit=0):
    """Start a re-classification run without holding up the caller; progress shows in /v0/stats."""
    if ex.refile and ex.refile.get("running"):
        raise ValueError("a re-file is already running")
    if only == "rules" and ex.engine.name == "rules":
        return {"skipped": "the node's classifier is the keyword engine; add TYPESAFE_API_KEY to re-file with Jev"}
    n = ex.db.execute("SELECT COUNT(*) FROM labels" + (" WHERE engine = 'rules'" if only == "rules" else "")).fetchone()[0]
    threading.Thread(target=ex.reclassify, kwargs={"only": only, "limit": limit}, daemon=True, name="refile").start()
    return {"started": True, "traces": min(n, limit) if limit else n, "engine": ex.engine.name}


def run_epochs(ex, hours):
    """Clear the auctions and settle every `hours`, so a hosted node runs without an operator at the keyboard."""
    ex.epoch_hours = hours

    def loop():
        while True:
            time.sleep(hours * 3600)
            try:
                ex.clear()
                s = ex.settle()
                print(f"epoch {s['epoch']} settled: root {s['root'][:18]}… total {s['total_micros']} micros", flush=True)
            except Exception as e:                       # keep the clock running; the next epoch retries
                print(f"epoch run failed: {type(e).__name__}: {e}", flush=True)
    threading.Thread(target=loop, daemon=True, name="epochs").start()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):           # a Windows console can't print every character the feed uses
        stream.reconfigure(errors="backslashreplace")
    env = os.environ.get
    ap = argparse.ArgumentParser(description="traceX exchange node: API, MCP endpoint and website in one process")
    ap.add_argument("--host", default=env("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(env("PORT", 8787)))
    ap.add_argument("--db", default=env("TRACEX_DB", "exchange.db"))
    ap.add_argument("--k", type=int, default=int(env("TRACEX_K", 10)))
    ap.add_argument("--reserve-micros", type=int, default=int(env("TRACEX_RESERVE_MICROS", 1_000)),
                    help="reserve price per licence in a lot auction")
    ap.add_argument("--public", action="store_true", default=env("TRACEX_PUBLIC") == "1",
                    help="rate limits on; settle, clear, checkers, learnings, claims and transfers need TRACEX_ADMIN_TOKEN")
    ap.add_argument("--seed", action="store_true", default=env("TRACEX_SEED") == "1",
                    help="on an empty database, load the example traces, bounties and learnings")
    ap.add_argument("--test-credits", type=int, default=int(env("TRACEX_TEST_CREDITS", 0)),
                    help="testnet: micros of test credits each new wallet can take once (0 = off)")
    ap.add_argument("--economy", choices=("usdc", "coin"), default=env("TRACEX_ECONOMY", "usdc"),
                    help="usdc: contributors are paid in dollars (v0.1). coin: they earn TXC, payments buy and burn it")
    ap.add_argument("--max-db-mb", type=int, default=int(env("TRACEX_MAX_DB_MB", 0)),
                    help="stop taking new traces and bounties when the database file passes this size (0 = no limit)")
    ap.add_argument("--epoch-hours", type=float, default=float(env("TRACEX_EPOCH_HOURS", 0)),
                    help="clear and settle automatically every N hours (0 = only when the operator calls it)")
    ap.add_argument("--fee-to", default=env("TRACEX_FEE_TO") or None,
                    help="the address that receives the transaction fees (it pays for the electricity); default: 'network'")
    a = ap.parse_args()
    token = env("TRACEX_ADMIN_TOKEN") or None
    if a.public and not token:
        print("warning: --public without TRACEX_ADMIN_TOKEN: operator calls are switched off", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.db)), exist_ok=True)
    ex, srv = serve(a.port, a.db, host=a.host, public=a.public, admin_token=token, k=a.k, economy=a.economy,
                    reserve_micros=a.reserve_micros, test_credits=a.test_credits, max_db_bytes=a.max_db_mb * 1024 * 1024,
                    fee_to=a.fee_to)
    if a.seed:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from seed import seed_if_empty
        print(seed_if_empty(ex), flush=True)
    if a.epoch_hours > 0:
        run_epochs(ex, a.epoch_hours)
    if ex.engine.name != "rules" and ex.db.execute("SELECT 1 FROM labels WHERE engine = 'rules' LIMIT 1").fetchone():
        print("re-filing keyword-classified traces with", ex.engine.name, refile_in_background(ex), flush=True)
    mode = "public" if a.public else "local"
    print(f"traceX node on http://{a.host}:{a.port}  (db {a.db}, {mode}, {a.economy} economy, "
          f"{'testnet' if a.test_credits else 'USDC settlement'}, epochs {a.epoch_hours or 'manual'}h)", flush=True)
    srv.serve_forever()
