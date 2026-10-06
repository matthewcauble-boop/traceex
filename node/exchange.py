"""Reference exchange node (spec section 7). Standard library only: http.server + sqlite3.

    python node/exchange.py --port 8787 --db exchange.db                 # local
    python node/exchange.py --host 0.0.0.0 --public --seed --economy sats --test-credits 30000000 --epoch-hours 24

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
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk", "python"))
from traceex import canonical, object_id  # noqa: E402
from traceex.client import privacy_leaks  # noqa: E402
from traceex.auction import Bid, clear_shared  # noqa: E402
from traceex.royalty import split_trace_sale, split_usage, pro_rata, MAX_DEPTH  # noqa: E402
from traceex.merkle import leaf, build_tree, proof  # noqa: E402
from traceex.classify import classify, default_engine, nodes, RulesEngine, TAXONOMY_VERSION  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import Registry  # noqa: E402  (v0.7: the failure registry and fix tracking)
from composition import Composition  # noqa: E402  (v0.8: step graphs and cross-agent composition, after TROPIC)
from leviathan_search import Query, best_snippet, render as render_cards  # noqa: E402  (v0.7: the Leviathan-style index)

# A bounty's pledges, when a learning solves it: the solver is paid most, the traces it was built from still earn.
# Backers get the solution and nothing else: no token, no share of its revenue, nothing to trade.
BOUNTY_SPLIT = {"trainer": 0.70, "traces": 0.20, "checkers": 0.05, "validators": 0.05}
# The standard transaction fee: 58 millisatoshis (TX_FEE_MSATS), about $0.00005 with bitcoin at $85,962 (Coinbase
# spot, 2026-10-05) and about 125 times the electricity of the dearest transaction examples/fees/measure.py finds
# (registering a learning, its 5.5 KB kept in three copies for ten years: about $0.0000004), far more than any other.
# The margin is the point: the fee is the whole income of the operator that served the transaction (paid to it in
# sats, at once) and prices out spam at machine scale, where a billion junk transactions cost 58 million sats (about
# $50,000) instead of about $400. This is the one setting; 100x to 200x the measured electricity keeps both jobs. It is
# fixed in sats, so its dollar value floats with bitcoin; a sats node can re-peg it to a dollar target every N epochs
# (sats.Params.fee_repeg_epochs, off by default). A millisatoshi is the smallest amount Lightning moves, so the fee is a
# whole number of them and nothing carries over.
TX_FEE_MSATS = 58
# Retired with the v0.1 dollar node (USDC settlement is no longer the mainnet path; see SPEC 5): its fee in
# nano-dollars, $0.00005, billed in whole micro-dollars.
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
                                     pledged INT DEFAULT 0, deadline INT, status TEXT, winner TEXT, learning TEXT,
                                     epoch INT, key TEXT, note TEXT, failure_id TEXT);
CREATE INDEX IF NOT EXISTS bounties_key ON bounties(key, status);
CREATE TABLE IF NOT EXISTS pledges  (id INTEGER PRIMARY KEY, bounty INT, backer TEXT, amount INT, epoch INT,
                                     payment INT);
CREATE INDEX IF NOT EXISTS pledges_bounty ON pledges(bounty);
CREATE TABLE IF NOT EXISTS fees     (account TEXT PRIMARY KEY, nanos INT);
CREATE TABLE IF NOT EXISTS events   (id INTEGER PRIMARY KEY, at TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS grants   (account TEXT PRIMARY KEY, micros INT, at TEXT, source TEXT);
"""
ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}")
PATH = re.compile(r"[a-z_]+(/[a-z_]+){0,5}")
SORTS = ("relevant", "new", "bounty")
KINDS = ("trace", "failure", "all")
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


class PaymentRequired(ValueError):
    """An account can't cover a payment. Over HTTP this is 402 Payment Required with an L402 challenge (a Lightning
    invoice for what is missing and a macaroon); on the testnet the invoice is a placeholder and the faucet pays."""

    def __init__(self, message, account="", msats=0):
        super().__init__(message)
        self.account, self.msats = account, max(int(msats or 0), 0)


def msats_in(body, key="", required=False):
    """An amount sent to a sats-priced node: `{key}_msats` (integer millisatoshis) or `{key}_sats` (integer sats), or
    `msats` / `sats` when key is empty. Dollar amounts (`_micros`) are refused, so nobody pays in sats believing they
    paid dollars. Returns msats (0 when absent, unless required)."""
    pre = f"{key}_" if key else ""
    for name, scale in ((pre + "msats", 1), (pre + "sats", 1000)):
        v = body.get(name)
        if v not in (None, "", 0, "0"):
            try:
                n = int(str(v))
            except ValueError:
                raise ValueError(f"{name} is a whole number (millisatoshis or satoshis), not {v!r}")
            return n * scale
    if body.get(pre + "micros") not in (None, "", 0, "0"):
        raise ValueError(f"this node prices everything in bitcoin: send {pre}msats (millisatoshis) or {pre}sats, "
                         f"not {pre}micros")
    if required:
        raise ValueError(f"send {pre}msats (millisatoshis) or {pre}sats")
    return 0


def need_address(a, what):
    """Anything that can be paid must be a real address: one malformed account would make the epoch's Merkle root
    impossible to build, and settlement would stall for everyone."""
    if not ADDRESS.fullmatch(str(a or "")):
        raise ValueError(f"{what} must be a 0x address (40 hex characters)")
    return a


def wire(obj):
    """What goes into JSON for any client: every amount is an integer (msats or sats); one JavaScript would round
    (2**53 msats and up, about 90,000 BTC) travels as a decimal string."""
    if isinstance(obj, dict):
        return {k: wire(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [wire(v) for v in obj]
    if isinstance(obj, int) and not isinstance(obj, bool) and abs(obj) >= SAFE_INT:
        return str(obj)
    return obj


def bounty_key(path, failure="", base_model=""):
    """What a bounty is for, as the classifier files it: taxonomy branch, failure mode, base model. A new post with the
    key of an open bounty backs that bounty instead of opening a duplicate."""
    return "|".join((str(path or "").strip("/"), str(failure or "").strip(), str(base_model or "").strip()))


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


class Exchange(Registry, Composition):
    """The v0.1 reference node: contributors paid in dollars (micros). Retired as a mainnet path (USDC settlement and
    x402 gave way to bitcoin over Lightning, SPEC 5); kept as the library's default for its tests and examples. The sats
    node (node/sats.py, v0.6) pays everything in millisatoshis, with no token."""
    money = "micros"                              # the unit every amount key ends in: micros here, msats on a sats node

    def _fmt(self, amount):
        """An amount for people, in this node's money."""
        return f"${amount / 1e6:,.2f}"

    def __init__(self, path=":memory:", *, k=10, reserve_micros=1_000, validators=("0x" + "5" * 40,),
                 tx_fee_nanos=None, fee_to=None, engine=None, test_credits=0, max_db_bytes=0, unbacked_epochs=3):
        """test_credits: run as a testnet. Each new wallet can take this much test money once (micros here; msats on a
        sats node), and every spend (pledges, bids, metered usage, the transaction fee) must be covered by the wallet's
        balance. 0 = settlement is external, the reference behaviour. A node that keeps wallets charges every transaction the
        standard fee (TX_FEE_NANOS unless `tx_fee_nanos` says otherwise) and pays it to `fee_to`, whoever runs it and
        so pays its electricity ("network" until the operator names an address). unbacked_epochs: a bounty nobody has
        pledged to expires after this many epochs (and open bounties nobody backs stay out of the default search)."""
        self.path, self.max_db_bytes = path, int(max_db_bytes)
        self.tx_fee_nanos = int(TX_FEE_NANOS if tx_fee_nanos is None and test_credits else tx_fee_nanos or 0)
        self.fee_to = fee_to or "network"
        self.lock = threading.RLock()
        self.db = SafeDB(sqlite3.connect(path, check_same_thread=False), self.lock)
        try:
            self.db.executescript(SCHEMA)
        except sqlite3.OperationalError as e:        # a database from before v0.6 (bounty coins, or v0.5's token)
            was = None
            try:
                was = self.db.execute("SELECT v FROM meta WHERE k='coin_version'").fetchone()
            except sqlite3.OperationalError:
                pass
            self.db.close()
            what = f"a testnet v{was[0]} coin economy (TXC)" if was else "an older node (before v0.6)"
            raise ValueError(f"this database holds {what}; v0.6 has no token and pays everything in sats: start it "
                             "on an empty database") from e
        self.test_credits = int(test_credits)
        self.quiet = False                        # bulk loads (the seed) write one summary event instead of one per row
        self.refile = None                        # status of the last re-classification run
        self.engine = engine or default_engine()
        self.k, self.reserve, self.validators = k, reserve_micros, list(validators)
        self.unbacked_epochs = int(unbacked_epochs)
        if self._meta("epoch") is None:
            self._set_meta("epoch", "1")
        self._open_registry()                     # v0.7: the failure registry and the search index (registry.py)

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
            raise ValueError(f"not enough test credits: {account[:10]}… has {self._fmt(self._funds(account) - pending)}"
                             f", this needs {self._fmt(micros)} (POST /v0/faucet opens a wallet)")

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
            raise ValueError("this node has no test wallets (it is not a testnet)")
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
            self._event(f"a new wallet joined with {self._fmt(self.test_credits)} of test credits")
            self.db.commit()
        return self.wallet(account)

    def wallet(self, account):
        g = self.db.execute("SELECT micros FROM grants WHERE account=?", (account,)).fetchone()
        pledges = {str(b): int(m) for b, m in self.db.execute(
            "SELECT p.bounty, SUM(p.amount) FROM pledges p JOIN bounties b ON b.id = p.bounty "
            "WHERE p.backer=? AND b.status='open' GROUP BY p.bounty", (account,))}
        return {"account": account, "opened": bool(g), f"grant_{self.money}": g[0] if g else 0,
                f"balance_{self.money}": self._funds(account), f"pledged_{self.money}": pledges,
                "testnet": bool(self.test_credits)}

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
                f"pools_open_{self.money}": one("SELECT COALESCE(SUM(pool), 0) FROM bounties WHERE status='open'"),
                f"pools_paid_{self.money}": one("SELECT COALESCE(SUM(pool), 0) FROM bounties WHERE status='solved'"),
                "last_root": {"epoch": r[0], "root": r[1], f"total_{self.money}": r[2]} if r else None,
                "testnet": bool(self.test_credits), "epoch_hours": getattr(self, "epoch_hours", 0),
                "fees": self.fees(), "registry": self.registry_stats(), "composition": self.composition_stats(),
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
            return {"id": tid, "lot": lot, "duplicate": True, "failure_id": self._failure_of(tid)}
        c = classify(t, self.engine)            # may call a hosted engine: never while holding the write lock
        with self.lock:
            if self.db.execute("SELECT 1 FROM traces WHERE id=?", (tid,)).fetchone():
                return {"id": tid, "lot": lot, "duplicate": True, "failure_id": self._failure_of(tid)}
            self._tx_fee(t["producer"])
            self.db.execute("INSERT INTO traces VALUES (?,?,?,?,?,?)",
                            (tid, lot, t["producer"], t["checker"]["id"], canonical(t).decode(), self.epoch))
            fid = self._index(tid, t, c)          # label, failure registry and search index: this same transaction
            self._event(f"trace filed under {c['path_str'] or 'uncategorised'}" + (f" · {c['signature']}" if c["signature"] else ""))
            self.db.commit()
        return {"id": tid, "lot": lot, "epoch": self.epoch, "classified": c, "failure_id": fid,
                "bounties": self._matching_bounties(c, t, fid)}

    def _failure_of(self, tid):
        r = self.db.execute("SELECT failure FROM occurrences WHERE trace=?", (tid,)).fetchone()
        return r[0] if r else None

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

    def search(self, q="", path="", failure="", model="", limit=20, sort="", offset=0, facets=False, kind="trace",
               fmt="json"):
        """Find traces (and failures) by words, taxonomy branch, failure mode and model: the Leviathan-style index
        (leviathan_search.py). Words are matched with porter stemming, any of them may match, ranked by BM25 (title 2x);
        the branch and every filter are synthetic tokens inside the match. A branch resolves in tiers and never guesses
        (several candidates: status ambiguous_branch, with the candidates); when nothing matches inside it, cards from
        its nearest ancestor (or elsewhere) come back in `other_branches`, marked OTHER.

        sort: "relevant" (default with words), "new" (default without) or "bounty" (traces that feed the richest open
        bounty first). kind: "trace" (default), "failure" (the registry) or "all". fmt "text": compact cited cards, what
        an agent reads. Backed open bounties on the branch come back too, and with words, the top failures they match.
        facets=True adds counts per branch and per failure mode."""
        q, path = (q or "").strip(), (path or "").strip().strip("/")
        words = not Query(q).empty()
        sort = sort or ("relevant" if words else "new")
        if sort not in SORTS:
            raise ValueError(f"sort is one of {SORTS}")
        if kind not in KINDS:
            raise ValueError(f"kind is one of {KINDS}")
        limit, offset = max(1, min(int(limit), 1000)), max(0, int(offset))
        where = {"kind": [("kind", kind)]} if kind != "all" else {}
        if failure:
            where["failure"] = [("failure", failure)]
        if model:
            where["model"] = [("model", model), ("family", model)]
        open_all = [b for b in self.bounties(status="open")["bounties"] if b[f"pool_{self.money}"] > 0]   # backed ones
        by_pool = sort == "bounty"
        res = self.index.search(q, branch=path or None, where=where, sort="newest" if sort == "new" else "relevance",
                                limit=limit, offset=offset, everything=by_pool)
        rows = res["rows"]
        if by_pool:
            docs = self.index.docs([r[0] for r in rows])
            pools = {r[0]: self._pool_for(docs[r[0]], open_all) for r in rows}
            rows = sorted(rows, key=lambda r: -pools[r[0]])[offset:offset + limit]     # stable: ties keep their order
        hl = Query(q).highlight()
        boiler = self.index.boilerplate() if hl else frozenset()
        docs = self.index.docs([r[0] for r in rows] + [r[0] for r in res["other"]])
        hits = [self._hit(docs[r], rel, hl, boiler, open_all) for r, _, rel in rows]
        other = [dict(self._hit(docs[r], rel, hl, boiler, open_all), other_branch=True) for r, _, rel in res["other"]]
        branch = res["branch"]
        key = branch["key"] if branch else ""
        first, last = (offset + 1, offset + len(hits)) if hits else (0, 0)
        out = {"status": res["status"], "query": q, "branch": branch, "candidates": res["candidates"],
               "results": hits, "count": len(hits), "total": res["total"], "sort": sort, "offset": offset, "kind": kind,
               "shown": f"{first}-{last} of {res['total']}" if hits else f"0 of {res['total']}",
               "other_branches": other, "fallback_from": res["fallback_from"], "notes": res["notes"],
               "bounties": [b for b in open_all if not key or key == b["path"] or key.startswith(b["path"] + "/")
                            or b["path"].startswith(key + "/")]}
        if kind == "trace" and res["status"] == "ok" and (words or key):
            fw = {k: v for k, v in where.items() if k != "kind"}
            fres = self.index.search(q, branch=key or None, where=dict(fw, kind=[("kind", "failure")]), limit=3,
                                     fallback=False)
            fdocs = self.index.docs([r[0] for r in fres["rows"]])
            out["failures"] = [self._hit(fdocs[r], rel, hl, boiler, open_all) for r, _, rel in fres["rows"]]
        if facets:
            base = {k: v for k, v in where.items() if k != "failure"}
            paths, _ = self._facet_counts(self.index.matches(q, None, base))
            _, modes = self._facet_counts(self.index.matches(q, key or None, base)) if res["status"] == "ok" else ({}, {})
            out["facets"] = {"paths": paths, "modes": modes}
        if fmt == "text":
            return {"text": render_cards(dict(out, cards=hits, other_cards=other, indexed=self.index.count(),
                                              filters={"failure": failure, "model": model,
                                                       "kind": kind if kind != "trace" else ""}))}
        return out

    def _facet_counts(self, matches):
        """{branch: n} and {failure mode: n} over a set of matching records."""
        paths, modes = {}, {}
        ids = [m[0] for m in matches]
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            for grp, fs in self.db.execute(f"SELECT grp, facets FROM lx_records WHERE rowid IN ({','.join('?' * len(chunk))})",
                                           chunk).fetchall():
                paths[grp or ""] = paths.get(grp or "", 0) + 1
                for f, v in json.loads(fs):
                    if f == "failure":
                        modes[v] = modes.get(v, 0) + 1
        return paths, modes

    def _pool_for(self, doc, open_all):
        if doc.get("kind") == "failure":
            return max([b[f"pool_{self.money}"] for b in open_all if b.get("failure_id") == doc["id"]] or [0])
        return max([b[f"pool_{self.money}"] for b in open_all
                    if (b.get("failure_id") and b["failure_id"] == doc.get("failure_id"))
                    or (not b.get("failure_id") and self._feeds(b, doc["path"], doc["model"], doc["modes"]))] or [0])

    def _hit(self, doc, rel, hl, boiler, open_all):
        """One search result: the API's fields, plus the card an agent reads (title, fields, the matching sentence)."""
        texts = doc.get("texts") or []
        if doc.get("kind") == "failure":
            rate = f" {doc['pass_rate']:.0%}" if doc.get("pass_rate") is not None and doc["status"] == "partly_fixed" else ""
            fields = {"status": doc["status"].replace("_", " ") + rate,
                      "seen": f"{doc['occurrences']} case{'s' if doc['occurrences'] != 1 else ''}, "
                              f"{doc['reporters']} verified reporter{'s' if doc['reporters'] != 1 else ''}",
                      "family": doc["family"]}
            hit = {"id": doc["id"], "kind": "failure", "short_id": doc["id"], "path": doc["path"], "title": doc["title"],
                   "status": doc["status"], "pass_rate": doc.get("pass_rate"), "reporters": doc["reporters"],
                   "occurrences": doc["occurrences"], "family": doc["family"], "models": doc.get("models", []),
                   "date": (doc.get("last_seen") or "")[:10] or None, "fields": fields}
            shown = [doc["title"], *fields.values()]
        else:
            fields = {"failure": doc["signature"], "model": doc["model"], "registry": doc.get("failure_id")}
            hit = {k: doc.get(k) for k in ("id", "path", "signature", "model", "task", "lot", "fixed_fields", "snippet",
                                           "producer", "privacy", "created", "failure_id")}
            hit.update(kind="trace", short_id=doc["id"][:23], title=doc["snippet"], fields=fields,
                       date=(doc.get("created") or "")[:10] or None)
            shown = [doc["snippet"], *[str(v) for v in fields.values()]]
            feeds = self._pool_for(doc, open_all) if open_all else 0
            if feeds:
                hit[f"bounty_pool_{self.money}"] = feeds
        if rel is not None:
            hit["relevance"] = hit["score"] = rel
        if hl:
            hit["match"] = best_snippet(texts, hl, shown, 160, boiler)
        return hit

    @staticmethod
    def _feeds(b, path, model, modes):
        """Would a trace on `path` from `model` with these failure modes count toward bounty `b`?"""
        on_branch = path == b["path"] or path.startswith(b["path"] + "/")
        return on_branch and (not b["base_model"] or b["base_model"] == model) and (not b["failure"] or b["failure"] in modes)

    def _seed_amount(self, b):
        """A post's optional first pledge, in this node's money."""
        return int(b.get("seed_micros") or b.get("reward_micros") or 0)

    def post_bounty(self, b):
        """Posting is free (the transaction fee only). A bounty names a taxonomy branch (optionally a failure mode and
        base model), a hidden eval set by hash and the score a solution must reach. It is a refundable pledge escrow:
        anyone adds money to it, a solve pays the solver and the traces it was built from, and an unsolved bounty
        refunds every backer what it put in. Backers get no token, no share and nothing to trade. A post whose branch,
        failure and model match an open bounty backs that bounty instead of opening a duplicate. `seed_<money>` makes
        the poster's first pledge in the same call."""
        fid = str(b.get("failure_id") or "").strip().upper()
        if fid:                                   # v0.7: a bounty on a registry failure takes its branch and mode
            f = self._failure_row(fid)
            if not f:
                raise ValueError(f"unknown failure {fid!r}")
            b = dict(b, path=f["path"] or "uncategorised", failure=(f["modes"] or [""])[0], base_model="")
        if not b.get("eval_set") or not b.get("path"):
            raise ValueError("a bounty needs a path (or a failure_id) and an eval_set hash")
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
        seed = self._seed_amount(b)
        key = f"failure|{fid}" if fid else bounty_key(path, b.get("failure", ""), b.get("base_model", ""))
        with self.lock:
            if seed > 0:
                self._need_funds(b["poster"], seed)          # before anything is written, so a failed seed leaves nothing
            same = self.db.execute("SELECT id, deadline FROM bounties WHERE key=? AND status='open' ORDER BY id LIMIT 1",
                                   (key,)).fetchone()
            if same:                                         # the same problem is already posted: back it instead
                self._tx_fee(b["poster"])
                self.db.commit()
                what = fid or key.replace('|', ' / ').strip(' /')
                out = {"id": same[0], "status": "open", "merged": True, "deadline_epoch": same[1],
                       "note": f"bounty #{same[0]} is already open for {what}: this post "
                               "backs it instead of opening a duplicate (its poster's hidden eval decides the solve)"}
                if seed > 0:
                    out["pledge"] = self.pledge(same[0], b["poster"], seed)
                return out
            if self.test_credits:
                n = self.db.execute("SELECT COUNT(*) FROM bounties WHERE poster=? AND status='open'",
                                    (b["poster"],)).fetchone()[0]
                if n >= LIMITS["open_bounties_per_poster"]:
                    raise ValueError(f"{LIMITS['open_bounties_per_poster']} open bounties per poster")
            self._tx_fee(b["poster"])
            cur = self.db.execute(
                "INSERT INTO bounties (poster,title,path,failure,base_model,eval_set,target,deadline,status,epoch,key,"
                "failure_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (b["poster"], b.get("title", ""), path, b.get("failure", ""), b.get("base_model", ""),
                 str(b["eval_set"])[:200], target, self.epoch + epochs, "open", self.epoch, key, fid or None))
            bid = cur.lastrowid
            self._event(f"bounty #{bid} posted free on {fid or path}: {b.get('title') or 'untitled'}")
            if fid:
                self._index_failure(fid)
            self.db.commit()
        out = {"id": bid, "status": "open", "deadline_epoch": self.epoch + epochs,
               "unbacked_expires_epoch": self.epoch + self.unbacked_epochs, "failure_id": fid or None}
        if seed > 0:
            out["pledge"] = self.pledge(bid, b["poster"], seed)
        return out

    def _bounty(self, bounty_id):
        r = self.db.execute("SELECT status, pool, pledged FROM bounties WHERE id=?", (bounty_id,)).fetchone()
        if not r:
            raise KeyError(f"bounty {bounty_id}")
        return r

    def _backers(self, bounty_id):
        """{backer: what it pledged} for a bounty."""
        return {a: int(m) for a, m in self.db.execute(
            "SELECT backer, SUM(amount) FROM pledges WHERE bounty=? GROUP BY backer", (bounty_id,)).fetchall()}

    def pledge(self, bounty_id, backer, amount):
        """Add money to a bounty's escrow. It stays there until a solve pays it out or the bounty ends unsolved and it
        comes back. A pledge buys nothing: no token, no share of the solution's revenue, nothing to sell or transfer."""
        amount = int(amount)
        if amount <= 0:
            raise ValueError("a pledge must be more than 0")
        need_address(backer, "backer")
        with self.lock:
            status, pool, _ = self._bounty(bounty_id)
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}")
            self._need_funds(backer, amount)
            self._tx_fee(backer)
            self._credit(backer, -amount, f"bounty {bounty_id} pledge")
            self.db.execute("INSERT INTO pledges (bounty, backer, amount, epoch) VALUES (?,?,?,?)",
                            (bounty_id, backer, amount, self.epoch))
            self.db.execute("UPDATE bounties SET pool=pool+?, pledged=pledged+? WHERE id=?", (amount, amount, bounty_id))
            self._event(f"bounty #{bounty_id} backed with {self._fmt(amount)}; it now holds {self._fmt(pool + amount)}")
            self.db.commit()
        return {"bounty": bounty_id, f"pledged_{self.money}": amount, f"pool_{self.money}": pool + amount,
                "backers": len(self._backers(bounty_id)), "refund": "everything you pledged comes back if it ends unsolved"}

    def backers(self, bounty_id):
        status, pool, pledged = self._bounty(bounty_id)
        return {"bounty": bounty_id, "status": status, f"pool_{self.money}": pool, f"pledged_{self.money}": pledged,
                "backers": self._backers(bounty_id)}

    def bounties(self, path="", status=""):
        rows = self.db.execute("SELECT id,poster,title,path,failure,base_model,eval_set,target,pool,pledged,deadline,"
                               "status,winner,learning,epoch,note,failure_id FROM bounties WHERE status != 'removed' "
                               "ORDER BY id").fetchall()
        keys = ["id", "poster", "title", "path", "failure", "base_model", "eval_set", "target", f"pool_{self.money}",
                f"pledged_{self.money}", "deadline_epoch", "status", "winner", "learning", "posted_epoch", "note",
                "failure_id"]
        out = [dict(zip(keys, r)) for r in rows]
        counts = dict(self.db.execute("SELECT bounty, COUNT(DISTINCT backer) FROM pledges GROUP BY bounty").fetchall())
        for b in out:
            b["backers"] = counts.get(b["id"], 0)
        if status:
            out = [b for b in out if b["status"] == status]
        if path:
            p = path.strip("/")
            out = [b for b in out if p == b["path"] or p.startswith(b["path"] + "/") or b["path"].startswith(p + "/")]
        return {"bounties": out}

    def _matching_bounties(self, c, t, fid=None):
        """Open bounties a new trace feeds: the ones on its failure, and the ones on its branch, mode and model."""
        return [b["id"] for b in self.bounties(status="open")["bounties"]
                if (b["failure_id"] and b["failure_id"] == fid) or (
                    not b["failure_id"] and (c["path_str"] == b["path"] or c["path_str"].startswith(b["path"] + "/"))
                    and (not b["base_model"] or b["base_model"] == t["base_model"]["name"])
                    and (not b["failure"] or b["failure"] in c["failure_modes"].values()))]

    def claim_bounty(self, bounty_id, learning_id, attestation=None):
        """A learning claims a bounty when its validator attested the bounty's own eval set and reached the target
        (or the bounty's poster measured it there: `attestation`, as a failure bounty's automatic payout passes it).
        The pledges pay the solver and, through the learning's family tree, the traces it was built from (70 / 20 /
        5 / 5). Backers get the solution, nothing more."""
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
            poster = self.db.execute("SELECT poster FROM bounties WHERE id=?", (bounty_id,)).fetchone()[0]
            if attestation and attestation.get("validator") == poster and attestation.get("eval_set") == eval_set:
                a = attestation
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
            rest = pool - sum(payout.values())
            if rest > 0:                                         # rounding the tree couldn't place: back to the backers
                self._refund(bounty_id, rest, "rounding")
            self.db.execute("UPDATE bounties SET pool=0 WHERE id=?", (bounty_id,))
            self._event(f"bounty #{bounty_id} solved: {a.get('metric') or 'score'} {float(a['after']):.1%} beat the "
                        f"{target:.1%} target; its {self._fmt(pool)} of pledges paid the solver and the traces")
            self.db.commit()
        return {"bounty": bounty_id, "status": "solved", "winner": L["trainer"], f"pool_{self.money}": pool,
                "payout": payout}

    def _tree(self):
        trace_info = {tid: {"producer": p, "checker_author": self._checker_author(c)}
                      for tid, p, c in self.db.execute("SELECT id, producer, checker FROM traces")}
        learnings = {lid: json.loads(b) for lid, b in self.db.execute("SELECT id, body FROM learnings")}
        return self._join_credit(trace_info), learnings     # v0.8: a path trace pays the step traces it is built from

    def _refund(self, bounty_id, amount, why):
        """Give a bounty's money back to its backers: each what it pledged, pro rata if less than everything is left."""
        for h, m in pro_rata(amount, self._backers(bounty_id)).items():
            self._credit(h, m, f"bounty {bounty_id} refund: {why}")

    def _expire_bounties(self):
        """At settlement: a bounty unsolved past its deadline gives every backer back what it pledged; one nobody
        backed within `unbacked_epochs` expires too, and leaves the default search."""
        for bid, pool, pledged, deadline, posted in self.db.execute(
                "SELECT id, pool, pledged, deadline, epoch FROM bounties WHERE status='open'").fetchall():
            unbacked = not pledged and self.epoch - posted >= self.unbacked_epochs
            if deadline >= self.epoch and not unbacked:
                continue
            self._expire(bid, pool, "unbacked" if unbacked and deadline >= self.epoch else "deadline")

    def _expire(self, bid, pool, why):
        if pool > 0:
            self._refund(bid, pool, "unsolved")
        self.db.execute("UPDATE bounties SET status='expired', pool=0, note=? WHERE id=?", (why, bid))
        self._event(f"bounty #{bid} expired unsolved" + (f": {self._fmt(pool)} back to its backers, each what it put in"
                                                        if pool else (" with no backers" if why == "unbacked" else "")))

    def remove(self, kind, oid):
        """Operator takedown. A bounty is withdrawn and its pool refunded to its backers, by what each put in; a trace
        leaves the index and search (its record stays, so royalties already owed down a learning's tree still add up)."""
        with self.lock:
            if kind == "bounty":
                r = self.db.execute("SELECT status, pool FROM bounties WHERE id=?", (int(oid),)).fetchone()
                if not r:
                    raise KeyError(f"bounty {oid}")
                if r[0] == "open" and r[1] > 0:
                    self._refund(int(oid), r[1], "taken down")
                self.db.execute("UPDATE bounties SET status='removed', pool=0, title='(removed)' WHERE id=?", (int(oid),))
                self.db.execute("DELETE FROM events WHERE text LIKE ?", (f"bounty #{int(oid)} %",))
            elif kind == "trace":
                if not self.db.execute("SELECT 1 FROM labels WHERE id=?", (oid,)).fetchone():
                    raise KeyError(f"trace {oid}")
                self.db.execute("DELETE FROM labels WHERE id=?", (oid,))
                body = self.db.execute("SELECT body FROM traces WHERE id=?", (oid,)).fetchone()
                self.index.delete(oid, json.loads(body[0])["input"].splitlines() if body else ())
                fid = self._failure_of(oid)
                self.db.execute("DELETE FROM occurrences WHERE trace=?", (oid,))
                self.db.execute("UPDATE occurrences SET canonical=trace WHERE canonical=?", (oid,))
                if fid:
                    self._index_failure(fid)
            else:
                raise ValueError("kind is 'bounty' or 'trace'")
            self.db.commit()
        return {"removed": kind, "id": oid}

    def lots(self):
        rows = self.db.execute("SELECT lot, COUNT(*), COUNT(DISTINCT producer) FROM traces GROUP BY lot").fetchall()
        return {"epoch": self.epoch, "k": self.k, f"reserve_{self.money}": self.reserve,
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
                credit = self._join_credit({t[0]: {"producer": t[1]} for t in traces})
                for w in winners:
                    self.db.execute("INSERT INTO licences VALUES (?,?,?,?,?)", (lot, w, price, e, json.dumps([t[0] for t in traces])))
                    self._credit(w, -price, f"licence {lot}")
                    per, dust = divmod(price, len(traces))
                    for i, (tid, producer, checker) in enumerate(traces):
                        amt = per + (dust if i == 0 else 0)
                        author = self._checker_author(checker)
                        sale = split_trace_sale(amt, credit[tid]["producer"], author, self.validators)
                        for acct, m in sale.items():
                            self._credit(acct, m, f"sale {tid[:19]}")
                        if amt - sum(sale.values()) > 0:  # v0.8: a path trace with no passing trace: back to the buyer
                            self._credit(w, amt - sum(sale.values()), f"licence {lot}: share with no payee")
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
            self._registry_settle()               # v0.7: failure bounties whose fix became payable
            self._expire_bounties()
            trace_info, learnings = self._tree()
            for lid, consumer, calls in self.db.execute(
                    "SELECT learning, consumer, SUM(calls) FROM usage WHERE epoch=? GROUP BY learning, consumer", (e,)).fetchall():
                L = learnings[lid]
                amount = calls * L["royalty"]["per_call_micros"]
                self._credit(consumer, -amount, f"usage {lid[:19]}")
                paid = split_usage(amount, L, trace_info, self.validators, learnings)
                for acct, m in paid.items():
                    self._credit(acct, m, f"royalty {lid[:19]}")
                if amount - sum(paid.values()) > 0:      # v0.8: a share with no one to pay goes back to the consumer
                    self._credit(consumer, amount - sum(paid.values()), f"usage {lid[:19]}: share with no payee")
            self._bill_fees()
            # debits were collected up front (a payment channel); the root pays out every credit, gross
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
                        f"per_call_{self.money}": L["royalty"].get(f"per_call_{self.money}"), "artifact": L["artifact"],
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
                       "credits_micros": self.test_credits} if self.test_credits
                      else {"asset": "dollars (v0.1 reference node, retired: mainnet settles in bitcoin, SPEC 5)"})
        return {"protocol": "trace-exchange/0.1", "name": "traceX", "api": "/v0", "mcp": "/mcp",
                "taxonomy": TAXONOMY_VERSION, "classifier": self.engine.name, "epoch": self.epoch,
                "settlement": settlement, "privacy": ["skeleton", "open"],
                "fee_per_transaction_nanos": self.tx_fee_nanos,
                "start_here": ["GET /v0/failures", "GET /v0/search", "GET /v0/taxonomy", "GET /v0/learnings",
                               "GET /v0/bounties"],
                "registry": {"failures": "GET /v0/failures?path=&failure=&model=&status=&sort=frequency|growth|bounty|new",
                             "failure": "GET /v0/failures/{TXF-id} (and /history)", "fixes": "POST /v0/fixes",
                             "models": "GET /v0/models/{version}/report"},
                "step_traces": {"file": "POST /v0/failures/{TXF-id}/fragments (steps/0.1: replayed, then composed)",
                                "restart_from": "GET /v0/failures/{TXF-id}/frontier",
                                "verified_paths": "GET /v0/failures/{TXF-id}/joins (and /graphs, /basis)"}}

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
            j = self._join_of(oid)
            if j:                                  # v0.8: a verified path composed from step traces
                return {"id": oid, "type": "trace", "composed": True, "lot": r[1], "failure_id": j["failure_id"],
                        "origin": j["origin"], "fragments": j["fragments"], "producers": j["producers"]}
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
# them could steer payouts. Bounty claims and validator stakes wait for signed messages for the same reason.
ADMIN_ROUTES = {"/v0/epochs/clear", "/v0/epochs/settle", "/v0/checkers", "/v0/learnings", "/v0/admin/remove",
                "/v0/admin/reclassify", "/v0/validators", "/v0/decoys", "/v0/decoys/unseal", "/v0/licences/direct",
                "/v0/admin/btc-price", "/v0/models"}
ADMIN_LEARNING_ACTIONS = {"commits", "reveals"}   # validator messages: operator-relayed until they are signed
ADMIN_ACTIONS = {"claims", "measurements"}       # a bounty poster's measurement, too, until it is signed
GONE = ("bounty coins were removed in v0.6: a bounty is a refundable pledge escrow with no token. Pledge with "
        "POST /v0/bounties/{id}/pledges {backer, msats or sats}")


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
    sats_mode = getattr(ex, "economy", "") == "sats"

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

        def _send_text(self, code, text):
            body = text.encode()
            self.send_response(code)
            self._headers("text/plain; charset=utf-8", len(body), {"Cache-Control": "no-store"})
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
                    fmt = q.get("format", "json")
                    r = ex.search(q.get("q", ""), q.get("path", ""), q.get("failure", ""), q.get("model", ""),
                                  int(q.get("limit", 20)), q.get("sort", ""), int(q.get("offset", 0)),
                                  q.get("facets") in ("1", "true"), q.get("kind", "trace"), fmt)
                    return self._send_text(200, r["text"]) if fmt == "text" else self._send(200, r)
                if u.path == "/v0/failures":
                    return self._send(200, ex.failures(q.get("path", ""), q.get("failure", ""), q.get("model", ""),
                                                       q.get("status", ""), q.get("sort", "frequency"),
                                                       int(q.get("limit", 20)), int(q.get("offset", 0))))
                mg = re.fullmatch(r"/v0/failures/([A-Za-z0-9-]+)/(graphs|frontier|joins|basis)", u.path)
                if mg:                                   # v0.8: step graphs, where to restart, verified paths
                    if mg[2] == "frontier":
                        return self._send(200, ex.frontier(mg[1], q.get("root", ""), int(q.get("k", 5))))
                    if mg[2] == "basis":
                        return self._send(200, ex.tropic_basis(mg[1], max(1, min(int(q.get("size", 2)), 16))))
                    return self._send(200, ex.step_graphs(mg[1]) if mg[2] == "graphs" else ex.joins(mg[1]))
                mf = re.fullmatch(r"/v0/failures/([A-Za-z0-9-]+)(/history)?", u.path)
                if mf:
                    return self._send(200, ex.failure_history(mf[1]) if mf[2] else ex.get_failure(mf[1]))
                if u.path == "/v0/fixes":
                    return self._send(200, ex.fixes(q.get("failure", ""), q.get("status", ""), int(q.get("limit", 50))))
                mx = re.fullmatch(r"/v0/fixes/([A-Za-z0-9-]+)", u.path)
                if mx:
                    return self._send(200, ex.get_fix(mx[1]))
                if u.path == "/v0/models":
                    return self._send(200, ex.models())
                mm = re.fullmatch(r"/v0/models/(.+)/report", u.path)
                if mm:
                    return self._send(200, ex.model_report(unquote(mm[1])))
                if u.path.startswith("/v0/traces/"):
                    return self._send(200, ex.get_trace(unquote(u.path.split("/", 3)[3])))
                if u.path.startswith("/v0/reporters/") and sats_mode:
                    return self._send(200, ex.reporter(u.path.split("/", 3)[3]))
                if u.path == "/v0/bounties":
                    return self._send(200, ex.bounties(q.get("path", ""), q.get("status", "")))
                mv = re.fullmatch(r"/v0/learnings/([^/]+)/verdict", u.path)
                if mv and sats_mode:
                    return self._send(200, ex.verdict(mv[1]))
                if u.path.startswith("/v0/learnings/"):
                    return self._send(200, ex.get_learning(u.path.split("/", 3)[3]))
                if u.path == "/v0/economy" and sats_mode:
                    return self._send(200, ex.economy_stats())
                if u.path in ("/v0/coin", "/v0/quote"):
                    return self._send(410, {"error": "v0.6 has no token: there is no coin, pool or price. Everything is "
                                                     "paid in sats; GET /v0/economy shows the money"})
                if u.path == "/v0/fees" and hasattr(ex, "fees"):
                    return self._send(200, ex.fees())
                if u.path == "/v0/validators" and sats_mode:
                    return self._send(200, ex.validators_list())
                if u.path == "/v0/learnings":
                    extra = {"include_pending": q.get("all") in ("1", "true")} if sats_mode else {}
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
                mh = re.fullmatch(r"/v0/bounties/(\d+)/(backers|holders)", u.path)
                if mh:
                    if mh[2] == "holders":
                        return self._send(410, {"error": GONE})
                    return self._send(200, ex.backers(int(mh[1])))
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
                          "/v0/faucet": lambda b: ex.faucet(b.get("address"), self._client() if public else ""),
                          "/v0/fixes": ex.claim_fix, "/v0/models": ex.register_model}
                if self.path in ("/v0/swap", "/v0/credits"):
                    return self._send(410, {"error": "v0.6 has no token: no swaps and no credits. Every call is paid "
                                                     "in sats (testnet: POST /v0/faucet; mainnet: Lightning, L402)"})
                if sats_mode:
                    routes.update({"/v0/admin/btc-price": lambda b: ex.set_btc_usd(b.get("usd_per_btc")),
                                   "/v0/validators": lambda b: ex.register_validator(b.get("address"),
                                                                                     msats_in(b, "stake", required=True)),
                                   "/v0/decoys": lambda b: ex.register_decoy(b["learning"], b["digest"], b["funder"]),
                                   "/v0/decoys/unseal": lambda b: ex.unseal_decoy(b["learning"], b["gain"], b["salt"]),
                                   "/v0/licences/direct": lambda b: ex.direct_licence(b["lot"], b["buyer"], b["traces"]),
                                   "/v0/reporters": lambda b: ex.post_reporter_bond(b.get("address")),
                                   "/v0/reporters/withdraw": lambda b: ex.withdraw_reporter(b.get("address"))})
                mfx = re.fullmatch(r"/v0/fixes/([A-Za-z0-9-]+)/(commits|reveals)", self.path)
                if mfx:                                  # validator messages about a fix: operator-relayed until signed
                    if not self._admin():
                        return self._send(403, {"error": "validator messages are relayed by the operator until they are signed"})
                    b = self._body()
                    if mfx[2] == "commits":
                        return self._send(200, ex.commit_fix(mfx[1], b["validator"], b["digest"]))
                    return self._send(200, ex.reveal_fix(mfx[1], b["validator"], b["measurement"], b.get("salt", "")))
                mfr = re.fullmatch(r"/v0/failures/([A-Za-z0-9-]+)/fragments", self.path)
                if mfr:                                  # v0.8: a step trace, replayed and composed on arrival
                    return self._send(200, ex.submit_fragment(mfr[1], self._body()))
                mrp = re.fullmatch(r"/v0/failures/([A-Za-z0-9-]+)/repro", self.path)
                if mrp:
                    if not self._admin():
                        return self._send(403, {"error": "validator messages are relayed by the operator until they are signed"})
                    b = self._body()
                    return self._send(200, ex.repro_check(mrp[1], b["validator"], b["results"]))
                ml = re.fullmatch(r"/v0/learnings/([^/]+)/(commits|reveals|challenges)", self.path)
                if ml and sats_mode:
                    lid, act = ml[1], ml[2]
                    admin = act in ADMIN_LEARNING_ACTIONS
                    fn = {"commits": lambda b: ex.commit(lid, b["validator"], b["digest"], b.get("round")),
                          "reveals": lambda b: ex.reveal(lid, b["validator"], b["attestation"], b["salt"], b.get("round")),
                          "challenges": lambda b: ex.challenge(lid, b["challenger"])}[act]
                    if admin and not self._admin():
                        return self._send(403, {"error": "validator messages are relayed by the operator until they are signed"})
                    return self._send(200, fn(self._body()))
                fn, admin = routes.get(self.path), self.path in ADMIN_ROUTES
                m = re.fullmatch(r"/v0/bounties/(\d+)/(claims|pledges|measurements|buy|sell|transfer)", self.path)
                if m:
                    i, act = int(m[1]), m[2]
                    if act in ("buy", "sell", "transfer"):
                        return self._send(410, {"error": GONE})
                    admin = act in ADMIN_ACTIONS
                    fn = {"claims": (lambda b: ex.claim_bounty(i, b["learning"], b.get("attestation"))) if sats_mode
                          else (lambda b: ex.claim_bounty(i, b["learning"])),
                          "pledges": (lambda b: ex.pledge(i, b.get("backer"), msats_in(b, required=True)))
                          if sats_mode else (lambda b: ex.pledge(i, b.get("backer"), b.get("micros") or 0)),
                          "measurements": lambda b: ex.poster_measure(i, b["fix"], b["attestation"])}[act]
                if not fn:
                    return self._send(404, {"error": "not found"})
                if admin and not self._admin():
                    return self._send(403, {"error": "on this public node that call is kept to the operator "
                                                     "(attestations and claims are not signed yet)"})
                self._send(200, fn(self._body()))
            except PaymentRequired as e:                 # L402: a Lightning invoice for what is missing, and a macaroon
                challenge = ex.l402(e.account, e.msats) if hasattr(ex, "l402") else {}
                head = {"WWW-Authenticate": f'L402 macaroon="{challenge["macaroon"]}", invoice="{challenge["invoice"]}"'} \
                    if challenge else {}
                self._send(402, {"error": str(e), "l402": challenge} if challenge else {"error": str(e)}, head)
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
        raise ValueError("v0.6 has no coin: run --economy sats (everything paid directly in sats, SPEC 4e)")
    if economy == "sats":
        from sats import SatsExchange                  # v0.6: node/sats.py
        ex = SatsExchange(db, **kw)
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
                total = next((f"{v} {k.split('_', 1)[1]}" for k, v in s.items() if k.startswith("total_")), "")
                print(f"epoch {s['epoch']} settled: root {s['root'][:18]}… total {total}", flush=True)
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
                    help="dollar node (v0.1, retired): reserve price per licence in a lot auction")
    ap.add_argument("--reserve-msats", type=int, default=int(env("TRACEX_RESERVE_MSATS", 50_000)),
                    help="sats node: reserve price per licence in a lot auction, in millisatoshis (50,000 = 50 sats)")
    ap.add_argument("--btc-usd", type=int, default=int(env("TRACEX_BTC_USD", 0) or 0),
                    help="sats node: dollars per bitcoin, for the approximate dollar figures shown beside sats (and the "
                         "fee re-peg, if on); never used in any amount")
    ap.add_argument("--public", action="store_true", default=env("TRACEX_PUBLIC") == "1",
                    help="rate limits on; settle, clear, checkers, learnings, claims and transfers need TRACEX_ADMIN_TOKEN")
    ap.add_argument("--seed", action="store_true", default=env("TRACEX_SEED") == "1",
                    help="on an empty database, load the example traces, bounties and learnings")
    ap.add_argument("--test-credits", type=int, default=int(env("TRACEX_TEST_CREDITS", 0)),
                    help="testnet: test money each new wallet can take once, in the node's unit: msats on a sats node "
                         "(30000000 = 30,000 test sats), micros on the dollar node (0 = off)")
    ap.add_argument("--economy", choices=("usdc", "sats", "coin"), default=env("TRACEX_ECONOMY", "usdc"),
                    help="sats: v0.6, every payment in sats, split at once to the work it paid for; no token (SPEC "
                         "4e). coin: refused (v0.5's token is gone). "
                         "usdc: the v0.1 dollar node, retired as a mainnet path, kept for the library's tests")
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
    money = ({"reserve_msats": a.reserve_msats, **({"btc_usd": a.btc_usd} if a.btc_usd else {})}
             if a.economy == "sats" else {"reserve_micros": a.reserve_micros})
    ex, srv = serve(a.port, a.db, host=a.host, public=a.public, admin_token=token, k=a.k, economy=a.economy,
                    test_credits=a.test_credits, max_db_bytes=a.max_db_mb * 1024 * 1024, fee_to=a.fee_to, **money)
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
          f"{'testnet' if a.test_credits else 'no test wallets'}, epochs {a.epoch_hours or 'manual'}h)", flush=True)
    srv.serve_forever()
