"""Reference exchange node (spec section 7). Standard library only: http.server + sqlite3.

    python node/exchange.py --port 8787 --db exchange.db

One process plays the off-chain half of the protocol: it accepts skeleton traces, groups them into lots, takes sealed
bids, clears each epoch, registers learnings with validator attestations, meters usage, and at settlement computes
every address's earnings and the epoch's Merkle payout root (the one value that goes on-chain).
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk", "python"))
from traceex import canonical, object_id, find_pii  # noqa: E402
from traceex.auction import Bid, clear_shared  # noqa: E402
from traceex.royalty import split_trace_sale, split_usage  # noqa: E402
from traceex.merkle import leaf, build_tree, proof  # noqa: E402
from traceex.classify import classify, default_engine, nodes, TAXONOMY_VERSION  # noqa: E402
from traceex import bountycoin as coin  # noqa: E402

# A bounty's pool, when a learning claims it: the solver is paid most, the traces it was built from still earn.
BOUNTY_SPLIT = {"trainer": 0.70, "traces": 0.20, "checkers": 0.05, "validators": 0.05}

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
"""
FTS = "CREATE VIRTUAL TABLE IF NOT EXISTS trace_fts USING fts5(id UNINDEXED, path, signature, model, task, body)"


class Exchange:
    def __init__(self, path=":memory:", *, k=10, reserve_micros=1_000, validators=("0x" + "5" * 40,),
                 fee_micros=0, engine=None):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        try:
            self.db.execute(FTS)
            self.fts = True
        except sqlite3.OperationalError:          # SQLite built without FTS5: search falls back to LIKE
            self.fts = False
        self.engine = engine or default_engine()
        self.lock = threading.Lock()
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

    # --- API -----------------------------------------------------------------------------------------------------
    def register_checker(self, checker_id, author):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO checkers VALUES (?,?)", (checker_id, author))
            self.db.commit()
        return {"checker": checker_id, "author": author}

    def submit_trace(self, t):
        if t.get("v") != "trace/0.1":
            raise ValueError("unknown trace version")
        if t.get("privacy") == "skeleton":
            leaks = find_pii(t.get("input", ""))
            if leaks:
                raise ValueError(f"rejected: personal data outside placeholders: {leaks[:3]}")
        if not t.get("fixed_fields"):
            raise ValueError("rejected: a trace must record at least one fixed field")
        tid = object_id(t)
        ck = f"{t['checker']['id']}@{t['checker']['version']}"
        lot = f"{t['task']}|{t['base_model']['name']}|{ck}"
        with self.lock:
            dup = self.db.execute("SELECT id FROM traces WHERE id=?", (tid,)).fetchone()
            if dup:
                return {"id": tid, "lot": lot, "duplicate": True}
            self.db.execute("INSERT INTO traces VALUES (?,?,?,?,?,?)",
                            (tid, lot, t["producer"], t["checker"]["id"], canonical(t).decode(), self.epoch))
            c = classify(t, self.engine)
            self.db.execute("INSERT INTO labels VALUES (?,?,?,?,?,?,?,?)",
                            (tid, c["path_str"], c["confidence"], c["engine"], c["signature"], json.dumps(c["failure_modes"]),
                             t["base_model"]["name"], t["task"]))
            if self.fts:
                self.db.execute("INSERT INTO trace_fts VALUES (?,?,?,?,?,?)",
                                (tid, c["path_str"].replace("/", " "), c["signature"], t["base_model"]["name"], t["task"],
                                 t["input"] + " " + " ".join(t["fixed_fields"]).replace("_", " ")))
            self.db.commit()
        return {"id": tid, "lot": lot, "epoch": self.epoch, "classified": c, "bounties": self._matching_bounties(c, t)}

    # --- classifier, search, bounties -----------------------------------------------------------------------------
    def taxonomy(self):
        counts = dict(self.db.execute("SELECT path, COUNT(*) FROM labels GROUP BY path").fetchall())
        out = []
        for path, desc in nodes():
            p = "/".join(path)
            n = sum(v for k, v in counts.items() if k == p or k.startswith(p + "/"))
            out.append({"path": p, "description": desc, "traces": n})
        return {"version": TAXONOMY_VERSION, "engine": self.engine.name, "nodes": out}

    def search(self, q="", path="", failure="", model="", limit=20):
        """Find traces by words, taxonomy branch, failure mode and base model. Open bounties on the same branch come back
        with the results, so a trainer sees both the supply (traces) and the demand (bounties)."""
        sql, args = ["SELECT l.id, l.path, l.signature, l.model, l.task, t.lot, t.body FROM labels l JOIN traces t ON t.id=l.id"], []
        where = []
        if q:
            if self.fts:
                terms = " ".join('"' + w.replace('"', '') + '"' for w in q.split())
                where.append("l.id IN (SELECT id FROM trace_fts WHERE trace_fts MATCH ?)")
                args.append(terms)
            else:
                where.append("(t.body LIKE ? OR l.path LIKE ?)")
                args += [f"%{q}%", f"%{q}%"]
        if path:
            where.append("(l.path = ? OR l.path LIKE ?)")
            args += [path, path + "/%"]
        if failure:
            where.append("l.signature LIKE ?")
            args.append(f"%{failure}:%")
        if model:
            where.append("l.model = ?")
            args.append(model)
        if where:
            sql.append("WHERE " + " AND ".join(where))
        sql.append("ORDER BY l.path, l.id LIMIT ?")
        args.append(int(limit))
        hits = []
        for tid, p, sig, m, task, lot, body in self.db.execute(" ".join(sql), args):
            b = json.loads(body)
            hits.append({"id": tid, "path": p, "signature": sig, "model": m, "task": task, "lot": lot,
                         "fixed_fields": b["fixed_fields"], "snippet": b["input"].splitlines()[0][:140]})
        return {"results": hits, "count": len(hits), "bounties": self.bounties(path=path, status="open")["bounties"]}

    def post_bounty(self, b):
        """Posting is free. A bounty names a taxonomy branch (optionally a failure mode and base model), a hidden eval
        set by hash and the score a solution must reach, and mints a coin on the bonding curve. `seed_micros` lets the
        poster buy the first coins in the same call."""
        if not b.get("eval_set") or not b.get("path"):
            raise ValueError("a bounty needs a path and an eval_set hash")
        epochs = int(b.get("epochs", 4))
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO bounties (poster,title,path,failure,base_model,eval_set,target,deadline,status,epoch)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (b["poster"], b.get("title", ""), b["path"].strip("/"), b.get("failure", ""), b.get("base_model", ""),
                 b["eval_set"], float(b["target"]), self.epoch + epochs, "open", self.epoch))
            bid = cur.lastrowid
            self.db.commit()
        out = {"id": bid, "status": "open", "deadline_epoch": self.epoch + epochs, "price_micros": coin.price(0)}
        seed = int(b.get("seed_micros") or b.get("reward_micros") or 0)
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

    def buy_coins(self, bounty_id, buyer, micros):
        """Spend `micros` on the curve. Every micro goes into the bounty's pool."""
        micros = int(micros)
        if micros <= 0:
            raise ValueError("spend must be positive")
        with self.lock:
            status, pool, supply = self._bounty(bounty_id)
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}; buy its coins from a holder")
            n = coin.coins_for(supply, micros)
            self._set_holding(bounty_id, buyer, self._holding(bounty_id, buyer) + n)
            self.db.execute("UPDATE bounties SET pool=pool+?, supply=supply+? WHERE id=?", (micros, n, bounty_id))
            self._credit(buyer, -micros, f"bounty {bounty_id} coins")
            self.db.commit()
        return {"bounty": bounty_id, "coins": round(n, 6), "avg_price_micros": round(micros / n),
                "next_price_micros": round(coin.price(supply + n)), "pool_micros": pool + micros}

    def sell_coins(self, bounty_id, seller, coins):
        """While a bounty is open, sell coins back to the curve at the current price."""
        with self.lock:
            status, pool, supply = self._bounty(bounty_id)
            if status != "open":
                raise ValueError(f"bounty {bounty_id} is {status}; its coins now earn from the solution")
            have = self._holding(bounty_id, seller)
            coins = min(float(coins), have)
            if coins <= 0:
                raise ValueError("no coins to sell")
            value = min(int(coin.sell_value(supply, coins)), pool)
            self._set_holding(bounty_id, seller, have - coins)
            self.db.execute("UPDATE bounties SET pool=pool-?, supply=supply-? WHERE id=?", (value, coins, bounty_id))
            self._credit(seller, value, f"bounty {bounty_id} sell")
            self.db.commit()
        return {"bounty": bounty_id, "sold": round(coins, 6), "paid_micros": value,
                "next_price_micros": round(coin.price(supply - coins))}

    def transfer_coins(self, bounty_id, sender, to, coins):
        """Move coins between holders, any time. (A live network checks the sender's signature; the contract does.)"""
        with self.lock:
            self._bounty(bounty_id)
            have = self._holding(bounty_id, sender)
            coins = float(coins)
            if coins <= 0 or coins > have + 1e-9:
                raise ValueError(f"{sender} holds {have:.6f} coins")
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
                               "status,winner,learning FROM bounties ORDER BY id").fetchall()
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
            trace_info, learnings = self._tree()
            payout = split_usage(pool, dict(L, royalty=dict(L["royalty"], split=BOUNTY_SPLIT)), trace_info,
                                 self.validators, learnings)
            for acct, m in payout.items():
                self._credit(acct, m, f"bounty {bounty_id}")
            self.db.execute("UPDATE bounties SET status='solved', winner=?, learning=? WHERE id=?",
                            (L["trainer"], learning_id, bounty_id))
            self.db.commit()
        return {"bounty": bounty_id, "status": "solved", "winner": L["trainer"], "pool_micros": pool, "payout": payout,
                "holders_now_earn": f"{coin.HOLDER_CUT:.0%} of every use of {learning_id[:19]}"}

    def _tree(self):
        trace_info = {tid: {"producer": p, "checker_author": self._checker_author(c)}
                      for tid, p, c in self.db.execute("SELECT id, producer, checker FROM traces")}
        learnings = {lid: json.loads(b) for lid, b in self.db.execute("SELECT id, body FROM learnings")}
        return trace_info, learnings

    def _expire_bounties(self):
        """At settlement, a bounty unsolved past its deadline returns its pool to coin holders, pro rata."""
        for bid, pool in self.db.execute(
                "SELECT id, pool FROM bounties WHERE status='open' AND deadline < ?", (self.epoch,)).fetchall():
            holds = dict(self.db.execute("SELECT holder, coins FROM holdings WHERE bounty=?", (bid,)).fetchall())
            for h, m in coin.pro_rata(pool, holds).items():
                self._credit(h, m, f"bounty {bid} refund")
            self.db.execute("UPDATE bounties SET status='expired', pool=0 WHERE id=?", (bid,))

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
        with self.lock:
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
            self.db.commit()
        return {"epoch": e, "cleared": out}

    def _checker_author(self, checker_id):
        r = self.db.execute("SELECT author FROM checkers WHERE id=?", (checker_id,)).fetchone()
        return r[0] if r else self.validators[0]

    def register_learning(self, l):
        a = l.get("attestation") or {}
        if not (a.get("validator") in self.validators and float(a.get("after", 0)) > float(a.get("before", 1))):
            raise ValueError("rejected: needs a known validator's attestation showing after > before")
        for p in l["parents"]:
            known = self.db.execute("SELECT 1 FROM traces WHERE id=? UNION SELECT 1 FROM learnings WHERE id=?",
                                    (p["trace"], p["trace"])).fetchone()
            if not known:
                raise ValueError(f"rejected: unknown parent {p['trace']}")
        lid = object_id(l)
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO learnings VALUES (?,?,?)", (lid, canonical(l).decode(), self.epoch))
            self.db.commit()
        return {"id": lid}

    def usage(self, u):
        with self.lock:
            self.db.execute("INSERT INTO usage VALUES (?,?,?,?)", (u["learning"], u["consumer"], int(u["calls"]), self.epoch))
            self.db.commit()
        return {"metered": int(u["calls"])}

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
            # debits were collected up front (x402 / payment channel); the root pays out every credit, gross
            payouts = {a: m for a, m in self.db.execute(
                "SELECT account, SUM(micros) FROM ledger WHERE epoch=? AND micros > 0 GROUP BY account", (e,))}
            leaves = {a: leaf(e, a, m) for a, m in payouts.items()}
            levels = build_tree(list(leaves.values()))
            root = levels[-1][0].hex()
            claims = {a: {"amount_micros": m, "proof": ["0x" + h.hex() for h in proof(levels, leaves[a])]}
                      for a, m in payouts.items()}
            self.db.execute("INSERT OR REPLACE INTO roots VALUES (?,?,?,?)",
                            (e, "0x" + root, sum(payouts.values()), json.dumps(claims)))
            self._set_meta("epoch", e + 1)
            self.db.commit()
        return {"epoch": e, "root": "0x" + root, "total_micros": sum(payouts.values()), "claims": claims}

    def provenance(self, oid):
        r = self.db.execute("SELECT body FROM learnings WHERE id=?", (oid,)).fetchone()
        if r:
            L = json.loads(r[0])
            return {"id": oid, "type": "learning", "trainer": L["trainer"], "attestation": L["attestation"],
                    "parents": [dict(p, **self.provenance(p["trace"])) for p in L["parents"]]}
        r = self.db.execute("SELECT producer, lot FROM traces WHERE id=?", (oid,)).fetchone()
        if r:
            return {"id": oid, "type": "trace", "producer": r[0], "lot": r[1]}
        raise KeyError(oid)

    def balance(self, account):
        rows = self.db.execute("SELECT epoch, SUM(micros) FROM ledger WHERE account=? GROUP BY epoch", (account,)).fetchall()
        return {"account": account, "by_epoch": {str(e): m for e, m in rows}, "total_micros": sum(m for _, m in rows)}


def make_handler(ex):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, indent=1).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/v0/lots":
                    return self._send(200, ex.lots())
                if u.path == "/v0/taxonomy":
                    return self._send(200, ex.taxonomy())
                if u.path == "/v0/search":
                    return self._send(200, ex.search(q.get("q", ""), q.get("path", ""), q.get("failure", ""),
                                                      q.get("model", ""), int(q.get("limit", 20))))
                if u.path == "/v0/bounties":
                    return self._send(200, ex.bounties(q.get("path", ""), q.get("status", "")))
                mh = re.fullmatch(r"/v0/bounties/(\d+)/holders", u.path)
                if mh:
                    return self._send(200, ex.holders(int(mh[1])))
                if self.path.startswith("/v0/provenance/"):
                    return self._send(200, ex.provenance(self.path.split("/", 3)[3]))
                if self.path.startswith("/v0/balances/"):
                    return self._send(200, ex.balance(self.path.split("/", 3)[3]))
                self._send(404, {"error": "not found"})
            except KeyError as e:
                self._send(404, {"error": f"unknown {e}"})

        def do_POST(self):
            routes = {"/v0/traces": ex.submit_trace, "/v0/bids": ex.bid, "/v0/learnings": ex.register_learning,
                      "/v0/usage": ex.usage, "/v0/epochs/clear": lambda _: ex.clear(),
                      "/v0/epochs/settle": lambda _: ex.settle(),
                      "/v0/checkers": lambda b: ex.register_checker(b["id"], b["author"]),
                      "/v0/bounties": ex.post_bounty}
            fn = routes.get(self.path)
            m = re.fullmatch(r"/v0/bounties/(\d+)/(claims|buy|sell|transfer)", self.path)
            if m:
                i, act = int(m[1]), m[2]
                fn = {"claims": lambda b: ex.claim_bounty(i, b["learning"]),
                      "buy": lambda b: ex.buy_coins(i, b["buyer"], b["micros"]),
                      "sell": lambda b: ex.sell_coins(i, b["seller"], b["coins"]),
                      "transfer": lambda b: ex.transfer_coins(i, b["from"], b["to"], b["coins"])}[act]
            if not fn:
                return self._send(404, {"error": "not found"})
            try:
                self._send(200, fn(self._body()))
            except (ValueError, KeyError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *a):
            pass
    return H


def serve(port=8787, db="exchange.db", **kw):
    ex = Exchange(db, **kw)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(ex))
    return ex, srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--db", default="exchange.db")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    _, srv = serve(a.port, a.db, k=a.k)
    print(f"trace exchange node on http://127.0.0.1:{a.port}  (db {a.db})")
    srv.serve_forever()
