"""The node's search index: Leviathan-style full-text search over traces and failures, inside the node's own database.

Adapted from Leviathan 0.1.0 (https://github.com/elstongun/leviathan, commit eb045bb), Copyright 2026 The Leviathan
Authors, licensed under the Apache License, Version 2.0 (see NOTICE). Re-implemented in Python for traceX and modified;
no Leviathan source is copied verbatim. What it borrows from Leviathan's design:
  * one SQLite FTS5 index (porter stemming over unicode61) with weighted columns, ranked by BM25 (title 2x);
  * the branch (Leviathan's "group") and every filter value indexed as one synthetic token each in a `tags` column, so
    scoping and filtering are posting-list intersections inside the match, not post-filters;
  * branch resolution in tiers, exact > case-insensitive > name > contains > fuzzy (Sorensen-Dice >= 0.6), that lists
    the candidates when more than one matches instead of guessing;
  * a labelled fallback: when nothing matches inside the branch, cards from elsewhere come back marked OTHER;
  * stopwords dropped and every word re-quoted, so query syntax typed by a person or an agent is inert;
  * compact cited cards ("shown N of M", the best matching sentence, "next: get <id>").
What traceX changes ([traceX] below): records live in the node's own database and are written in the same transaction
as the trace or failure they index; two record kinds (traces and failures) share the index; the taxonomy is a tree, so
a branch token is indexed for every ancestor and a branch scopes its subtree, falling back to its nearest ancestor
first; decimal synthetic tokens (the porter stemmer can rewrite hex tokens ending in "ed"); prompt boilerplate that
recurs across traces is never chosen as a snippet; a substring fallback when SQLite has no FTS5.
"""
import json
import re
import sqlite3
from collections import defaultdict

MAX_LIMIT = 50
FUZZY_CUTOFF = 0.6
MATCH_CAP = 50_000
COLUMNS = ("title", "body", "code", "names", "labels", "tags")
WEIGHTS = {"title": 2.0, "body": 1.0, "code": 0.3, "names": 0.5, "labels": 1.0, "tags": 0.0}
BM25 = "bm25(lx_fts, " + ", ".join(str(WEIGHTS[c]) for c in COLUMNS) + ")"
STOPWORDS = frozenset((
    "a an and are as at be been but by did do does for from had has have how i if in into is it its last me my of on "
    "or our so that the their then there this time to was we were what when where which who why will with you").split())
TOKEN = re.compile(r"[^\W_][\w.\-]*")            # starts with a letter or digit; then letters, digits, _ . -

SCHEMA = """
CREATE TABLE IF NOT EXISTS lx_records (rowid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, kind TEXT, grp TEXT,
                                       date TEXT, facets TEXT, doc TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS lx_records_kind_date ON lx_records (kind, date);
CREATE INDEX IF NOT EXISTS lx_records_grp ON lx_records (grp);
CREATE TABLE IF NOT EXISTS lx_branches (key TEXT PRIMARY KEY, key_lower TEXT, key_flat TEXT, name TEXT,
                                        name_normalized TEXT);
CREATE TABLE IF NOT EXISTS lx_lines (line TEXT PRIMARY KEY, n INT NOT NULL);
"""
FTS = ("CREATE VIRTUAL TABLE IF NOT EXISTS lx_fts USING fts5(" + ", ".join(COLUMNS) +
       ", tokenize = 'porter unicode61')")


# --- text helpers --------------------------------------------------------------------------------------------------
def words_in(text):
    for m in TOKEN.finditer(text or ""):
        w = m.group(0).rstrip("._-").lower()
        if w:
            yield w


def query_words(text):
    return [w for w in words_in(text) if w not in STOPWORDS]


def normalize_name(value):
    return " ".join((value or "").split()).lower()


def fnv1a(*parts):
    h = 0xcbf29ce484222325
    for i, part in enumerate(parts):
        if i:
            h = ((h ^ 0x1f) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        for b in part.encode():
            h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def group_token(key):                     # [traceX] decimal digits: no stemmer rule can touch them
    return "g%020d" % fnv1a(key)


def facet_token(field, value):
    return "f%020d" % fnv1a(field, str(value).strip().lower())


def truncate(value, n):
    value = (value or "").strip()
    if len(value) <= n:
        return value
    return value[:max(n - 1, 0)].rstrip() + "…"


def one_line(s):
    return " ".join(str(s).split())


def sorensen_dice(a, b):
    """strsim::sorensen_dice: whitespace ignored, multiset bigram overlap."""
    a, b = "".join((a or "").split()), "".join((b or "").split())
    if a == b:
        return 1.0
    if len(a) < 2 or len(b) < 2:
        return 0.0
    grams = defaultdict(int)
    for x, y in zip(a, a[1:]):
        grams[(x, y)] += 1
    inter = 0
    for x, y in zip(b, b[1:]):
        if grams[(x, y)] > 0:
            grams[(x, y)] -= 1
            inter += 1
    return 2 * inter / (len(a) + len(b) - 2)


class Query:
    """Words (ranked; any may match), "quoted phrases" and -exclusions, all re-quoted for FTS5 so nothing a caller
    types is ever read as query syntax."""

    def __init__(self, text):
        self.words, self.phrases, self.excluded = [], [], []
        seen, rest = set(), text or ""
        while True:
            rest = rest.lstrip()
            if not rest:
                break
            negate = rest.startswith("-") and len(rest) > 1
            body = rest[1:] if negate else rest
            if body.startswith('"'):
                inner = body[1:]
                end = inner.find('"')
                end = len(inner) if end < 0 else end
                chunk, phrase, rest = inner[:end], True, inner[end + 1:]
            else:
                m = re.search(r"\s", body)
                end = m.start() if m else len(body)
                chunk, phrase, rest = body[:end], False, body[end:]
            toks = list(words_in(chunk)) if phrase else query_words(chunk)
            if not toks:
                continue
            if negate:
                self.excluded.append(" ".join(toks))
            elif phrase and len(toks) > 1:
                self.phrases.append(" ".join(toks))
            else:
                for t in toks:
                    if t not in seen:
                        seen.add(t)
                        self.words.append(t)

    def empty(self):
        return not self.words and not self.phrases

    def fts(self):
        terms = ['"%s"' % t.replace('"', "") for t in (self.words + self.phrases)[:64]]
        return " OR ".join(terms) if terms else None

    def not_fts(self):
        terms = ['"%s"' % t.replace('"', "") for t in self.excluded[:32]]
        return " OR ".join(terms) if terms else None

    def highlight(self):
        return self.words + [w for p in self.phrases for w in p.split()]


def _stem(word):
    n = len(word)
    return word if n <= 4 else word[:max(n - 2, 4)]


def _sentences(s):
    out, start = [], 0
    for i, c in enumerate(s):
        if c in ".!?;" and i + 1 < len(s) and s[i + 1].isspace():
            out.append(s[start:i + 1].strip())
            start = i + 1
    out.append(s[start:].strip())
    return [p for p in out if p]


def best_snippet(texts, highlight, shown, cap=160, boilerplate=frozenset()):
    """The sentence-sized segment that matches the most distinct query words. [traceX] Lines are segments too (a
    skeleton's asserts, a traceback), and lines that recur across many traces (prompt boilerplate) are never chosen."""
    stems = [_stem(w) for w in highlight]
    best = None
    for text in texts:
        for line in str(text).splitlines():
            flat = one_line(line)
            if not flat or flat in boilerplate or any(flat in s for s in shown if s):
                continue
            for seg in _sentences(flat):
                if seg in boilerplate or any(seg in s for s in shown if s):
                    continue
                hits, first = set(), None
                for m in re.finditer(r"[^\W_]+", seg.lower()):
                    for k, st in enumerate(stems):
                        if m.group(0).startswith(st):
                            hits.add(k)
                            first = m.start() if first is None else first
                            break
                if hits and (best is None or len(hits) > best[0]):
                    best = (len(hits), seg, first or 0)
    if not best:
        return None
    _, seg, first = best
    if len(seg) <= cap:
        return seg
    lead = max(first - 40, 0)
    cut = truncate(seg[lead:], cap)
    return ("…" + cut) if lead else cut


def record(rid, kind, group, group_name, date, cols, facets, doc, lines=()):
    """What one trace or failure puts in the index. cols: the six FTS columns' text (tags are added here); facets:
    [(field, value)], each indexed as one synthetic token; doc: the card, as JSON; lines: text lines that count toward
    the boilerplate statistics."""
    seen, fs = set(), []
    for f, v in facets:
        v = str(v).strip()[:200]
        if v and (f, v.lower()) not in seen:
            seen.add((f, v.lower()))
            fs.append((f, v))
    gtags = []
    if group:                                            # [traceX] a token per ancestor: a branch scopes its subtree
        parts = group.split("/")
        gtags = [group_token("/".join(parts[:i])) for i in range(1, len(parts) + 1)]
    cols = dict(cols, tags=" ".join(gtags + [facet_token(f, v) for f, v in fs]))
    return {"id": rid, "kind": kind, "group": group or None, "group_name": group_name, "date": date or "",
            "facets": fs, "cols": {c: cols.get(c) or "" for c in COLUMNS}, "doc": doc, "lines": list(lines)}


class SearchIndex:
    """Records live in four tables of the node's own database (lx_records, lx_fts, lx_branches, lx_lines). Every
    write goes through the node's connection under its lock and is committed by the caller, together with the write
    it indexes."""

    def __init__(self, db):
        self.db = db
        db.executescript(SCHEMA)
        try:
            db.execute(FTS)
            self.fts = True
        except sqlite3.OperationalError:                 # SQLite built without FTS5: substring search instead
            self.fts = False

    # ----------------------------------------------------------------------------------------------- writing
    def set_branches(self, branches):
        """branches: [(key, description)] for every taxonomy node, with traces or not."""
        for key, desc in branches:
            self.db.execute("INSERT OR REPLACE INTO lx_branches VALUES (?,?,?,?,?)",
                            (key, key.lower(), re.sub(r"[/_\s]", "", key.lower()), desc, normalize_name(desc)))

    def upsert(self, rec):
        old = self.db.execute("SELECT rowid FROM lx_records WHERE id=?", (rec["id"],)).fetchone()
        row = (rec["kind"], rec["group"], rec["date"], json.dumps(rec["facets"]), json.dumps(rec["doc"]))
        if old:
            rowid = old[0]
            self.db.execute("UPDATE lx_records SET kind=?, grp=?, date=?, facets=?, doc=? WHERE rowid=?", (*row, rowid))
            if self.fts:
                self.db.execute("DELETE FROM lx_fts WHERE rowid=?", (rowid,))
        else:
            rowid = self.db.execute("INSERT INTO lx_records (id, kind, grp, date, facets, doc) VALUES (?,?,?,?,?,?)",
                                    (rec["id"], *row)).lastrowid
            self._lines(rec["lines"], +1)
        if rec["group"] and not self.db.execute("SELECT 1 FROM lx_branches WHERE key=?", (rec["group"],)).fetchone():
            self.set_branches([(rec["group"], rec["group_name"] or "")])          # filed outside the taxonomy
        if self.fts:
            self.db.execute(f"INSERT INTO lx_fts (rowid, {', '.join(COLUMNS)}) VALUES (?,?,?,?,?,?,?)",
                            (rowid, *(rec["cols"][c] for c in COLUMNS)))
        return rowid

    def delete(self, rid, lines=()):
        r = self.db.execute("SELECT rowid FROM lx_records WHERE id=?", (rid,)).fetchone()
        if r:
            self.db.execute("DELETE FROM lx_records WHERE rowid=?", (r[0],))
            if self.fts:
                self.db.execute("DELETE FROM lx_fts WHERE rowid=?", (r[0],))
            self._lines(lines, -1)

    def _lines(self, lines, d):
        for ln in {one_line(x) for x in lines if x and x.strip()}:
            if len(ln) > 400:
                continue
            self.db.execute("INSERT INTO lx_lines VALUES (?, ?) ON CONFLICT (line) DO UPDATE SET n = n + ?",
                            (ln, max(d, 0), d))

    def boilerplate(self):
        """Lines repeated in at least 20% of traces (and at least 5): prompt templates, never a snippet."""
        n = self.count("trace")
        return frozenset(ln for (ln,) in self.db.execute("SELECT line FROM lx_lines WHERE n >= ?",
                                                          (max(5, int(0.2 * n)),)).fetchall())

    def count(self, kind=None):
        sql, args = "SELECT COUNT(*) FROM lx_records", ()
        if kind:
            sql, args = sql + " WHERE kind=?", (kind,)
        return self.db.execute(sql, args).fetchone()[0]

    def doc(self, rid):
        r = self.db.execute("SELECT doc FROM lx_records WHERE id=?", (rid,)).fetchone()
        return json.loads(r[0]) if r else None

    # ----------------------------------------------------------------------------------------------- branches
    def resolve(self, query, limit=10, kind="trace"):
        """A branch key or name -> candidates, best tier first: exact, case-insensitive, description, key contains,
        description contains, fuzzy. [traceX] Every taxonomy node is a candidate, traces or not; when every candidate
        of a tier lies under one of them, that node is the answer (its subtree is in scope). More than one other
        candidate: the caller asks which one. It never guesses."""
        query = (query or "").strip().strip("/")
        if not query:
            return []
        lower, norm = query.lower(), normalize_name(query)
        flat = re.sub(r"[/_\s]", "", lower)
        counts = defaultdict(int)
        for grp, n in self.db.execute("SELECT grp, COUNT(*) FROM lx_records WHERE kind=? AND grp IS NOT NULL GROUP BY grp",
                                      (kind,)).fetchall():
            parts = grp.split("/")
            for i in range(1, len(parts) + 1):
                counts["/".join(parts[:i])] += n

        def rows(clause, *args):
            got = [{"key": k, "name": n, "records": counts.get(k, 0)} for k, n in self.db.execute(
                f"SELECT key, name FROM lx_branches WHERE {clause}", args).fetchall()]
            return sorted(got, key=lambda g: (-g["records"], g["key"]))

        def settle(found, how):
            for g in found:
                g["match_type"] = how
            top = [g for g in found if all(o["key"] == g["key"] or o["key"].startswith(g["key"] + "/") for o in found)]
            if len(found) > 1 and top:
                top[0]["match_type"] = how + "+subtree"
                return top[:1]
            return found[:limit]
        esc = norm.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        tiers = (("exact", lambda: rows("key = ?", query)),
                 ("case_insensitive", lambda: rows("key_lower = ?", lower)),
                 ("name", lambda: rows("name_normalized = ?", norm)),
                 ("contains", lambda: rows("key_lower LIKE '%' || ? || '%' ESCAPE '\\'", esc)),
                 ("name_contains", lambda: rows("name_normalized LIKE '%' || ? || '%' ESCAPE '\\'", esc)))
        for how, found in tiers:
            got = found()
            if got:
                return settle(got, how)
        scored = []
        for k, n, kl, kf, nn in self.db.execute(
                "SELECT key, name, key_lower, key_flat, name_normalized FROM lx_branches").fetchall():
            s = max(sorensen_dice(norm, kl), sorensen_dice(flat, kf), sorensen_dice(norm, nn or ""))
            if s >= FUZZY_CUTOFF:
                scored.append({"key": k, "name": n, "records": counts.get(k, 0), "similarity": round(s, 2)})
        if not scored:
            return []
        best = max(g["similarity"] for g in scored)
        scored = sorted((g for g in scored if g["similarity"] >= best - 0.05),
                        key=lambda g: (-g["similarity"], -g["records"]))
        return settle(scored, "fuzzy")

    # ----------------------------------------------------------------------------------------------- searching
    @staticmethod
    def _filters(where):
        """where: {name: [(field, value), ...]}: alternatives inside a name are OR'ed, names are AND'ed."""
        out = []
        for alts in (where or {}).values():
            alts = [(f, v) for f, v in alts if str(v).strip()]
            if alts:
                out.append("(" + " OR ".join(f'tags : "{facet_token(f, v)}"' for f, v in alts) + ")")
        return out

    @staticmethod
    def _expr(pos, neg, gtok, fclauses, scope):
        parts = []
        if gtok and scope == "group":
            parts.append(f'tags : "{gtok}"')
        parts += fclauses
        if pos:
            # inside a branch the branch's own name matches every record, so `names` is left out of scoped matching
            parts.append(f"{{title body code labels}} : ({pos})" if (scope == "group" and gtok) else f"({pos})")
        expr = " AND ".join(parts) if parts else None
        if gtok and scope == "others":
            expr = f'({expr}) NOT tags : "{gtok}"'
        if neg and expr:
            expr = f"({expr}) NOT ({neg})"
        return expr

    def _run(self, expr, ranked):
        """[(rowid, bm25, date)] for every record matching `expr` (capped)."""
        if not expr:
            return []
        rank = BM25 if ranked else "0"
        try:
            return self.db.execute(f"SELECT r.rowid, {rank}, r.date FROM lx_fts JOIN lx_records r ON r.rowid = lx_fts.rowid "
                                   f"WHERE lx_fts MATCH ? LIMIT ?", (expr, MATCH_CAP)).fetchall()
        except sqlite3.OperationalError:                 # never expected: every caller word is quoted
            return []

    def matches(self, q="", branch=None, where=None):
        """Every record matching the words (any of them), the branch's subtree and the filters: [(rowid, bm25, date)]."""
        query = Query(q)
        gtok = group_token(branch) if branch else None
        if not self.fts:
            return self._substring(query, branch, where)
        expr = self._expr(query.fts(), query.not_fts(), gtok, self._filters(where), "group")
        if not expr:                                     # nothing to match on: every record
            return [(r, 0.0, d) for r, d in self.db.execute("SELECT rowid, date FROM lx_records LIMIT ?",
                                                             (MATCH_CAP,)).fetchall()]
        return self._run(expr, not query.empty())

    def _substring(self, query, branch, where):
        """No FTS5: a plain substring match over the stored cards, filters applied in Python. Correct, not fast."""
        want = [w for w in query.words + query.phrases]
        out = []
        for rowid, grp, date, facets, doc in self.db.execute(
                "SELECT rowid, grp, date, facets, doc FROM lx_records LIMIT ?", (MATCH_CAP,)).fetchall():
            if branch and not (grp == branch or (grp or "").startswith(branch + "/")):
                continue
            fs = {(f, str(v).lower()) for f, v in json.loads(facets)}
            if not all(any((f, str(v).lower()) in fs for f, v in alts) for alts in (where or {}).values() if alts):
                continue
            low = doc.lower()
            hits = sum(w in low for w in want)
            if want and not hits:
                continue
            out.append((rowid, -float(hits), date))
        return out

    def search(self, q="", branch=None, where=None, sort="relevance", limit=5, offset=0, fallback=True, everything=False):
        """Ranked records. Returns {status, branch, candidates, total, rows, other, fallback_from, notes, query};
        rows and other are [(rowid, id, relevance)]. sort: relevance (BM25 x weights, newest on ties) or newest.
        everything=True returns every match in order (for callers that re-sort, e.g. by bounty)."""
        limit, offset = max(1, min(int(limit), MAX_LIMIT if not everything else MATCH_CAP)), max(0, int(offset))
        out = {"status": "ok", "query": q or "", "branch": None, "candidates": [], "total": 0, "rows": [], "other": [],
               "fallback_from": None, "notes": []}
        if branch:
            found = self.resolve(branch, 10)
            if len(found) > 1:
                out.update(status="ambiguous_branch", candidates=found)
                out["notes"].append(f"{len(found)} branches match {branch!r}; ask which one, then retry with its exact key")
                return out
            if not found:
                out["status"] = "unknown_branch"
                out["notes"].append(f"no branch matches {branch!r}; try part of its name, or search without one")
                return out
            out["branch"] = found[0]
        query = Query(q)
        if query.empty() and (q or "").strip() and not query.excluded:
            out["notes"].append("the query has no searchable words; listing newest instead")
        key = out["branch"]["key"] if out["branch"] else None
        hits = self.matches(q, key, where)
        out["total"] = len(hits)
        ranked = not query.empty() and sort != "newest"
        order = (lambda h: (h[1], _neg(h[2]), -h[0])) if ranked else (lambda h: (_neg(h[2]), -h[0]))
        hits.sort(key=order)
        page = hits if everything else hits[offset:offset + limit]
        out["rows"] = self._ids(page, ranked)
        if key and fallback and not hits and offset == 0 and not query.empty() and self.fts:
            gtok = group_token(key)
            pos, neg, fcl = query.fts(), query.not_fts(), self._filters(where)
            parts = key.split("/")
            for i in range(len(parts) - 1, 0, -1):          # [traceX] nearest ancestor first, then everything else
                anc = "/".join(parts[:i])
                expr = f'({self._expr(pos, neg, group_token(anc), fcl, "group")}) NOT tags : "{gtok}"'
                other = sorted(self._run(expr, True), key=order)[:limit]
                if other:
                    out["other"], out["fallback_from"] = self._ids(other, True), anc
                    break
            if not out["other"]:
                other = sorted(self._run(self._expr(pos, neg, gtok, fcl, "others"), True), key=order)[:limit]
                out["other"] = self._ids(other, True)
            if out["other"]:
                where_from = f"the rest of {out['fallback_from']}" if out["fallback_from"] else "OTHER branches"
                out["notes"].append(f"nothing matched in {key}; the cards marked OTHER come from {where_from}: say so, "
                                    "and keep each card's branch visible")
        return out

    def _ids(self, hits, ranked):
        if not hits:
            return []
        ids = dict(self.db.execute(f"SELECT rowid, id FROM lx_records WHERE rowid IN ({','.join('?' * len(hits))})",
                                   [h[0] for h in hits]).fetchall())
        return [(h[0], ids[h[0]], round(-h[1], 2) if ranked else None) for h in hits if h[0] in ids]

    def docs(self, rowids):
        if not rowids:
            return {}
        return {r: json.loads(d) for r, d in self.db.execute(
            f"SELECT rowid, doc FROM lx_records WHERE rowid IN ({','.join('?' * len(rowids))})", list(rowids)).fetchall()}


def _neg(date):
    """Sort key that puts later ISO dates first."""
    return tuple(-ord(c) for c in (date or ""))


# --- compact text, the default an agent reads ------------------------------------------------------------------------
def render_card(n, c, show_branch=True, hoisted=()):
    head = f"[{n}] {c.get('short_id') or c['id']}"
    if c.get("other_branch"):
        head += " · OTHER BRANCH"
    if (show_branch or c.get("other_branch")) and c.get("path"):
        head += f" · {c['path']}"
    if c.get("date"):
        head += f" · {c['date']}"
    if c.get("relevance") is not None:
        head += f" · rel {c['relevance']:.{2 if abs(c['relevance']) < 1 else 1}f}"
    lines = [head]
    if c.get("title"):
        lines.append(f"  {c['title']}")
    own = {k: v for k, v in (c.get("fields") or {}).items() if k not in hoisted and v not in (None, "")}
    short = [(k, v) for k, v in own.items() if len(str(v)) <= 40]
    long_ = [(k, v) for k, v in own.items() if len(str(v)) > 40]
    if short:
        lines.append("  " + " · ".join(f"{k}: {v}" for k, v in short))
    lines += [f"  {k}: {v}" for k, v in long_]
    if c.get("match"):
        lines.append(f"  match: {c['match']}")
    return "\n".join(lines) + "\n"


def render(o):
    """A search result as compact cited cards: what an agent reads by default."""
    out = "tracex search"
    if o.get("status", "ok") != "ok":
        out += "".join(f" · {n}" for n in o.get("notes", [])) + "\n"
        for g in o.get("candidates", []):
            out += f"  {g['key']} · {g['records']} traces · {g['match_type']}\n"
        return out
    if o.get("branch"):
        out += f" · branch {o['branch']['key']} ({o['branch']['records']} traces)"
    if (o.get("query") or "").strip():
        out += f" · query {json.dumps(o['query'].strip())}"
    for k in ("failure", "model", "kind"):
        if o.get("filters", {}).get(k):
            out += f" · {k}={o['filters'][k]}"
    cards = o.get("cards", [])
    first = o["offset"] + 1 if cards else 0
    last = o["offset"] + len(cards)
    shown = f"{first}-{last}" if o["offset"] else str(len(cards))
    out += f" · shown {shown} of {o['total']:,} · {o.get('indexed', 0):,} records indexed\n"
    every = cards + o.get("other_cards", [])
    hoisted = {}                                     # [traceX] fields every shown card shares, said once
    if len(every) > 1:
        for k, v in (every[0].get("fields") or {}).items():
            if v not in (None, "") and all((c.get("fields") or {}).get(k) == v for c in every):
                hoisted[k] = v
    if hoisted:
        out += "all shown: " + " · ".join(f"{k}: {v}" for k, v in hoisted.items()) + "\n"
    for i, c in enumerate(cards):
        out += render_card(o["offset"] + i + 1, c, not o.get("branch"), hoisted)
    if not cards:
        out += "no matching records (none found, not none exist: try fewer or different words)\n"
    for i, c in enumerate(o.get("other_cards", [])):
        out += render_card(i + 1, c, True, hoisted)
    out += "".join(f"note: {n}\n" for n in o.get("notes", []))
    if cards or o.get("other_cards"):
        more = f" · offset {last} for more" if o["total"] > last else ""
        out += f"next: GET /v0/traces/<id> or /v0/failures/<id> for the full record{more}\n"
    return out
