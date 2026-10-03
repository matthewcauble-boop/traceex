"""The classifier engine: file every trace under a task taxonomy and name *how* the model failed, so traces can be
searched, grouped into lots, and matched to bounties.

Two parts:
  1. Where it belongs: a path down TAXONOMY (e.g. extract > travel > flight). Two interchangeable engines:
       RulesEngine  keyword scoring, standard library only, always available
       JevEngine    TypeSafe Jev hierarchical beam search (one `choice` question per level, keep the best `beam`
                    branches), the engine jevbox uses for document search. Needs TYPESAFE_API_KEY.
     classify_path() uses Jev when a key is present and falls back to rules otherwise.
  2. How it failed: failure modes read exactly from the skeleton, no model needed. Because placeholders are
     consistent across input and outputs, we can tell a swapped field from a made-up value:
       role_swap      the model put a value the verified answer assigns to a *different* field
       wrong_span     the model copied some other value from the input
       type_mismatch  ...and it was the wrong kind of value (a code where a number belongs)
       invented       the model produced a value that is nowhere in the input
       omission       the model left the field empty
       normalised     the right value, in a form the skeleton can't link (dates, money), fixed by formatting
"""
import json
import os
import re
import urllib.request

TAXONOMY_VERSION = "taxonomy/0.1"
# node: (description, keywords, children)
TAXONOMY = {
    "extract": ("Pull structured fields out of a document, email, form or page.", ["extract", "parse", "field", "form", "email"], {
        "travel": ("Trips: flights, hotels, cars, trains, itineraries.", ["flight", "airport", "depart", "arriv", "hotel", "check-in", "itinerary", "boarding", "rail", "car rental", "confirmation"], {
            "flight": ("Flight bookings and boarding passes.", ["flight", "airport", "depart", "arriv", "boarding", "airline", "_code", "seat"], {}),
            "lodging": ("Hotel and rental stays.", ["hotel", "check-in", "check-out", "room", "nights", "stay"], {}),
            "ground": ("Cars, trains and transfers.", ["car rental", "pickup", "train", "rail", "station", "transfer"], {}),
        }),
        "commerce": ("Money documents: receipts, invoices, orders, shipping.", ["invoice", "receipt", "order", "total", "subtotal", "tax", "ship", "tracking", "amount due"], {
            "invoice": ("Bills to pay: vendor, amount due, due date, terms.", ["invoice", "amount due", "due date", "net 30", "terms", "bill to", "remit"], {}),
            "receipt": ("Proof of purchase: items, totals, payment.", ["receipt", "subtotal", "paid", "card ending", "items"], {}),
            "shipping": ("Shipments and deliveries.", ["tracking", "shipped", "carrier", "delivery", "ups", "fedex", "usps"], {}),
        }),
        "schedule": ("Events, appointments, school and work notices.", ["event", "appointment", "meeting", "rsvp", "field trip", "permission", "pickup time", "calendar"], {
            "event": ("Invitations and events with a time and place.", ["event", "rsvp", "party", "invite", "venue"], {}),
            "school": ("School notices: trips, forms, closures.", ["school", "field trip", "permission slip", "teacher", "class", "early release"], {}),
            "appointment": ("Bookings with a provider.", ["appointment", "doctor", "dentist", "booking", "reschedule"], {}),
        }),
        "identity": ("People and organisations: contacts, signatures, IDs.", ["name", "phone", "address", "contact", "signature", "company"], {}),
    }),
    "tool_call": ("Choose a tool and fill its arguments.", ["tool", "function", "call", "arguments", "api", "intent"], {
        "api": ("HTTP / SaaS API calls.", ["endpoint", "http", "request", "api", "webhook"], {}),
        "query": ("Database and search queries.", ["sql", "query", "select", "filter", "where"], {}),
        "device": ("Phone, home and OS actions.", ["alarm", "timer", "call", "text", "lights", "reminder", "set"], {}),
    }),
    "code": ("Write, fix or test code.", ["code", "function", "bug", "test", "compile", "stack trace", "def ", "error"], {
        "repair": ("Fix failing code.", ["fix", "bug", "error", "traceback", "exception", "failing"], {}),
        "generate": ("Write new code.", ["implement", "write", "generate", "create"], {}),
        "test": ("Write or fix tests.", ["test", "assert", "unittest", "pytest"], {}),
    }),
    "reasoning": ("Answers that need calculation or logic.", ["calculate", "how many", "total", "date", "convert", "units", "compare"], {
        "arithmetic": ("Sums, totals, percentages.", ["sum", "total", "percent", "multiply", "add"], {}),
        "temporal": ("Dates, times, durations, time zones.", ["date", "time", "duration", "timezone", "before", "after", "days"], {}),
    }),
    "dialog": ("Conversation with a person: support, sales, tutoring.", ["customer", "reply", "support", "chat", "user said"], {}),
}
SLOT_RE = re.compile(r"^\{([A-Z]+)_\d+\}$")


def nodes(tree=TAXONOMY, prefix=()):
    """Every taxonomy node as (path tuple, description)."""
    for name, (desc, _, kids) in tree.items():
        p = prefix + (name,)
        yield p, desc
        yield from nodes(kids, p)


def _children(path):
    tree = TAXONOMY
    for p in path:
        tree = tree[p][2]
    return tree


def trace_text(trace):
    """What the path engines read: task name, field names and the skeleton (no personal data by construction)."""
    fields = " ".join(sorted(set(trace.get("verified_output", {})) | set(trace.get("fixed_fields", []))))
    return f"task: {trace.get('task', '')}\nfields: {fields.replace('_', ' ')}\n{trace.get('input', '')}"


# --- how it failed ------------------------------------------------------------------------------------------------
def failure_modes(trace):
    """{field: mode} for every fixed field, read exactly from the placeholders."""
    m_out, v_out = trace.get("model_output", {}), trace.get("verified_output", {})
    text = trace.get("input", "")
    fixed_by = trace.get("fixed_by", {})
    owner = {}
    for f, v in v_out.items():
        owner.setdefault(str(v), set()).add(f)
    out = {}
    for f in trace.get("fixed_fields", []):
        said, right = m_out.get(f), v_out.get(f)
        if said in (None, ""):
            out[f] = "omission"
            continue
        said = str(said)
        st, rt = SLOT_RE.match(said), SLOT_RE.match(str(right))
        if st and rt and st[1] != rt[1]:
            out[f] = "type_mismatch"            # the wrong kind of value: the most telling error, checked first
        elif said in owner and f not in owner[said]:
            out[f] = "role_swap"
        elif said in text:
            out[f] = "wrong_span"
        elif str(right) not in text and fixed_by.get(f) == "rule":
            out[f] = "normalised"
        else:
            out[f] = "invented"
    return out


def signature(modes):
    """A short, searchable label for a trace's failure pattern, e.g. 'role_swap:2 wrong_span:1'."""
    counts = {}
    for m in modes.values():
        counts[m] = counts.get(m, 0) + 1
    return " ".join(f"{m}:{n}" for m, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


# --- where it belongs ---------------------------------------------------------------------------------------------
class RulesEngine:
    """Beam search down the taxonomy with keyword scores. Deterministic, offline, good enough to bootstrap."""
    name = "rules"

    def __init__(self, beam=2, min_score=1):
        self.beam, self.min_score = beam, min_score

    @staticmethod
    def _vocab(node):
        """A branch's vocabulary is its own keywords plus every descendant's, so 'extract' hears 'airport'."""
        _, kws, kids = node
        out = set(kws)
        for k in kids.values():
            out |= RulesEngine._vocab(k)
        return out

    @staticmethod
    def _score(text, keywords):
        low = text.lower()
        return sum(min(low.count(k), 3) for k in keywords)   # capped, so one repeated word can't win alone

    def classify(self, text):
        frontier = [((), 1.0)]
        best = ((), 0.0)
        while frontier:
            nxt = []
            for path, p in frontier:
                kids = _children(path)
                if not kids:
                    continue
                scores = {k: self._score(text, self._vocab(v)) for k, v in kids.items()}
                total = sum(scores.values())
                if total < self.min_score:
                    continue
                for k, s in sorted(scores.items(), key=lambda kv: -kv[1])[:self.beam]:
                    if s >= self.min_score:
                        nxt.append((path + (k,), p * s / total))
            for cand in nxt:
                if len(cand[0]) > len(best[0]) or (len(cand[0]) == len(best[0]) and cand[1] > best[1]):
                    best = cand
            frontier = sorted(nxt, key=lambda c: -c[1])[:self.beam]
        return {"path": list(best[0]), "confidence": round(best[1], 3), "engine": self.name}


class JevEngine:
    """TypeSafe Jev hierarchical beam search: one `choice` question per taxonomy level, keep the `beam` most likely
    branches, stop when a level's best probability falls below `stop_below` (the trace stays at the parent)."""
    name = "jev"

    def __init__(self, api_key, beam=2, stop_below=0.35, model=None, base_url=None, timeout=10.0):
        self.key, self.beam, self.stop_below, self.timeout = api_key, beam, stop_below, timeout
        self.model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        self.base = (base_url or os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")).rstrip("/")

    def request(self, text, path):
        kids = _children(path)
        where = " > ".join(path) or "the top level"
        return {"model": self.model,
                "state": {"trace": text[:6000], "context": "A verified fix to an AI model's output, with every personal "
                          "value replaced by a typed placeholder like {NUM_1}. Classify the task the model was doing."},
                "questions": {"category": {"type": "choice",
                                           "instructions": f"Which category under {where} best describes the task?",
                                           "criteria": {k: v[0] for k, v in kids.items()}}}}

    def _ask(self, req):
        http = urllib.request.Request(f"{self.base}/v1/systemone", data=json.dumps(req).encode(), method="POST",
                                      headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                                               "User-Agent": "trace-exchange-classifier/0.1"})
        with urllib.request.urlopen(http, timeout=self.timeout) as r:
            ans = json.loads(r.read())["answers"]["category"]
        return ans.get("probabilities") or {ans["choice"]: float(ans.get("confidence", 1.0))}

    def classify(self, text, ask=None):
        ask = ask or self._ask
        frontier, best = [((), 1.0)], ((), 0.0)
        while frontier:
            nxt = []
            for path, p in frontier:
                if not _children(path):
                    continue
                probs = ask(self.request(text, path))
                for k, q in sorted(probs.items(), key=lambda kv: -kv[1])[:self.beam]:
                    if q >= self.stop_below and k in _children(path):
                        nxt.append((path + (k,), p * q))
            for cand in nxt:
                if len(cand[0]) > len(best[0]) or (len(cand[0]) == len(best[0]) and cand[1] > best[1]):
                    best = cand
            frontier = sorted(nxt, key=lambda c: -c[1])[:self.beam]
        return {"path": list(best[0]), "confidence": round(best[1], 3), "engine": self.name}


def default_engine():
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    return JevEngine(key) if key else RulesEngine()


def classify(trace, engine=None):
    """Everything the exchange indexes about a trace."""
    engine = engine or default_engine()
    try:
        where = engine.classify(trace_text(trace))
    except Exception as e:                      # a hosted engine being down must never block a submission
        where = dict(RulesEngine().classify(trace_text(trace)), fallback=f"{engine.name}: {type(e).__name__}")
    modes = failure_modes(trace)
    return {"taxonomy": TAXONOMY_VERSION, **where, "path_str": "/".join(where["path"]),
            "failure_modes": modes, "signature": signature(modes)}
