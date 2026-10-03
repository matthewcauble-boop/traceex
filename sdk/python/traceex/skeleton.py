"""Skeletons: replace every concrete value with a typed placeholder, consistently across the input and both outputs.

The placeholder -> value mapping stays on the producer's machine. What leaves is the *structure* of the failure:
    "Flight 1123 departs Austin (AUS) 8:05 AM"   ->   "Flight {NUM_1} departs {CITY_1} ({CODE_1}) {TIME_1}"
A trainer refills placeholders with synthetic values to make as many concrete training pairs as it likes.
"""
import re

MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
# Order matters: specific before general. Each entry: (slot type, regex).
DETECTORS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("url", re.compile(r"https?://\S+")),
    ("phone", re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")),
    ("money", re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")),
    ("date", re.compile(rf"\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*,?\s+{MONTHS}\s+\d{{1,2}}(?:,\s*\d{{4}})?|\b{MONTHS}\s+\d{{1,2}}(?:,\s*\d{{4}})?|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b", re.I)),
    ("time", re.compile(r"\b\d{1,2}:\d{2}(?:\s?[AaPp]\.?[Mm]\.?)?")),
    ("code", re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{5,8}\b")),       # booking codes
    ("airport", re.compile(r"\b[A-Z]{3}\b")),
    ("name", re.compile(r"\b[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+\b|\b[A-Z]{2,}/[A-Z]{2,}\b")),  # "Jordan Parker", "PARKER/CASEY"
    ("number", re.compile(r"\b\d[\d,.]*\b")),
]
SLOT_NAMES = {"email": "EMAIL", "url": "URL", "phone": "PHONE", "money": "MONEY", "date": "DATE", "time": "TIME",
              "code": "CODE", "airport": "CODE", "name": "NAME", "number": "NUM", "value": "VAL"}
PLACEHOLDER = re.compile(r"\{[A-Z]+_\d+\}")
# Words that look like names but are part of the document's furniture, not personal data.
KEEP = {"Total", "Flight", "Depart", "Departs", "Arrive", "Arrives", "Return", "Outbound", "Passenger", "Passengers",
        "Confirmation", "Booking", "Subject", "Fare", "Taxes", "Traveler", "Number", "Code", "Reference", "Hi",
        "Hello", "Dear", "Thanks", "Your", "Guest", "Trip", "Amount", "Airlines", "Lines"}


def _kind_of(value: str):
    for kind, rx in DETECTORS:
        if rx.fullmatch(value.strip()):
            return kind
    return "value"


def skeletonize(text: str, *outputs: dict, private_terms=()):
    """Return (skeleton_text, [skeleton_outputs...], slots). Values found in outputs are replaced first, so a field's
    placeholder in the output points at the same placeholder in the input."""
    mapping, slots, counters = {}, {}, {}

    def placeholder(value: str, kind: str):
        key = value.strip()
        if key in mapping:
            return mapping[key]
        name = SLOT_NAMES.get(kind, "VAL")
        counters[name] = counters.get(name, 0) + 1
        ph = f"{{{name}_{counters[name]}}}"
        mapping[key] = ph
        slots[ph[1:-1]] = kind
        return ph

    # 1. values that appear in the outputs (longest first so "1532" doesn't eat "UA 1532")
    values = set()
    for out in outputs:
        for v in (out or {}).values():
            if v not in (None, "") and len(str(v).strip()) >= 2:
                values.add(str(v).strip())
    values.update(t for t in private_terms if t)
    s = text
    for v in sorted(values, key=lambda x: (-len(x), x)):   # deterministic: same fix, same id, on every run
        if v in s:
            s = s.replace(v, placeholder(v, _kind_of(v)))

    # 2. everything else the detectors recognise
    for kind, rx in DETECTORS:
        def sub(m, kind=kind):
            val = m.group(0)
            if PLACEHOLDER.fullmatch(val):
                return val
            if kind == "name":   # keep document furniture ("Departs Austin" -> "Departs {NAME_1}")
                words = val.split()
                lead, tail = 0, len(words)
                while lead < tail and words[lead] in KEEP:
                    lead += 1
                while tail > lead and words[tail - 1] in KEEP:
                    tail -= 1
                if lead == tail:
                    return val
                return " ".join(words[:lead] + [placeholder(" ".join(words[lead:tail]), "name")] + words[tail:])
            return placeholder(val, kind)
        parts = PLACEHOLDER.split(s)
        holes = PLACEHOLDER.findall(s)
        s = "".join(rx.sub(sub, p) + (holes[i] if i < len(holes) else "") for i, p in enumerate(parts))

    # 3. lone capitalised words mid-sentence (single-word names and places)
    parts, holes = PLACEHOLDER.split(s), PLACEHOLDER.findall(s)
    out = []
    for i, p in enumerate(parts):
        last = 0
        for m in _lone_names(p):
            out.append(p[last:m.start()] + placeholder(m.group(0), "name"))
            last = m.end()
        out.append(p[last:] + (holes[i] if i < len(holes) else ""))
    s = "".join(out)

    sk_outputs = []
    for out in outputs:
        sk = {}
        for k, v in (out or {}).items():
            if v in (None, ""):
                sk[k] = v
            else:
                sv = str(v).strip()
                sk[k] = mapping.get(sv) or placeholder(sv, _kind_of(sv))
        sk_outputs.append(sk)
    return s, sk_outputs, slots


SECRETS = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("api_key", re.compile(r"\b(?:sk|pk|rk)[-_](?:live[-_]|test[-_]|ant[-_]|proj[-_])?[A-Za-z0-9_-]{20,}\b")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("password", re.compile(r"(?i)\b(?:password|passwd|secret|api_key|apikey|token)\s*[:=]\s*['\"][^'\"\s]{6,}['\"]")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*")),
]


def find_secrets(text: str):
    """Keys, tokens and passwords: never allowed in any trace, whatever its privacy level."""
    return [(kind, m.group(0)[:12] + "…") for kind, rx in SECRETS for m in rx.finditer(text or "")]


def find_open_risks(text: str):
    """What an `open` trace (code, maths, public text) must not contain: secrets, plus emails and phone numbers.
    Names are allowed (code is full of capitalised identifiers); personal content belongs in a skeleton trace."""
    hits = find_secrets(text)
    for kind, rx in DETECTORS[:3]:            # email, url, phone
        if kind != "url":
            hits += [(kind, m.group(0)) for m in rx.finditer(text or "")]
    return hits


CAP_WORD = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _sentence_start(text, i):
    """True when position i begins a line or a sentence, where a capital letter says nothing about names."""
    j = i - 1
    while j >= 0 and text[j] in " \t":
        j -= 1
    return j < 0 or text[j] in "\n.!?"      # not after ':' '-' '(' etc: "Passenger: Riley", "- Austin, TX"


def _lone_names(text):
    """Single capitalised words mid-sentence ("departs Austin", "call Riley") are treated as names: redacting a
    harmless word costs a little signal, missing a real name leaks it."""
    return [m for m in CAP_WORD.finditer(text) if m.group(0) not in KEEP and not _sentence_start(text, m.start())]


def find_pii(text: str):
    """What a node checks before accepting a trace: any detector hit outside a placeholder is a leak."""
    hits = []
    for part in PLACEHOLDER.split(text):
        for kind, rx in DETECTORS[:6]:              # email, url, phone, money, date, time
            hits += [(kind, m.group(0)) for m in rx.finditer(part)]
        for m in DETECTORS[8][1].finditer(part):    # names
            if not all(w in KEEP for w in m.group(0).split()):
                hits.append(("name", m.group(0)))
        hits += [("name", m.group(0)) for m in _lone_names(part)]
    return hits
