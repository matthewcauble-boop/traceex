"""Task 'extract.flight': pull a flight booking out of a confirmation email, and check every field against the email.

The checker is pure rules, grounded in the email's own text, so it can say *which* field is wrong without knowing the
right answer. That property is what turns a model's mistakes into verified fixes (traces).
"""
import re

TASK, CHECKER = "extract.flight", "flight-rules@1"
S = lambda d: {"type": "string", "description": d}
FIELDS = {
    "confirmation_number": S("booking code, e.g. QX7PLM"),
    "outbound_flight_number": S("number of the first flight, digits only, e.g. 1123"),
    "outbound_date": S("date of the first flight, YYYY-MM-DD"),
    "outbound_departure_time": S("clock time the first flight departs, e.g. 8:05 AM"),
    "origin_airport_code": S("3-letter code the first flight departs from"),
    "destination_airport_code": S("3-letter code the first flight arrives at"),
    "return_flight_number": S("number of the return flight, digits only"),
    "return_date": S("date of the return flight, YYYY-MM-DD"),
    "return_departure_time": S("clock time the return flight departs"),
    "total_paid": {"type": "number", "description": "total price in dollars"},
}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
CODE = re.compile(r"\b[A-Z]{3}\b")
DEPART_TIME = re.compile(r"depart\w*[^0-9]{0,40}?(\d{1,2}:\d{2}\s*[AaPp]\.?[Mm]?)", re.I)


def schema(names):
    return {"name": "record_flight", "description": "Record a flight booking from a confirmation email.",
            "parameters": {"type": "object", "properties": {k: FIELDS[k] for k in names}, "required": list(names)}}


def norm_time(t):
    v = str(t or "")
    m = re.search(r"T(\d{2}:\d{2})", v)
    if m:
        v = m[1]
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]?\.?\s*$", v)
    if m:
        return f"{int(m[1])}:{m[2]} {m[3].upper()}M"
    m = re.match(r"^\s*(\d{1,2}):(\d{2})(:\d{2})?\s*$", v)
    if m:
        h = int(m[1])
        return f"{(h + 11) % 12 + 1}:{m[2]} {'AM' if h < 12 else 'PM'}"
    return None


def clean(r):
    """Format fixes are not model errors: normalise before judging."""
    r = dict(r or {})
    for f in ("outbound_flight_number", "return_flight_number"):
        m = re.fullmatch(r"[A-Za-z]{0,3}\s*#?\s*(\d{1,4})", str(r.get(f) or "").strip())
        if m:
            r[f] = m[1]
    for f in ("outbound_departure_time", "return_departure_time"):
        n = norm_time(r.get(f))
        if n:
            r[f] = n
    for f in ("origin_airport_code", "destination_airport_code"):
        if r.get(f):
            r[f] = str(r[f]).strip().upper()
    return r


def lines_for_date(email, iso):
    try:
        _, mo, d = map(int, str(iso).split("-"))
        month = MONTHS[mo - 1]
    except Exception:
        return []
    pats = [rf"\b{month}\s+0?{d}\b", rf"\b{month[:3]}\.?\s+0?{d}\b", rf"\b0?{mo}/0?{d}\b"]
    return [ln for ln in email.splitlines() if any(re.search(p, ln, re.I) for p in pats)]


def leg_line(email, iso):
    ls = lines_for_date(email, iso)
    return ls[0] if ls else ""


def code_for(line, word):
    """Airport code belonging to 'depart'/'arriv': the first code after the word unless a time or the other word sits
    in between ("Departs Austin (AUS)", "arrives AUS"), else the nearest code before it ("ATL departs")."""
    m = re.search(word, line, re.I)
    if not m:
        return None
    after = CODE.search(line, m.end())
    if after and not re.search(r"\d{1,2}:\d{2}|depart|arriv", line[m.end():after.start()], re.I):
        return after[0]
    before = CODE.findall(line[:m.start()])
    return before[-1] if before else None


def context_for(email, field, r):
    """The narrowest text that answers a field: that leg's line, else the whole email."""
    leg = "outbound" if field.startswith("outbound") or field.endswith("airport_code") else \
          "return" if field.startswith("return") else None
    if leg and field != f"{leg}_date" and r.get(f"{leg}_date"):
        line = leg_line(email, r[f"{leg}_date"])
        if line:
            return line
    return email


def evidence(email, r):
    ev, out = {}, leg_line(email, r.get("outbound_date"))
    ev["origin_airport_code"], ev["destination_airport_code"] = code_for(out, "depart"), code_for(out, "arriv")
    for leg in ("outbound", "return"):
        line = leg_line(email, r.get(f"{leg}_date"))
        if not line:
            continue
        nums = re.findall(r"\b(?:flight|flt|[A-Z]{2})\s*#?\s*(\d{1,4})\b", line, re.I)
        if len(set(nums)) == 1:
            ev[f"{leg}_flight_number"] = nums[0]
        m = DEPART_TIME.search(line)
        if m:
            ev[f"{leg}_departure_time"] = norm_time(m[1])
    return {k: v for k, v in ev.items() if v}


def check(email, r):
    bad = {}

    def need(f):
        if r.get(f) in (None, ""):
            bad[f] = "missing"
            return False
        return True
    if need("confirmation_number") and (not re.fullmatch(r"[A-Z0-9]{5,8}", str(r["confirmation_number"]))
                                        or str(r["confirmation_number"]) not in email):
        bad["confirmation_number"] = "not a 5-8 character code in the email"
    out = leg_line(email, r.get("outbound_date"))
    for f, word in (("origin_airport_code", "depart"), ("destination_airport_code", "arriv")):
        if not need(f):
            continue
        want = code_for(out, word)
        if want is None:
            bad[f] = f"can't find the '{word}' airport on the outbound line"
        elif r[f] != want:
            bad[f] = f"the '{word}' airport on the outbound line is {want}, not {r[f]}"
    for leg in ("outbound", "return"):
        df, nf, tf = f"{leg}_date", f"{leg}_flight_number", f"{leg}_departure_time"
        if not need(df):
            continue
        line = leg_line(email, r[df])
        if not line:
            bad[df] = "no line in the email has this date"
            continue
        if need(nf) and not (re.fullmatch(r"\d{1,4}", str(r[nf])) and
                             re.search(rf"\b(?:flight|flt|[A-Z]{{2}})\s*#?\s*{r[nf]}\b", line, re.I)):
            bad[nf] = f"not the flight number on the {leg} line"
        if need(tf):
            m = DEPART_TIME.search(line)
            if not m or norm_time(m[1]) != r[tf]:
                bad[tf] = f"not the departure time on the {leg} line"
    if r.get("outbound_date") and r.get("return_date") and str(r["return_date"]) < str(r["outbound_date"]):
        bad["return_date"] = "earlier than the outbound date"
    if need("total_paid"):
        amts = [float(a.replace(",", "")) for a in re.findall(r"\$\s?([\d,]+\.\d{2})", email)]
        try:
            ok = any(abs(a - float(r["total_paid"])) < 0.01 for a in amts)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            bad["total_paid"] = "not a dollar amount in the email"
    return bad


# Producer emails: the agent's own traffic. Fixes here become traces.
TRAIN = {
"southwest": """Subject: Your trip confirmation: AUS to DEN
Hi Jordan, you're all set! Confirmation # QX7PLM
Thursday, October 15, 2026  Flight 1123  Departs Austin (AUS) 8:05 AM  Arrives Denver (DEN) 9:35 AM
Sunday, October 18, 2026  Flight 2240  Departs Denver (DEN) 5:40 PM  Arrives Austin (AUS) 8:55 PM
Passengers: Jordan Parker, Casey Parker, Ada Parker, Leo Parker
Total paid: $1,284.40. Check in opens 24 hours before departure.""",
"united": """Your eTicket itinerary and receipt. Confirmation Number: HB2K9T
Traveler: PARKER/CASEY
Flight UA 1532 - Wed, Nov 25, 2026 - Austin, TX (AUS) Depart 6:40 AM - Chicago, IL (ORD) Arrive 10:05 AM
Flight UA 2210 - Sun, Nov 29, 2026 - Chicago, IL (ORD) Depart 3:15 PM - Austin, TX (AUS) Arrive 6:30 PM
Fare $412.30  Taxes and fees $58.70  Total $471.00 USD""",
"delta": """DELTA AIR LINES  |  Booking reference: GXW4RZ
Outbound  Dec 19  DL 768  ATL departs 11:20 AM, arrives AUS 12:55 PM
Return    Dec 27  DL 1940  AUS departs 1:45 PM, arrives ATL 5:10 PM
Total charged: $689.20. Passenger: Jordan Parker. Year of travel: 2026.""",
}

# Held-out emails: nobody trains on these. The validator measures before/after here.
EVAL = {
"american": """American Airlines - Your trip receipt. Record locator: MZKQ4D
Fri, Jan 8, 2027 AA 2417 depart DFW 7:15 AM arrive LGA 11:48 AM
Mon, Jan 11, 2027 AA 1288 depart LGA 4:05 PM arrive DFW 7:20 PM
Passenger: Riley Okafor   Total: $538.60""",
"alaska": """Thanks for flying Alaska! Confirmation code: PLW7XN
Outbound: Feb 3, 2027  Flight 1405  Departs Seattle (SEA) 9:50 AM  Arrives San Diego (SAN) 12:31 PM
Return: Feb 9, 2027  Flight 1416  Departs San Diego (SAN) 6:20 PM  Arrives Seattle (SEA) 9:05 PM
Guest: Morgan Lee.  Amount paid $312.40""",
"jetblue": """JetBlue booking confirmed - code TRVQ8B
Leaving 3/14/2027: Flight 623, Boston (BOS) departs 2:10 PM, New Orleans (MSY) arrives 5:22 PM
Coming home 3/18/2027: Flight 624, New Orleans (MSY) departs 6:05 PM, Boston (BOS) arrives 10:31 PM
Traveler Sam Rivera. Trip total $401.18""",
}

# New traffic for an agent running on autopilot: the numeric-date format the routing learning doesn't fix.
LIVE = {
"spirit": """Spirit Airlines receipt - confirmation KD4M9P
Departing 4/2/2027: Flight 1287, Las Vegas (LAS) departs 7:10 AM, Denver (DEN) arrives 10:02 AM
Returning 4/6/2027: Flight 1288, Denver (DEN) departs 4:45 PM, Las Vegas (LAS) arrives 5:41 PM
Traveler Pat Morgan. Total $212.48""",
"frontier": """Frontier booking - code FT7Q2W
Going 5/11/2027: Flight 2213, Orlando (MCO) departs 6:30 AM, Raleigh (RDU) arrives 8:18 AM
Back 5/15/2027: Flight 2214, Raleigh (RDU) departs 9:05 PM, Orlando (MCO) arrives 10:57 PM
Passenger Lee Ortiz. Paid $158.30""",
"suncountry": """Sun Country itinerary, record locator SC8H3N
Outbound 6/20/2027: Flight 405, Minneapolis (MSP) departs 8:15 AM, Phoenix (PHX) arrives 10:20 AM
Return 6/27/2027: Flight 406, Phoenix (PHX) departs 11:40 AM, Minneapolis (MSP) arrives 4:05 PM
Guest Dana Reyes. Total $389.16""",
}
