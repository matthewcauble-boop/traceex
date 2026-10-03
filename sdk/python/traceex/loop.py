"""The extract -> check -> send back -> verify loop that turns a model's mistakes into verified fixes.

Works with any model: you pass the extraction function, the checker, and (optionally) a way to narrow the context
for one field and a way to name the value the rules point to when the model can't get it.
"""
import time


def extract_checked(text, extract, check, *, fields, clean=lambda r: r, narrow=None, evidence=None, rounds=3):
    """extract(text, field_names) -> dict; check(text, result) -> {field: reason} for failing fields.
    narrow(text, field, result) -> smaller context for one field; evidence(text, result) -> {field: value}.

    Returns a dict:
      first      the model's first answer (cleaned)        result   the verified answer
      failing    fields still unverified (hand these to a bigger model)
      by_model   fields the model got right after being sent back
      by_rule    fields filled from the checker's evidence
      rounds     per-round pass counts, seconds
    """
    t0 = time.time()
    first = clean(extract(text, list(fields)) or {})
    r, history = dict(first), []
    for i in range(rounds + 1):
        bad = check(text, r)
        history.append(len(fields) - len(bad))
        if not bad or i == rounds:
            break
        for f in bad:
            ctx = narrow(text, f, r) if narrow else text
            got = extract(ctx, [f]) or {}
            if got.get(f) not in (None, ""):
                r[f] = got[f]
        r = clean(r)
    bad = check(text, r)
    by_rule = []
    if bad and evidence:
        ev = evidence(text, r)
        for f in list(bad):
            if f in ev:
                r[f] = ev[f]
                by_rule.append(f)
        bad = check(text, r)
    first_bad = check(text, first)
    by_model = [f for f in first_bad if f not in bad and f not in by_rule]
    return {"first": first, "result": r, "failing": sorted(bad), "by_model": sorted(by_model), "by_rule": sorted(by_rule),
            "rounds": history, "seconds": round(time.time() - t0, 3)}
