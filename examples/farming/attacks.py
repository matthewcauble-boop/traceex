"""Farming attacks against the coin economy (v0.4), each run on a real node (node/coin.py), with its profit or loss.

    python examples/farming/attacks.py              # every attack once, with what happened
    python examples/farming/attacks.py --seeds 30   # each on 30 random draws: mean, best run, how often it paid

Setup per attack: a fresh testnet node with 7 staked validators ($15 of TXC each), quorum 3, validators drawn from the
beacon published after each submission, an honest operator collecting the fees, and honest neighbours: a producer whose
traces real learnings use; a user who tries every accepted learning on its own traffic and pays for one ($10 of calls)
only when the gain clears its own noise; a watchdog that re-measures accepted learnings, looks at their parents, and
challenges what it can show is fake; bounty posters who measure claims on their own hidden evals; licence buyers whose
learnings name the traces they used. Honest validators report the true gain plus sampling noise (600-problem eval sets,
paired standard error 0.026), audit 20 parents and spend $0.05 of GPU time a measurement; bribed ones report whatever
the attacker says. P&L counts the dollars, credits (at face value, though they can only be spent here) and TXC (liquid,
vesting, staked) the attack moved, TXC at the price when the run began, less GPU time, so nobody shows a profit just
because someone else bought TXC. "vs honest" compares with the same trainer doing honest work (or, for a validator or
an operator, with the same one behaving). The node's clock is simulated: an epoch is a day where an attack needs time
to pass (the time-weighted price), and otherwise stands still.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node")]
from traceex import Trace, Learning, attest  # noqa: E402
from coin import CoinExchange, Params, UNIT, PQ, attestation_digest, decoy_digest, fmt_txc, units_for  # noqa: E402

A = lambda c: "0x" + c * 40
V = lambda i: "0x" + f"{0xa0 + i:02x}" * 20
ATTACKER, HONEST, WATCHDOG, BACKER, USER, POSTER, BUYER, OPERATOR = (A(c) for c in "6bd89743")
INVESTOR, FEES = A("1"), "0x" + "0f" * 20                 # someone buying TXC; the honest node operator
N_VALIDATORS, TRUE_GAIN, GPU = 7, 0.08, 50_000            # $0.05 of GPU time per honest measurement
DAY = 86_400
TASK = "Write a function number {} that adds two numbers."
usd = lambda m: f"{'-' if m < 0 else '+'}${abs(m) / 1e6:,.{6 if 0 < abs(m) < 10_000 else 2}f}"   # sub-cent: 6 places


class Clock:
    """The node's clock, in seconds: it moves only when an attack says time passes."""

    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def fund(ex, who, micros):
    ex.db.execute("INSERT OR REPLACE INTO grants VALUES (?,?,?,?)", (who, micros, "sim", "sim"))
    ex.db.commit()


def node(seed=7, attacker_validators=0, lazy=(), operator=FEES):
    """A node with 7 validators: the first `attacker_validators` answer to the attacker; `lazy` ones never measure.
    Licences clear at no less than $2."""
    clock = Clock()
    ex = CoinExchange(":memory:", test_credits=25_000_000, beacon_delay=1, reserve_micros=2_000_000,
                      params=Params(quorum=3), clock=clock, fee_to=operator)
    ex.seed, ex.time = seed, clock
    vals = [V(i) for i in range(N_VALIDATORS)]
    for v in vals:
        fund(ex, v, 25_000_000)
        ex.swap(v, "buy", 16_000_000)
        ex.register_validator(v, 1_500 * UNIT)
    for who in (ATTACKER, HONEST, WATCHDOG, BACKER, USER, POSTER, BUYER, OPERATOR, INVESTOR):
        fund(ex, who, 200_000_000)
    ex.register_checker("unit-tests", HONEST)
    ex.register_checker("attacker-tests", ATTACKER)
    ex.corrupt, ex.lazy, ex.measured = set(vals[:attacker_validators]), set(lazy), {}
    return ex


def stream(ex, lid, what):
    """The noise for one learning's validation round, or for its user's trial: keyed by the run's seed and the
    learning's name, so a learning named the same draws the same noise in an attack and in its honest twin, and
    "vs honest" measures the attack, not luck."""
    name = json.loads(ex.db.execute("SELECT body FROM learnings WHERE id=?", (lid,)).fetchone()[0])["artifact"]["uri"]
    return random.Random(f"{ex.seed}|{name}|{what}")


def trace(producer, text, checker="unit-tests@1", code=None):
    return Trace.from_fix(task="code.python", base_model="qwen", input=text, model_output={"code": "bad"},
                          verified_output={"code": code or f"fixed: {text}"}, checker=checker, producer=producer,
                          created="2026-10-04T00:00:00Z", privacy="open", failure_modes={"code": "wrong_answer"})


def parents_of(ex, producer=HONEST, n=20, checker="unit-tests@1"):
    return [ex.submit_trace(dict(trace(producer, TASK.format(i), checker)))["id"] for i in range(n)]


def submit(ex, trainer, parents, claim, name, eval_set="sha256:claim", by=None):
    ex.swap(trainer, "buy", 6_000_000)                    # enough TXC for the $5 bond
    att = attest(by or V(0), eval_set, "pass@1", 0.30, round(0.30 + claim, 4))
    L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": name, "hash": None},
                       parents=[(p, 1) for p in parents], trainer=trainer, attestation=att, per_call_micros=200)
    return ex.register_learning(L)["id"]


def validate(ex, lid, true_gain, claim, rnd=0, audit_bad=0.0):
    """Assigned validators commit, then reveal. Honest ones measure (the truth plus noise, a real parent audit, $0.05 of
    GPU); bribed ones report the attacker's number with a clean audit; lazy ones repeat the trainer's claim unmeasured."""
    reports, rng = {}, stream(ex, lid, f"round {rnd}")
    for v in ex.verdict(lid)["assigned"]:
        noise, before = rng.gauss(0, 0.026), 0.30 + rng.uniform(-0.03, 0.03)     # drawn for every seat, in order
        honest = v not in ex.corrupt and v not in ex.lazy
        if honest:
            g = true_gain + noise
            ex.measured[v] = ex.measured.get(v, 0) + 1
        else:
            g = claim if v in ex.corrupt else ex._claimed_gain(lid)
        att = {"validator": v, "eval_set": f"sha256:{v[-4:]}{rnd}", "metric": "pass@1", "before": round(before, 4),
               "after": round(min(max(before + g, 0), 1), 4), "n": 600, "se": 0.026,
               "audit": {"checked": 20, "bad": round(20 * audit_bad) if honest else 0}}
        ex.commit(lid, v, attestation_digest(att, "s" + v), rnd)
        reports[v] = att
    for v, att in reports.items():
        ex.reveal(lid, v, att, "s" + v, rnd)


def watch(ex, lid, true_gain, junk=0.0):
    """The honest watchdog: if an accepted learning shows no gain when it measures it, or its parents are padding, it
    stakes $2 of TXC on a challenge; fresh validators are drawn from the next beacon."""
    v = ex.verdict(lid)
    fake = true_gain < ex.p.min_gain
    padded = junk > ex.p.audit_max_bad and (v["audit_bad"] or 0) <= ex.p.audit_max_bad
    if v["status"] == "accepted" and (fake or padded):
        ex.swap(WATCHDOG, "buy", 5_000_000)
        ex.challenge(lid, WATCHDOG)
        ex.settle()
        validate(ex, lid, true_gain, claim=0.30, rnd=ex.verdict(lid)["round"], audit_bad=junk)


def use(ex, lid, true_gain, calls=50_000):
    """An honest user tries an accepted learning on its own traffic first (its own standard error: 0.015) and pays for
    $10 of calls only when the gain clears twice its noise."""
    if ex.verdict(lid)["status"] == "accepted" and true_gain + stream(ex, lid, "user").gauss(0, 0.015) - 0.03 >= ex.p.min_gain:
        ex.usage({"learning": lid, "consumer": USER, "calls": calls})
        return True
    return False


def endow(ex, who, units):
    """TXC an account earned long ago (outside this run), kept on the books like any minted TXC."""
    ex._add("minted", units)
    ex._coin(who, units, "earned before this run")
    ex.db.commit()


def worth(ex, accounts, price):
    total = 0
    for a in accounts:
        w = ex.wallet(a)
        total += w["balance_micros"] + w["credits_micros"] + (w["coin_units"] + w["vesting_units"] + w["stake_units"]) * price // UNIT
    return total


def top_up(ex):
    """Honest validators the price pushed under the dollar minimum top their stake up (they have stake_grace_epochs)."""
    floor = ex.min_stake_units()
    for v in (V(i) for i in range(N_VALIDATORS)):
        short = floor - ex._stake(v)
        if v not in ex.corrupt and short > 0:
            ex.swap(v, "buy", max(short * ex.price() // UNIT * 11 // 10, 1))
            ex.register_validator(v, min(short, ex.wallet(v)["coin_units"]))


def accepted(ex, trainer, parents, name):
    """A real learning, resubmitted (a fresh validation draw each time, every bond counted) until it is accepted."""
    for k in range(5):
        lid = submit(ex, trainer, parents, TRUE_GAIN, name if k == 0 else f"{name}-{k}")
        ex.settle()
        validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
        if ex.verdict(lid)["status"] == "accepted":
            break
    return lid


def real_learning(ex, name="real", own=True):
    """The attacker's own real learning (its traces, its checker, so every share but the validators' is its own),
    validated and accepted."""
    parents = parents_of(ex, producer=ATTACKER, checker="attacker-tests@1") if own else parents_of(ex)
    lid = accepted(ex, ATTACKER, parents, name)
    for _ in range(ex.p.vest_epochs + 1):                 # its bond comes home before the attack is measured
        ex.settle()
    return lid


def run(attack, attacker_validators=0, seed=7, lazy=(), mine=None, operator=FEES):
    ex = node(seed, attacker_validators, lazy, operator)
    mine = mine or [ATTACKER] + sorted(ex.corrupt | ex.lazy)
    price = ex.price()
    start = worth(ex, mine, price)
    out = attack(ex)
    note, override = out if isinstance(out, tuple) else (out, None)
    for _ in range(ex.p.vest_epochs + 2):                  # let vesting, bonds, licences and challenges play out
        ex.settle()
    assert ex.audit()["balanced"]
    gpu = sum(ex.measured.get(v, 0) for v in mine) * GPU
    pnl = override if override is not None else worth(ex, mine, price) - start - gpu
    return pnl, note, ex


# --- the v0.3 attacks, re-run ---------------------------------------------------------------------------------------
def trace_spam(ex):
    for i in range(1_000):
        ex.submit_trace(dict(trace(ATTACKER, f"junk task {i} that no model fails on", "attacker-tests@1")))
    return "1,000 junk traces; no learning uses them, nobody pays for them; each paid the $0.00005 fee"


def stuff_lot(ex):
    real = parents_of(ex)
    for i in range(1_000):                                    # junk that claims the honest checker lands in its lot
        ex.submit_trace(dict(trace(ATTACKER, f"Write a function number {i} that returns its input.")))
    lot = next(x["lot"] for x in ex.lots()["lots"] if x["lot"].endswith("|unit-tests@1"))
    for who in (BUYER, BACKER):
        ex.bid({"lot": lot, "bidder": who, "price_micros": 2_000_000})
    ex.clear()
    lid = submit(ex, BUYER, real, TRUE_GAIN, "built-on-the-lot")        # one buyer's learning uses the 20 real traces
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    ex.direct_licence(lot, BACKER, real)                                 # the other names the traces it used
    return "1,000 junk traces stuffed into an honest lot; two $2 licences count for the 20 traces the buyers used"


def copies(reworded):
    def attack(ex):
        originals = parents_of(ex)
        if reworded:                                            # the same fixes, the task said in other words
            texts = [f"Create a Python function (number {i}) which sums two numbers." for i in range(20)]
        else:                                                   # case, spacing
            texts = [TASK.format(i).upper().replace(" ", "  ") for i in range(20)]
        dupes = [ex.submit_trace(dict(trace(ATTACKER, t, code=f"fixed: {TASK.format(i)}")))["id"]
                 for i, t in enumerate(texts)]
        lid = submit(ex, HONEST, originals + dupes, TRUE_GAIN, "honest-with-copies")   # even if a trainer cites them
        ex.settle()
        validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
        use(ex, lid, TRUE_GAIN)
        return f"20 {'reworded copies' if reworded else 'near-copies'} of honest traces, cited by a real learning; " \
               "every copy pays the original producer"
    return attack


def fake_learning(ex):
    lid = submit(ex, ATTACKER, parents_of(ex, n=5), 0.30, "fake")
    ex.settle()
    validate(ex, lid, 0.0, claim=0.30)
    watch(ex, lid, 0.0)
    return f"claims +30 points, true gain 0; verdict: {ex.verdict(lid)['status']}"


def fake_learnings_with_bribes(ex):
    parents, outcomes = parents_of(ex, n=5), []
    for k in range(12):                                       # keep resubmitting, hoping to draw both bribed validators
        lid = submit(ex, ATTACKER, parents, 0.30, f"fake-{k}")
        ex.settle()
        validate(ex, lid, 0.0, claim=0.30)
        watch(ex, lid, 0.0)
        use(ex, lid, 0.0)
        outcomes.append(ex.verdict(lid)["status"])
    return "12 fake learnings, 2 of 7 validators bribed: " + ", ".join(f"{outcomes.count(s)} {s}" for s in sorted(set(outcomes)))


def wash_usage(ex):
    lid = real_learning(ex)
    price = ex.price()
    before = worth(ex, [ATTACKER], price)
    ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 500_000})       # $100 of usage paid to itself
    for _ in range(ex.p.vest_epochs + 1):
        ex.settle()
    return f"price above equilibrium (the cap binds): $100 of usage of its own real learning (its traces, its " \
           f"checker); that money alone: {usd(worth(ex, [ATTACKER], price) - before)}"


def self_bounty(ex):
    b = ex.post_bounty({"poster": ATTACKER, "path": "code", "eval_set": "sha256:mine", "target": 0.35, "title": "mine"})
    ex.buy_coins(b["id"], ATTACKER, 50_000_000)
    lid = submit(ex, ATTACKER, parents_of(ex), TRUE_GAIN, "real")
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    use(ex, lid, TRUE_GAIN)
    if ex.verdict(lid)["status"] == "accepted":              # as poster, it confirms its own solution
        ex.claim_bounty(b["id"], lid, attest(ATTACKER, "sha256:mine", "pass@1", 0.30, 0.38))
    return "posts and backs its own bounty with $50, solves it with a real learning, confirms it as the poster"


def dilution(audit_bad):
    def attack(ex):
        real = parents_of(ex)
        junk = [ex.submit_trace(dict(trace(ATTACKER, f"padding trace {i}", "attacker-tests@1")))["id"] for i in range(200)]
        lid = submit(ex, ATTACKER, real + junk, TRUE_GAIN, "real")
        ex.settle()
        validate(ex, lid, TRUE_GAIN, TRUE_GAIN, audit_bad=audit_bad)
        use(ex, lid, TRUE_GAIN)                              # a real user pays before anyone looks at the parents
        if audit_bad == 0:                                   # lazy validators passed it; the watchdog looks at them
            ex.settle()
            watch(ex, lid, TRUE_GAIN, junk=200 / 220)
        return (f"real learning padded with 200 junk parents of its own; validators' audit finds {audit_bad:.0%} bad"
                + ("; the watchdog's audit challenge catches it" if audit_bad == 0 else ""))
    return attack


def wrap_traces(ex):
    """Its real learning cites a learning of its own (terms: 100% to its trainer) wrapped around the honest traces,
    instead of the traces themselves, hoping their share flows through its wrapper to it. Generous: the wrapper passes
    validation too."""
    greedy = {"trainer": 1.0, "traces": 0.0, "checkers": 0.0, "validators": 0.0}
    ex.swap(ATTACKER, "buy", 6_000_000)
    L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": "wrapper", "hash": None},
                       parents=[(p, 1) for p in parents_of(ex)], trainer=ATTACKER, per_call_micros=200, split=greedy,
                       attestation=attest(V(0), "sha256:claim", "pass@1", 0.30, 0.38))
    wrapper = ex.register_learning(L)["id"]
    ex.settle()
    validate(ex, wrapper, TRUE_GAIN, TRUE_GAIN)
    lid = submit(ex, ATTACKER, [wrapper], TRUE_GAIN, "real")
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    use(ex, lid, TRUE_GAIN)
    return "its real learning cites its own wrapper (100% to itself) instead of the honest traces; a cited " \
           "learning passes its share through to its own traces, so they are paid in full"


def pump_and_dump(ex):
    b = ex.post_bounty({"poster": ATTACKER, "path": "code", "eval_set": "sha256:lure", "target": 0.9, "title": "lure"})
    early = ex.buy_coins(b["id"], ATTACKER, 3_000_000)["coins"]               # the first, cheapest coins
    ex.buy_coins(b["id"], BACKER, 60_000_000)                                 # an honest backer buys dearer ones
    ex.sell_coins(b["id"], ATTACKER, early)                                   # and the attacker sells into it
    return "backs its own bounty first with $3, waits for an honest backer's $60, sells back into the raised price: " \
           "a sale returns at most what the coins cost"


def honest_trainer(ex):
    lid = submit(ex, ATTACKER, parents_of(ex), TRUE_GAIN, "real")
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    use(ex, lid, TRUE_GAIN)
    return "for comparison: the same trainer, real learning, real parents, one real user paying $10"


def validator_work(ex):
    """24 learnings: 18 real (from an honest trainer; a user pays for each that helps) and 6 decoys the operator seals
    with their true gain, 0, behind a +15 point claim."""
    parents, caught = parents_of(ex), 0
    for k in range(24):
        if k % 4 == 3:
            ex.swap(OPERATOR, "buy", 6_000_000)
            att = attest(V(0), "sha256:claim", "pass@1", 0.30, 0.45)
            L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": f"d{k}", "hash": None},
                               parents=[(p, 1) for p in parents], trainer="0x" + f"{0xd0 + k:02x}" * 20,
                               attestation=att, per_call_micros=200)
            lid = ex.register_decoy(L, decoy_digest(0.0, f"salt{k}"), OPERATOR)["id"]
            ex.settle()
            validate(ex, lid, 0.0, claim=0.15)
            caught += V(6) in ex.unseal_decoy(lid, 0.0, f"salt{k}")["caught"]
        else:
            lid = submit(ex, HONEST, parents, TRUE_GAIN, f"real-{k}")
            ex.settle()
            validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
            use(ex, lid, TRUE_GAIN)
    return (f"repeats each claim instead of measuring it (saves $0.05 a time) over 18 real learnings and 6 sealed "
            f"decoys; caught on {caught}" if V(6) in ex.lazy else f"measures every learning; caught on {caught}")


def majority_fakes(ex):
    own = parents_of(ex, producer=ATTACKER, checker="attacker-tests@1")
    outcomes = []
    for k in range(6):
        lid = submit(ex, ATTACKER, own, 0.30, f"fake-{k}")
        ex.settle()
        validate(ex, lid, 0.0, claim=0.30)
        watch(ex, lid, 0.0)
        use(ex, lid, 0.0)
        outcomes.append(ex.verdict(lid)["status"])
    return "owns 4 of 7 validator seats (57% of stake); 6 fake learnings: " + \
           ", ".join(f"{outcomes.count(s)} {s}" for s in sorted(set(outcomes))) + "; no verdict mints anything"


def majority_wash(ex):
    own = parents_of(ex, producer=ATTACKER, checker="attacker-tests@1")
    for k in range(6):
        lid = submit(ex, ATTACKER, own, 0.30, f"fake-{k}")
        ex.settle()
        validate(ex, lid, 0.0, claim=0.30)
        if ex.verdict(lid)["status"] == "accepted":
            break
    if ex.verdict(lid)["status"] == "accepted":
        ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 500_000})   # $100 of usage of its own accepted fake
    return "gets its own fake accepted, then pays $100 of usage of it to collect the mint that usage earns"


def majority_bounty(ex):
    b = ex.post_bounty({"poster": POSTER, "path": "code", "eval_set": "sha256:hidden", "target": 0.35, "title": "theirs"})
    ex.buy_coins(b["id"], BACKER, 50_000_000)
    own = parents_of(ex, producer=ATTACKER, checker="attacker-tests@1")
    for k in range(6):                                        # a fake "attested" on the bounty's eval by its own validator
        lid = submit(ex, ATTACKER, own, 0.30, f"fake-{k}", eval_set="sha256:hidden", by=V(1))
        ex.settle()
        validate(ex, lid, 0.0, claim=0.30)
        if ex.verdict(lid)["status"] == "accepted":
            break
    try:
        ex.claim_bounty(b["id"], lid, attest(V(2), "sha256:hidden", "pass@1", 0.30, 0.60))
        got = "paid"
    except ValueError:
        got = "refused: only the poster's own measurement on its hidden eval counts"
    tries = "first try" if k == 0 else f"{k + 1} tries"
    return f"its fake, attested on the bounty's eval by its own validator, is accepted on the {tries}; the claim on an " \
           f"honest $50 bounty: {got}"


def majority_grief(ex):
    parents, outcomes = parents_of(ex), []
    bonds = worth(ex, [HONEST], ex.price())
    for k in range(4):
        lid = submit(ex, HONEST, parents, TRUE_GAIN, f"honest-{k}")
        ex.settle()
        validate(ex, lid, TRUE_GAIN, claim=0.0)                       # its validators score honest work at zero
        if ex.verdict(lid)["status"] == "accepted":                   # and it challenges whatever gets through
            ex.swap(ATTACKER, "buy", 5_000_000)
            ex.challenge(lid, ATTACKER)
            ex.settle()
            validate(ex, lid, TRUE_GAIN, claim=0.0, rnd=ex.verdict(lid)["round"])
        outcomes.append(ex.verdict(lid)["status"])
    lost = worth(ex, [HONEST], ex.price()) - bonds
    return ("blocks 4 honest learnings (" + ", ".join(f"{outcomes.count(s)} {s}" for s in sorted(set(outcomes)))
            + f"): the honest trainer is down {usd(lost)[1:]}, and none of it reaches the attacker")


# --- attacks the v0.4 design invites -----------------------------------------------------------------------------------
def wash_below_equilibrium(ex):
    """Honest users pay $2,000 an epoch for an honest learning: more than the emission is worth at the pool's price,
    so the emission is over-subscribed and shared out below the cap. The attacker adds $100 of usage of its own."""
    honest = accepted(ex, HONEST, parents_of(ex), "busy")
    lid = real_learning(ex, "real")
    price = ex.price()
    before = worth(ex, [ATTACKER], price)
    fund(ex, USER, 10_000_000_000)
    ex.usage({"learning": honest, "consumer": USER, "calls": 10_000_000})     # $2,000 of honest usage
    ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 500_000})       # $100 to itself
    ex.settle()
    m = ex.last_mint()
    rate = m["minted_units"]["work"] * ex.ref_q() // UNIT // PQ / max(m["credits_burned"]["work"], 1)
    for _ in range(ex.p.vest_epochs):
        ex.settle()
    return (f"price below equilibrium: $2,100 of usage against an emission worth ${m['emission_units'] * price // UNIT / 1e6:,.0f} "
            f"at the pool's price, so each $1 burned earned {rate:.2f} of TXC; its $100 to itself, that money alone: "
            f"{usd(worth(ex, [ATTACKER], price) - before)}")


def twap_lag(ex):
    """The price climbs through the epoch (an investor buys TXC hour after hour), so the time-weighted price lags the
    spot. At the end the attacker pays $100 to its own learning at the high spot: credits valued at the lagging TWAP
    are worth more TXC than its dollars burned. Valued at the price when it acts, against doing nothing."""
    lid = real_learning(ex)
    fund(ex, INVESTOR, 5_000_000_000)
    for _ in range(24):
        ex.time.advance(DAY / 24)
        ex.swap(INVESTOR, "buy", 100_000_000)                     # $2,400 over the day: the price climbs about 50%
    price = ex.price()
    before = worth(ex, [ATTACKER], price)
    burned0 = ex._m("burned")
    ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 500_000})
    burned = ex._m("burned") - burned0
    ex.settle()
    twap = ex.last_mint()["twap_q"]
    rows = ex.db.execute("SELECT credits, basis FROM work WHERE epoch=? AND account=?", (ex.epoch - 1, ATTACKER)).fetchall()
    twap_only = sum(units_for(c, twap) for c, _ in rows)            # what the TWAP alone would have allowed
    capped = sum(min(units_for(c, twap), int(b)) for c, b in rows)  # what the creation-price bound allows
    for _ in range(ex.p.vest_epochs + 1):
        ex.settle()
    pnl = worth(ex, [ATTACKER], price) - before
    return (f"spot {(price * PQ / twap - 1):.0%} above the epoch's time-weighted price when it paid $100: the TWAP alone "
            f"would mint it {twap_only / burned:.3f} TXC per TXC burned ({usd((twap_only - burned) * price // UNIT)}); the "
            f"bound by what its credits burned held it to {capped / burned:.3f}; that money alone: {usd(pnl)}", pnl)


def cheap_credits(ex):
    """Make credits cheap: buy TXC, pump the pool's spot, burn the first TXC for credits at the pumped price, unwind."""
    ex.swap(ATTACKER, "buy", 20_000_000)
    held = ex.wallet(ATTACKER)["coin_units"]
    ex.swap(ATTACKER, "buy", 80_000_000)                          # the pump
    spot = ex.price()
    made = ex.buy_credits(ATTACKER, units=held)["credits_micros"]
    ex.swap(ATTACKER, "sell", ex.wallet(ATTACKER)["coin_units"])   # unwinds it
    return (f"burns its TXC with the spot pumped to ${spot / 1e6:.4f}: credits come at the lower reference price "
            f"(${ex.ref_q() / PQ / 1e6:.4f}), {made:,} credits for TXC that cost $20; dollars on-ramp 1:1 whatever the pool does")


def inflate_mint(ex):
    """Inflate a capped mint: a large holder dumps 300,000 TXC it earned long ago to hold the pool's price down for a
    whole epoch (so the time-weighted price is low and its credits are worth more TXC), pays $100 to its own learning
    there, then buys its TXC back. Measured, at the price before the dump, from just before the dump."""
    lid = real_learning(ex)
    endow(ex, ATTACKER, 300_000 * UNIT)
    p0 = ex.price()
    before = worth(ex, [ATTACKER], p0)
    cash = ex.swap(ATTACKER, "sell", 300_000 * UNIT)["paid_micros"]     # the price falls...
    low = ex.price()
    ex.time.advance(DAY)                                          # ...and stays down all epoch (nobody buys the dip)
    ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 500_000})
    ex.settle()
    m = ex.last_mint()
    ex.swap(ATTACKER, "buy", cash)                                # buys its TXC back
    for _ in range(ex.p.vest_epochs):
        ex.settle()
    return (f"dumps 300,000 TXC to hold the pool {1 - low / p0:.0%} down for a whole epoch, pays itself $100 there, buys "
            f"back: the cap is the lower of the credits at that low time-weighted price and the TXC they burned, so it "
            f"gets back what it burned ({fmt_txc(m['minted_units']['work'])} TXC) less every share not its own, and pays "
            f"the spread both ways", worth(ex, [ATTACKER], p0) - before)


def operator_self_deal(dealing):
    def attack(ex):
        """The attacker runs the node (it collects the operator share). Honest traffic: a producer's 20 traces and a
        user's $10. Self-dealing: 2,000 junk transactions of its own, to claim more of the operator's 10%."""
        lid = submit(ex, HONEST, parents_of(ex), TRUE_GAIN, "real")
        ex.settle()
        validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
        use(ex, lid, TRUE_GAIN)
        if dealing:
            for i in range(2_000):
                ex.submit_trace(dict(trace(ATTACKER, f"self-dealt transaction {i}", "attacker-tests@1")))
        return ("runs the node and sends it 2,000 transactions of its own ($0.10 of fees) to claim more of the "
                "operator's 10%: it is minted at most what its fees burned" if dealing else
                "runs the node honestly, minted its share for the fees honest traffic paid")
    return attack


def fee_evasion(ex):
    """Pack many fixes into each transaction, batch usage into one report, then empty the wallet into TXC before
    settlement so the fees can't be billed."""
    packed = [ex.submit_trace(dict(trace(ATTACKER, " ".join(f"fix {i}-{j}: add numbers {j}" for j in range(400)),
                                         "attacker-tests@1")))["id"] for i in range(25)]   # 10,000 fixes, 25 fees
    lid = real_learning(ex)
    ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 5_000})                   # one report, 5,000 calls
    try:                                                                                 # drain before settlement
        ex.swap(ATTACKER, "buy", ex.wallet(ATTACKER)["balance_micros"] + ex.wallet(ATTACKER)["credits_micros"])
        drained = "it could"
    except ValueError:
        drained = "refused: an account can't spend what it owes"
    return (f"10,000 fixes in 25 traces (25 fees; they earn as 25 parents, not 10,000), 5,000 calls in one report (one fee, "
            f"every call still paid); emptying the wallet before the fee bill: {drained}; the biggest transaction allowed "
            f"(64 KB) still pays about 11x its electricity")


def curve_gaming(ex):
    """Bounty curve gaming under dollar pricing: push the pool up, then back its own bounty with dollars (fewer TXC go
    into the pool for the same dollars) hoping for coins worth more than the TXC it put in; lure an honest backer; sell
    back; unwind the pump."""
    b = ex.post_bounty({"poster": ATTACKER, "path": "code", "eval_set": "sha256:lure", "target": 0.9, "title": "lure"})
    ex.swap(ATTACKER, "buy", 80_000_000)                          # the pump
    got = ex.buy_coins(b["id"], ATTACKER, 10_000_000)
    ex.swap(ATTACKER, "sell", ex.wallet(ATTACKER)["coin_units"])   # unwinds it
    ex.buy_coins(b["id"], BACKER, 60_000_000)                     # an honest backer
    ex.sell_coins(b["id"], ATTACKER, got["coins"])
    return (f"backs its own bounty with $10 while it has the pool pumped: the coins count only the TXC that went in, at "
            f"the reference price (${got['spent_micros'] / 1e6:.2f} of the $10), and a sale returns that TXC, never more")


def whale_dump(ex):
    """A large holder dumps 300,000 TXC it earned long ago (30% of the pool's TXC) into the pool at once, in an epoch
    when honest users pay for an honest learning, and keeps the dollars."""
    lid = submit(ex, HONEST, parents_of(ex), TRUE_GAIN, "real")
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    endow(ex, ATTACKER, 300_000 * UNIT)
    p0 = ex.price()
    before = worth(ex, [ATTACKER], p0)
    ex.swap(ATTACKER, "sell", 300_000 * UNIT)
    low = ex.price()
    use(ex, lid, TRUE_GAIN)
    ex.time.advance(DAY)
    ex.settle()
    active = len(ex._active())
    return (f"dumps 300,000 TXC at once (the price falls {1 - low / p0:.0%}): it sells down its own price, credits still "
            f"cost $1 per million, and the federation keeps working ({active} of 7 validators keep their seats while "
            f"they top up to the dollar minimum)", worth(ex, [ATTACKER], p0) - before)


def stake_floor(ex):
    """Push honest validators under the dollar minimum and take the draw: dump 300,000 TXC earned long ago so the next
    reference price falls, stake 3 validators of its own at the new minimum, and submit fakes, hoping only its own
    validators still count as active."""
    mine = ["0x" + f"{0x90 + i:02x}" * 20 for i in range(3)]
    for v in mine:
        fund(ex, v, 50_000_000)
    endow(ex, ATTACKER, 300_000 * UNIT)
    p0 = ex.price()
    before = worth(ex, [ATTACKER] + mine, p0)
    ex.swap(ATTACKER, "sell", 300_000 * UNIT)
    ex.time.advance(DAY)
    ex.settle()                                                   # the reference price follows the dump
    for v in mine:
        ex.swap(v, "buy", 20_000_000)
        ex.register_validator(v, ex.min_stake_units())
    ex.corrupt |= set(mine)
    own, seats, outcomes = parents_of(ex, producer=ATTACKER, checker="attacker-tests@1"), [], []
    for k in range(4):
        top_up(ex)
        lid = submit(ex, ATTACKER, own, 0.30, f"fake-{k}")
        ex.settle()
        seats.append(sum(v in mine for v in ex.verdict(lid)["assigned"]))
        validate(ex, lid, 0.0, claim=0.30)
        watch(ex, lid, 0.0)
        use(ex, lid, 0.0)
        outcomes.append(ex.verdict(lid)["status"])
    for _ in range(ex.p.vest_epochs):
        ex.settle()
    return (f"after the dump the honest validators keep their seats while they top up ({len(ex._active())} active); its 3 new validators took "
            f"{sum(seats)} of {3 * len(seats)} seats on its 4 fakes: " +
            ", ".join(f"{outcomes.count(s)} {s}" for s in sorted(set(outcomes))), worth(ex, [ATTACKER] + mine, p0) - before)


def sybil(ex):
    """Split across many accounts: 10 trainer identities with 10 learnings of 20 traces each, 10 consumer accounts
    paying $10 each, and its stake split into 5 validators at the minimum; caps, shares and draws are all linear."""
    trainers = ["0x" + f"{0x60 + i:02x}" * 20 for i in range(10)]
    payers = ["0x" + f"{0x70 + i:02x}" * 20 for i in range(10)]
    sybils = ["0x" + f"{0x80 + i:02x}" * 20 for i in range(5)]
    for who in trainers + payers + sybils:
        fund(ex, who, 50_000_000)
    for v in sybils:
        ex.swap(v, "buy", 11_000_000)
        ex.register_validator(v, ex.wallet(v)["coin_units"])
    ex.corrupt |= set(sybils)
    lids = []
    for i, t in enumerate(trainers):
        own = [ex.submit_trace(dict(trace(t, f"{TASK.format(j)} variant {i}", "attacker-tests@1")))["id"] for j in range(20)]
        ex.swap(t, "buy", 6_000_000)
        L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": f"sybil-{i}", "hash": None},
                           parents=[(p, 1) for p in own], trainer=t, per_call_micros=200,
                           attestation=attest(V(0), "sha256:claim", "pass@1", 0.30, 0.38))
        lids.append(ex.register_learning(L)["id"])
    ex.settle()
    for lid in lids:
        validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    ex.settle()
    paid = 0
    for lid, payer in zip(lids, payers):
        if ex.verdict(lid)["status"] == "accepted":
            ex.usage({"learning": lid, "consumer": payer, "calls": 50_000})
            paid += 1
    ex.sybil_accounts = trainers + payers + sybils
    return f"10 trainers, 10 payers ({paid} paid $10 each to the sybils' own accepted learnings) and 5 minimum-stake validators"


ATTACKS = [
    ("Trace spam", trace_spam, 0),
    ("Stuff an honest lot with junk", stuff_lot, 0),
    ("Copy honest traces", copies(False), 0),
    ("Reworded copies of honest traces", copies(True), 0),
    ("Fake learning", fake_learning, 0),
    ("Fake learning, 1 bribed validator", fake_learning, 1),
    ("Fake learnings, 2 of 7 bribed", fake_learnings_with_bribes, 2),
    ("Wash usage, price above equilibrium", wash_usage, 0),
    ("Self-funded bounty", self_bounty, 0),
    ("Pad a real learning, honest audits", dilution(0.9), 0),
    ("Pad a real learning, lazy audits", dilution(0.0), 0),
    ("Wrap honest traces in its own learning", wrap_traces, 0),
    ("Pump and dump a bounty's coins", pump_and_dump, 0),
    ("Lazy validator (never measures)", validator_work, 0),
    ("(honest trainer, for scale)", honest_trainer, 0),
    ("Majority: fake learnings", majority_fakes, 4),
    ("Majority: wash its own fake", majority_wash, 4),
    ("Majority: take an honest bounty", majority_bounty, 4),
    ("Majority: block honest work", majority_grief, 4),
    ("Wash usage, price below equilibrium", wash_below_equilibrium, 0),
    ("Wash usage while the price climbs", twap_lag, 0),
    ("Pump the pool to make credits cheap", cheap_credits, 0),
    ("Hold the pool low to inflate a mint", inflate_mint, 0),
    ("Operator self-dealing", operator_self_deal(True), 0),
    ("Fee evasion: pack, batch, drain", fee_evasion, 0),
    ("Bounty curve gaming (dollar curve)", curve_gaming, 0),
    ("Large holder dumps into the pool", whale_dump, 0),
    ("Sybil: split across 25 accounts", sybil, 0),
    ("Dump to push validators under the minimum", stake_floor, 0),
]
REAL = {"Wash usage, price above equilibrium", "Self-funded bounty", "Pad a real learning, honest audits",
        "Pad a real learning, lazy audits", "Wrap honest traces in its own learning", "(honest trainer, for scale)",
        "Wash usage, price below equilibrium"}


def play(name, attack, bribed, seed=7, honest=None):
    """One run of one attack: (P&L, vs honest, what happened)."""
    if attack is validator_work:                              # the same validator, lazy and then honest
        pnl, note, _ = run(attack, lazy=(V(6),), mine=[V(6)], seed=seed)
        return pnl, pnl - run(attack, mine=[V(6)], seed=seed)[0], note
    if name == "Operator self-dealing":                       # the same operator, self-dealing and then not
        pnl, note, _ = run(attack, seed=seed, operator=ATTACKER)
        return pnl, pnl - run(operator_self_deal(False), seed=seed, operator=ATTACKER)[0], note
    if attack is stake_floor:                                 # the attacker and its new validators together
        pnl, note, ex = run(attack, seed=seed)
        return pnl, pnl, note
    if attack is sybil:                                       # all the sybil's accounts together
        ex = node(seed)
        price = ex.price()
        note = attack(ex)
        mine = ex.sybil_accounts
        start = len(mine) * 50_000_000
        for _ in range(ex.p.vest_epochs + 2):
            ex.settle()
        assert ex.audit()["balanced"]
        pnl = worth(ex, mine, price) - start
        return pnl, pnl, note
    pnl, note, _ = run(attack, bribed, seed=seed)
    return pnl, (pnl - honest if name in REAL else pnl), note


def main(argv=None):
    """python attacks.py: every attack once (seed 7), with what happened. --seeds 30: each attack on 30 different
    random draws (validators, noise, users), with the mean result, the best run for the attacker, and how often it paid."""
    argv = sys.argv[1:] if argv is None else argv
    seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 1
    print(__doc__.split("\n\n")[1].strip(), "\n")
    rows = []
    if seeds == 1:
        honest = run(honest_trainer)[0]
        print(f"{'attack':40} {'P&L':>11} {'vs honest':>11}   what happened")
        for name, attack, bribed in ATTACKS:
            pnl, extra, note = play(name, attack, bribed, honest=honest)
            rows.append((name, pnl, extra, note))
            print(f"{name:40} {usd(pnl):>11} {usd(extra):>11}   {note}")
    else:                      # each run against its honest twin: same seed, and its real learning draws the same noise
        honest = {s: run(honest_trainer, seed=s)[0] for s in range(1, seeds + 1)}
        print(f"{'attack':40} {'mean vs honest':>15} {'best run':>11} {'runs it paid':>13}")
        for name, attack, bribed in ATTACKS:
            xs = [play(name, attack, bribed, s, honest[s])[1] for s in range(1, seeds + 1)]
            rows.append((name, sum(xs) / len(xs), max(xs), sum(x > 0 for x in xs)))
            print(f"{name:40} {usd(sum(xs) / len(xs)):>15} {usd(max(xs)):>11} {sum(x > 0 for x in xs):>6} of {seeds}")
    print("\nEvery strategy loses, including owning most of the validator stake: a majority can still block honest work, "
          "because it controls the vote, but no verdict moves money to it. Nobody is minted more than the credits burned "
          "on their own work were worth, at the time-weighted price and never above the TXC those credits burned; bounties "
          "pay on the poster's own measurement; licence credits count for the traces the buyer's own learnings use; and "
          "forfeits burn.")
    return rows


if __name__ == "__main__":
    main()
