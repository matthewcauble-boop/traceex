"""Farming attacks against traceX v0.6 (no token: every payment in sats, every payout a split of a real payment), each
run on a real node (node/sats.py), with its profit or loss.

    python examples/farming/attacks.py              # every attack once, with what happened
    python examples/farming/attacks.py --seeds 30   # each on 30 random draws: mean, best run, how often it paid
    python examples/farming/attacks.py --decoys     # the decoy test's false-positive rate, measured

Setup per attack: a fresh testnet node with 7 validators staking 15,000 sats each, quorum 3, validators drawn from the
beacon published after each submission, an honest operator collecting the 58-msat fees, and honest neighbours: a
producer whose traces real learnings use; a user who tries every accepted learning on its own traffic and pays for one
(10,000 sats of calls) only when the gain clears its own noise; a watchdog that re-measures accepted learnings, looks at
their parents, and challenges what it can show is fake; bounty posters who measure claims on their own hidden evals;
licence buyers whose learnings name the traces they used. Honest validators report the true gain plus sampling noise
(600-problem eval sets, paired standard error 0.026), audit 20 parents and spend 50 sats of GPU time a measurement;
bribed ones report whatever the attacker says. P&L counts the sats the attack moved (spendable, in escrow for it, and
staked), less GPU time. "vs honest" compares with the same trainer doing honest work (or, for a validator, with the
same one measuring). Results are in sats, with a dollar equivalent at $85,962 a bitcoin (Coinbase spot, 2026-10-05)
that is only approximate. v0.5's attacks on the token (pool pumps, dumps, TWAP lag, cheap credits, inflated mints,
curve gaming, coin pump-and-dump, the operator's emission share) are gone with the token.
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node")]
from traceex import Trace, Learning, attest  # noqa: E402
from sats import SatsExchange, Params, attestation_digest, decoy_digest, tolerance  # noqa: E402

A = lambda c: "0x" + c * 40
V = lambda i: "0x" + f"{0xa0 + i:02x}" * 20
ATTACKER, HONEST, WATCHDOG, BACKER, USER, POSTER, BUYER, OPERATOR = (A(c) for c in "6bd89743")
FEES = "0x" + "0f" * 20                                    # the honest node operator
N_VALIDATORS, TRUE_GAIN, GPU, SE = 7, 0.08, 50_000, 0.026  # 50 sats (about $0.04) of GPU time per honest measurement
STAKE = 15_000_000
TASK = "Write a function number {} that adds two numbers."
BTC_USD = 85_962                                           # dollars a bitcoin, for the approximate dollar column
DECOY_LOG = []                                             # honest validators' decoy measurements: (missed, slashed)


def sats(m):
    """msats as signed sats: '+58 sats', '-0.116 sats', '-720,530 sats'."""
    v = abs(m) / 1000
    body = f"{v:,.0f}" if v >= 100 else (f"{v:,.2f}" if v >= 1 else f"{v:.3f}")
    return f"{'-' if m < 0 else '+'}{body} sats"


def usd(m):
    """msats as approximate signed dollars at BTC_USD (sub-cent: 6 places)."""
    d = abs(m) * BTC_USD / 1e11
    return f"{'-' if m < 0 else '+'}${d:,.{6 if 0 < d < 0.01 else 2}f}"


both = lambda m: f"{sats(m)} ({usd(m)})"


def fund(ex, who, msats):
    ex.db.execute("INSERT OR REPLACE INTO grants VALUES (?,?,?,?)", (who, msats, "sim", "sim"))
    ex.db.commit()


def node(seed=7, attacker_validators=0, lazy=()):
    """A node with 7 validators: the first `attacker_validators` answer to the attacker; `lazy` ones never measure.
    Licences clear at no less than 2,000 sats."""
    ex = SatsExchange(":memory:", test_credits=30_000_000, beacon_delay=1, reserve_msats=2_000_000,
                      params=Params(quorum=3), fee_to=FEES)
    ex.seed = seed
    vals = [V(i) for i in range(N_VALIDATORS)]
    for v in vals:
        fund(ex, v, 25_000_000)
        ex.register_validator(v, STAKE)
    for who in (ATTACKER, HONEST, WATCHDOG, BACKER, USER, POSTER, BUYER, OPERATOR):
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
    att = attest(by or V(0), eval_set, "pass@1", 0.30, round(0.30 + claim, 4))
    L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": name, "hash": None},
                       parents=[(p, 1) for p in parents], trainer=trainer, attestation=att, per_call_msats=200)
    return ex.register_learning(L)["id"]


def validate(ex, lid, true_gain, claim, rnd=0, audit_bad=0.0):
    """Assigned validators commit, then reveal. Honest ones measure (the truth plus noise, a real parent audit, 50 sats of
    GPU); bribed ones report the attacker's number with a clean audit; lazy ones repeat the trainer's claim unmeasured."""
    reports, rng = {}, stream(ex, lid, f"round {rnd}")
    for v in ex.verdict(lid)["assigned"]:
        noise, before = rng.gauss(0, SE), 0.30 + rng.uniform(-0.03, 0.03)     # drawn for every seat, in order
        honest = v not in ex.corrupt and v not in ex.lazy
        if honest:
            g = true_gain + noise
            ex.measured[v] = ex.measured.get(v, 0) + 1
        else:
            g = claim if v in ex.corrupt else ex._claimed_gain(lid)
        att = {"validator": v, "eval_set": f"sha256:{v[-4:]}{rnd}", "metric": "pass@1", "before": round(before, 4),
               "after": round(min(max(before + g, 0), 1), 4), "n": 600, "se": SE,
               "audit": {"checked": 20, "bad": round(20 * audit_bad) if honest else 0}}
        ex.commit(lid, v, attestation_digest(att, "s" + v), rnd)
        reports[v] = att
    for v, att in reports.items():
        ex.reveal(lid, v, att, "s" + v, rnd)


def watch(ex, lid, true_gain, junk=0.0):
    """The honest watchdog: if an accepted learning shows no gain when it measures it, or its parents are padding, it
    stakes 2,000 sats on a challenge; fresh validators are drawn from the next beacon."""
    v = ex.verdict(lid)
    fake = true_gain < ex.p.min_gain
    padded = junk > ex.p.audit_max_bad and (v["audit_bad"] or 0) <= ex.p.audit_max_bad
    if v["status"] == "accepted" and (fake or padded):
        ex.challenge(lid, WATCHDOG)
        ex.settle()
        validate(ex, lid, true_gain, claim=0.30, rnd=ex.verdict(lid)["round"], audit_bad=junk)


def use(ex, lid, true_gain, calls=50_000):
    """An honest user tries an accepted learning on its own traffic first (its own standard error: 0.015) and pays for
    10,000 sats of calls only when the gain clears twice its noise."""
    if ex.verdict(lid)["status"] == "accepted" and true_gain + stream(ex, lid, "user").gauss(0, 0.015) - 0.03 >= ex.p.min_gain:
        ex.usage({"learning": lid, "consumer": USER, "calls": calls})
        return True
    return False


def worth(ex, accounts):
    """Sats an account set holds: spendable, waiting in escrow for it (vesting), and staked."""
    total = 0
    for a in accounts:
        w = ex.wallet(a)
        total += w["balance_msats"] + w["vesting_msats"] + w["stake_msats"]
    return total


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


def run(attack, attacker_validators=0, seed=7, lazy=(), mine=None):
    ex = node(seed, attacker_validators, lazy)
    mine = mine or [ATTACKER] + sorted(ex.corrupt | ex.lazy)
    start = worth(ex, mine)
    out = attack(ex)
    note, override = out if isinstance(out, tuple) else (out, None)
    for _ in range(ex.p.vest_epochs + 2):                  # let escrow, bonds, licences, bounties and challenges play out
        ex.settle()
    assert ex.audit()["balanced"], ex.audit()
    gpu = sum(ex.measured.get(v, 0) for v in mine) * GPU
    pnl = override if override is not None else worth(ex, mine) - start - gpu
    return pnl, note, ex


# --- spam, copies, padding, wrapping -------------------------------------------------------------------------------
def trace_spam(ex):
    for i in range(1_000):
        ex.submit_trace(dict(trace(ATTACKER, f"junk task {i} that no model fails on", "attacker-tests@1")))
    return "1,000 junk traces; no learning uses them, nobody pays for them; each paid the 58-msat fee"


def stuff_lot(ex):
    real = parents_of(ex)
    for i in range(1_000):                                    # junk that claims the honest checker lands in its lot
        ex.submit_trace(dict(trace(ATTACKER, f"Write a function number {i} that returns its input.")))
    lot = next(x["lot"] for x in ex.lots()["lots"] if x["lot"].endswith("|unit-tests@1"))
    for who in (BUYER, BACKER):
        ex.bid({"lot": lot, "bidder": who, "price_msats": 2_000_000})
    ex.clear()
    lid = submit(ex, BUYER, real, TRUE_GAIN, "built-on-the-lot")        # one buyer's learning uses the 20 real traces
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    ex.direct_licence(lot, BACKER, real)                                 # the other names the traces it used
    return "1,000 junk traces stuffed into an honest lot; two 2,000-sat licences pay only the 20 traces the buyers used"


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
                + ("; the watchdog's audit challenge sends the escrowed parents' share back to the user"
                   if audit_bad == 0 else "; the parents' share goes back to the user"))
    return attack


def wrap_traces(ex):
    """Its real learning cites a learning of its own (terms: 100% to its trainer) wrapped around the honest traces,
    instead of the traces themselves, hoping their share flows through its wrapper to it. Generous: the wrapper passes
    validation too."""
    greedy = {"trainer": 1.0, "traces": 0.0, "checkers": 0.0, "validators": 0.0}
    L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": "wrapper", "hash": None},
                       parents=[(p, 1) for p in parents_of(ex)], trainer=ATTACKER, per_call_msats=200, split=greedy,
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


# --- fake learnings and bought validators --------------------------------------------------------------------------
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
           ", ".join(f"{outcomes.count(s)} {s}" for s in sorted(set(outcomes))) + "; no verdict pays anything"


def majority_wash(ex):
    own = parents_of(ex, producer=ATTACKER, checker="attacker-tests@1")
    for k in range(6):
        lid = submit(ex, ATTACKER, own, 0.30, f"fake-{k}")
        ex.settle()
        validate(ex, lid, 0.0, claim=0.30)
        if ex.verdict(lid)["status"] == "accepted":
            break
    if ex.verdict(lid)["status"] == "accepted":
        ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 500_000})   # 100,000 sats of its own usage
    return "gets its own fake accepted, then pays 100,000 sats to use it: every share comes out of its own payment"


def majority_bounty(ex):
    b = ex.post_bounty({"poster": POSTER, "path": "code", "eval_set": "sha256:hidden", "target": 0.35, "title": "theirs"})
    ex.pledge(b["id"], BACKER, 50_000_000)
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
    tries = "on the first try" if k == 0 else f"after {k + 1} tries"
    return f"its fake, attested on the bounty's eval by its own validator, is accepted {tries}; the claim on an " \
           f"honest 50,000-sat bounty: {got}"


def majority_grief(ex):
    parents, outcomes = parents_of(ex), []
    bonds = worth(ex, [HONEST])
    for k in range(4):
        lid = submit(ex, HONEST, parents, TRUE_GAIN, f"honest-{k}")
        ex.settle()
        validate(ex, lid, TRUE_GAIN, claim=0.0)                       # its validators score honest work at zero
        if ex.verdict(lid)["status"] == "accepted":                   # and it challenges whatever gets through
            ex.challenge(lid, ATTACKER)
            ex.settle()
            validate(ex, lid, TRUE_GAIN, claim=0.0, rnd=ex.verdict(lid)["round"])
        outcomes.append(ex.verdict(lid)["status"])
    lost = worth(ex, [HONEST]) - bonds
    return ("blocks 4 honest learnings (" + ", ".join(f"{outcomes.count(s)} {s}" for s in sorted(set(outcomes)))
            + f"): the honest trainer is down {sats(lost)[1:]}, and none of it reaches the attacker")


def challenge_grief(ex):
    """Without a majority: challenge three honest, accepted learnings, hoping to pause their payouts and claw back
    their escrowed shares. Fresh validators reproduce each gain, so each 2,000-sat stake is destroyed."""
    parents, lids = parents_of(ex), []
    for k in range(3):
        lids.append(accepted(ex, HONEST, parents, f"honest-{k}"))
    for lid in lids:
        use(ex, lid, TRUE_GAIN)
    ex.settle()
    outcomes = []
    for lid in lids:
        if ex.verdict(lid)["status"] != "accepted":
            continue
        ex.challenge(lid, ATTACKER)
        ex.settle()
        validate(ex, lid, TRUE_GAIN, claim=0.0, rnd=ex.verdict(lid)["round"])
        outcomes.append(ex.verdict(lid)["status"])
    return (f"challenges {len(outcomes)} honest learnings: " + ", ".join(f"{outcomes.count(s)} {s}"
                                                                       for s in sorted(set(outcomes)))
            + "; each failed challenge destroys its 2,000-sat stake, and the escrowed shares are paid once it ends")


def validator_work(ex):
    """24 learnings: 18 real (from an honest trainer; a user pays for each that helps) and 6 decoys the operator seals
    with their true gain, 0, behind a +15 point claim."""
    parents, struck, caught = parents_of(ex), 0, 0
    for k in range(24):
        if k % 4 == 3:
            fund(ex, OPERATOR, 200_000_000)
            att = attest(V(0), "sha256:claim", "pass@1", 0.30, 0.45)
            L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": f"d{k}", "hash": None},
                               parents=[(p, 1) for p in parents], trainer="0x" + f"{0xd0 + k:02x}" * 20,
                               attestation=att, per_call_msats=200)
            lid = ex.register_decoy(L, decoy_digest(0.0, f"salt{k}"), OPERATOR)["id"]
            ex.settle()
            validate(ex, lid, 0.0, claim=0.15)
            r = ex.unseal_decoy(lid, 0.0, f"salt{k}")
            struck += V(6) in r["struck"]
            caught += V(6) in r["caught"]
            for v in ex.verdict(lid)["assigned"]:                  # every honest measurement of a decoy, logged
                if v not in ex.lazy and v not in ex.corrupt:
                    DECOY_LOG.append((v in r["struck"], v in r["caught"]))
        else:
            lid = submit(ex, HONEST, parents, TRUE_GAIN, f"real-{k}")
            ex.settle()
            validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
            use(ex, lid, TRUE_GAIN)
    who = "repeats each claim instead of measuring it (saves 50 sats a time)" if V(6) in ex.lazy else "measures every learning"
    return f"{who} over 18 real learnings and 6 sealed decoys; struck on {struck}, slashed {caught} time(s)"


# --- paying yourself ---------------------------------------------------------------------------------------------------
def wash_usage(ex):
    lid = real_learning(ex)
    before = worth(ex, [ATTACKER])
    ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 500_000})       # 100,000 sats of usage paid to itself
    for _ in range(ex.p.vest_epochs + 1):
        ex.settle()
    return f"pays 100,000 sats to use its own real learning (its traces, its checker): it gets back every share but the " \
           f"validators' 5%; that money alone: {both(worth(ex, [ATTACKER]) - before)}"


def self_bounty(ex):
    b = ex.post_bounty({"poster": ATTACKER, "path": "code", "eval_set": "sha256:mine", "target": 0.35, "title": "mine"})
    ex.pledge(b["id"], ATTACKER, 50_000_000)
    lid = submit(ex, ATTACKER, parents_of(ex), TRUE_GAIN, "real")
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    use(ex, lid, TRUE_GAIN)
    if ex.verdict(lid)["status"] == "accepted":              # as poster, it confirms its own solution
        ex.claim_bounty(b["id"], lid, attest(ATTACKER, "sha256:mine", "pass@1", 0.30, 0.38))
    return "posts and pledges 50,000 sats to its own bounty, solves it with a real learning, confirms it as the poster: " \
           "the traces' 20%, checkers' 5% and validators' 5% are paid out of its own pledge"


def duplicate_bounty(ex):
    """An honest poster's bounty, backed by an honest backer. The attacker posts the same problem (same branch, failure
    and model) with its own eval set, hoping to solve its own copy cheaply and confirm the solve itself, so the honest
    backers' interest goes to its copy. The node merges the post: it backs the honest bounty."""
    honest = ex.post_bounty({"poster": POSTER, "path": "code/generate", "failure": "wrong_answer", "base_model": "qwen",
                             "eval_set": "sha256:hidden", "target": 0.9, "title": "theirs"})
    ex.pledge(honest["id"], BACKER, 20_000_000)
    dup = ex.post_bounty({"poster": ATTACKER, "path": "code/generate", "failure": "wrong_answer", "base_model": "qwen",
                          "eval_set": "sha256:mine", "target": 0.31, "title": "mine", "seed_sats": 1})
    lid = submit(ex, ATTACKER, parents_of(ex, n=5), 0.30, "cheap", eval_set="sha256:mine")
    ex.settle()
    validate(ex, lid, 0.0, claim=0.30)
    watch(ex, lid, 0.0)
    try:
        ex.claim_bounty(dup["id"], lid, attest(ATTACKER, "sha256:mine", "pass@1", 0.30, 0.60))
        got = "paid"
    except ValueError as e:
        got = f"refused ({str(e)[:40]}…)"
    return (f"its post {'was merged into' if dup.get('merged') else 'opened beside'} bounty #{honest['id']} (its 1-sat seed backs "
            f"the honest one); its claim with its own measurement: {got}")


def refund_gaming(ex):
    """Gaming the refund: a lure bounty with a target nobody reaches, pledged first with 1,000 sats, an honest backer's
    60,000 sats after it, and 1,000 one-msat pledges hoping refunds round up; then it lets it expire."""
    b = ex.post_bounty({"poster": ATTACKER, "path": "code", "eval_set": "sha256:lure", "target": 0.99, "title": "lure"})
    ex.pledge(b["id"], ATTACKER, 1_000_000)
    ex.pledge(b["id"], BACKER, 60_000_000)
    for _ in range(1_000):
        ex.pledge(b["id"], ATTACKER, 1)
    for _ in range(6):
        ex.settle()
    return ("lures 60,000 sats of an honest backer's pledges, pledges 1,000 times 1 msat, lets the bounty expire: each "
            "pledge comes back exactly, and each paid its fee")


def honest_trainer(ex):
    lid = submit(ex, ATTACKER, parents_of(ex), TRUE_GAIN, "real")
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    use(ex, lid, TRUE_GAIN)
    return "for comparison: the same trainer, real learning, real parents, one real user paying 10,000 sats"


def fee_evasion(ex):
    """Pack many fixes into each transaction, batch usage into one report, then try to empty the wallet before the
    fee is taken."""
    for i in range(25):
        ex.submit_trace(dict(trace(ATTACKER, " ".join(f"fix {i}-{j}: add numbers {j}" for j in range(400)),
                                   "attacker-tests@1")))                  # 10,000 fixes, 25 fees
    lid = real_learning(ex)
    ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 5_000})       # one report, 5,000 calls
    return ("10,000 fixes in 25 traces (25 fees; they earn as 25 parents, not 10,000), 5,000 calls in one report (one "
            "fee, every call still paid); the fee is taken with each transaction, so there is nothing to drain before "
            "a bill; the biggest transaction allowed (64 KB) still pays about 11x its electricity")


def sybil(ex):
    """Split across many accounts: 10 trainer identities with 10 learnings of 20 traces each, 10 consumer accounts
    paying 10,000 sats each to them, and 5 validators at the minimum stake; shares and draws are all linear."""
    trainers = ["0x" + f"{0x60 + i:02x}" * 20 for i in range(10)]
    payers = ["0x" + f"{0x70 + i:02x}" * 20 for i in range(10)]
    sybils = ["0x" + f"{0x80 + i:02x}" * 20 for i in range(5)]
    for who in trainers + payers + sybils:
        fund(ex, who, 50_000_000)
    for v in sybils:
        ex.register_validator(v, 10_000_000)
    ex.corrupt |= set(sybils)
    ex.register_checker("sybil-tests", trainers[0])
    lids = []
    for i, t in enumerate(trainers):
        own = [ex.submit_trace(dict(trace(t, f"{TASK.format(j)} variant {i}", "sybil-tests@1")))["id"] for j in range(20)]
        L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": f"sybil-{i}", "hash": None},
                           parents=[(p, 1) for p in own], trainer=t, per_call_msats=200,
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
    return f"10 trainers, 10 payers ({paid} paid 10,000 sats each to the sybils' own accepted learnings) and 5 minimum-stake validators"


ATTACKS = [
    ("Trace spam", trace_spam, 0),
    ("Stuff an honest lot with junk", stuff_lot, 0),
    ("Copy honest traces", copies(False), 0),
    ("Reworded copies of honest traces", copies(True), 0),
    ("Fake learning", fake_learning, 0),
    ("Fake learning, 1 bribed validator", fake_learning, 1),
    ("Fake learnings, 2 of 7 bribed", fake_learnings_with_bribes, 2),
    ("Wash usage (self-dealing)", wash_usage, 0),
    ("Self-funded bounty", self_bounty, 0),
    ("Duplicate bounty", duplicate_bounty, 0),
    ("Bounty refund gaming", refund_gaming, 0),
    ("Pad a real learning, honest audits", dilution(0.9), 0),
    ("Pad a real learning, lazy audits", dilution(0.0), 0),
    ("Wrap honest traces in its own learning", wrap_traces, 0),
    ("Lazy validator (never measures)", validator_work, 0),
    ("Challenge griefing", challenge_grief, 0),
    ("Fee evasion: pack, batch, drain", fee_evasion, 0),
    ("Sybil: split across 25 accounts", sybil, 0),
    ("(honest trainer, for scale)", honest_trainer, 0),
    ("Majority: fake learnings", majority_fakes, 4),
    ("Majority: wash its own fake", majority_wash, 4),
    ("Majority: take an honest bounty", majority_bounty, 4),
    ("Majority: block honest work", majority_grief, 4),
]
REAL = {"Wash usage (self-dealing)", "Self-funded bounty", "Pad a real learning, honest audits",
        "Pad a real learning, lazy audits", "Wrap honest traces in its own learning", "(honest trainer, for scale)"}


def play(name, attack, bribed, seed=7, honest=None):
    """One run of one attack: (P&L, vs honest, what happened)."""
    if attack is validator_work:                              # the same validator, lazy and then honest
        pnl, note, _ = run(attack, lazy=(V(6),), mine=[V(6)], seed=seed)
        return pnl, pnl - run(attack, mine=[V(6)], seed=seed)[0], note
    if attack is sybil:                                       # all the sybil's accounts together
        ex = node(seed)
        note = attack(ex)
        mine = ex.sybil_accounts
        start = len(mine) * 50_000_000
        for _ in range(ex.p.vest_epochs + 2):
            ex.settle()
        assert ex.audit()["balanced"]
        pnl = worth(ex, mine) - start
        return pnl, pnl, note
    pnl, note, _ = run(attack, bribed, seed=seed)
    return pnl, (pnl - honest if name in REAL else pnl), note


def decoy_false_positives(measurements=1_000_000, seed=2026):
    """The decoy test on honest validators: how often an honest measurement lands beyond 4 of its own standard errors
    from the sealed truth (a strike), and so how often two strikes inside the window would slash an honest validator
    (v0.6) against how often one would (v0.5). Same noise model as the attacks: 600-item evals, paired SE 0.026."""
    rng = random.Random(seed)
    p = Params()
    misses = 0
    for _ in range(measurements):
        before = 0.30 + rng.uniform(-0.03, 0.03)
        g = 0.0 + rng.gauss(0, SE)
        att = {"before": round(before, 4), "after": round(min(max(before + g, 0), 1), 4), "n": 600, "se": SE}
        g = att["after"] - att["before"]
        misses += abs(g) > tolerance(att, p.decoy_z, p.min_tol)
    q = misses / measurements
    exact = math.erfc(p.decoy_z / math.sqrt(2))                  # the normal tail beyond 4 standard errors
    rate = max(q, exact)
    window = {k: (1 - (1 - rate) ** k, 1 - (1 - rate) ** k - k * rate * (1 - rate) ** (k - 1)) for k in (10, 100, 1_000)}
    return {"measurements": measurements, "misses": misses, "miss_rate": q, "normal_tail": exact, "window": window}


def main(argv=None):
    """python attacks.py: every attack once (seed 7), with what happened. --seeds 30: each attack on 30 different
    random draws (validators, noise, users), with the mean result, the best run for the attacker, and how often it paid.
    --decoys: the decoy test's false-positive rate."""
    argv = sys.argv[1:] if argv is None else argv
    if "--decoys" in argv:
        d = decoy_false_positives()
        print(f"honest decoy measurements simulated: {d['measurements']:,}; beyond 4 standard errors (a strike): "
              f"{d['misses']} ({d['miss_rate']:.2e}; the normal tail is {d['normal_tail']:.2e})")
        for k, (one, two) in d["window"].items():
            print(f"  {k:>5} decoy measurements in a window: slashed by v0.5's one-miss rule {one:.2e}, by v0.6's "
                  f"two-strike rule {two:.2e}")
        return d
    seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 1
    print(__doc__.split("\n\n")[1].strip(), "\n")
    rows = []
    DECOY_LOG.clear()
    if seeds == 1:
        honest = run(honest_trainer)[0]
        print(f"{'attack':40} {'P&L, sats':>14} {'vs honest':>14} {'about $':>10}   what happened")
        for name, attack, bribed in ATTACKS:
            pnl, extra, note = play(name, attack, bribed, honest=honest)
            rows.append((name, pnl, extra, note))
            print(f"{name:40} {sats(pnl):>14} {sats(extra):>14} {usd(extra):>10}   {note}")
    else:                      # each run against its honest twin: same seed, and its real learning draws the same noise
        honest = {s: run(honest_trainer, seed=s)[0] for s in range(1, seeds + 1)}
        print(f"{'attack':40} {'mean vs honest':>18} {'about $':>10} {'best run':>16} {'runs it paid':>13}")
        for name, attack, bribed in ATTACKS:
            xs = [play(name, attack, bribed, s, honest[s])[1] for s in range(1, seeds + 1)]
            rows.append((name, sum(xs) / len(xs), max(xs), sum(x > 0 for x in xs)))
            print(f"{name:40} {sats(sum(xs) / len(xs)):>18} {usd(sum(xs) / len(xs)):>10} {sats(max(xs)):>16} "
                  f"{sum(x > 0 for x in xs):>6} of {seeds}")
    if DECOY_LOG:
        print(f"\nhonest validators' decoy measurements in these runs: {len(DECOY_LOG)}; strikes "
              f"{sum(m for m, _ in DECOY_LOG)}; slashes {sum(s for _, s in DECOY_LOG)}")
    print("\nEvery strategy loses, including owning most of the validator stake: a majority can still block honest work, "
          "because it controls the vote, but no verdict moves money to it. Every payout is a split of a real payment, "
          "so paying yourself returns at most what you paid, less every share that isn't yours and the fee; bounties "
          "pay on the poster's own measurement and refund exactly what each backer pledged; licence money pays only "
          "the traces the buyer's own learnings use; and forfeits are destroyed.")
    return rows


if __name__ == "__main__":
    main()
