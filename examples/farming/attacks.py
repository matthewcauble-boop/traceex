"""Farming attacks against the coin economy, each run on a real node (node/coin.py), with its profit or loss.

    python examples/farming/attacks.py

Setup per attack: a fresh testnet node with 7 staked validators (1,500 TXC each), quorum 3, validators drawn from the
beacon published after each submission, and honest neighbours: a producer whose traces real learnings use; a user who
tries every accepted learning on its own traffic and pays for one ($10 of calls) only when the gain clears its own noise;
a watchdog that re-measures accepted learnings, looks at their parents, and challenges what it can show is fake; bounty
posters who measure claims on their own hidden evals; licence buyers whose learnings name the traces they used. Honest
validators report the true gain plus sampling noise (600-problem eval sets, paired standard error 0.026), audit 20
parents and spend $0.05 of GPU time a measurement; bribed ones report whatever the attacker says. P&L counts the
dollars and coins (liquid, vesting, staked) the attack moved, coins at the price when the run began, less GPU time, so
nobody shows a profit just because someone else bought TXC. "vs honest" compares with the same trainer doing honest
work (or, for a validator, with the same validator measuring).
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node")]
from traceex import Trace, Learning, attest  # noqa: E402
from coin import CoinExchange, Params, UNIT, attestation_digest, decoy_digest  # noqa: E402

A = lambda c: "0x" + c * 40
V = lambda i: "0x" + f"{0xa0 + i:02x}" * 20
ATTACKER, HONEST, WATCHDOG, BACKER, USER, POSTER, BUYER, OPERATOR = (A(c) for c in "6bd89743")
N_VALIDATORS, TRUE_GAIN, GPU = 7, 0.08, 50_000            # $0.05 of GPU time per honest measurement
TASK = "Write a function number {} that adds two numbers."
usd = lambda m: f"{'-' if m < 0 else '+'}${abs(m) / 1e6:,.{6 if 0 < abs(m) < 10_000 else 2}f}"   # sub-cent: 6 places


def fund(ex, who, micros):
    ex.db.execute("INSERT OR REPLACE INTO grants VALUES (?,?,?,?)", (who, micros, "sim", "sim"))
    ex.db.commit()


def node(seed=7, attacker_validators=0, lazy=()):
    """A node with 7 validators: the first `attacker_validators` answer to the attacker; `lazy` ones never measure.
    Licences clear at no less than $2."""
    ex = CoinExchange(":memory:", test_credits=25_000_000, beacon_delay=1, reserve_micros=2_000_000,
                      params=Params(quorum=3))
    ex.seed = seed
    vals = [V(i) for i in range(N_VALIDATORS)]
    for v in vals:
        fund(ex, v, 25_000_000)
        ex.swap(v, "buy", 16_000_000)
        ex.register_validator(v, 1_500 * UNIT)
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
    ex.swap(trainer, "buy", 6_000_000)                    # enough TXC for the 500 TXC bond
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
    stakes 200 TXC on a challenge; fresh validators are drawn from the next beacon."""
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


def worth(ex, accounts, price):
    total = 0
    for a in accounts:
        w = ex.wallet(a)
        total += w["balance_micros"] + (w["coin_units"] + w["vesting_units"] + w["stake_units"]) * price // UNIT
    return total


def run(attack, attacker_validators=0, seed=7, lazy=(), mine=None):
    ex = node(seed, attacker_validators, lazy)
    mine = mine or [ATTACKER] + sorted(ex.corrupt | ex.lazy)
    price = ex.price()
    start = worth(ex, mine, price)
    note = attack(ex)
    for _ in range(ex.p.vest_epochs + 2):                  # let vesting, bonds, licences and challenges play out
        ex.settle()
    assert ex.audit()["balanced"]
    gpu = sum(ex.measured.get(v, 0) for v in mine) * GPU
    return worth(ex, mine, price) - start - gpu, note, ex


# --- the attacks --------------------------------------------------------------------------------------------------
def trace_spam(ex):
    for i in range(1_000):
        ex.submit_trace(dict(trace(ATTACKER, f"junk task {i} that no model fails on", "attacker-tests@1")))
    return "1,000 junk traces; no learning uses them, nobody pays for them"


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
    return "1,000 junk traces stuffed into an honest lot; two $2 licences go to the 20 traces the buyers used"


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
    own = parents_of(ex, producer=ATTACKER, checker="attacker-tests@1")     # every share of it is the attacker's
    lid = submit(ex, ATTACKER, own, TRUE_GAIN, "real")
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    ex.settle()
    price = ex.price()
    before = worth(ex, [ATTACKER], price)
    ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 500_000})       # $100 of usage paid to itself
    for _ in range(ex.p.vest_epochs + 1):
        ex.settle()
    return f"$100 of usage of its own real learning (its traces, its checker); that money alone: " \
           f"{usd(worth(ex, [ATTACKER], price) - before)}"


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
    ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 500_000})       # $100 of usage of its own accepted fake
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


ATTACKS = [
    ("Trace spam", trace_spam, 0),
    ("Stuff an honest lot with junk", stuff_lot, 0),
    ("Copy honest traces", copies(False), 0),
    ("Reworded copies of honest traces", copies(True), 0),
    ("Fake learning", fake_learning, 0),
    ("Fake learning, 1 bribed validator", fake_learning, 1),
    ("Fake learnings, 2 of 7 bribed", fake_learnings_with_bribes, 2),
    ("Wash usage", wash_usage, 0),
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
]
REAL = {"Wash usage", "Self-funded bounty", "Pad a real learning, honest audits", "Pad a real learning, lazy audits",
        "Wrap honest traces in its own learning", "(honest trainer, for scale)"}


def play(name, attack, bribed, seed=7, honest=None):
    """One run of one attack: (P&L, vs honest, what happened)."""
    if attack is validator_work:                              # the same validator, lazy and then honest
        pnl, note, _ = run(attack, lazy=(V(6),), mine=[V(6)], seed=seed)
        return pnl, pnl - run(attack, mine=[V(6)], seed=seed)[0], note
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
        print(f"{'attack':40} {'P&L':>10} {'vs honest':>10}   what happened")
        for name, attack, bribed in ATTACKS:
            pnl, extra, note = play(name, attack, bribed, honest=honest)
            rows.append((name, pnl, extra, note))
            print(f"{name:40} {usd(pnl):>10} {usd(extra):>10}   {note}")
    else:                      # each run against its honest twin: same seed, and its real learning draws the same noise
        honest = {s: run(honest_trainer, seed=s)[0] for s in range(1, seeds + 1)}
        print(f"{'attack':40} {'mean vs honest':>15} {'best run':>10} {'runs it paid':>13}")
        for name, attack, bribed in ATTACKS:
            xs = [play(name, attack, bribed, s, honest[s])[1] for s in range(1, seeds + 1)]
            rows.append((name, sum(xs) / len(xs), max(xs), sum(x > 0 for x in xs)))
            print(f"{name:40} {usd(sum(xs) / len(xs)):>15} {usd(max(xs)):>10} {sum(x > 0 for x in xs):>6} of {seeds}")
    print("\nEvery strategy loses, including owning most of the validator stake: a majority can still block honest work, "
          "because it controls the vote, but no verdict moves money to it. Coins are minted only to match what users "
          "pay, bounties pay on the poster's own measurement, licence money follows the buyer's own learnings, and "
          "forfeits burn.")
    return rows


if __name__ == "__main__":
    main()
