"""Farming attacks against the coin economy, each run on a real node (node/coin.py), with its profit or loss.

    python examples/farming/attacks.py

Setup per attack: a fresh testnet node, 7 staked validators (1,500 TXC each), quorum 3, validators drawn from the
beacon published after each submission, an honest watchdog that re-measures every accepted learning (and looks at its
parents) and challenges it if the gain or the parents aren't real. Honest validators report the true gain plus
sampling noise (600-problem eval sets, paired standard error 0.026) and audit 20 parents; bribed ones report whatever
the attacker claims. P&L counts dollars, coins, vesting coins and stake at the pool's final price, before the price
impact of selling (generous to the attacker). "vs honest" compares with the same trainer doing honest work.
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node")]
from traceex import Trace, Learning, attest  # noqa: E402
from coin import CoinExchange, Params, UNIT, attestation_digest  # noqa: E402

A = lambda c: "0x" + c * 40
V = lambda i: "0x" + f"{0xa0 + i:02x}" * 20
ATTACKER, HONEST, WATCHDOG, BACKER = A("6"), A("b"), A("d"), A("8")
N_VALIDATORS, TRUE_GAIN = 7, 0.08
usd = lambda m: f"{'-' if m < 0 else '+'}${abs(m) / 1e6:,.2f}"


def fund(ex, who, micros):
    ex.db.execute("INSERT OR REPLACE INTO grants VALUES (?,?,?,?)", (who, micros, "sim", "sim"))
    ex.db.commit()


def node(rng, attacker_validators=0):
    """A node with 7 validators; the first `attacker_validators` of them answer to the attacker."""
    ex = CoinExchange(":memory:", test_credits=25_000_000, beacon_delay=1, params=Params(quorum=3))
    vals = [V(i) for i in range(N_VALIDATORS)]
    for v in vals:
        fund(ex, v, 25_000_000)
        ex.swap(v, "buy", 16_000_000)
        ex.register_validator(v, 1_500 * UNIT)
    for who in (ATTACKER, HONEST, WATCHDOG, BACKER):
        fund(ex, who, 200_000_000)
    ex.register_checker("unit-tests", HONEST)
    ex.register_checker("attacker-tests", ATTACKER)
    ex.corrupt, ex.rng = set(vals[:attacker_validators]), rng
    return ex


def trace(producer, text, checker="unit-tests@1"):
    return Trace.from_fix(task="code.python", base_model="qwen", input=text, model_output={"code": "bad"},
                          verified_output={"code": "good " + text[-12:]}, checker=checker, producer=producer,
                          created="2026-10-04T00:00:00Z", privacy="open", failure_modes={"code": "wrong_answer"})


def submit(ex, trainer, parents, claim, name):
    ex.swap(trainer, "buy", 6_000_000)
    att = attest(V(0), "sha256:claim", "pass@1", 0.30, 0.30 + claim)
    L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": name, "hash": None},
                       parents=[(p, 1) for p in parents], trainer=trainer, attestation=att, per_call_micros=200)
    return ex.register_learning(L)["id"]


def validate(ex, lid, true_gain, claim, rnd=0, audit_bad=0.0):
    """Assigned validators commit, then reveal. Honest: the truth plus noise, and a real parent audit. Bribed: the
    attacker's claim and a clean audit."""
    reports = {}
    for v in ex.verdict(lid)["assigned"]:
        g = claim if v in ex.corrupt else true_gain + ex.rng.gauss(0, 0.026)
        before = 0.30 + ex.rng.uniform(-0.03, 0.03)
        att = {"validator": v, "eval_set": f"sha256:{v[-4:]}{rnd}", "metric": "pass@1", "before": round(before, 4),
               "after": round(min(max(before + g, 0), 1), 4), "n": 600, "se": 0.026,
               "audit": {"checked": 20, "bad": 0 if v in ex.corrupt else round(20 * audit_bad)}}
        ex.commit(lid, v, attestation_digest(att, "s" + v), rnd)
        reports[v] = att
    for v, att in reports.items():
        ex.reveal(lid, v, att, "s" + v, rnd)


def watch(ex, lid, true_gain, junk=0.0):
    """The honest watchdog: if a learning shows no gain when it measures it, or its parents are padding, it challenges."""
    v = ex.verdict(lid)
    fake = true_gain < ex.p.min_gain
    padded = junk > ex.p.audit_max_bad and (v["audit_bad"] or 0) <= ex.p.audit_max_bad
    if v["status"] == "accepted" and (fake or padded):
        ex.swap(WATCHDOG, "buy", 5_000_000)
        ex.challenge(lid, WATCHDOG)
        ex.settle()                                       # fresh validators are drawn from the next beacon
        validate(ex, lid, true_gain, claim=0.30, rnd=ex.verdict(lid)["round"], audit_bad=junk)


def worth(ex, accounts):
    total = 0
    for a in accounts:
        w = ex.wallet(a)
        total += w["balance_micros"] + (w["coin_units"] + w["vesting_units"] + w["stake_units"]) * ex.price() // UNIT
    return total


def honest_parents(ex, n=20):
    return [ex.submit_trace(dict(trace(HONEST, f"Write a function number {i} that adds two numbers.")))["id"]
            for i in range(n)]


def run(attack, attacker_validators=0, seed=7):
    ex = node(random.Random(seed), attacker_validators)
    mine = [ATTACKER] + sorted(ex.corrupt)
    start = worth(ex, mine)
    note = attack(ex)
    for _ in range(ex.p.vest_epochs + 2):                  # let vesting, bonds and challenges play out
        ex.settle()
    assert ex.audit()["balanced"]
    return worth(ex, mine) - start, note, ex


# --- the attacks --------------------------------------------------------------------------------------------------
def trace_spam(ex):
    for i in range(1_000):
        ex.submit_trace(dict(trace(ATTACKER, f"junk task {i} that no model fails on", "attacker-tests@1")))
    return "1,000 junk traces; nothing that passes validation ever cites them"


def copy_traces(ex):
    originals = honest_parents(ex, 20)
    copies = [ex.submit_trace(dict(trace(ATTACKER, f"write  a function NUMBER {i} that adds two numbers.")))["id"]
              for i in range(20)]
    lid = submit(ex, HONEST, originals + copies, TRUE_GAIN, "honest-with-copies")     # even if a trainer cites them
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    return "20 near-copies of honest traces, cited by a real learning; copies pay the original producer"


def fake_learning(ex):
    parents = honest_parents(ex, 5)
    lid = submit(ex, ATTACKER, parents, 0.30, "fake")
    ex.settle()
    validate(ex, lid, 0.0, claim=0.30)
    watch(ex, lid, 0.0)
    return f"claims +30 points, true gain 0; verdict: {ex.verdict(lid)['status']}"


def fake_learnings_with_bribes(ex):
    parents, outcomes = honest_parents(ex, 5), []
    for k in range(12):                                       # keep resubmitting, hoping to draw both bribed validators
        lid = submit(ex, ATTACKER, parents, 0.30, f"fake-{k}")
        ex.settle()
        validate(ex, lid, 0.0, claim=0.30)
        watch(ex, lid, 0.0)
        outcomes.append(ex.verdict(lid)["status"])
    return "12 fake learnings, 2 of 7 validators bribed: " + ", ".join(f"{outcomes.count(s)} {s}" for s in sorted(set(outcomes)))


def wash_usage(ex):
    parents = honest_parents(ex, 5)
    lid = submit(ex, ATTACKER, parents, TRUE_GAIN, "real")
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    ex.settle()                                               # its honest acceptance reward is granted here...
    baseline = worth(ex, [ATTACKER])
    ex.usage({"learning": lid, "consumer": ATTACKER, "calls": 500_000})       # ...then $100 of usage paid to itself
    ex.settle()
    return f"$100 of usage of its own learning; that epoch alone: {usd(worth(ex, [ATTACKER]) - baseline)}"


def self_bounty(ex):
    parents = honest_parents(ex, 5)
    b = ex.post_bounty({"poster": ATTACKER, "path": "code", "eval_set": "sha256:claim", "target": 0.35, "title": "mine"})
    ex.buy_coins(b["id"], ATTACKER, 50_000_000)
    lid = submit(ex, ATTACKER, parents, TRUE_GAIN, "real")
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    ex.claim_bounty(b["id"], lid)
    return "posts and backs its own bounty with $50, solves it with a real learning"


def dilution(audit_bad):
    def attack(ex):
        real = honest_parents(ex, 20)
        junk = [ex.submit_trace(dict(trace(ATTACKER, f"padding trace {i}", "attacker-tests@1")))["id"] for i in range(200)]
        lid = submit(ex, ATTACKER, real + junk, TRUE_GAIN, "padded")
        ex.settle()
        validate(ex, lid, TRUE_GAIN, TRUE_GAIN, audit_bad=audit_bad)
        if audit_bad == 0:                                # lazy validators passed it; the watchdog looks at the parents
            ex.settle()
            watch(ex, lid, TRUE_GAIN, junk=200 / 220)
        return (f"real learning padded with 200 own junk parents; validators' audit finds {audit_bad:.0%} bad"
                + ("" if audit_bad else "; the watchdog's audit challenge catches it"))
    return attack


def honest_trainer(ex):
    lid = submit(ex, ATTACKER, honest_parents(ex, 20), TRUE_GAIN, "honest")
    ex.settle()
    validate(ex, lid, TRUE_GAIN, TRUE_GAIN)
    return "for comparison: the same trainer, real learning, real parents"


def majority(ex):
    parents = honest_parents(ex, 5)
    for k in range(6):
        lid = submit(ex, ATTACKER, parents, 0.30, f"fake-{k}")
        ex.settle()
        validate(ex, lid, 0.0, claim=0.30)
        watch(ex, lid, 0.0)
    return "owns 4 of 7 validator seats (57% of stake); 6 fake learnings"


ATTACKS = [
    ("Trace spam", trace_spam, 0),
    ("Copy honest traces", copy_traces, 0),
    ("Fake learning", fake_learning, 0),
    ("Fake learning, 1 bribed validator", fake_learning, 1),
    ("Fake learnings, 2 of 7 bribed", fake_learnings_with_bribes, 2),
    ("Wash usage", wash_usage, 0),
    ("Self-funded bounty", self_bounty, 0),
    ("Pad a real learning, honest audits", dilution(0.9), 0),
    ("Pad a real learning, lazy audits", dilution(0.0), 0),
    ("(honest trainer, for scale)", honest_trainer, 0),
    ("Majority of validator stake", majority, 4),
]


REAL = {"Wash usage", "Self-funded bounty", "Pad a real learning, honest audits", "Pad a real learning, lazy audits",
        "(honest trainer, for scale)"}


def main():
    print(__doc__.split("\n\n")[1].strip(), "\n")
    honest = run(honest_trainer)[0]
    rows = []
    print(f"{'attack':36} {'P&L':>11} {'vs honest':>11}   what happened")
    for name, attack, bribed in ATTACKS:
        pnl, note, ex = run(attack, bribed)
        extra = pnl - honest if name in REAL else pnl
        rows.append((name, pnl, extra, note))
        print(f"{name:36} {usd(pnl):>11} {usd(extra):>11}   {note}")
    seats = 4 * 1_500 * UNIT * ex.price() // UNIT
    per_epoch = int(ex.emission() * ex.p.pool_improve * ex.p.max_share_per_learning) * ex.price() // UNIT
    print(f"\nThe one way through is a majority of validator stake, as in any proof-of-stake system. On this testnet that "
          f"is 4 seats of 1,500 TXC (about {usd(seats)[1:]} at the pool price), against at most {usd(per_epoch)[1:]} of "
          f"emissions per fake learning per epoch. Mainnet needs honest stake worth many epochs of emissions.")
    return rows


if __name__ == "__main__":
    main()
