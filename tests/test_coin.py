"""The coin economy (node/coin.py): python -m unittest tests.test_coin

Each test is one rule of the economy, most of them a rule that makes a farming strategy lose. Every test ends with
the books balanced: minted - burned == everything held by accounts and escrows + the pool's reserve."""
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node")]

from traceex import Trace, Learning, attest  # noqa: E402
from coin import CoinExchange, Params, UNIT, attestation_digest, decoy_digest  # noqa: E402

A = lambda c: "0x" + c * 40
V = lambda i: "0x" + f"{0xa0 + i:02x}" * 20           # validator addresses
TRAINER, PRODUCER, CONSUMER, CHALLENGER = A("c"), A("b"), A("e"), A("d")


def make_ex(validators=3, quorum=3, delay=0, **params):
    ex = CoinExchange(":memory:", test_credits=25_000_000, beacon_delay=delay, params=Params(quorum=quorum, **params))
    vals = []
    for i in range(validators):
        v = V(i)
        ex.faucet(v)
        ex.swap(v, "buy", 15_000_000)                     # $15 of TXC, then stake 1,000 of it
        ex.register_validator(v, 1_000 * UNIT)
        vals.append(v)
    for who in (TRAINER, PRODUCER, CONSUMER, CHALLENGER):
        ex.faucet(who)
    ex.register_checker("unit-tests", PRODUCER)
    return ex, vals


def trace(text="Write a function to add two numbers.\nassert add(1, 2) == 3", producer=PRODUCER, code="return a + b"):
    return Trace.from_fix(task="code.python", base_model="qwen", input=text, model_output={"code": "return a - b"},
                          verified_output={"code": code}, checker="unit-tests@1", producer=producer,
                          created="2026-10-04T00:00:00Z", privacy="open", failure_modes={"code": "wrong_answer"})


def learning(ex, parents, trainer=TRAINER, validator=None, eval_set="sha256:claimed", before=0.6, after=0.7, name="x",
             weights=None, split=None):
    ex.swap(trainer, "buy", 6_000_000)                    # enough TXC for the 500 TXC bond
    att = attest(validator or V(0), eval_set, "pass@1", before, after)
    L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": name, "hash": weights},
                       parents=[(p, 1) for p in parents], trainer=trainer, attestation=att, per_call_micros=200,
                       split=split)
    return ex.register_learning(L)["id"]


def validate(ex, lid, scores, rnd=None, bad=0):
    """scores: {validator: (before, after, n)}. Everyone commits, then everyone reveals (with a parent audit)."""
    atts = {}
    for v, (b, a, n) in scores.items():
        att = {"validator": v, "eval_set": f"sha256:private-{v[-4:]}-{rnd}", "metric": "pass@1", "before": b, "after": a,
               "n": n, "audit": {"checked": 10, "bad": bad}}
        ex.commit(lid, v, attestation_digest(att, "salt" + v[-2:]), rnd)
        atts[v] = att
    for v, att in atts.items():
        ex.reveal(lid, v, att, "salt" + v[-2:], rnd)
    return ex.verdict(lid)


def worth(ex, account, price=None):
    """An account's coins (liquid + vesting) at a fixed price (the pool's current one by default), plus its dollars."""
    w = ex.wallet(account)
    return w["balance_micros"] + (w["coin_units"] + w["vesting_units"]) * (price or ex.price()) // UNIT


class Pool(unittest.TestCase):
    def test_swaps_move_the_price_and_burn_a_fee(self):
        ex, _ = make_ex(validators=0)
        p0 = ex.price()
        r = ex.swap(CONSUMER, "buy", 5_000_000)
        self.assertGreater(ex.price(), p0)
        self.assertGreater(ex.coin_stats()["burned_units"], 0)
        ex.swap(CONSUMER, "sell", r["bought_units"])
        self.assertLess(ex.wallet(CONSUMER)["balance_micros"], 25_000_000)   # a round trip costs the fees
        self.assertTrue(ex.audit()["balanced"])
        with self.assertRaisesRegex(ValueError, "not enough TXC"):
            ex.swap(CONSUMER, "sell", 10 * UNIT)

    def test_trace_fee_is_burned_and_near_duplicates_pay_the_first_producer(self):
        ex, _ = make_ex(validators=0)
        burned = ex.coin_stats()["burned_units"]
        t1 = ex.submit_trace(dict(trace()))
        self.assertEqual(ex.wallet(PRODUCER)["balance_micros"], 25_000_000 - 500)
        self.assertGreater(ex.coin_stats()["burned_units"], burned)
        copier = A("7")
        ex.faucet(copier)
        t2 = ex.submit_trace(dict(trace(text="write a function to  add two numbers.\nassert add(1, 2) == 3", producer=copier)))
        self.assertEqual(t2["near_duplicate_of"], t1["id"])
        info, _ = ex._tree()
        self.assertEqual(info[t2["id"]]["producer"], PRODUCER)              # the copy's earnings go to the original
        self.assertTrue(ex.audit()["balanced"])

    def test_a_reworded_copy_of_a_fix_shares_the_originals_slot(self):
        ex, _ = make_ex(validators=0)
        code = "def add(a, b):\n    return a + b"
        t1 = ex.submit_trace(dict(trace("Write a function to add two numbers.", code=code)))
        copier = A("7")
        ex.faucet(copier)
        t2 = ex.submit_trace(dict(trace("Create a function that adds two numbers.", producer=copier, code=code)))
        self.assertEqual(t2.get("near_duplicate_of"), t1["id"])             # the same code, the task in other words
        t3 = ex.submit_trace(dict(trace("Write a function to subtract two numbers.", producer=copier,
                                        code="def sub(a, b):\n    return a - b")))
        self.assertNotIn("near_duplicate_of", t3)                           # a different fix is its own trace
        sk = lambda text: dict(trace(text, code="{CODE_1}"), privacy="skeleton")
        t4, t5 = ex.submit_trace(sk("Flight {NUM_1} departs {CITY_1}")), ex.submit_trace(sk("Your flight {NUM_1} from {CITY_1}"))
        self.assertNotIn("near_duplicate_of", t5)                           # skeletons: "{CODE}" identifies nothing
        self.assertNotEqual(t4["id"], t5["id"])


class Federation(unittest.TestCase):
    def setUp(self):
        self.ex, self.vals = make_ex()
        self.tid = self.ex.submit_trace(dict(trace()))["id"]

    def test_commit_then_reveal_and_the_median_decides(self):
        ex, vals = self.ex, self.vals
        lid = learning(ex, [self.tid])
        self.assertEqual(sorted(ex.verdict(lid)["assigned"]), sorted(vals))
        with self.assertRaises(PermissionError):                            # assignment is random, not chosen
            ex.commit(lid, A("9"), "x")
        att = {"validator": vals[0], "eval_set": "sha256:p", "metric": "m", "before": .6, "after": .7, "n": 200,
               "audit": {"checked": 1, "bad": 0}}
        ex.commit(lid, vals[0], attestation_digest(att, "s"))
        with self.assertRaisesRegex(ValueError, "every assigned validator has committed"):
            ex.reveal(lid, vals[0], att, "s")                               # nobody sees a score before all commit
        v = validate(ex, lid, {vals[1]: (.62, .71, 200), vals[2]: (.58, .70, 200)})
        self.assertEqual(v["status"], "pending")
        with self.assertRaisesRegex(ValueError, "does not match"):
            ex.reveal(lid, vals[0], dict(att, after=.9), "s")
        v = ex.reveal(lid, vals[0], att, "s")
        self.assertEqual(v["status"], "accepted")
        self.assertAlmostEqual(v["median_gain"], 0.10, places=6)
        self.assertTrue(ex.audit()["balanced"])

    def test_one_bought_validator_cannot_fake_a_gain_and_the_bond_burns(self):
        ex, vals = self.ex, self.vals
        lid = learning(ex, [self.tid])
        bought = vals[2]
        coins, burned = ex.wallet(TRAINER)["coin_units"], ex.coin_stats()["burned_units"]
        v = validate(ex, lid, {vals[0]: (.60, .60, 200), vals[1]: (.61, .60, 200), bought: (.60, .90, 200)})
        self.assertEqual(v["status"], "rejected")
        self.assertEqual(ex.wallet(TRAINER)["coin_units"], coins)                     # the 500 TXC bond is gone...
        self.assertEqual(ex.coin_stats()["burned_units"], burned + ex.p.learning_bond)  # ...to nobody: it burned
        self.assertEqual(sum(ex.validator(x)["slashed_units"] for x in vals), 0)       # disagreeing isn't a fault
        self.assertEqual([r["agreed"] for r in v["reveals"] if r["validator"] == bought], [False])
        self.assertEqual(ex.find_learnings()["count"], 0)
        self.assertTrue(ex.audit()["balanced"])

    def test_a_claim_far_beyond_the_measured_gain_forfeits_the_bond(self):
        ex, vals = self.ex, self.vals
        fake = learning(ex, [self.tid], before=.30, after=.60, name="claims-30")
        v = validate(ex, fake, {vals[0]: (.30, .32, 600), vals[1]: (.30, .31, 600), vals[2]: (.30, .60, 600)})
        self.assertEqual((v["status"], v["note"]), ("rejected", "overclaimed"))  # a bribed vote lifted the median; no use
        noisy = learning(ex, [self.tid], before=.30, after=.50, name="small-eval")  # claimed on 100 items, measured +8
        v = validate(ex, noisy, {x: (.30, .38, 600) for x in vals})
        self.assertEqual(v["status"], "accepted")                             # an honest, noisy claim is no overclaim
        self.assertTrue(ex.audit()["balanced"])

    def test_a_gain_inside_the_noise_is_inconclusive_not_punished(self):
        ex, vals = self.ex, self.vals
        lid = learning(ex, [self.tid])
        coins = ex.wallet(TRAINER)["coin_units"]
        v = validate(ex, lid, {vals[0]: (.6, .8, 10), vals[1]: (.7, .8, 10), vals[2]: (.6, .6, 10)})   # 10-item evals
        self.assertEqual(v["status"], "inconclusive")                       # +10 points on 10 items proves nothing
        self.assertTrue(all(r["agreed"] for r in v["reveals"]))
        self.assertEqual(sum(ex.validator(x)["slashed_units"] for x in vals), 0)
        self.assertEqual(ex.wallet(TRAINER)["coin_units"], coins + ex.p.learning_bond * 9 // 10)
        self.assertEqual(ex.settle()["granted"]["match"], 0)
        self.assertTrue(ex.audit()["balanced"])

    def test_every_reveal_audits_parents(self):
        ex, vals = self.ex, self.vals
        lid = learning(ex, [self.tid])
        att = {"validator": vals[0], "eval_set": "sha256:p", "metric": "m", "before": .6, "after": .7, "n": 200}
        for v in vals:
            ex.commit(lid, v, attestation_digest(dict(att, validator=v), "s"))
        with self.assertRaisesRegex(ValueError, "audits at least 1"):
            ex.reveal(lid, vals[0], att, "s")

    def test_validators_are_drawn_after_submission(self):
        ex, vals = make_ex(validators=7, delay=1)
        tid = ex.submit_trace(dict(trace()))["id"]
        lid = learning(ex, [tid])
        self.assertEqual(ex.verdict(lid)["assigned"], [])                    # unknown until the next beacon
        ex.settle()
        drawn = ex.verdict(lid)["assigned"]
        self.assertEqual(len(drawn), 3)
        self.assertTrue(set(drawn) <= set(vals))
        outsider = next(v for v in vals if v not in drawn)
        with self.assertRaises(PermissionError):
            ex.commit(lid, outsider, "x")

    def test_the_same_weights_cannot_be_registered_twice(self):
        ex = self.ex
        first = learning(ex, [self.tid], weights="sha256:w", name="mine")
        with self.assertRaisesRegex(ValueError, "already learning"):
            learning(ex, [self.tid], trainer=CONSUMER, weights="sha256:w", name="copied")
        with self.assertRaisesRegex(ValueError, "already learning"):                 # citing it changes nothing
            learning(ex, [self.tid, first], trainer=CONSUMER, weights="sha256:w", name="cites-it")
        self.assertTrue(learning(ex, [self.tid, first], trainer=CONSUMER, weights="sha256:w2", name="built-on-it"))


class Nesting(unittest.TestCase):
    def test_a_cited_learning_passes_its_share_through_to_its_own_traces(self):
        ex, vals = make_ex()
        tid = ex.submit_trace(dict(trace()))["id"]
        greedy = {"trainer": 1.0, "traces": 0, "checkers": 0, "validators": 0}
        wrapper = learning(ex, [tid], name="wrapper", split=greedy)                 # 100% to its trainer, it says
        validate(ex, wrapper, {v: (.6, .7, 300) for v in vals})
        tip = learning(ex, [wrapper], name="tip")                                   # cites the wrapper, not the trace
        validate(ex, tip, {v: (.6, .7, 300) for v in vals})
        ex.usage({"learning": tip, "consumer": CONSUMER, "calls": 50_000})
        coins = ex.wallet(TRAINER)["coin_units"]
        ex.settle()
        got = ex.wallet(TRAINER)["coin_units"] - coins
        self.assertGreater(ex.wallet(PRODUCER)["vesting_units"], 2 * got)          # the trace still gets its 60%+10%
        self.assertTrue(ex.audit()["balanced"])

    def test_learnings_nest_at_most_32_deep(self):
        from exchange import MAX_DEPTH
        ex, _ = make_ex(validators=0, quorum=0)                                     # no federation, no bonds: quick
        link = lambda parent, name: Learning.build(kind="lora", task="code.python", base_model="qwen",
                                                   artifact={"uri": name, "hash": None}, parents=[(parent, 1)],
                                                   trainer=TRAINER, per_call_micros=200,
                                                   attestation=attest(V(0), "sha256:c", "pass@1", .6, .7))
        prev = ex.submit_trace(dict(trace()))["id"]
        for i in range(MAX_DEPTH):
            prev = ex.register_learning(link(prev, f"L{i}"))["id"]
        with self.assertRaisesRegex(ValueError, "nest at most 32 deep"):
            ex.register_learning(link(prev, "one-too-many"))
        self.assertEqual(len(ex.provenance(prev)["parents"]), 1)                    # and provenance still answers


class Dissent(unittest.TestCase):
    def test_validators_outvoted_by_a_bribed_majority_keep_their_stake(self):
        ex, vals = make_ex(validators=6)
        tid = ex.submit_trace(dict(trace()))["id"]
        lid = learning(ex, [tid])
        first = ex.verdict(lid)["assigned"]
        honest = first[2]
        v = validate(ex, lid, {first[0]: (.5, .8, 300), first[1]: (.5, .8, 300), honest: (.5, .5, 300)})
        self.assertEqual(v["status"], "accepted")                            # two bribed validators carry it...
        stake = ex.validator(honest)["stake_units"]
        ex.swap(CHALLENGER, "buy", 5_000_000)
        ex.challenge(lid, CHALLENGER)
        validate(ex, lid, {x: (.5, .5, 300) for x in ex.verdict(lid)["assigned"]}, rnd=1)
        self.assertEqual(ex.verdict(lid)["status"], "clawed back")           # ...until fresh validators re-measure it
        for _ in range(6):
            ex.settle()
        self.assertEqual(ex.validator(honest)["stake_units"], stake)          # the dissenter was right: no slash
        self.assertLess(ex.validator(first[0])["stake_units"], 1_000 * UNIT)
        self.assertTrue(ex.audit()["balanced"])

    def test_padded_parents_lose_their_share_to_an_audit_challenge_any_time(self):
        ex, vals = make_ex(validators=6)
        real = ex.submit_trace(dict(trace()))["id"]
        pad = [ex.submit_trace(dict(trace(text=f"padding {i}", producer=TRAINER, code=f"x{i}")))["id"] for i in range(9)]
        lid = learning(ex, [real] + pad)
        validate(ex, lid, {v: (.6, .7, 300) for v in ex.verdict(lid)["assigned"]})   # lazy audits: "0 bad"
        for _ in range(ex.p.vest_epochs + 1):                                 # long after the old window would close
            ex.settle()
        ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 50_000})     # a real user pays $10
        ex.settle()
        share = lambda role: ex.db.execute("SELECT COALESCE(SUM(units - released), 0) FROM vesting WHERE learning=? "
                                           "AND role=? AND status='vesting'", (lid, role)).fetchone()[0]
        self.assertGreater(share("parents"), 0)                             # the parents' share vests...
        bond = ex.verdict(lid)["bond_units"]
        ex.swap(CHALLENGER, "buy", 5_000_000)
        ex.challenge(lid, CHALLENGER)
        ex.settle()
        validate(ex, lid, {x: (.6, .7, 300) for x in ex.verdict(lid)["assigned"]}, rnd=1, bad=9)
        self.assertEqual((ex.verdict(lid)["status"], ex.verdict(lid)["note"]), ("accepted", "padded"))
        self.assertEqual(share("parents"), 0)                                # ...so the audit can still claw it back
        self.assertGreater(share("learner"), 0)                              # the gain is real; the padding is not
        self.assertEqual(bond, 0)                                             # (the bond had already gone home)
        self.assertTrue(ex.audit()["balanced"])


class Minting(unittest.TestCase):
    def setUp(self):
        self.ex, self.vals = make_ex()
        self.tid = self.ex.submit_trace(dict(trace()))["id"]
        self.lid = learning(self.ex, [self.tid])
        validate(self.ex, self.lid, {v: (.6, .7, 300) for v in self.vals})

    def test_a_verdict_mints_nothing_usage_mints_a_match(self):
        ex = self.ex
        minted = ex.coin_stats()["minted_units"]
        self.assertEqual(ex.settle()["granted"]["match"], 0)                 # accepted, and nobody paid: nothing
        ex.usage({"learning": self.lid, "consumer": CONSUMER, "calls": 50_000})   # $10 of usage
        burned = ex.coin_stats()["burned_units"]
        s = ex.settle()
        burn = ex.db.execute("SELECT SUM(units) FROM burns WHERE ref=?", (self.lid,)).fetchone()[0]
        self.assertGreater(s["granted"]["match"], burn * ex.p.match * 0.99)   # half of what the usage burned...
        self.assertLessEqual(s["granted"]["match"], burn * ex.p.match)
        self.assertGreater(ex.coin_stats()["burned_units"], burned)
        self.assertEqual(ex.coin_stats()["minted_units"], minted)            # ...promised, minted only as it vests
        for _ in range(4):
            ex.settle()
        self.assertGreater(ex.coin_stats()["minted_units"], minted + s["granted"]["match"] - 10)
        self.assertTrue(ex.audit()["balanced"])

    def test_royalties_follow_the_protocol_split_whatever_the_trainer_asks(self):
        ex = self.ex
        greedy = learning(ex, [self.tid], name="greedy", split={"traces": 0, "trainer": 1.0, "checkers": 0, "validators": 0})
        validate(ex, greedy, {v: (.6, .7, 300) for v in self.vals})
        ex.usage({"learning": greedy, "consumer": CONSUMER, "calls": 50_000})
        coins = ex.wallet(TRAINER)["coin_units"]
        ex.settle()
        got = ex.wallet(TRAINER)["coin_units"] - coins                       # the trainer's quarter, paid now
        self.assertGreater(got, 0)
        self.assertGreater(ex.wallet(PRODUCER)["vesting_units"], 2 * got)    # the trace's 60%, vesting
        self.assertTrue(ex.audit()["balanced"])

    def test_paying_for_your_own_learning_loses_money(self):
        ex = self.ex
        farmer = A("f")
        ex.faucet(farmer)
        for _ in range(5):                                                   # the trainer's bond is home
            ex.settle()
        price = ex.price()
        ring = lambda: sum(worth(ex, who, price) for who in (farmer, PRODUCER, TRAINER))
        before = ring()
        ex.usage({"learning": self.lid, "consumer": farmer, "calls": 50_000})   # $10 of usage, to itself
        for _ in range(5):                                                   # royalties and the match, all vested
            ex.settle()
        self.assertLess(ring(), before - 2_000_000)                          # the whole ring is down more than $2
        self.assertTrue(ex.audit()["balanced"])


class Forfeits(unittest.TestCase):
    def test_a_failed_challenge_burns_the_stake_and_pays_nobody(self):
        ex, vals = make_ex(validators=6)
        tid = ex.submit_trace(dict(trace()))["id"]
        lid = learning(ex, [tid])
        validate(ex, lid, {v: (.6, .7, 300) for v in ex.verdict(lid)["assigned"]})
        ex.swap(CHALLENGER, "buy", 5_000_000)
        coins, burned = ex.wallet(TRAINER)["coin_units"], ex.coin_stats()["burned_units"]
        stake = ex.wallet(CHALLENGER)["coin_units"]
        ex.challenge(lid, CHALLENGER)
        v = validate(ex, lid, {x: (.6, .71, 300) for x in ex.verdict(lid)["assigned"]}, rnd=1)
        self.assertEqual(v["status"], "accepted")
        self.assertEqual(ex.wallet(TRAINER)["coin_units"], coins)            # the trainer gains nothing from it...
        self.assertEqual(ex.wallet(CHALLENGER)["coin_units"], stake - ex.p.challenge_stake)
        self.assertEqual(ex.coin_stats()["burned_units"], burned + ex.p.challenge_stake)   # ...it burned
        self.assertTrue(ex.audit()["balanced"])


class Bounties(unittest.TestCase):
    def setUp(self):
        self.ex, self.vals = make_ex(validators=6)
        ex = self.ex
        self.tid = ex.submit_trace(dict(trace()))["id"]
        self.b = ex.post_bounty({"poster": CONSUMER, "path": "code", "eval_set": "sha256:B", "target": .7, "title": "t"})
        self.backer = A("8")
        ex.faucet(self.backer)
        ex.buy_coins(self.b["id"], self.backer, 5_000_000)                   # $5, bought as TXC on the way in
        self.pool = ex.bounties()["bounties"][0]["pool_units"]

    def test_only_the_posters_own_measurement_pays_a_bounty(self):
        ex = self.ex
        lid = learning(ex, [self.tid], validator=self.vals[0], eval_set="sha256:B", before=.5, after=.8)
        first = ex.verdict(lid)["assigned"]
        validate(ex, lid, {v: (.5, .8, 300) for v in first})                 # three colluding validators accept it
        with self.assertRaisesRegex(ValueError, "poster's own measurement"):  # and attest the bounty's eval: no use
            ex.claim_bounty(self.b["id"], lid, attest(first[0], "sha256:B", "pass@1", .5, .8))
        with self.assertRaisesRegex(ValueError, "poster's own measurement"):  # the poster measured it short of target
            ex.claim_bounty(self.b["id"], lid, attest(CONSUMER, "sha256:B", "pass@1", .5, .6))
        r = ex.claim_bounty(self.b["id"], lid, attest(CONSUMER, "sha256:B", "pass@1", .5, .75))
        self.assertEqual(r["status"], "solved")
        self.assertGreater(r["vesting"][TRAINER], self.pool // 2)             # the pool vests to the solver...
        self.assertLessEqual(sum(r["vesting"].values()), self.pool)           # ...and the traces, never more
        self.assertTrue(ex.audit()["balanced"])

    def test_early_backers_cannot_cash_out_later_backers(self):
        ex = self.ex
        lure = ex.post_bounty({"poster": TRAINER, "path": "code", "eval_set": "sha256:L", "target": .9, "title": "l",
                               "epochs": 1})
        early = ex.buy_coins(lure["id"], TRAINER, 3_000_000)                  # the first, cheapest coins
        late = ex.buy_coins(lure["id"], self.backer, 10_000_000)              # dearer ones, later
        sold = ex.sell_coins(lure["id"], TRAINER, early["coins"])             # sold into the raised price...
        self.assertLessEqual(sold["paid_units"], early["spent_units"])       # ...for no more than they cost
        before = ex.wallet(self.backer)["coin_units"]
        for _ in range(3):                                                    # unsolved: it expires
            ex.settle()
        self.assertEqual(ex.bounties(status="")["bounties"][-1]["status"], "expired")
        back = ex.wallet(self.backer)["coin_units"] - before
        self.assertGreaterEqual(back, late["spent_units"] * 99 // 100)        # what it put in, less the 0.1% fee
        self.assertTrue(ex.audit()["balanced"])

    def test_a_fake_solve_is_clawed_back_and_the_backers_refunded(self):
        ex = self.ex
        lid = learning(ex, [self.tid], validator=self.vals[0], eval_set="sha256:B", before=.5, after=.8)
        first = ex.verdict(lid)["assigned"]
        validate(ex, lid, {v: (.5, .8, 300) for v in first})                 # colluders, and a poster it fooled
        ex.claim_bounty(self.b["id"], lid, attest(CONSUMER, "sha256:B", "pass@1", .5, .72))
        ex.swap(CHALLENGER, "buy", 5_000_000)                                # a challenge stakes 200 TXC
        ex.challenge(lid, CHALLENGER)
        fresh = ex.verdict(lid)["assigned"]
        self.assertFalse(set(fresh) & set(first))                            # a different draw re-measures it
        stake = {v: ex.validator(v)["stake_units"] for v in first}
        v = validate(ex, lid, {x: (.5, .5, 300) for x in fresh}, rnd=1)
        self.assertEqual(v["status"], "clawed back")
        self.assertEqual(ex.bounties(status="")["bounties"][0]["status"], "clawed back")
        self.assertGreaterEqual(ex.wallet(self.backer)["coin_units"], self.pool * 99 // 100)   # the pool is back
        for x in first:
            self.assertEqual(ex.validator(x)["stake_units"], stake[x] - stake[x] // 4)
        self.assertGreater(ex.wallet(CHALLENGER)["coin_units"], 0)
        self.assertTrue(ex.audit()["balanced"])


class Licences(unittest.TestCase):
    def test_licence_money_goes_to_the_traces_its_buyer_uses_not_to_junk(self):
        ex, vals = make_ex()
        ex.reserve = 2_000_000                                                # licences clear at $2
        real = [ex.submit_trace(dict(trace(f"Write a function number {i} that adds two numbers.", code=f"return a+b+{i}")))["id"]
                for i in range(4)]
        junk_maker = A("7")
        ex.faucet(junk_maker)
        junk = [ex.submit_trace(dict(trace(f"junk {i}", producer=junk_maker, code=f"junk {i} " * 5)))["id"]
                for i in range(40)]                                          # 40 junk traces in the same lot
        lot = ex.lots()["lots"][0]["lot"]
        for who in (TRAINER, CONSUMER):
            ex.bid({"lot": lot, "bidder": who, "price_micros": 2_000_000})
        ex.clear()
        self.assertGreater(ex.coin_stats()["licence_escrow_units"], 0)        # the money waits for the buyers
        with self.assertRaisesRegex(ValueError, "none of your licence money"):
            ex.direct_licence(lot, junk_maker, junk)                          # nobody steers another buyer's money
        lid = learning(ex, real)                                              # the trainer's learning uses the 4
        validate(ex, lid, {v: (.6, .7, 300) for v in vals})
        ex.settle()
        ex.direct_licence(lot, CONSUMER, real[:2])                            # the other buyer used two of them
        self.assertEqual(ex.coin_stats()["licence_escrow_units"], 0)
        self.assertEqual(ex.wallet(junk_maker)["coin_units"], 0)              # 40 junk traces earned nothing
        self.assertGreater(ex.wallet(PRODUCER)["coin_units"], 0)
        self.assertTrue(ex.audit()["balanced"])


class Decoys(unittest.TestCase):
    def test_a_decoy_catches_a_validator_that_never_measures(self):
        ex, vals = make_ex()
        operator, trainer = A("3"), "0x" + "d1" * 20
        ex.faucet(operator)
        ex.swap(operator, "buy", 6_000_000)
        tid = ex.submit_trace(dict(trace()))["id"]
        L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": "d", "hash": None},
                           parents=[(tid, 1)], trainer=trainer, attestation=attest(V(0), "sha256:c", "pass@1", .30, .45),
                           per_call_micros=200)
        lid = ex.register_decoy(L, decoy_digest(0.0, "salt"), operator)["id"]
        bond = ex.wallet(operator)["coin_units"]
        lazy = vals[0]
        v = validate(ex, lid, {lazy: (.30, .45, 600), vals[1]: (.30, .30, 600), vals[2]: (.30, .29, 600)})
        self.assertEqual(v["status"], "rejected")                             # it looks like any other verdict
        self.assertEqual(ex.wallet(operator)["coin_units"], bond)             # but no money has moved yet
        with self.assertRaisesRegex(ValueError, "not the truth"):
            ex.unseal_decoy(lid, 0.05, "salt")
        r = ex.unseal_decoy(lid, 0.0, "salt")
        self.assertEqual(r["caught"], [lazy])                                 # repeated the claim, never measured
        self.assertEqual(ex.validator(lazy)["stake_units"], 750 * UNIT)
        self.assertEqual(ex.validator(vals[1])["slashed_units"], 0)
        self.assertEqual(ex.wallet(operator)["coin_units"], bond + ex.p.learning_bond)   # the decoy's bond comes home
        self.assertEqual(ex.verdict(lid)["status"], "decoy")
        self.assertNotIn(lid, [x["id"] for x in ex.find_learnings(include_pending=True)["learnings"]])
        self.assertTrue(ex.audit()["balanced"])


class Attacks(unittest.TestCase):
    def test_every_farming_strategy_loses_even_with_most_of_the_stake(self):
        import contextlib
        import io
        sys.path.insert(0, os.path.join(ROOT, "examples", "farming"))
        import attacks
        with contextlib.redirect_stdout(io.StringIO()):
            rows = attacks.main()
        for name, pnl, extra, note in rows:
            if name.startswith("(honest"):
                self.assertEqual(extra, 0)
            else:
                self.assertLess(extra, 0, f"{name} should lose money against honest work: {note}")


class Hosted(unittest.TestCase):
    def setUp(self):
        import threading
        from exchange import serve
        self.ex, self.srv = serve(0, ":memory:", public=True, admin_token="op", economy="coin", test_credits=25_000_000,
                                  beacon_delay=0)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_coin_api(self):
        from traceex import Client
        me = Client(self.url, CONSUMER)
        me.faucet()
        self.assertEqual(me.describe()["settlement"]["asset"], "TXC")
        got = me.swap("buy", 2_000_000)["bought_units"]
        self.assertGreater(got, 150 * UNIT)
        b = me.post_bounty(title="t", path="code", eval_set="sha256:E", target=.5)
        r = me.back_with_coins(b["id"], 100)
        self.assertAlmostEqual(r["coins"], 73.1, delta=0.2)                    # 1 TXC, rising 0.01 TXC a coin
        self.assertGreater(me.buy_coins(b["id"], 1_000_000)["coins"], 0)       # dollars are swapped on the way in
        self.assertEqual(me.coin()["symbol"], "TXC")
        me.submit(trace(producer=CONSUMER))
        hits = me.search(sort="bounty")                                     # bounties price in TXC here
        self.assertEqual((hits["total"], len(hits["bounties"])), (1, 1))
        self.assertEqual(me.wallet()["coin_units"], got - 100 * UNIT)
        for path in ("/v0/validators", "/v0/learnings/x/commits", "/v0/learnings/x/reveals", "/v0/decoys",
                     "/v0/decoys/unseal", "/v0/licences/direct", "/v0/bounties/1/claims"):
            with self.assertRaisesRegex(RuntimeError, "^403"):
                me._call("POST", path, {})
        self.assertTrue(self.ex.audit()["balanced"])

    def test_operator_routes_for_decoys_licences_and_claims(self):
        from traceex import Client
        ex = self.ex
        vals = [V(i) for i in range(3)]
        for v in vals:
            ex.faucet(v)
            ex.swap(v, "buy", 15_000_000)
            ex.register_validator(v, 1_000 * UNIT)
        for who in (TRAINER, PRODUCER, CONSUMER, A("3")):
            ex.faucet(who)
        ex.register_checker("unit-tests", PRODUCER)
        tid = ex.submit_trace(dict(trace()))["id"]
        op = lambda who=None: Client(self.url, who, token="op")
        b = ex.post_bounty({"poster": CONSUMER, "path": "code", "eval_set": "sha256:B", "target": .7, "title": "t"})
        lid = learning(ex, [tid], before=.5, after=.8)
        validate(ex, lid, {v: (.5, .8, 300) for v in vals})
        with self.assertRaisesRegex(RuntimeError, "poster's own measurement"):
            op().claim_bounty(b["id"], lid)
        r = op().claim_bounty(b["id"], lid, attest(CONSUMER, "sha256:B", "pass@1", .5, .75))
        self.assertEqual(r["status"], "solved")
        lot = ex.lots()["lots"][0]["lot"]
        ex.bid({"lot": lot, "bidder": CONSUMER, "price_micros": 50_000})
        ex.clear()
        self.assertGreater(op(CONSUMER).direct_licence(lot, [tid])["paid_units"], 0)
        ex.swap(A("3"), "buy", 6_000_000)                                     # the operator's decoy fund
        L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": "d", "hash": None},
                           parents=[(tid, 1)], trainer="0x" + "d2" * 20, per_call_micros=200,
                           attestation=attest(V(0), "sha256:c", "pass@1", .30, .45))
        decoy = op().register_decoy(L, decoy_digest(0.0, "s"), A("3"))["id"]
        validate(ex, decoy, {vals[0]: (.30, .45, 600), vals[1]: (.30, .30, 600), vals[2]: (.30, .29, 600)})
        self.assertEqual(op().unseal_decoy(decoy, 0.0, "s")["caught"], [vals[0]])
        self.assertTrue(ex.audit()["balanced"])

    def test_one_operator_runs_three_validators_from_one_folder(self):
        import contextlib
        import io
        import tempfile
        import validator
        vals = [V(i) for i in range(3)]
        for v in vals:
            self.ex.faucet(v)
            self.ex.swap(v, "buy", 15_000_000)
            self.ex.register_validator(v, 1_000 * UNIT)
        for who in (TRAINER, PRODUCER):
            self.ex.faucet(who)
        self.ex.register_checker("unit-tests", PRODUCER)
        lid = learning(self.ex, [self.ex.submit_trace(dict(trace()))["id"]])
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        for _ in ("commit", "reveal"):                                      # the same command twice per validator
            for v in vals:
                with contextlib.redirect_stdout(io.StringIO()):
                    validator.main(["--url", self.url, "--token", "op", "--address", v, "--learning", lid,
                                    "--eval-set", "sha256:" + v[-4:], "--before", "0.6", "--after", "0.7", "--n", "300",
                                    "--se", "0.03", "--audit-checked", "1",
                                    "--state", os.path.join(folder.name, "salts.json")])
        v = self.ex.verdict(lid)
        self.assertEqual((v["status"], len(v["reveals"])), ("accepted", 3))
        self.assertTrue(self.ex.audit()["balanced"])

    def test_parallel_requests_never_collide(self):
        import concurrent.futures
        import threading
        from exchange import serve
        from traceex import Client
        ex, srv = serve(0, ":memory:", economy="coin", test_credits=25_000_000)    # no rate limits: all from one IP
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        people = [Client(url, "0x" + f"{i:02x}" * 20) for i in range(8)]
        for c in people:                                                     # (the faucet allows 3 per network a day)
            ex.db.execute("INSERT INTO grants VALUES (?,?,?,?)", (c.address, 25_000_000, "t", "test"))

        def work(c):
            for _ in range(12):
                c.stats(); c.bounties(status=""); c.wallet(); c.coin()
                c.swap("buy", 50_000)
            return True
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            self.assertTrue(all(pool.map(work, people)))                     # one connection, many threads
        self.assertTrue(ex.audit()["balanced"])


class Seed(unittest.TestCase):
    def test_the_coin_seed_runs_the_federation_for_real(self):
        from seed import seed_if_empty
        ex = CoinExchange(":memory:", test_credits=25_000_000, reserve_micros=50_000)
        seed_if_empty(ex)
        c = ex.coin_stats()
        self.assertEqual(c["learnings"], {"accepted": 1, "inconclusive": 1})   # LoRA v2 proven; 3 emails are not proof
        self.assertEqual((ex.epoch, c["validators"]), (3, 3))
        self.assertGreater(c["burned_units"], 0)
        self.assertGreater(c["vesting_units"], 0)                             # the host's usage, matched
        self.assertEqual(c["licence_escrow_units"], 0)                        # every buyer named what it used
        self.assertEqual([b["status"] for b in ex.bounties()["bounties"]], ["open", "open"])
        self.assertTrue(ex.audit()["balanced"])


if __name__ == "__main__":
    unittest.main()
