"""The sats economy (node/sats.py, testnet v0.6, no token): python -m pytest -q tests/test_sats.py

Each test is one rule, most of them a rule that makes a farming strategy lose. Every test ends with the books
balanced (audit()): the ledger is double-entry and sums to zero, no escrow is negative, every validator's stake is what
its stake account holds, and for every payment, what was paid out plus what it still holds is exactly what its payer
paid in less the fee. Amounts are msats; each wallet takes 30,000 test sats."""
import json
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node")]

from traceex import Trace, Learning, attest  # noqa: E402
from sats import (SatsExchange, Params, BURN, InvariantError, attestation_digest, decoy_digest, _frac)  # noqa: E402

A = lambda c: "0x" + c * 40
V = lambda i: "0x" + f"{0xa0 + i:02x}" * 20           # validator addresses
TRAINER, PRODUCER, CONSUMER, CHALLENGER, OPERATOR = A("c"), A("b"), A("e"), A("d"), A("f")
FEE = 58


def make_ex(validators=3, quorum=3, delay=0, fee_to=OPERATOR, **params):
    ex = SatsExchange(":memory:", test_credits=30_000_000, beacon_delay=delay, params=Params(quorum=quorum, **params),
                      fee_to=fee_to)
    vals = []
    for i in range(validators):
        v = V(i)
        ex.faucet(v)
        ex.register_validator(v, 15_000_000)               # 15,000 sats staked (the minimum is 10,000)
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
             weights=None, split=None, per_call=200):
    att = attest(validator or V(0), eval_set, "pass@1", before, after)
    L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": name, "hash": weights},
                       parents=[(p, 1) for p in parents], trainer=trainer, attestation=att, per_call_msats=per_call,
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


def accepted(ex, vals, parents=None, **kw):
    """A real learning, accepted by the federation."""
    parents = parents or [ex.submit_trace(dict(trace()))["id"]]
    lid = learning(ex, parents, **kw)
    validate(ex, lid, {v: (.6, .7, 300) for v in ex.verdict(lid)["assigned"] or vals})
    return lid


def bal(ex, who):
    """What an account holds and can spend (sats), plus what waits for it in escrow."""
    w = ex.wallet(who)
    return w["balance_msats"] + w["vesting_msats"]


def payouts_ok(test, ex):
    a = ex.audit()
    test.assertTrue(a["balanced"], a)
    for pid, gross, fee, out in ex.db.execute("SELECT id, gross, fee, out FROM payments").fetchall():
        test.assertLessEqual(out, gross - fee, f"payment {pid}")


class Invariant(unittest.TestCase):
    """Nothing is ever paid out that a payer didn't pay in."""

    def test_every_payout_is_a_split_of_a_real_payment(self):
        ex, vals = make_ex(validators=6)
        lid = accepted(ex, vals)
        b = ex.post_bounty({"poster": CONSUMER, "path": "code", "eval_set": "sha256:B", "target": .7, "title": "t"})
        ex.pledge(b["id"], CHALLENGER, 3_000_000)
        ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 10_000})     # 2,000 sats
        lot = ex.lots()["lots"][0]["lot"]
        ex.bid({"lot": lot, "bidder": CONSUMER, "price_msats": 50_000})
        ex.clear()
        ex.direct_licence(lot, CONSUMER, [ex.db.execute("SELECT id FROM traces").fetchone()[0]])
        for _ in range(7):
            ex.settle()
        st = ex.economy_stats()
        paid_in = st["paid_in_msats"]
        self.assertEqual(paid_in, 2_000_000 + 3_000_000 + ex.reserve)          # the licence clears at the reserve
        self.assertEqual(st["paid_out_msats"] + st["refunded_msats"] + st["escrow_msats"]["payments"], paid_in)
        self.assertGreaterEqual(st["refunded_msats"], 3_000_000)              # the unsolved pledge (and split dust)
        self.assertLess(st["refunded_msats"], 3_000_010)
        payouts_ok(self, ex)

    def test_disburse_refuses_to_pay_out_more_than_was_paid_in(self):
        ex, vals = make_ex()
        lid = accepted(ex, vals)
        pid = ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 1_000})["payment"]   # 200,000 msats
        with self.assertRaises(InvariantError):
            ex._disburse(pid, TRAINER, 200_001, "too much")
        ex._disburse(pid, TRAINER, 150_000, "part")
        with self.assertRaises(InvariantError):
            ex._disburse(pid, TRAINER, 50_001, "past the payment")
        real = ex._shares
        ex._shares = lambda *a, **k: {"trainer": {TRAINER: 10 ** 9}}           # a bug that would mint money...
        with self.assertRaises(InvariantError):
            ex._split(pid, lid, None, ex._split_tree())                        # ...is refused before anything moves
        ex._shares = real
        with self.assertRaises(InvariantError):
            ex._credit(BURN, -1, "spend the burn")                             # forfeits can never come back
        ex.db.conn.rollback()

    def test_nothing_is_minted_the_books_sum_to_zero(self):
        ex, vals = make_ex()
        lid = accepted(ex, vals)
        for _ in range(3):
            ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 5_000})
            ex.settle()
        grants = ex.db.execute("SELECT SUM(micros) FROM grants").fetchone()[0]
        everyone = [r for (r,) in ex.db.execute("SELECT DISTINCT account FROM sledger")]
        self.assertEqual(sum(ex._bal(a) for a in everyone), 0)                 # double-entry: every row is a move
        wallets = {a for (a,) in ex.db.execute("SELECT account FROM grants")} | {a for a in everyone if a.startswith("0x")}
        held = sum(ex._funds(a) for a in wallets)                              # (the operator's fees included)
        escrow = sum(ex._bal(a) for a in everyone if not a.startswith("0x"))
        self.assertEqual(held + escrow, grants)                                # all the money there is: the faucet's
        self.assertIsNone(ex.economy_stats()["token"])
        payouts_ok(self, ex)


class Usage(unittest.TestCase):
    def test_use_pays_out_at_once_traces_60_trainer_25_checkers_10_validators_5(self):
        ex, vals = make_ex()
        writer = A("7")                                                       # the checker's author, apart
        ex.faucet(writer)
        ex.register_checker("unit-tests", writer)
        lid = accepted(ex, vals)
        before = {w: bal(ex, w) for w in (TRAINER, PRODUCER, writer, *vals)}
        ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 50_000})     # 10,000 sats
        ex.settle()
        got = {w: bal(ex, w) - before[w] for w in before}
        self.assertEqual(got[TRAINER], 2_500_000)                              # paid now
        self.assertEqual(sum(got[v] for v in vals), 500_000)
        self.assertEqual((got[PRODUCER], got[writer]), (6_000_000, 1_000_000))
        self.assertEqual(ex.wallet(PRODUCER)["vesting_msats"], 6_000_000)       # waits in escrow...
        for _ in range(ex.p.vest_epochs - 1):
            ex.settle()
        self.assertEqual(ex.wallet(PRODUCER)["vesting_msats"], 6_000_000)
        ex.settle()
        self.assertEqual(ex.wallet(PRODUCER)["vesting_msats"], 0)              # ...4 epochs, then it is paid
        payouts_ok(self, ex)

    def test_sellers_set_prices_and_dollar_prices_are_refused(self):
        ex, vals = make_ex()
        cheap = accepted(ex, vals, name="cheap", per_call=7)
        dear = accepted(ex, vals, name="dear", per_call=1_000)
        self.assertEqual(ex.usage({"learning": cheap, "consumer": CONSUMER, "calls": 3})["paid_msats"], 21)
        self.assertEqual(ex.usage({"learning": dear, "consumer": CONSUMER, "calls": 3})["paid_msats"], 3_000)
        tid = ex.submit_trace(dict(trace("Write a function to divide.", code="return a / b")))["id"]
        L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": "d", "hash": None},
                           parents=[(tid, 1)], trainer=TRAINER, per_call_micros=200,
                           attestation=attest(V(0), "sha256:c", "pass@1", .6, .7))
        with self.assertRaisesRegex(ValueError, "per_call_msats"):
            ex.register_learning(L)
        payouts_ok(self, ex)

    def test_paying_for_your_own_learning_loses(self):
        ex, vals = make_ex()
        farmer = A("7")
        ex.faucet(farmer)
        ex.register_checker("farm-tests", farmer)
        own = [ex.submit_trace(dict(Trace.from_fix(
            task="code.python", base_model="qwen", input=f"Write function {i}.", model_output={"code": "x"},
            verified_output={"code": f"y{i}"}, checker="farm-tests@1", producer=farmer, created="2026-10-04T00:00:00Z",
            privacy="open", failure_modes={"code": "wrong_answer"})))["id"] for i in range(3)]
        lid = accepted(ex, vals, parents=own, trainer=farmer)
        for _ in range(5):
            ex.settle()                                                       # its bond is home
        start = bal(ex, farmer)
        ex.usage({"learning": lid, "consumer": farmer, "calls": 50_000})       # 10,000 sats to itself
        for _ in range(6):
            ex.settle()
        self.assertEqual(start - bal(ex, farmer), 500_000 + FEE)               # the validators' 5% and the fee, gone
        payouts_ok(self, ex)

    def test_a_learning_clawed_back_before_settlement_refunds_its_users(self):
        ex, vals = make_ex(validators=6)
        lid = accepted(ex, vals)
        sats = ex.wallet(CONSUMER)["balance_msats"]
        ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 5_000})
        ex.challenge(lid, CHALLENGER)
        ex.settle()                                                           # challenged: the payment waits
        validate(ex, lid, {x: (.6, .6, 300) for x in ex.verdict(lid)["assigned"]}, rnd=1)
        ex.settle()
        self.assertEqual(ex.verdict(lid)["status"], "clawed back")
        self.assertEqual(ex.wallet(CONSUMER)["balance_msats"], sats - FEE)   # every sat back but the fee
        payouts_ok(self, ex)


class Copies(unittest.TestCase):
    def test_near_duplicates_pay_the_first_producer(self):
        ex, _ = make_ex(validators=0)
        t1 = ex.submit_trace(dict(trace()))
        copier = A("7")
        ex.faucet(copier)
        t2 = ex.submit_trace(dict(trace(text="write a function to  add two numbers.\nassert add(1, 2) == 3", producer=copier)))
        self.assertEqual(t2["near_duplicate_of"], t1["id"])
        info, _ = ex._tree()
        self.assertEqual(info[t2["id"]]["producer"], PRODUCER)              # the copy's earnings go to the original

    def test_a_reworded_copy_of_a_fix_shares_the_originals_slot(self):
        ex, _ = make_ex(validators=0)
        code = "def add(a, b):\n    return a + b"
        t1 = ex.submit_trace(dict(trace("Write a function to add two numbers.", code=code)))
        copier = A("7")
        ex.faucet(copier)
        t2 = ex.submit_trace(dict(trace("Create a function that adds two numbers.", producer=copier, code=code)))
        self.assertEqual(t2.get("near_duplicate_of"), t1["id"])
        t3 = ex.submit_trace(dict(trace("Write a function to subtract two numbers.", producer=copier,
                                        code="def sub(a, b):\n    return a - b")))
        self.assertNotIn("near_duplicate_of", t3)
        sk = lambda text: dict(trace(text, code="{CODE_1}"), privacy="skeleton")
        t4, t5 = ex.submit_trace(sk("Flight {NUM_1} departs {CITY_1}")), ex.submit_trace(sk("Your flight {NUM_1} from {CITY_1}"))
        self.assertNotIn("near_duplicate_of", t5)
        self.assertNotEqual(t4["id"], t5["id"])


class Fee(unittest.TestCase):
    def test_one_fee_per_transaction_paid_at_once_to_the_operator(self):
        from exchange import TX_FEE_MSATS
        self.assertEqual(TX_FEE_MSATS, 58)                                   # about $0.00005 at $85,962 a bitcoin
        ex, _ = make_ex(validators=0)
        op0 = ex.wallet(OPERATOR)["balance_msats"]
        for i in range(3):
            ex.submit_trace(dict(trace(f"Write a function number {i} to add two numbers.", code=f"return a + b + {i}")))
        self.assertEqual(ex.wallet(PRODUCER)["balance_msats"], 30_000_000 - 3 * 58)
        self.assertEqual(ex.wallet(OPERATOR)["balance_msats"] - op0, 3 * 58)  # its whole income: the fees it served
        self.assertEqual(ex.fees()["paid_msats"] - 0, ex.economy_stats()["fees_msats"])
        from exchange import PaymentRequired
        with self.assertRaises(PaymentRequired):
            ex.submit_trace(dict(trace(producer=A("9"))))                    # no wallet, no transaction
        payouts_ok(self, ex)

    def test_the_fee_is_fixed_in_sats_unless_the_operator_turns_on_the_repeg(self):
        ex, _ = make_ex(validators=0)
        ex.set_btc_usd(171_924)                                              # bitcoin doubles...
        for _ in range(4):
            ex.settle()
        self.assertEqual(ex.tx_fee_msats, 58)                                # ...the fee stays 58 msats (now ~$0.0001)
        ex, _ = make_ex(validators=0, fee_repeg_epochs=2)
        ex.set_btc_usd(171_924)
        ex.settle()
        self.assertEqual(ex.tx_fee_msats, 58)                                # re-pegs only every 2nd epoch...
        ex.settle()
        self.assertEqual(ex.tx_fee_msats, 29)                                # ...back to $0.00005: 29 msats
        self.assertEqual(ex.fees()["per_transaction_msats"], 29)


class Stakes(unittest.TestCase):
    def test_bonds_stakes_and_challenges_are_sats(self):
        ex, vals = make_ex()
        self.assertEqual((ex.learning_bond_msats(), ex.min_stake_msats(), ex.challenge_stake_msats()),
                         (5_000_000, 10_000_000, 2_000_000))
        with self.assertRaisesRegex(ValueError, "at least 10,000 sats"):
            ex.register_validator(CONSUMER, 9_999_999)
        sats = ex.wallet(TRAINER)["balance_msats"]
        lid = learning(ex, [ex.submit_trace(dict(trace()))["id"]])
        self.assertEqual(ex.wallet(TRAINER)["balance_msats"], sats - 5_000_000 - FEE)
        self.assertEqual(ex.verdict(lid)["bond_msats"], 5_000_000)
        self.assertFalse(hasattr(ex.p, "stake_grace_epochs"))                 # no price, so no price-drop grace

    def test_forfeits_are_destroyed_in_a_batch_for_an_unspendable_output(self):
        ex, vals = make_ex()
        lid = learning(ex, [ex.submit_trace(dict(trace()))["id"]])
        validate(ex, lid, {v: (.60, .60, 300) for v in vals})                 # no gain: the bond is forfeit
        self.assertEqual(ex.verdict(lid)["status"], "rejected")
        self.assertEqual(ex._bal(BURN), 5_000_000)
        s = ex.settle()
        batch = s["summary"]["burn_batch"]
        self.assertEqual((batch["msats"], batch["items"]), (5_000_000, 1))
        self.assertTrue(batch["digest"].startswith("sha256:"))
        self.assertIn("OP_RETURN", ex.economy_stats()["burn"]["destination"])
        self.assertEqual(ex.economy_stats()["forfeited_msats"], 5_000_000)
        payouts_ok(self, ex)

    def test_a_validator_slashed_under_the_minimum_loses_its_seat_at_once(self):
        ex, vals = make_ex(validators=4)
        ex._slash(vals[0], 0.40, "test")                                      # 15,000 -> 9,000 sats
        self.assertFalse(ex.validator(vals[0])["active"])
        self.assertNotIn(vals[0], [a for a, _ in ex._active()])
        ex.register_validator(vals[0], 1_000_000)                             # tops up to 10,000: back
        self.assertTrue(ex.validator(vals[0])["active"])
        payouts_ok(self, ex)


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
        payouts_ok(self, ex)

    def test_one_bought_validator_cannot_fake_a_gain_and_the_bond_is_destroyed(self):
        ex, vals = self.ex, self.vals
        lid = learning(ex, [self.tid])
        bought = vals[2]
        v = validate(ex, lid, {vals[0]: (.60, .60, 200), vals[1]: (.61, .60, 200), bought: (.60, .90, 200)})
        self.assertEqual(v["status"], "rejected")
        self.assertEqual(ex._bal(BURN), 5_000_000)                            # the bond went to nobody
        self.assertEqual(sum(ex.validator(x)["slashed_msats"] for x in vals), 0)   # disagreeing isn't a fault
        self.assertEqual([r["agreed"] for r in v["reveals"] if r["validator"] == bought], [False])
        self.assertEqual(ex.find_learnings()["count"], 0)
        payouts_ok(self, ex)

    def test_a_claim_far_beyond_the_measured_gain_forfeits_the_bond(self):
        ex, vals = self.ex, self.vals
        fake = learning(ex, [self.tid], before=.30, after=.60, name="claims-30")
        v = validate(ex, fake, {vals[0]: (.30, .32, 600), vals[1]: (.30, .31, 600), vals[2]: (.30, .60, 600)})
        self.assertEqual((v["status"], v["note"]), ("rejected", "overclaimed"))
        noisy = learning(ex, [self.tid], before=.30, after=.50, name="small-eval")
        v = validate(ex, noisy, {x: (.30, .38, 600) for x in vals})
        self.assertEqual(v["status"], "accepted")                             # an honest, noisy claim is no overclaim

    def test_a_gain_inside_the_noise_is_inconclusive_not_punished(self):
        ex, vals = self.ex, self.vals
        sats = ex.wallet(TRAINER)["balance_msats"]
        lid = learning(ex, [self.tid])
        v = validate(ex, lid, {vals[0]: (.6, .8, 10), vals[1]: (.7, .8, 10), vals[2]: (.6, .6, 10)})   # 10-item evals
        self.assertEqual(v["status"], "inconclusive")
        self.assertEqual(sum(ex.validator(x)["slashed_msats"] for x in vals), 0)
        self.assertEqual(ex.wallet(TRAINER)["balance_msats"], sats - FEE - _frac(5_000_000, 0.10))   # bond back - 10%
        payouts_ok(self, ex)

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
        with self.assertRaises(PermissionError):
            ex.commit(lid, next(v for v in vals if v not in drawn), "x")

    def test_the_same_weights_cannot_be_registered_twice(self):
        ex = self.ex
        first = learning(ex, [self.tid], weights="sha256:w", name="mine")
        with self.assertRaisesRegex(ValueError, "already learning"):
            learning(ex, [self.tid], trainer=CONSUMER, weights="sha256:w", name="copied")
        with self.assertRaisesRegex(ValueError, "already learning"):
            learning(ex, [self.tid, first], trainer=CONSUMER, weights="sha256:w", name="cites-it")
        self.assertTrue(learning(ex, [self.tid, first], trainer=CONSUMER, weights="sha256:w2", name="built-on-it"))

    def test_only_an_accepted_learning_can_be_paid_for(self):
        lid = learning(self.ex, [self.tid])
        with self.assertRaisesRegex(ValueError, "only a learning the validator federation accepted"):
            self.ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 10})


class Nesting(unittest.TestCase):
    def test_a_cited_learning_passes_its_share_through_to_its_own_traces(self):
        ex, vals = make_ex()
        tid = ex.submit_trace(dict(trace()))["id"]
        greedy = {"trainer": 1.0, "traces": 0, "checkers": 0, "validators": 0}
        wrapper = learning(ex, [tid], name="wrapper", split=greedy)                 # 100% to its trainer, it says
        validate(ex, wrapper, {v: (.6, .7, 300) for v in vals})
        tip = learning(ex, [wrapper], name="tip")                                   # cites the wrapper, not the trace
        validate(ex, tip, {v: (.6, .7, 300) for v in vals})
        for _ in range(5):
            ex.settle()                                                              # bonds home
        t0, p0 = bal(ex, TRAINER), bal(ex, PRODUCER)
        ex.usage({"learning": tip, "consumer": CONSUMER, "calls": 50_000})          # 10,000 sats
        ex.settle()
        self.assertEqual(bal(ex, TRAINER) - t0, 2_500_000)                          # its own 25%, nothing via the wrapper
        self.assertEqual(bal(ex, PRODUCER) - p0, 7_000_000)                         # the trace still gets 60% + 10%
        payouts_ok(self, ex)

    def test_learnings_nest_at_most_32_deep(self):
        from exchange import MAX_DEPTH
        ex, _ = make_ex(validators=0, quorum=0)                                     # no federation, no bonds: quick
        link = lambda parent, name: Learning.build(kind="lora", task="code.python", base_model="qwen",
                                                   artifact={"uri": name, "hash": None}, parents=[(parent, 1)],
                                                   trainer=TRAINER, per_call_msats=200,
                                                   attestation=attest(V(0), "sha256:c", "pass@1", .6, .7))
        prev = ex.submit_trace(dict(trace()))["id"]
        for i in range(MAX_DEPTH):
            prev = ex.register_learning(link(prev, f"L{i}"))["id"]
        with self.assertRaisesRegex(ValueError, "nest at most 32 deep"):
            ex.register_learning(link(prev, "one-too-many"))
        self.assertEqual(len(ex.provenance(prev)["parents"]), 1)
        ex.usage({"learning": prev, "consumer": CONSUMER, "calls": 1_000})          # 32 deep still pays, exactly
        for _ in range(6):
            ex.settle()
        payouts_ok(self, ex)


class Challenges(unittest.TestCase):
    def test_validators_outvoted_by_a_bribed_majority_keep_their_stake(self):
        ex, vals = make_ex(validators=6)
        tid = ex.submit_trace(dict(trace()))["id"]
        lid = learning(ex, [tid])
        first = ex.verdict(lid)["assigned"]
        honest = first[2]
        v = validate(ex, lid, {first[0]: (.5, .8, 300), first[1]: (.5, .8, 300), honest: (.5, .5, 300)})
        self.assertEqual(v["status"], "accepted")                            # two bribed validators carry it...
        stake = ex.validator(honest)["stake_msats"]
        ex.challenge(lid, CHALLENGER)
        validate(ex, lid, {x: (.5, .5, 300) for x in ex.verdict(lid)["assigned"]}, rnd=1)
        self.assertEqual(ex.verdict(lid)["status"], "clawed back")           # ...until fresh validators re-measure it
        self.assertEqual(ex.validator(honest)["stake_msats"], stake)          # the dissenter was right: no slash
        self.assertEqual(ex.validator(first[0])["stake_msats"], 15_000_000 - 3_750_000)
        payouts_ok(self, ex)

    def test_padded_parents_lose_their_escrowed_share_to_an_audit_challenge(self):
        ex, vals = make_ex(validators=6)
        real = ex.submit_trace(dict(trace()))["id"]
        pad = [ex.submit_trace(dict(trace(text=f"padding {i}", producer=TRAINER, code=f"x{i}")))["id"] for i in range(9)]
        lid = learning(ex, [real] + pad)
        validate(ex, lid, {v: (.6, .7, 300) for v in ex.verdict(lid)["assigned"]})   # lazy audits: "0 bad"
        sats = ex.wallet(CONSUMER)["balance_msats"]
        ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 50_000})     # a real user pays 10,000 sats
        ex.settle()
        self.assertEqual(sum(ex.wallet(w)["vesting_msats"] for w in (TRAINER, PRODUCER)), 7_000_000)
        ex.challenge(lid, CHALLENGER)
        ex.settle()
        validate(ex, lid, {x: (.6, .7, 300) for x in ex.verdict(lid)["assigned"]}, rnd=1, bad=9)
        self.assertEqual((ex.verdict(lid)["status"], ex.verdict(lid)["note"]), ("accepted", "padded"))
        self.assertEqual(ex._vesting_of(), 0)                                 # the escrowed parents' share...
        self.assertEqual(ex.wallet(CONSUMER)["balance_msats"], sats - FEE - 3_000_000)   # ...went back to the user
        payouts_ok(self, ex)

    def test_a_failed_challenge_destroys_the_stake_and_pays_nobody(self):
        ex, vals = make_ex(validators=6)
        lid = learning(ex, [ex.submit_trace(dict(trace()))["id"]])
        validate(ex, lid, {v: (.6, .7, 300) for v in ex.verdict(lid)["assigned"]})
        t0, c0 = ex.wallet(TRAINER)["balance_msats"], ex.wallet(CHALLENGER)["balance_msats"]
        ex.challenge(lid, CHALLENGER)
        v = validate(ex, lid, {x: (.6, .71, 300) for x in ex.verdict(lid)["assigned"]}, rnd=1)
        self.assertEqual(v["status"], "accepted")
        self.assertEqual(ex.wallet(TRAINER)["balance_msats"], t0)            # the trainer gains nothing from it
        self.assertEqual(ex.wallet(CHALLENGER)["balance_msats"], c0 - 2_000_000 - FEE)
        self.assertEqual(ex._bal(BURN), 2_000_000)                           # it was destroyed
        payouts_ok(self, ex)


class Decoys(unittest.TestCase):
    def decoy(self, ex, tid, k):
        L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": f"d{k}", "hash": None},
                           parents=[(tid, 1)], trainer="0x" + f"{0xd0 + k:02x}" * 20, per_call_msats=200,
                           attestation=attest(V(0), "sha256:c", "pass@1", .30, .45))
        return ex.register_decoy(L, decoy_digest(0.0, f"salt{k}"), A("3"))["id"]

    def test_two_decoy_misses_not_one_cost_a_validator_its_stake(self):
        ex, vals = make_ex()
        ex.faucet(A("3"))
        tid = ex.submit_trace(dict(trace()))["id"]
        lazy = vals[0]
        first = self.decoy(ex, tid, 1)
        bond_home = ex.wallet(A("3"))["balance_msats"]
        v = validate(ex, first, {lazy: (.30, .45, 600), vals[1]: (.30, .30, 600), vals[2]: (.30, .29, 600)})
        self.assertEqual(v["status"], "rejected")                             # it looks like any other verdict
        self.assertEqual(ex.wallet(A("3"))["balance_msats"], bond_home)        # but no money has moved yet
        with self.assertRaisesRegex(ValueError, "not the truth"):
            ex.unseal_decoy(first, 0.05, "salt1")
        r = ex.unseal_decoy(first, 0.0, "salt1")
        self.assertEqual((r["struck"], r["caught"]), ([lazy], []))             # one miss: a strike, no slash
        self.assertEqual(ex.validator(lazy)["stake_msats"], 15_000_000)
        self.assertEqual(ex.validator(lazy)["decoy_strikes"], 1)
        self.assertEqual(ex.wallet(A("3"))["balance_msats"], bond_home + 5_000_000)   # the decoy's bond comes home
        second = self.decoy(ex, tid, 2)
        validate(ex, second, {lazy: (.30, .45, 600), vals[1]: (.30, .31, 600), vals[2]: (.30, .30, 600)})
        r = ex.unseal_decoy(second, 0.0, "salt2")
        self.assertEqual(r["caught"], [lazy])                                 # the second miss: 25% of its stake
        self.assertEqual(ex.validator(lazy)["stake_msats"], 11_250_000)
        self.assertEqual(ex.validator(vals[1])["slashed_msats"], 0)
        self.assertEqual(ex.verdict(second)["status"], "decoy")
        self.assertNotIn(second, [x["id"] for x in ex.find_learnings(include_pending=True)["learnings"]])
        payouts_ok(self, ex)

    def test_strikes_expire(self):
        ex, vals = make_ex(strike_window=3)
        ex.faucet(A("3"))
        tid = ex.submit_trace(dict(trace()))["id"]
        d1 = self.decoy(ex, tid, 1)
        validate(ex, d1, {vals[0]: (.30, .45, 600), vals[1]: (.30, .30, 600), vals[2]: (.30, .29, 600)})
        ex.unseal_decoy(d1, 0.0, "salt1")
        for _ in range(3):
            ex.settle()                                                       # an old miss ages out of the window
        d2 = self.decoy(ex, tid, 2)
        ex.settle()
        validate(ex, d2, {v: (.30, .45, 600) if v == vals[0] else (.30, .30, 600) for v in ex.verdict(d2)["assigned"]})
        self.assertEqual(ex.unseal_decoy(d2, 0.0, "salt2")["caught"], [])
        self.assertEqual(ex.validator(vals[0])["slashed_msats"], 0)


class Bounties(unittest.TestCase):
    def setUp(self):
        self.ex, self.vals = make_ex(validators=6)
        ex = self.ex
        self.tid = ex.submit_trace(dict(trace()))["id"]
        self.b = ex.post_bounty({"poster": CONSUMER, "path": "code", "eval_set": "sha256:B", "target": .7, "title": "t"})
        self.backer = A("8")
        ex.faucet(self.backer)
        ex.pledge(self.b["id"], self.backer, 5_000_000)

    def test_only_the_posters_own_measurement_pays_a_bounty(self):
        ex = self.ex
        lid = learning(ex, [self.tid], validator=self.vals[0], eval_set="sha256:B", before=.5, after=.8)
        first = ex.verdict(lid)["assigned"]
        validate(ex, lid, {v: (.5, .8, 300) for v in first})                 # three colluding validators accept it
        with self.assertRaisesRegex(ValueError, "poster's own measurement"):
            ex.claim_bounty(self.b["id"], lid, attest(first[0], "sha256:B", "pass@1", .5, .8))
        with self.assertRaisesRegex(ValueError, "poster's own measurement"):
            ex.claim_bounty(self.b["id"], lid, attest(CONSUMER, "sha256:B", "pass@1", .5, .6))
        r = ex.claim_bounty(self.b["id"], lid, attest(CONSUMER, "sha256:B", "pass@1", .5, .75))
        self.assertEqual(r["status"], "solved")
        self.assertEqual(r["vesting_msats"][TRAINER], 3_500_000)              # 70% to the solver...
        self.assertEqual(sum(r["vesting_msats"].values()), 5_000_000)        # ...and the tree: the pledges, exactly
        self.assertNotIn(self.backer, r["vesting_msats"])                    # backers take no share
        payouts_ok(self, ex)

    def test_unsolved_refunds_every_backer_what_it_put_in(self):
        ex = self.ex
        late = A("7")
        ex.faucet(late)
        ex.pledge(self.b["id"], late, 1_234_567)
        start = {w: ex.wallet(w)["balance_msats"] for w in (self.backer, late)}
        for _ in range(6):
            ex.settle()
        self.assertEqual(ex.bounties()["bounties"][0]["status"], "expired")
        self.assertEqual(ex.wallet(self.backer)["balance_msats"] - start[self.backer], 5_000_000)
        self.assertEqual(ex.wallet(late)["balance_msats"] - start[late], 1_234_567)
        payouts_ok(self, ex)

    def test_a_fake_solve_is_clawed_back_and_the_backers_refunded(self):
        ex = self.ex
        lid = learning(ex, [self.tid], validator=self.vals[0], eval_set="sha256:B", before=.5, after=.8)
        first = ex.verdict(lid)["assigned"]
        validate(ex, lid, {v: (.5, .8, 300) for v in first})                 # colluders, and a poster it fooled
        ex.claim_bounty(self.b["id"], lid, attest(CONSUMER, "sha256:B", "pass@1", .5, .72))
        sats = ex.wallet(self.backer)["balance_msats"]
        ex.challenge(lid, CHALLENGER)
        fresh = ex.verdict(lid)["assigned"]
        self.assertFalse(set(fresh) & set(first))                            # a different draw re-measures it
        v = validate(ex, lid, {x: (.5, .5, 300) for x in fresh}, rnd=1)
        self.assertEqual(v["status"], "clawed back")
        self.assertEqual(ex.bounties(status="")["bounties"][0]["status"], "clawed back")
        self.assertEqual(ex.wallet(self.backer)["balance_msats"] - sats, 5_000_000)   # every sat back
        for x in first:
            self.assertEqual(ex.validator(x)["stake_msats"], 15_000_000 - 3_750_000)
        payouts_ok(self, ex)

    def test_a_duplicate_post_backs_the_open_bounty_and_unbacked_ones_expire(self):
        ex = self.ex
        r = ex.post_bounty({"poster": TRAINER, "path": "code/", "eval_set": "sha256:other", "target": .9,
                            "title": "same problem", "seed_sats": 1_000})
        self.assertEqual((r["id"], r["merged"]), (self.b["id"], True))
        self.assertEqual(ex.backers(self.b["id"])["backers"], {self.backer: 5_000_000, TRAINER: 1_000_000})
        lonely = ex.post_bounty({"poster": TRAINER, "path": "extract", "eval_set": "sha256:L", "target": .5,
                                 "epochs": 20})["id"]
        self.assertNotIn(lonely, [b["id"] for b in ex.search()["bounties"]])  # unbacked: not in the default search
        for _ in range(4):
            ex.settle()
        st = {b["id"]: b["status"] for b in ex.bounties()["bounties"]}
        self.assertEqual((st[lonely], st[self.b["id"]]), ("expired", "open"))
        payouts_ok(self, ex)

    def test_a_pledge_is_no_security(self):
        ex = self.ex
        for gone in ("buy_coins", "sell_coins", "transfer_coins", "holders", "swap", "buy_credits", "quote",
                     "coin_stats"):
            self.assertFalse(hasattr(ex, gone), gone)
        lid = learning(ex, [self.tid], validator=self.vals[0], eval_set="sha256:B", before=.5, after=.8)
        validate(ex, lid, {v: (.5, .8, 300) for v in ex.verdict(lid)["assigned"]})
        ex.claim_bounty(self.b["id"], lid, attest(CONSUMER, "sha256:B", "pass@1", .5, .75))
        sats = ex.wallet(self.backer)["balance_msats"]
        ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 50_000})   # the solution earns...
        for _ in range(6):
            ex.settle()
        self.assertEqual(ex.wallet(self.backer)["balance_msats"], sats)      # ...and the backer gets none of it
        payouts_ok(self, ex)


class Licences(unittest.TestCase):
    def test_licence_money_goes_to_the_traces_its_buyer_uses_not_to_junk(self):
        ex, vals = make_ex()
        ex.reserve = 2_000_000                                                # licences clear at 2,000 sats
        real = [ex.submit_trace(dict(trace(f"Write a function number {i} that adds two numbers.", code=f"return a+b+{i}")))["id"]
                for i in range(4)]
        junk_maker = A("7")
        ex.faucet(junk_maker)
        for i in range(40):                                                   # 40 junk traces in the same lot
            ex.submit_trace(dict(trace(f"junk {i}", producer=junk_maker, code=f"junk {i} " * 5)))
        lot = ex.lots()["lots"][0]["lot"]
        for who in (TRAINER, CONSUMER):
            ex.bid({"lot": lot, "bidder": who, "price_msats": 2_000_000})
        ex.clear()
        self.assertEqual(ex.economy_stats()["escrow_msats"]["payments"], 4_000_000)   # paid in sats, waiting
        with self.assertRaisesRegex(ValueError, "none of your licence money"):
            ex.direct_licence(lot, junk_maker, [])                           # nobody steers another buyer's money
        lid = learning(ex, real)                                              # the trainer's learning uses the 4
        validate(ex, lid, {v: (.6, .7, 300) for v in vals})
        p0 = ex.wallet(PRODUCER)["balance_msats"]
        ex.settle()
        self.assertEqual(ex.wallet(PRODUCER)["balance_msats"] - p0, 2_000_000 * 95 // 100)   # producer 85 + checker 10
        ex.direct_licence(lot, CONSUMER, real[:2])                            # the other buyer used two of them
        ex.settle()
        self.assertEqual(ex.economy_stats()["escrow_msats"]["payments"], 0)
        self.assertEqual(ex.wallet(junk_maker)["balance_msats"], 30_000_000 - 40 * FEE)   # junk earned nothing
        payouts_ok(self, ex)


class Storage(unittest.TestCase):
    def test_detail_is_pruned_after_the_challenge_window_and_balances_stay_exact(self):
        def run(keep):
            ex, vals = make_ex(keep_epochs=keep)
            lid = accepted(ex, vals)
            for i in range(12):                                              # an epoch of activity, twelve times
                ex.submit_trace(dict(trace(f"Write a function number {i} to add.", code=f"return a+b+{i}")))
                ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 5_000})
                ex.settle()
            return ex
        pruned, kept = run(6), run(1_000)
        for who in (TRAINER, PRODUCER, CONSUMER, OPERATOR, V(0)):            # the same balances, to the msat
            self.assertEqual(pruned.wallet(who), kept.wallet(who))
        cut = pruned.epoch - 1 - pruned.p.keep_epochs + 1
        old = lambda table: pruned.db.execute(f"SELECT COUNT(*) FROM {table} WHERE epoch < ?", (cut,)).fetchone()[0]
        self.assertEqual([old(t) for t in ("usage", "payments", "forfeits")], [0, 0, 0])
        self.assertEqual(pruned.db.execute("SELECT COUNT(*) FROM sledger WHERE epoch < ? AND memo != 'carried forward'",
                                           (cut,)).fetchone()[0], 0)
        self.assertLess(pruned.db.execute("SELECT COUNT(*) FROM sledger").fetchone()[0],
                        kept.db.execute("SELECT COUNT(*) FROM sledger").fetchone()[0])
        roots = pruned.db.execute("SELECT epoch, root, claims FROM sats_roots ORDER BY epoch").fetchall()
        self.assertEqual(len(roots), pruned.epoch - 1)                       # every epoch's root, for good
        self.assertTrue(all(json.loads(c) == {} for e, _, c in roots if e < cut))
        self.assertTrue(any(json.loads(c) for e, _, c in roots if e >= cut))
        self.assertTrue(pruned.audit()["balanced"] and kept.audit()["balanced"])


class Attacks(unittest.TestCase):
    def test_every_farming_strategy_loses_even_with_most_of_the_stake(self):
        import contextlib
        import io
        sys.path.insert(0, os.path.join(ROOT, "examples", "farming"))
        import attacks
        with contextlib.redirect_stdout(io.StringIO()):
            rows = attacks.main([])
        for name, pnl, extra, note in rows:
            if name.startswith("(honest"):
                self.assertEqual(extra, 0)
            elif name in attacks.NEUTRAL:
                # v0.8: padding a passing step trace (loops, or a detour nobody shortens) leaves the same single passing
                # trace, the same parents and the same fee as the unpadded twin; payment is per whole trace, so the
                # difference is exactly 0 by construction, never a gain
                self.assertEqual(extra, 0, f"{name} should earn exactly what the unpadded twin earns: {note}")
            else:
                self.assertLess(extra, 0, f"{name} should lose money against honest work: {note}")


class Scaling(unittest.TestCase):
    def test_payouts_scale_linearly_with_paid_usage(self):
        import contextlib
        import io
        sys.path.insert(0, os.path.join(ROOT, "examples", "scaling"))
        import simulate
        with contextlib.redirect_stdout(io.StringIO()):
            out = simulate.main(["--brief"])
        rows = out["rows"]
        for r in rows:                                                       # every paid msat is paid out, no more
            self.assertEqual(r["paid_out_msats"] + r["refunded_msats"], r["paid_in_msats"])
            self.assertLess(r["refunded_msats"], 10)                         # rounding, back to the payer
            self.assertEqual(r["traces"] + r["trainer"] + r["checkers"] + r["validators"], r["paid_out_msats"])
            self.assertEqual((r["escrow_left"], r["balanced"]), (0, True))
        ratio = [r["paid_out_msats"] / r["paid_in_msats"] for r in rows]
        self.assertTrue(max(ratio) - min(ratio) < 1e-6)                      # the same share at every scale
        self.assertAlmostEqual(out["machine_sats_per_day"] / 1e12, 1.1633, places=3)   # $1B a day at $85,962 a BTC


class Hosted(unittest.TestCase):
    def setUp(self):
        import threading
        from exchange import serve
        self.ex, self.srv = serve(0, ":memory:", public=True, admin_token="op", economy="sats", test_credits=30_000_000,
                                  beacon_delay=0)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_sats_api(self):
        from traceex import Client
        me = Client(self.url, CONSUMER)
        me.faucet()
        d = me.describe()
        self.assertEqual((d["settlement"]["unit"], d["settlement"]["token"]), ("msats", None))
        b = me.post_bounty(title="t", path="code", eval_set="sha256:E", target=.5, seed_msats=1_000_000)
        self.assertEqual(b["pledge"]["pool_msats"], 1_000_000)
        r = me.pledge(b["id"], msats=500_000)
        self.assertEqual(r["pool_msats"], 1_500_000)
        self.assertEqual(me.backers(b["id"])["backers"], {CONSUMER: 1_500_000})
        me.submit(trace(producer=CONSUMER))
        hits = me.search(sort="bounty")
        self.assertEqual((hits["total"], len(hits["bounties"])), (1, 1))
        w = me.wallet()
        self.assertEqual(w["balance_msats"], 30_000_000 - 1_500_000 - 4 * FEE)
        self.assertIsInstance(w["balance_msats"], int)
        e = me.economy()
        self.assertEqual((e["token"], e["paid_in_msats"], e["fees_msats"]), (None, 1_500_000, 4 * FEE))
        for path in ("/v0/validators", "/v0/learnings/x/commits", "/v0/learnings/x/reveals", "/v0/decoys",
                     "/v0/decoys/unseal", "/v0/licences/direct", "/v0/bounties/1/claims"):
            with self.assertRaisesRegex(RuntimeError, "^403"):
                me._call("POST", path, {})
        for path in ("/v0/swap", "/v0/credits", "/v0/bounties/1/buy", "/v0/bounties/1/sell"):
            with self.assertRaisesRegex(RuntimeError, "^410"):             # the token's routes are gone
                me._call("POST", path, {})
        for path in ("/v0/coin", "/v0/quote", "/v0/bounties/1/holders"):
            with self.assertRaisesRegex(RuntimeError, "^410"):
                me._call("GET", path)
        with self.assertRaisesRegex(RuntimeError, r"^400[\s\S]*prices everything in bitcoin"):   # dollars are refused
            me._call("POST", "/v0/bounties/1/pledges", {"backer": CONSUMER, "micros": 1_000_000})
        self.assertTrue(self.ex.audit()["balanced"])

    def test_a_payment_the_wallet_cannot_cover_answers_402_with_an_l402_challenge(self):
        from traceex import Client
        from traceex.client import PaymentRequired
        me = Client(self.url, CONSUMER)
        me.faucet()
        b = me.post_bounty(title="t", path="code", eval_set="sha256:E", target=.5)
        with self.assertRaises(PaymentRequired) as caught:
            me.pledge(b["id"], msats=40_000_000)                             # 40,000 sats; the wallet holds 30,000
        l402 = caught.exception.l402
        self.assertEqual(l402["scheme"], "L402")
        self.assertEqual(l402["amount_msats"], 40_000_000 + FEE - (30_000_000 - FEE))
        self.assertTrue(l402["invoice_is_placeholder"] and l402["invoice"].startswith("lntbs"))
        self.assertTrue(caught.exception.challenge.startswith('L402 macaroon="'))
        d = me.describe()
        self.assertEqual((d["payments"]["protocol"], d["settlement"]["priced_in"]), ("L402", "sats"))
        self.assertEqual(d["fee_per_transaction_msats"], 58)

    def test_operator_routes_for_validators_decoys_licences_and_claims(self):
        from traceex import Client
        ex = self.ex
        vals = [V(i) for i in range(3)]
        for v in vals:
            ex.faucet(v)
            Client(self.url, v, token="op").stake(15_000_000)
        for who in (TRAINER, PRODUCER, CONSUMER, A("3")):
            ex.faucet(who)
        ex.register_checker("unit-tests", PRODUCER)
        tid = ex.submit_trace(dict(trace()))["id"]
        op = lambda who=None: Client(self.url, who, token="op")
        b = ex.post_bounty({"poster": CONSUMER, "path": "code", "eval_set": "sha256:B", "target": .7, "title": "t"})
        ex.pledge(b["id"], CONSUMER, 1_000_000)
        lid = learning(ex, [tid], before=.5, after=.8)
        validate(ex, lid, {v: (.5, .8, 300) for v in vals})
        with self.assertRaisesRegex(RuntimeError, "poster's own measurement"):
            op().claim_bounty(b["id"], lid)
        r = op().claim_bounty(b["id"], lid, attest(CONSUMER, "sha256:B", "pass@1", .5, .75))
        self.assertEqual(r["status"], "solved")
        lot = ex.lots()["lots"][0]["lot"]
        ex.bid({"lot": lot, "bidder": CONSUMER, "price_msats": 50_000})
        ex.clear()
        self.assertGreater(op(CONSUMER).direct_licence(lot, [tid])["paid_msats"], 0)
        L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": "d", "hash": None},
                           parents=[(tid, 1)], trainer="0x" + "d2" * 20, per_call_msats=200,
                           attestation=attest(V(0), "sha256:c", "pass@1", .30, .45))
        decoy = op().register_decoy(L, decoy_digest(0.0, "s"), A("3"))["id"]
        validate(ex, decoy, {vals[0]: (.30, .45, 600), vals[1]: (.30, .30, 600), vals[2]: (.30, .29, 600)})
        self.assertEqual(op().unseal_decoy(decoy, 0.0, "s")["struck"], [vals[0]])
        self.assertTrue(ex.audit()["balanced"])

    def test_one_operator_runs_three_validators_from_one_folder(self):
        import contextlib
        import io
        import tempfile
        import validator
        vals = [V(i) for i in range(3)]
        for v in vals:
            self.ex.faucet(v)
            self.ex.register_validator(v, 15_000_000)
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
        ex, srv = serve(0, ":memory:", economy="sats", test_credits=30_000_000)    # no rate limits: all from one IP
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        people = [Client(url, "0x" + f"{i:02x}" * 20) for i in range(8)]
        for c in people:                                                     # (the faucet allows 3 per network a day)
            ex.db.execute("INSERT INTO grants VALUES (?,?,?,?)", (c.address, 30_000_000, "t", "test"))
        b = ex.post_bounty({"poster": people[0].address, "path": "code", "eval_set": "sha256:E", "target": .5})

        def work(c):
            for _ in range(12):
                c.stats(); c.bounties(status=""); c.wallet(); c.economy()
                c.pledge(b["id"], msats=50_000)
            return True
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            self.assertTrue(all(pool.map(work, people)))                     # one connection, many threads
        self.assertEqual(ex.backers(b["id"])["pool_msats"], 8 * 12 * 50_000)
        self.assertTrue(ex.audit()["balanced"])


class VercelPreview(unittest.TestCase):
    def test_the_vercel_function_serves_the_seed_and_refuses_every_write(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("vercel_node", os.path.join(ROOT, "api", "node.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.assertEqual(m.original_path("/api/node?x_route=/v0/search&q=add&path=code&x_tail=search"),
                         "/v0/search?q=add&path=code")
        self.assertEqual(m.original_path("/v0/faucet?x_route=%2Fv0%2Ffaucet&x_tail=faucet"), "/v0/faucet")
        self.assertEqual(m.ex.stats()["traces"], 244)                  # the code-repair runs; no flight example
        self.assertTrue(m.ex.describe()["read_only"])
        self.assertIsNone(m.ex.economy_stats()["token"])
        for name in m.WRITES:
            with self.assertRaisesRegex(PermissionError, "read-only preview"):
                getattr(m.ex, name)()
        for q in ("flight", "airline", "email", "travel"):
            self.assertEqual(m.ex.search(q=q)["total"], 0, q)
        self.assertFalse([x for x in m.ex.failures(limit=100)["failures"] if x["family"] != "qwen2.5"])
        self.assertTrue(all("flight" not in b["title"].lower() for b in m.ex.bounties()["bounties"]))
        self.assertEqual(len(m.ex.challenges()["challenges"]), 5)              # v0.8: open problems, read-only
        for name in ("post_challenge", "pledge_challenge", "submit_solution", "file_prior_art"):
            self.assertIn(name, m.WRITES)


class Seed(unittest.TestCase):
    def test_the_default_seed_is_the_code_repair_runs_only(self):
        from seed import seed_if_empty
        ex = SatsExchange(":memory:", test_credits=30_000_000, reserve_msats=50_000)
        seed_if_empty(ex)
        e = ex.economy_stats()
        self.assertEqual((ex.stats()["traces"], e["learnings"], ex.epoch), (244, {"accepted": 1}, 3))
        b = ex.bounties()["bounties"]
        self.assertEqual([x["status"] for x in b], ["open", "open"])        # the maintainer's, and one on a failure
        self.assertEqual(b[1]["failure_id"][:4], "TXF-")
        self.assertEqual([lot["lot"].split("|")[0] for lot in ex.lots()["lots"]], ["code.python"])
        self.assertGreater(e["paid_out_msats"], 0)
        ch = {c["key"]: c for c in ex.challenges()["challenges"]}           # v0.8: the imported sample, unfunded
        self.assertEqual(sorted(ch), ["alphaevolve:packing_circles_max_sum_of_radii:n=26",
                                      "alphaevolve:packing_circles_max_sum_of_radii:n=32",
                                      "alphaevolve:tammes_problem:n=25", "erdos:28", "erdos:3"])
        self.assertAlmostEqual(ch["alphaevolve:packing_circles_max_sum_of_radii:n=26"]["best"], 2.6359830849, places=9)
        self.assertEqual([len(ch[k]["sources"]) for k in ("erdos:3", "erdos:28")], [2, 2])   # database + Lean, merged
        self.assertEqual({c["verifier_kind"] for k, c in ch.items() if k.startswith("erdos")}, {"lean4"})
        self.assertEqual(sum(c["escrow_msats"] + c["backers"] for c in ch.values()), 0)     # no made-up pledges
        self.assertTrue(ex.audit()["balanced"])

    def test_the_sats_seed_runs_the_federation_for_real(self):
        from seed import seed_if_empty
        ex = SatsExchange(":memory:", test_credits=30_000_000, reserve_msats=50_000)
        seed_if_empty(ex, flight=True)
        self.assertTrue(all(json.loads(b)["royalty"].get("per_call_msats") is not None
                            for (b,) in ex.db.execute("SELECT body FROM learnings").fetchall()))
        e = ex.economy_stats()
        self.assertEqual(e["learnings"], {"accepted": 1, "inconclusive": 1})   # LoRA v2 proven; 3 emails are not proof
        self.assertEqual((ex.epoch, e["validators"]), (3, 3))
        self.assertGreater(e["paid_out_msats"], 0)                            # the host's usage, paid to the tree
        self.assertGreater(e["escrow_msats"]["vesting"], 0)                   # the traces' part, in escrow
        self.assertEqual([b["status"] for b in ex.bounties()["bounties"]], ["open", "open", "open"])   # v0.7: + a failure's
        self.assertEqual(ex.bounties()["bounties"][2]["failure_id"][:4], "TXF-")
        self.assertTrue(ex.audit()["balanced"])


class Versions(unittest.TestCase):
    def test_v06_refuses_an_older_database(self):
        import sqlite3
        import tempfile
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        v05 = os.path.join(folder.name, "v05.db")                             # a v0.5 testnet: TXC and bounty coins
        con = sqlite3.connect(v05)
        con.executescript("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT); INSERT INTO meta VALUES ('coin_version','0.5');"
                          "INSERT INTO meta VALUES ('pool_coin','1'); CREATE TABLE bounties (id INTEGER PRIMARY KEY, "
                          "poster TEXT, pool INT, supply REAL, status TEXT);")
        con.commit()
        con.close()
        with self.assertRaisesRegex(ValueError, r"v0\.5 coin economy \(TXC\); v0\.6 has no token"):
            SatsExchange(v05, test_credits=30_000_000)
        dollars = os.path.join(folder.name, "v01.db")                          # a dollar node's database, with data
        from exchange import Exchange
        old = Exchange(dollars)
        old._credit(A("1"), 5, "x")
        old.db.commit()
        old.db.close()
        with self.assertRaisesRegex(ValueError, "an older node.*no token"):
            SatsExchange(dollars, test_credits=30_000_000)
        mine = os.path.join(folder.name, "v06.db")
        SatsExchange(mine, test_credits=30_000_000).db.close()
        SatsExchange(mine, test_credits=30_000_000).db.close()                # its own database opens again

    def test_no_dollar_amounts_and_no_coin_economy(self):
        from exchange import serve
        with self.assertRaisesRegex(ValueError, "prices everything in sats"):
            SatsExchange(":memory:", reserve_micros=50_000)
        with self.assertRaisesRegex(ValueError, "no coin"):
            serve(0, ":memory:", economy="coin")
        ex, _ = make_ex(validators=0, quorum=0)
        ex.submit_trace(dict(trace()))
        with self.assertRaisesRegex(ValueError, "prices everything in bitcoin"):
            ex.bid({"lot": ex.lots()["lots"][0]["lot"], "bidder": CONSUMER, "price_micros": 5_000})
        with self.assertRaisesRegex(ValueError, "prices everything in bitcoin"):
            ex.post_bounty({"poster": CONSUMER, "path": "code", "eval_set": "sha256:E", "target": .5,
                            "seed_micros": 5_000})


if __name__ == "__main__":
    unittest.main()
