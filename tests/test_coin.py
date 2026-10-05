"""The coin economy (node/coin.py, testnet v0.5, everything priced in sats): python -m unittest tests.test_coin

Each test is one rule of the economy, most of them a rule that makes a farming strategy lose. Every test ends with
the books balanced: minted - burned == everything held by accounts and escrows + the pool's reserve, and every credit
made is spent or still held. Amounts are msats (1 credit = 1 msat); TXC opens at 10 sats (10,000 msats), so the
numbers here are the v0.4 tests' with micro-dollars read as msats, and each wallet takes 30,000 test sats."""
import json
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node")]

from traceex import Trace, Learning, attest  # noqa: E402
from coin import (CoinExchange, Params, UNIT, PQ, attestation_digest, decoy_digest, credits_for, units_for,  # noqa: E402
                  to_units, _frac)

A = lambda c: "0x" + c * 40
V = lambda i: "0x" + f"{0xa0 + i:02x}" * 20           # validator addresses
TRAINER, PRODUCER, CONSUMER, CHALLENGER, OPERATOR = A("c"), A("b"), A("e"), A("d"), A("f")


class Clock:
    """A clock the test moves (the time-weighted price needs time to pass); it stands still otherwise."""

    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make_ex(validators=3, quorum=3, delay=0, fee_to=OPERATOR, **params):
    ex = CoinExchange(":memory:", test_credits=30_000_000, beacon_delay=delay, params=Params(quorum=quorum, **params),
                      clock=Clock(), fee_to=fee_to)
    vals = []
    for i in range(validators):
        v = V(i)
        ex.faucet(v)
        ex.swap(v, "buy", 15_000_000)                     # 15,000 sats of TXC, then stake 1,000 of it (10,000 sats)
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
             weights=None, split=None, per_call=200):
    ex.swap(trainer, "buy", 6_000_000)                    # enough TXC for the 5,000-sat bond
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


def worth(ex, account, price=None):
    """An account's TXC (liquid + vesting) at a fixed price (the pool's current one by default), plus its sats and
    credits, in msats."""
    w = ex.wallet(account)
    return (w["balance_msats"] + w["credits_msats"]
            + (w["coin_units"] + w["vesting_units"]) * (price or ex.price()) // UNIT)


def minted(ex, kind="work"):
    return ex.last_mint()["minted_units"][kind]


class Pool(unittest.TestCase):
    def test_swaps_move_the_price_and_a_round_trip_costs_the_spread(self):
        ex, _ = make_ex(validators=0)
        p0 = ex.price()
        r = ex.swap(CONSUMER, "buy", 5_000_000)
        self.assertGreater(ex.price(), p0)
        ex.swap(CONSUMER, "sell", r["bought_units"])
        self.assertLess(ex.wallet(CONSUMER)["balance_msats"], 30_000_000)   # the pool keeps its 0.3% each way
        self.assertTrue(ex.audit()["balanced"])
        with self.assertRaisesRegex(ValueError, "not enough TXC"):
            ex.swap(CONSUMER, "sell", 10 * UNIT)

    def test_near_duplicates_pay_the_first_producer(self):
        ex, _ = make_ex(validators=0)
        t1 = ex.submit_trace(dict(trace()))
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


class Credits(unittest.TestCase):
    def test_a_sat_makes_a_thousand_credits_at_any_price_and_burns_the_txc_it_buys(self):
        ex, _ = make_ex(validators=0)
        burned0 = ex.coin_stats()["burned_units"]
        r = ex.buy_credits(CONSUMER, msats=1_000_000)
        self.assertEqual((r["credits_msats"], ex.wallet(CONSUMER)["credits_msats"]), (1_000_000, 1_000_000))
        self.assertEqual(ex.coin_stats()["burned_units"] - burned0, r["burned_units"])   # every TXC it bought, burned
        ex.swap(PRODUCER, "buy", 20_000_000)                                 # TXC gets dearer...
        r2 = ex.buy_credits(CONSUMER, msats=1_000_000)
        self.assertEqual(r2["credits_msats"], 1_000_000)                    # ...1,000 sats still make 1,000,000 credits
        self.assertLess(r2["burned_units"], r["burned_units"])               # by burning fewer TXC
        self.assertTrue(ex.audit()["balanced"])

    def test_credits_cannot_be_moved_or_turned_back(self):
        ex, _ = make_ex(validators=0)
        ex.buy_credits(CONSUMER, msats=1_000_000)
        self.assertFalse([m for m in dir(ex) if "credit" in m and any(w in m for w in ("transfer", "sell", "redeem",
                                                                                         "withdraw", "refund"))])
        sats = ex.wallet(CONSUMER)["balance_msats"]
        with self.assertRaisesRegex(ValueError, "not enough test sats"):    # credits never buy TXC or sats
            ex.swap(CONSUMER, "buy", sats + 500_000)
        self.assertEqual(ex.wallet(CONSUMER)["credits_msats"], 1_000_000)

    def test_burning_held_txc_counts_at_the_lower_of_spot_and_reference(self):
        ex, _ = make_ex(validators=0)
        units = ex.swap(CONSUMER, "buy", 1_000_000)["bought_units"]
        ex.swap(PRODUCER, "buy", 20_000_000)                                 # someone pumps the spot
        self.assertGreater(ex._price_q(), ex.ref_q())
        made = ex.buy_credits(CONSUMER, units=units)["credits_msats"]
        self.assertEqual(made, credits_for(units, ex.ref_q()))              # the reference price, not the pumped spot
        self.assertLess(made, credits_for(units, ex._price_q()))
        self.assertTrue(ex.audit()["balanced"])

    def test_payments_spend_credits_first_and_sats_top_them_up(self):
        ex, vals = make_ex()
        lid = accepted(ex, vals)
        ex.buy_credits(CONSUMER, msats=3_000_000)
        sats = ex.wallet(CONSUMER)["balance_msats"]
        ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 50_000})   # 10,000 sats of calls: 3,000 in credits
        w = ex.wallet(CONSUMER)
        self.assertEqual((w["credits_msats"], w["balance_msats"]), (0, sats - 7_000_000))
        self.assertTrue(ex.audit()["balanced"])


class BurnAndMint(unittest.TestCase):
    def test_a_verdict_a_submission_or_a_stake_mints_nothing(self):
        ex, vals = make_ex()
        accepted(ex, vals)
        s = ex.settle()
        self.assertEqual(s["mint"]["minted_units"]["work"], 0)              # accepted, staked, nobody paid for it
        fees = s["mint"]["credits_burned"]["operator"]                      # only the operator's fees earned anything
        self.assertLessEqual(s["mint"]["minted_units"]["operator"], units_for(fees, ex.last_mint()["twap_q"]))
        self.assertTrue(ex.audit()["balanced"])

    def test_above_equilibrium_nobody_is_minted_more_than_their_work_burned(self):
        ex, vals = make_ex()
        lid = accepted(ex, vals)
        burned0 = ex.coin_stats()["burned_units"]
        ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 50_000})   # $10, far less than the emission is worth
        burned = ex.coin_stats()["burned_units"] - burned0
        s = ex.settle()
        m = s["mint"]
        self.assertTrue(all(m["capped"].values()))                           # the cap binds for every recipient...
        work = ex.db.execute("SELECT credits, basis FROM work WHERE epoch=?", (s["epoch"],)).fetchall()
        twap = ex.last_mint()["twap_q"]
        cap = sum(min(units_for(c, twap), int(b)) for c, b in work)
        self.assertEqual(m["minted_units"]["work"] + m["minted_units"]["operator"], cap)   # ...and that is all it mints
        self.assertLessEqual(m["minted_units"]["work"], burned * 1004 // 1000)   # the TXC burned, before the spread
        self.assertEqual(ex.emission(), ex.p.emission)                       # unearned emission never rolls over
        self.assertTrue(ex.audit()["balanced"])

    def test_below_equilibrium_the_emission_is_shared_by_credits_burned(self):
        ex, vals = make_ex(emission=100 * UNIT)                              # an emission worth $1 at $0.01
        a = accepted(ex, vals, name="a")
        b = accepted(ex, vals, parents=[ex.submit_trace(dict(trace("Write a function to multiply.", code="a*b")))["id"]],
                     name="b")
        ex.usage({"learning": a, "consumer": CONSUMER, "calls": 15_000})     # $3 on one
        ex.usage({"learning": b, "consumer": CONSUMER, "calls": 5_000})      # $1 on the other: $4 against $1
        m = ex.settle()["mint"]
        self.assertFalse(m["capped"]["work"])                                # below the cap: shared out by credits
        self.assertGreater(m["minted_units"]["work"], 90 * UNIT * 999 // 1000)   # the whole work share, but rounding
        self.assertLessEqual(m["minted_units"]["work"], 90 * UNIT)
        by_learning = {}
        for ref, units in ex.db.execute("SELECT learning, units FROM vesting WHERE role='parents'").fetchall():
            by_learning[ref] = by_learning.get(ref, 0) + int(units)
        self.assertAlmostEqual(by_learning[a] / by_learning[b], 3.0, places=2)   # pro rata to the credits
        self.assertTrue(ex.audit()["balanced"])

    def test_the_cap_uses_the_time_weighted_price_not_the_spot(self):
        ex, vals = make_ex()
        lid = accepted(ex, vals)
        ex.settle()
        ex.clock.advance(86_400)                                             # a day at the opening price...
        ex.swap(PRODUCER, "buy", 20_000_000)                                 # ...then the spot is pushed up at the end
        ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 50_000})
        s = ex.settle()
        twap = ex.last_mint()["twap_q"]
        self.assertLess(twap, ex._price_q())                                # the push had no time to count
        rows = ex.db.execute("SELECT credits, basis FROM work WHERE epoch=?", (s["epoch"],)).fetchall()
        self.assertEqual(minted(ex) + minted(ex, "operator"), sum(min(units_for(c, twap), int(b)) for c, b in rows if c))
        self.assertTrue(ex.audit()["balanced"])

    def test_credits_made_when_txc_was_cheap_earn_only_todays_value(self):
        ex, vals = make_ex()
        lid = accepted(ex, vals)
        ex.buy_credits(TRAINER, msats=5_000_000)                            # credits made at about $0.01...
        for _ in range(3):
            ex.swap(PRODUCER, "buy", 6_000_000)
            ex.clock.advance(86_400)
            ex.settle()                                                      # ...then TXC gets dearer, for days
        ex.usage({"learning": lid, "consumer": TRAINER, "calls": 25_000})    # spent on its own learning
        s = ex.settle()
        twap = ex.last_mint()["twap_q"]
        rows = ex.db.execute("SELECT credits, basis FROM work WHERE epoch=? AND account=?", (s["epoch"], TRAINER)).fetchall()
        self.assertTrue(rows)
        for c, b in rows:                                                    # today's value binds, not the old burn
            self.assertLess(units_for(c, twap), int(b))
        self.assertTrue(ex.audit()["balanced"])

    def test_paying_for_your_own_learning_loses_money(self):
        ex, vals = make_ex()
        lid = accepted(ex, vals)
        farmer = A("7")
        ex.faucet(farmer)
        for _ in range(5):                                                   # the trainer's bond is home
            ex.settle()
        price = ex.price()
        ring = lambda: sum(worth(ex, who, price) for who in (farmer, PRODUCER, TRAINER))
        before = ring()
        ex.usage({"learning": lid, "consumer": farmer, "calls": 50_000})     # $10 of usage, to itself
        for _ in range(5):                                                   # the mint, all vested
            ex.settle()
        self.assertLess(ring(), before - 400_000)                            # the validators' 5%, the spread, the fee
        self.assertTrue(ex.audit()["balanced"])

    def test_the_operator_is_minted_for_its_fees_and_no_more(self):
        ex, _ = make_ex(validators=0)
        for i in range(40):
            ex.submit_trace(dict(trace(f"Write a function number {i} that adds.", code=f"return a+b+{i}")))
        s = ex.settle()
        fees = s["mint"]["credits_burned"]["operator"]
        self.assertEqual(fees, 40 * 58)                                      # 40 transactions at 58 msats
        self.assertGreater(ex.wallet(OPERATOR)["coin_units"], 0)
        self.assertLessEqual(ex.wallet(OPERATOR)["coin_units"], units_for(fees, ex.last_mint()["twap_q"]))
        self.assertTrue(ex.audit()["balanced"])

    def test_halvings_stop_after_five(self):
        ex, _ = make_ex(validators=0)
        self.assertEqual([ex.emission(e) // UNIT for e in (1, 180, 181, 361, 901, 5_000)],
                         [50_000, 50_000, 25_000, 12_500, 1_562, 1_562])
        self.assertEqual(ex.emission(901), 1_562 * UNIT + UNIT // 2)         # 1,562.5 TXC an epoch, for good


class Fee(unittest.TestCase):
    def test_every_transaction_pays_the_standard_fee_in_credits_and_it_burns(self):
        from exchange import TX_FEE_MSATS
        self.assertEqual(TX_FEE_MSATS, 58)                                   # about $0.00005 at $85,962 a bitcoin
        ex, _ = make_ex(validators=0)
        owed = lambda who: ex.wallet(who)["owed_fee_msats"]
        for i in range(3):
            ex.submit_trace(dict(trace(f"Write a function number {i} to add two numbers.", code=f"return a + b + {i}")))
        self.assertEqual(owed(PRODUCER), 3 * TX_FEE_MSATS)
        self.assertEqual(ex.wallet(PRODUCER)["balance_msats"], 30_000_000)  # nothing billed until settlement
        burned0 = ex.coin_stats()["burned_units"]
        ex.settle()
        self.assertEqual(ex.wallet(PRODUCER)["balance_msats"], 30_000_000 - 174)   # 174 credits, topped up from sats
        self.assertEqual(owed(PRODUCER), 0)
        self.assertEqual(ex.fees()["burned_msats"], 174)
        self.assertGreater(ex.coin_stats()["burned_units"], burned0)         # the credits burned TXC
        with self.assertRaisesRegex(ValueError, "not enough test sats"):
            ex.submit_trace(dict(trace(producer=A("9"))))                    # no wallet, no transaction
        self.assertTrue(ex.audit()["balanced"])

    def test_an_account_cannot_spend_what_it_owes_before_settlement(self):
        ex, _ = make_ex(validators=0)
        for i in range(10):
            ex.submit_trace(dict(trace(f"Write a function number {i} to add.", code=f"return a+b+{i}")))
        sats = ex.wallet(PRODUCER)["balance_msats"]
        with self.assertRaisesRegex(ValueError, "not enough test sats"):
            ex.swap(PRODUCER, "buy", sats)                                   # the fees it owes are spoken for
        ex.swap(PRODUCER, "buy", sats - 1_000)
        ex.settle()
        self.assertEqual(ex.fees()["burned_msats"], 11 * 58)                # every fee was paid, the swap's too
        self.assertTrue(ex.audit()["balanced"])

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
        self.assertTrue(ex.audit()["balanced"])


class Divisibility(unittest.TestCase):
    def test_amounts_stay_payable_at_a_trillion_sats_a_txc(self):
        ex, vals = make_ex(validators=0, quorum=0, genesis_coins=10 * UNIT, genesis_msats=10 ** 16)   # 1e12 sats a TXC
        self.assertEqual(ex.price(), 10 ** 15)                               # msats per TXC
        r = ex.buy_credits(CONSUMER, msats=58)                              # one fee's worth of credits
        self.assertGreater(r["burned_units"], 10_000)                        # 58 credits burn ~58,000 base units
        self.assertIn(units_for(1, ex._price_q()), (999, 1_000))            # one credit = ~1,000 units; 6 decimals: 0
        self.assertEqual(1 * 10 ** 6 * PQ // ex._price_q(), 0)
        lid = learning(ex, [ex.submit_trace(dict(trace()))["id"]], per_call=1)
        ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 3})        # three calls at one credit each
        ex.settle()
        self.assertGreater(minted(ex), 0)                                    # still minted, in thousandths of a credit
        self.assertTrue(ex.audit()["balanced"])

    def test_json_carries_txc_amounts_as_strings(self):
        from exchange import wire
        w = wire({"coin_units": 12 * UNIT, "price_msats": 10_000, "claims": {A("a"): {"amount_units": 5, "proof": []}},
                  "vesting_units": {A("b"): 7 * UNIT}, "big": 2 ** 60})
        self.assertEqual(w["coin_units"], "12000000000000000000")
        self.assertEqual((w["price_msats"], w["claims"][A("a")]["amount_units"], w["vesting_units"][A("b")], w["big"]),
                         (10_000, "5", "7000000000000000000", str(2 ** 60)))
        self.assertEqual(to_units("1.5"), 15 * UNIT // 10)


class SatsPriced(unittest.TestCase):
    def test_bonds_stakes_and_challenges_cost_the_same_sats_at_any_price(self):
        ex, _ = make_ex(validators=0)
        at_genesis = (ex.learning_bond_units(), ex.min_stake_units(), ex.challenge_stake_units())
        self.assertEqual(at_genesis, (500 * UNIT, 1_000 * UNIT, 200 * UNIT))   # 5,000, 10,000, 2,000 sats at 10 sats
        ex.swap(PRODUCER, "buy", 10_000_000)
        ex.clock.advance(86_400)
        ex.settle()                                                          # TXC is dearer; the reference follows
        ref = ex.ref_q()
        self.assertGreater(ref, 10_000 * PQ)
        for units, msats in zip((ex.learning_bond_units(), ex.min_stake_units(), ex.challenge_stake_units()),
                                (5_000_000, 10_000_000, 2_000_000)):
            self.assertEqual(units, units_for(msats, ref))                   # fewer TXC, the same sats
            self.assertLess(units, at_genesis[0] * msats // 5_000_000)

    def test_a_sat_backs_a_bounty_with_the_same_coins_at_any_price(self):
        coins = []
        for pool in (10_000_000 * 1000, 10_000_000_000 * 1000):             # TXC at 10 sats, and at 10,000
            ex, _ = make_ex(validators=0, genesis_msats=pool)
            b = ex.post_bounty({"poster": CONSUMER, "path": "code", "eval_set": "sha256:B", "target": .7, "title": "t"})
            coins.append(ex.buy_coins(b["id"], CONSUMER, 1_000_000)["coins"])
            self.assertEqual(ex.bounties()["bounties"][0]["price_msats"], round(10_000 + 100 * coins[-1]))
        self.assertAlmostEqual(coins[0], coins[1], delta=0.5)                # about 73 coins either way
        self.assertAlmostEqual(coins[0], 73.2, delta=0.5)                    # 10 sats + 0.1 a coin: 10n + 0.05n^2 = 1,000

    def test_pumping_the_pool_buys_no_extra_bounty_coins(self):
        ex, _ = make_ex(validators=0)
        b = ex.post_bounty({"poster": CONSUMER, "path": "code", "eval_set": "sha256:B", "target": .7, "title": "t"})
        ex.swap(PRODUCER, "buy", 15_000_000)                                 # the pump
        r = ex.buy_coins(b["id"], CONSUMER, 5_000_000)
        self.assertLess(r["spent_msats"], 5_000_000)                        # only the TXC that went in counts
        self.assertEqual(r["spent_msats"], credits_for(r["spent_units"], ex.ref_q()))
        self.assertTrue(ex.audit()["balanced"])

    def test_validators_the_price_pushes_under_the_minimum_keep_their_seat_while_they_top_up(self):
        ex, vals = make_ex(validators=3)
        whale = A("7")
        ex.faucet(whale)
        ex._add("minted", 300_000 * UNIT)
        ex._coin(whale, 300_000 * UNIT, "earned before")
        ex.swap(whale, "sell", 300_000 * UNIT)                               # TXC falls about 40%
        ex.clock.advance(86_400)
        ex.settle()
        self.assertLess(ex._stake(vals[0]), ex.min_stake_units())           # under the sats minimum now...
        self.assertTrue(all(ex.validator(v)["active"] for v in vals))        # ...but still seated
        for _ in range(ex.p.stake_grace_epochs):
            ex.settle()
        self.assertFalse(ex.validator(vals[0])["active"])                    # the grace ran out
        ex.swap(vals[0], "buy", 5_000_000)
        ex.register_validator(vals[0], ex.min_stake_units() - ex._stake(vals[0]))
        self.assertTrue(ex.validator(vals[0])["active"])                     # topped up
        self.assertTrue(ex.audit()["balanced"])


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
        for who in (TRAINER, PRODUCER, CONSUMER, OPERATOR, V(0)):            # the same balances, to the unit
            self.assertEqual({k: v for k, v in pruned.wallet(who).items() if k != "price_msats"},
                             {k: v for k, v in kept.wallet(who).items() if k != "price_msats"})
        cut = pruned.epoch - 1 - pruned.p.keep_epochs + 1
        old = lambda table: pruned.db.execute(f"SELECT COUNT(*) FROM {table} WHERE epoch < ?", (cut,)).fetchone()[0]
        self.assertEqual([old(t) for t in ("usage", "paid", "work", "burns")], [0, 0, 0, 0])
        self.assertEqual(pruned.db.execute("SELECT COUNT(*) FROM coin_ledger WHERE epoch < ? AND memo != 'carried forward'",
                                           (cut,)).fetchone()[0], 0)
        self.assertLess(pruned.db.execute("SELECT COUNT(*) FROM coin_ledger").fetchone()[0],
                        kept.db.execute("SELECT COUNT(*) FROM coin_ledger").fetchone()[0])
        roots = pruned.db.execute("SELECT epoch, root, claims FROM coin_roots ORDER BY epoch").fetchall()
        self.assertEqual(len(roots), pruned.epoch - 1)                       # every epoch's root, for good
        self.assertTrue(all(json.loads(c) == {} for e, _, c in roots if e < cut))   # old claims folded away
        self.assertTrue(any(json.loads(c) for e, _, c in roots if e >= cut))
        self.assertEqual(pruned.stats()["traces"], kept.stats()["traces"])   # traces stay
        self.assertTrue(pruned.audit()["balanced"] and kept.audit()["balanced"])


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
        self.assertEqual(ex.wallet(TRAINER)["coin_units"], coins)                     # the $5 bond is gone...
        self.assertEqual(ex.coin_stats()["burned_units"], burned + ex.learning_bond_units())   # ...to nobody: it burned
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
        coins, bond = ex.wallet(TRAINER)["coin_units"], ex.verdict(lid)["bond_units"]
        v = validate(ex, lid, {vals[0]: (.6, .8, 10), vals[1]: (.7, .8, 10), vals[2]: (.6, .6, 10)})   # 10-item evals
        self.assertEqual(v["status"], "inconclusive")                       # +10 points on 10 items proves nothing
        self.assertTrue(all(r["agreed"] for r in v["reveals"]))
        self.assertEqual(sum(ex.validator(x)["slashed_units"] for x in vals), 0)
        self.assertEqual(ex.wallet(TRAINER)["coin_units"], coins + bond - _frac(bond, 0.10))
        self.assertEqual(ex.settle()["mint"]["minted_units"]["work"], 0)
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

    def test_only_an_accepted_learning_can_be_paid_for(self):
        ex = self.ex
        lid = learning(ex, [self.tid])
        with self.assertRaisesRegex(ValueError, "only a learning the validator federation accepted"):
            ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 10})


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
                                                   trainer=TRAINER, per_call_msats=200,
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
        share = lambda role: sum(int(u) - int(r) for u, r in ex.db.execute(
            "SELECT units, released FROM vesting WHERE learning=? AND role=? AND status='vesting'", (lid, role)).fetchall())
        self.assertGreater(share("parents"), 0)                             # the parents' share vests...
        bond = ex.verdict(lid)["bond_units"]
        ex.swap(CHALLENGER, "buy", 5_000_000)
        ex.challenge(lid, CHALLENGER)
        ex.settle()
        validate(ex, lid, {x: (.6, .7, 300) for x in ex.verdict(lid)["assigned"]}, rnd=1, bad=9)
        self.assertEqual((ex.verdict(lid)["status"], ex.verdict(lid)["note"]), ("accepted", "padded"))
        self.assertEqual(share("parents"), 0)                                # ...so the audit can still claw it back
        self.assertGreater(ex.wallet(TRAINER)["coin_units"], 0)              # the gain is real; the padding is not
        self.assertEqual(bond, 0)                                             # (the bond had already gone home)
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
        self.assertEqual(ex.wallet(CHALLENGER)["coin_units"], stake - ex.challenge_stake_units())
        self.assertEqual(ex.coin_stats()["burned_units"], burned + ex.challenge_stake_units())   # ...it burned
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
        self.assertGreater(r["vesting_units"][TRAINER], self.pool // 2)      # the pool vests to the solver...
        self.assertLessEqual(sum(r["vesting_units"].values()), self.pool)    # ...and the traces, never more
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
        self.assertGreaterEqual(back, late["spent_units"] * 99 // 100)        # the TXC it put in
        self.assertTrue(ex.audit()["balanced"])

    def test_a_fake_solve_is_clawed_back_and_the_backers_refunded(self):
        ex = self.ex
        lid = learning(ex, [self.tid], validator=self.vals[0], eval_set="sha256:B", before=.5, after=.8)
        first = ex.verdict(lid)["assigned"]
        validate(ex, lid, {v: (.5, .8, 300) for v in first})                 # colluders, and a poster it fooled
        ex.claim_bounty(self.b["id"], lid, attest(CONSUMER, "sha256:B", "pass@1", .5, .72))
        ex.swap(CHALLENGER, "buy", 5_000_000)                                # a challenge stakes $2 of TXC
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

    def test_coin_holders_are_minted_their_share_of_the_solutions_usage(self):
        ex = self.ex
        lid = learning(ex, [self.tid], validator=self.vals[0], eval_set="sha256:B", before=.5, after=.8)
        validate(ex, lid, {v: (.5, .8, 300) for v in ex.verdict(lid)["assigned"]})
        ex.claim_bounty(self.b["id"], lid, attest(CONSUMER, "sha256:B", "pass@1", .5, .75))
        coins = ex.wallet(self.backer)["coin_units"]
        ex.usage({"learning": lid, "consumer": CONSUMER, "calls": 50_000})   # $10 of use
        s = ex.settle()
        held = ex.db.execute("SELECT credits FROM work WHERE epoch=? AND role='holders'", (s["epoch"],)).fetchall()
        self.assertEqual(sum(c for (c,) in held), 2_000_000)                 # 20% of the credits burned on it
        self.assertGreater(ex.wallet(self.backer)["coin_units"], coins)
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
            ex.bid({"lot": lot, "bidder": who, "price_msats": 2_000_000})
        ex.clear()
        self.assertEqual(ex.coin_stats()["licence_escrow_msats"], 4_000_000)   # paid in credits, burned, waiting
        with self.assertRaisesRegex(ValueError, "none of your licence money"):
            ex.direct_licence(lot, junk_maker, junk)                          # nobody steers another buyer's money
        lid = learning(ex, real)                                              # the trainer's learning uses the 4
        validate(ex, lid, {v: (.6, .7, 300) for v in vals})
        ex.settle()
        self.assertGreater(ex.wallet(PRODUCER)["coin_units"], 0)
        ex.direct_licence(lot, CONSUMER, real[:2])                            # the other buyer used two of them
        ex.settle()
        self.assertEqual(ex.coin_stats()["licence_escrow_msats"], 0)
        self.assertEqual(ex.wallet(junk_maker)["coin_units"], 0)              # 40 junk traces earned nothing
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
                           per_call_msats=200)
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
        self.assertFalse(ex.validator(lazy)["active"])                       # slashed under the minimum: seat gone now
        self.assertEqual(ex.validator(vals[1])["slashed_units"], 0)
        self.assertEqual(ex.wallet(operator)["coin_units"], bond + ex.learning_bond_units())   # the decoy's bond comes home
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
            rows = attacks.main([])
        for name, pnl, extra, note in rows:
            if name.startswith("(honest"):
                self.assertEqual(extra, 0)
            else:
                self.assertLess(extra, 0, f"{name} should lose money against honest work: {note}")


class Scaling(unittest.TestCase):
    def test_txc_follows_the_usage_rate_not_the_sum_of_all_payments(self):
        import contextlib
        import io
        sys.path.insert(0, os.path.join(ROOT, "examples", "scaling"))
        import simulate
        with contextlib.redirect_stdout(io.StringIO()):
            out = simulate.main(["--brief"])
        g = out["growth"]
        for t in (180, 540, 900, 1080):                                      # growth: within 2% of P* = usage / emission
            v3, v4 = g[t]
            self.assertAlmostEqual(v4["price"] / v4["equilibrium"], 1, delta=0.02)
        self.assertGreater(g[1800][0]["price"] / g[1800][1]["price"], 1e6)    # v0.3 compounds on everything ever paid
        self.assertLess(g[1440][1]["price"], 2 * g[1440][1]["equilibrium"])  # v0.4 comes down after the crash
        self.assertGreater(g[1800][1]["take"], 0.99)                          # contributors realise what users pay
        self.assertLess(max(r[1]["price"] for r in out["machine"].values()), 1e12)  # sats a TXC: one msat is still
        #                                                                     10^3+ base units at every price reached
        self.assertAlmostEqual(out["machine_sats_per_day"] / 1e12, 1.1633, places=3)   # $1B a day at $85,962 a BTC


class Hosted(unittest.TestCase):
    def setUp(self):
        import threading
        from exchange import serve
        self.ex, self.srv = serve(0, ":memory:", public=True, admin_token="op", economy="coin", test_credits=30_000_000,
                                  beacon_delay=0, clock=Clock())
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
        self.assertIsInstance(got, str)                                      # 18 decimals travel as strings
        self.assertGreater(int(got), 150 * UNIT)
        b = me.post_bounty(title="t", path="code", eval_set="sha256:E", target=.5)
        r = me.back_with_coins(b["id"], "100")                               # 100 TXC, 1,000 sats at the reference price
        self.assertAlmostEqual(r["coins"], 73.2, delta=0.5)                  # the sats curve: 10 sats, +0.1 sat a coin
        self.assertGreater(me.buy_coins(b["id"], msats=1_000_000)["coins"], 0)       # sats are swapped on the way in
        made = me.buy_credits(msats=500_000)
        self.assertEqual(made["credits_msats"], 500_000)
        c = me.coin()
        self.assertEqual((c["symbol"], c["decimals"]), ("TXC", 18))
        self.assertIsInstance(c["supply_units"], str)
        me.submit(trace(producer=CONSUMER))
        hits = me.search(sort="bounty")
        self.assertEqual((hits["total"], len(hits["bounties"])), (1, 1))
        w = me.wallet()
        self.assertEqual((int(w["coin_units"]), w["credits_msats"]), (int(got) - 100 * UNIT, 500_000))
        for path in ("/v0/validators", "/v0/learnings/x/commits", "/v0/learnings/x/reveals", "/v0/decoys",
                     "/v0/decoys/unseal", "/v0/licences/direct", "/v0/bounties/1/claims"):
            with self.assertRaisesRegex(RuntimeError, "^403"):
                me._call("POST", path, {})
        with self.assertRaisesRegex(RuntimeError, r"^400[\s\S]*prices everything in bitcoin"):   # dollars are refused
            me._call("POST", "/v0/credits", {"account": CONSUMER, "micros": 1_000_000})
        self.assertIsInstance(c["price_msats"], int)                         # msats a TXC, an integer
        self.assertTrue(c["price"].endswith("sats") and c["price_usd_approx"].startswith("$"))
        self.assertTrue(self.ex.audit()["balanced"])

    def test_a_payment_the_wallet_cannot_cover_answers_402_with_an_l402_challenge(self):
        import urllib.error
        import urllib.request
        from traceex import Client
        from traceex.client import PaymentRequired
        me = Client(self.url, CONSUMER)
        me.faucet()
        with self.assertRaises(PaymentRequired) as caught:
            me.buy_credits(msats=40_000_000)                                 # 40,000 sats; the wallet holds 30,000
        l402 = caught.exception.l402
        self.assertEqual((l402["scheme"], l402["amount_msats"]), ("L402", 10_000_000))
        self.assertTrue(l402["invoice_is_placeholder"] and l402["invoice"].startswith("lntbs"))
        self.assertTrue(caught.exception.challenge.startswith('L402 macaroon="'))
        d = me.describe()
        self.assertEqual((d["payments"]["protocol"], d["settlement"]["priced_in"]), ("L402", "sats"))
        self.assertEqual(d["fee_per_transaction_msats"], 58)

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
        ex.bid({"lot": lot, "bidder": CONSUMER, "price_msats": 50_000})
        ex.clear()
        self.assertGreater(op(CONSUMER).direct_licence(lot, [tid])["paid_msats"], 0)
        ex.swap(A("3"), "buy", 6_000_000)                                     # the operator's decoy fund
        L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": "d", "hash": None},
                           parents=[(tid, 1)], trainer="0x" + "d2" * 20, per_call_msats=200,
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
        ex, srv = serve(0, ":memory:", economy="coin", test_credits=30_000_000)    # no rate limits: all from one IP
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        people = [Client(url, "0x" + f"{i:02x}" * 20) for i in range(8)]
        for c in people:                                                     # (the faucet allows 3 per network a day)
            ex.db.execute("INSERT INTO grants VALUES (?,?,?,?)", (c.address, 30_000_000, "t", "test"))

        def work(c):
            for _ in range(12):
                c.stats(); c.bounties(status=""); c.wallet(); c.coin()
                c.swap("buy", 50_000)
            return True
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            self.assertTrue(all(pool.map(work, people)))                     # one connection, many threads
        self.assertTrue(ex.audit()["balanced"])


class VercelPreview(unittest.TestCase):
    def test_the_vercel_function_serves_the_seed_and_refuses_every_write(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("vercel_node", os.path.join(ROOT, "api", "node.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.assertEqual(m.original_path("/api/node?x_route=/v0/search&q=add&path=code&x_tail=search"),
                         "/v0/search?q=add&path=code")                      # however Vercel hands the request over
        self.assertEqual(m.original_path("/v0/faucet?x_route=%2Fv0%2Ffaucet&x_tail=faucet"), "/v0/faucet")
        self.assertEqual(m.ex.stats()["traces"], 247)
        self.assertTrue(m.ex.describe()["read_only"])
        for name in m.WRITES:
            with self.assertRaisesRegex(PermissionError, "read-only preview"):
                getattr(m.ex, name)()
        self.assertGreater(m.ex.search(q="flight")["total"], 0)


class Seed(unittest.TestCase):
    def test_the_coin_seed_runs_the_federation_for_real(self):
        from seed import seed_if_empty
        ex = CoinExchange(":memory:", test_credits=30_000_000, reserve_msats=50_000)
        seed_if_empty(ex)
        self.assertTrue(all(json.loads(b)["royalty"].get("per_call_msats") is not None
                            for (b,) in ex.db.execute("SELECT body FROM learnings").fetchall()))   # priced in msats
        c = ex.coin_stats()
        self.assertEqual(c["learnings"], {"accepted": 1, "inconclusive": 1})   # LoRA v2 proven; 3 emails are not proof
        self.assertEqual((ex.epoch, c["validators"]), (3, 3))
        self.assertGreater(c["burned_units"], 0)
        self.assertGreater(c["vesting_units"], 0)                             # the host's usage, minted to its traces
        self.assertEqual(c["licence_escrow_msats"], 0)                       # every buyer named what it used
        self.assertEqual([b["status"] for b in ex.bounties()["bounties"]], ["open", "open"])
        self.assertAlmostEqual(ex.price() / 10_000, 1, delta=0.05)           # opens near 10 sats
        self.assertTrue(ex.audit()["balanced"])


class Versions(unittest.TestCase):
    def test_v05_refuses_an_older_database(self):
        import sqlite3
        import tempfile
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = os.path.join(folder.name, "v04.db")
        CoinExchange(path, test_credits=30_000_000).db.close()
        con = sqlite3.connect(path)
        con.execute("UPDATE meta SET v='0.4' WHERE k='coin_version'")        # a v0.4 (dollar-priced) testnet
        con.commit()
        con.close()
        with self.assertRaisesRegex(ValueError, r"v0\.4 coin economy \(priced in dollars\); v0\.5 prices everything in sats"):
            CoinExchange(path, test_credits=30_000_000)

    def test_a_coin_node_takes_no_dollar_amounts(self):
        with self.assertRaisesRegex(ValueError, "prices everything in sats"):
            CoinExchange(":memory:", reserve_micros=50_000)
        ex, _ = make_ex(validators=0, quorum=0)
        tid = ex.submit_trace(dict(trace()))["id"]
        L = Learning.build(kind="lora", task="code.python", base_model="qwen", artifact={"uri": "d", "hash": None},
                           parents=[(tid, 1)], trainer=TRAINER, per_call_micros=200,
                           attestation=attest(V(0), "sha256:c", "pass@1", .6, .7))
        with self.assertRaisesRegex(ValueError, "per_call_msats"):
            ex.register_learning(L)
        with self.assertRaisesRegex(ValueError, "prices everything in bitcoin"):
            ex.bid({"lot": ex.lots()["lots"][0]["lot"], "bidder": CONSUMER, "price_micros": 5_000})


if __name__ == "__main__":
    unittest.main()
