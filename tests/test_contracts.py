"""Contract tests on an in-process EVM (web3 + eth-tester/py-evm, solc 0.8.26 via py-solc-x).

Skipped cleanly when web3 / eth_tester / solcx are not importable, so `python -m unittest discover tests` still runs
with the standard library only. To run them:

    uv venv --python 3.12 .venv
    echo "safe-pysha3; sys_platform == 'never'" > ov.txt      # Windows: avoid a C build, use pycryptodome instead
    uv pip install --python .venv/Scripts/python.exe --override ov.txt web3 "eth-tester[py-evm]" \
        "eth-hash[pycryptodome]" py-solc-x
    .venv/Scripts/python.exe -c "import solcx; solcx.install_solc('0.8.26')"
    .venv/Scripts/python.exe -m unittest tests.test_contracts -v
"""
import glob
import math
import os
import sys
import unittest
from fractions import Fraction

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python")]

from traceex import merkle, bountycoin  # noqa: E402

try:
    import solcx
    from eth_tester import EthereumTester
    from web3 import Web3, EthereumTesterProvider
    from web3.exceptions import ContractLogicError
    from eth_tester.exceptions import TransactionFailed
    REVERT = (ContractLogicError, TransactionFailed)
    HAVE_EVM = True
except Exception:  # pragma: no cover - depends on the machine
    HAVE_EVM = False

SOLC = "0.8.26"
ONE = 10**6          # coin units per coin, and USDC micros per dollar
DAY = 86400
_BUILD = None


def _build():
    """Compile every contract once per process; returns {name: (abi, bytecode)}."""
    global _BUILD
    if _BUILD is None:
        if SOLC not in [str(v) for v in solcx.get_installed_solc_versions()]:
            solcx.install_solc(SOLC)
        cdir = os.path.abspath(os.path.join(ROOT, "contracts"))
        files = sorted(glob.glob(os.path.join(cdir, "*.sol")) + glob.glob(os.path.join(cdir, "test", "*.sol")))
        out = solcx.compile_files(files, output_values=["abi", "bin"], solc_version=SOLC, optimize=True)
        _BUILD = {k.split(":")[-1]: (v["abi"], v["bin"]) for k, v in out.items()}
    return _BUILD


@unittest.skipUnless(HAVE_EVM, "web3 / eth_tester / py-solc-x not installed")
class EVMCase(unittest.TestCase):
    def setUp(self):
        self.tester = EthereumTester()
        self.w3 = Web3(EthereumTesterProvider(self.tester))
        self.acct = self.w3.eth.accounts
        self.owner = self.acct[0]
        self.w3.eth.default_account = self.owner
        self.usdc = self.deploy("MockUSDC")

    # ---- helpers ---------------------------------------------------------------------------------------------
    def deploy(self, name, *args):
        abi, bytecode = _build()[name]
        f = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        rcpt = self.w3.eth.wait_for_transaction_receipt(f.constructor(*args).transact({"from": self.owner}))
        self.assertEqual(rcpt.status, 1)
        return self.w3.eth.contract(address=rcpt.contractAddress, abi=abi)

    def tx(self, fn, sender):
        rcpt = self.w3.eth.wait_for_transaction_receipt(fn.transact({"from": sender}))
        self.assertEqual(rcpt.status, 1)
        return rcpt

    def reverts(self, fn, sender, reason=None):
        with self.assertRaises(REVERT) as cm:
            fn.transact({"from": sender})
        if reason:
            self.assertIn(reason, str(cm.exception))

    def bal(self, who):
        return self.usdc.functions.balanceOf(who).call()

    def fund(self, who, amount, spender):
        self.tx(self.usdc.functions.mint(who, amount), who)
        self.tx(self.usdc.functions.approve(spender.address, 2**256 - 1), who)

    def now(self):
        return self.w3.eth.get_block("latest").timestamp

    def travel(self, seconds):
        self.tester.time_travel(self.now() + seconds)


# ======================================================================================================================
class PayoutDistributorTest(EVMCase):
    def setUp(self):
        super().setUp()
        self.poster = self.acct[1]
        self.pd = self.deploy("PayoutDistributor", self.usdc.address, self.poster)
        self.fund(self.poster, 10_000 * ONE, self.pd)

    def post(self, epoch, payouts, total=None):
        leaves = {a: merkle.leaf(epoch, a, amt) for a, amt in payouts.items()}
        levels = merkle.build_tree(list(leaves.values()))
        root = levels[-1][0]
        total = sum(payouts.values()) if total is None else total
        self.tx(self.pd.functions.postRoot(epoch, root, total), self.poster)
        return {a: merkle.proof(levels, lf) for a, lf in leaves.items()}, root

    def test_python_leaf_matches_solidity_and_claims_pay(self):
        epoch = 7
        payees = self.acct[2:9]                     # 7 payees: odd count exercises the promoted lone node
        payouts = {a: (i + 1) * 1_234_567 for i, a in enumerate(payees)}
        proofs, root = self.post(epoch, payouts)
        self.assertEqual(self.pd.functions.epochs(epoch).call()[0], root)
        self.assertEqual(self.bal(self.pd.address), sum(payouts.values()))

        # inside the challenge window nobody can claim
        a0 = payees[0]
        self.reverts(self.pd.functions.claim(epoch, a0, payouts[a0], proofs[a0]), a0, "challenge window")
        self.travel(DAY + 1)

        for i, a in enumerate(payees):
            before = self.bal(a)
            submitter = self.acct[9] if i % 2 else a   # anyone may submit; funds still go to the account
            self.tx(self.pd.functions.claim(epoch, a, payouts[a], proofs[a]), submitter)
            self.assertEqual(self.bal(a) - before, payouts[a])
            self.assertTrue(self.pd.functions.claimed(epoch, a).call())
        self.assertEqual(self.bal(self.pd.address), 0)

    def test_wrong_amount_and_replay_revert(self):
        payouts = {self.acct[2]: 5 * ONE, self.acct[3]: 7 * ONE, self.acct[4]: 1}
        proofs, _ = self.post(1, payouts)
        self.travel(DAY)
        a = self.acct[2]
        self.reverts(self.pd.functions.claim(1, a, payouts[a] + 1, proofs[a]), a, "bad proof")
        self.reverts(self.pd.functions.claim(1, self.acct[5], payouts[a], proofs[a]), a, "bad proof")
        self.reverts(self.pd.functions.claim(2, a, payouts[a], proofs[a]), a, "no root")
        self.tx(self.pd.functions.claim(1, a, payouts[a], proofs[a]), a)
        self.reverts(self.pd.functions.claim(1, a, payouts[a], proofs[a]), a, "claimed")

    def test_claim_many_across_epochs(self):
        a = self.acct[2]
        p1, _ = self.post(1, {a: 3 * ONE, self.acct[3]: ONE})
        p2, _ = self.post(2, {a: 4 * ONE})                  # single-leaf tree: empty proof
        self.assertEqual(p2[a], [])
        self.travel(DAY)
        self.tx(self.pd.functions.claimMany([1, 2], a, [3 * ONE, 4 * ONE], [p1[a], p2[a]]), self.acct[9])
        self.assertEqual(self.bal(a), 7 * ONE)

    def test_veto_inside_window_refunds_and_allows_repost(self):
        a = self.acct[2]
        start = self.bal(self.poster)
        self.post(3, {a: 9 * ONE})
        self.reverts(self.pd.functions.veto(3), a, "cannot veto")            # only the poster
        self.tx(self.pd.functions.veto(3), self.poster)
        self.assertEqual(self.bal(self.poster), start)
        self.reverts(self.pd.functions.veto(3), self.poster, "cannot veto")  # no double refund
        self.travel(DAY)
        proofs, _ = self.post(3, {a: 8 * ONE})                              # re-post the corrected epoch
        self.reverts(self.pd.functions.claim(3, a, 8 * ONE, proofs[a]), a, "challenge window")
        self.travel(DAY)
        self.reverts(self.pd.functions.veto(3), self.poster, "cannot veto")  # window closed
        self.tx(self.pd.functions.claim(3, a, 8 * ONE, proofs[a]), a)
        self.assertEqual(self.bal(a), 8 * ONE)

    def test_cannot_overwrite_live_epoch_or_post_zero_root_or_veto_unposted(self):
        self.post(4, {self.acct[2]: ONE})
        self.reverts(self.pd.functions.postRoot(4, b"\x11" * 32, ONE), self.poster, "epoch exists")
        self.reverts(self.pd.functions.postRoot(5, b"\x00" * 32, ONE), self.poster, "zero root")
        self.reverts(self.pd.functions.veto(6), self.poster, "cannot veto")
        self.reverts(self.pd.functions.postRoot(5, b"\x11" * 32, ONE), self.acct[2], "not poster")

    def test_root_cannot_pay_out_more_than_its_epoch_holds(self):
        """A root whose leaves sum past `total` must not drain other epochs' money."""
        a, b = self.acct[2], self.acct[3]
        good, _ = self.post(1, {b: 100 * ONE})
        bad, _ = self.post(2, {a: 100 * ONE}, total=1 * ONE)        # under-funded root
        self.travel(DAY)
        self.reverts(self.pd.functions.claim(2, a, 100 * ONE, bad[a]), a, "exceeds epoch")
        self.tx(self.pd.functions.claim(1, b, 100 * ONE, good[b]), b)
        self.assertEqual(self.bal(b), 100 * ONE)


# ======================================================================================================================
class BountyMarketTest(EVMCase):
    def setUp(self):
        super().setUp()
        self.settler = self.acct[1]
        self.bm = self.deploy("BountyMarket", self.usdc.address, self.settler)
        for who in self.acct[1:8]:
            self.fund(who, 1_000_000 * ONE, self.bm)
        self.A, self.B, self.C, self.D = self.acct[2:6]

    def post(self, epochs_open=7, poster=None):
        poster = poster or self.acct[6]
        before = self.bal(poster)
        self.tx(self.bm.functions.post(b"\x01" * 32, b"\x02" * 32, 9000, epochs_open), poster)
        self.assertEqual(self.bal(poster), before)                      # posting is free
        return self.bm.functions.count().call() - 1

    def b(self, bid):
        r = self.bm.functions.bounties(bid).call()
        return dict(zip(["poster", "pathHash", "evalSet", "targetBps", "deadline", "status", "supply", "pool",
                         "learning", "accPerCoin"], r))

    def buy(self, bid, who, coins):
        c = self.bm.functions.cost(self.b(bid)["supply"], coins).call()
        self.tx(self.bm.functions.buy(bid, coins, c), who)
        return c

    def test_cost_matches_python_curve(self):
        import random
        rng = random.Random(7)
        cases = [(0, ONE), (0, 1), (0, 1000 * ONE), (123_456_789, 3 * ONE + 7), (10**12, 1), (5 * ONE, 999_999)]
        cases += [(rng.randrange(10**10), rng.randrange(1, 10**9)) for _ in range(40)]
        for s, n in cases:
            exact = Fraction(bountycoin.BASE * n, ONE) + Fraction(bountycoin.SLOPE * (2 * s * n + n * n), 2 * ONE * ONE)
            got = self.bm.functions.cost(s, n).call()
            self.assertEqual(got, math.ceil(exact), (s, n))                          # rounds up for buyers
            self.assertEqual(self.bm.functions.sellValue(s, n).call(), math.floor(exact), (s, n))  # down for sellers
            self.assertAlmostEqual(got, bountycoin.cost(s / ONE, n / ONE), delta=1.0)  # same curve as the SDK
        self.assertEqual(self.bm.functions.cost(0, ONE).call(), 10_050)               # first whole coin: $0.01005

    def test_buy_slippage_sell_minout_and_full_exit_empties_pool(self):
        bid = self.post()
        self.assertEqual(self.bm.functions.priceNow(bid).call(), bountycoin.BASE)
        c = self.bm.functions.cost(0, 10 * ONE).call()
        self.reverts(self.bm.functions.buy(bid, 10 * ONE, c - 1), self.A, "slippage")
        self.tx(self.bm.functions.buy(bid, 10 * ONE, c), self.A)
        self.assertEqual(self.bm.functions.priceNow(bid).call(), bountycoin.BASE + bountycoin.SLOPE * 10)
        cb = self.buy(bid, self.B, 3 * ONE + 333_333)
        cc = self.buy(bid, self.C, 7)
        pool = self.b(bid)["pool"]
        self.assertEqual(self.bal(self.bm.address), pool)

        # A sells in odd slices after B and C bought dearer coins: it gets back what it paid, never more
        s = self.b(bid)["supply"]
        self.assertGreater(self.bm.functions.sellValue(s - 10 * ONE, 10 * ONE).call(), c)  # the curve alone would pay
        start_a = self.bal(self.A)                                             # A more, out of B's and C's money
        self.reverts(self.bm.functions.sell(bid, ONE, 10**12), self.A, "slippage")
        self.reverts(self.bm.functions.sell(bid, 11 * ONE, 0), self.A, "balance")
        self.reverts(self.bm.functions.sell(bid, 0, 0), self.A, "balance")
        pieces = [1, 333_333, 2_500_001, 7, 999_999, 1, 3, 5, 11]
        pieces.append(10 * ONE - sum(pieces))                                  # rest of A's 10 coins
        for piece in pieces:
            self.tx(self.bm.functions.sell(bid, piece, 0), self.A)
        self.assertEqual(self.bm.functions.balanceOf(bid, self.A).call(), 0)
        self.assertEqual(self.bal(self.A) - start_a, c)                        # the early backer can't sell into later buys

        # B and C exit with exactly what they paid
        for who, coins, paid in [(self.B, 3 * ONE + 333_333, cb), (self.C, 7, cc)]:
            before = self.bal(who)
            self.tx(self.bm.functions.sell(bid, coins, 0), who)
            self.assertEqual(self.bal(who) - before, paid)

        b = self.b(bid)
        self.assertEqual((b["supply"], b["pool"]), (0, 0))                      # every backer whole, nothing left over
        self.assertEqual(self.bal(self.bm.address), 0)

        # a lone round trip never profits, however it is sliced
        bid2 = self.post()
        before = self.bal(self.D)
        self.buy(bid2, self.D, 5 * ONE + 3)
        for piece in [1, 2, 5 * ONE]:
            self.tx(self.bm.functions.sell(bid2, piece, 0), self.D)
        self.assertLessEqual(self.bal(self.D), before)
        self.assertEqual(self.b(bid2)["supply"], 0)

    def test_settler_only_solve_then_holder_accrual_follows_the_coin(self):
        A, B, C, D = self.A, self.B, self.C, self.D
        bid = self.post()
        self.buy(bid, A, 3 * ONE)
        self.buy(bid, B, 1 * ONE)
        pool = self.b(bid)["pool"]
        self.reverts(self.bm.functions.payHolders(bid, ONE), self.settler, "cannot pay")   # not solved yet
        self.reverts(self.bm.functions.solve(bid, b"\x09" * 32), A, "cannot solve")
        settler_before = self.bal(self.settler)
        self.tx(self.bm.functions.solve(bid, b"\x09" * 32), self.settler)
        self.assertEqual(self.bal(self.settler) - settler_before, pool)
        self.reverts(self.bm.functions.solve(bid, b"\x09" * 32), self.settler, "cannot solve")
        self.reverts(self.bm.functions.buy(bid, ONE, 10**12), C, "not open")
        self.reverts(self.bm.functions.sell(bid, ONE, 0), A, "not open")
        self.assertEqual(self.b(bid)["status"], 1)

        self.reverts(self.bm.functions.payHolders(bid, 4 * ONE), A, "cannot pay")
        self.tx(self.bm.functions.payHolders(bid, 4 * ONE), self.settler)           # 1 USDC per coin
        pend = lambda w: self.bm.functions.pending(bid, w).call()
        self.assertEqual((pend(A), pend(B), pend(C)), (3 * ONE, ONE, 0))

        self.tx(self.bm.functions.transfer(bid, C, ONE), A)                          # C arrives after the payment
        self.tx(self.bm.functions.transfer(bid, D, ONE), B)                          # B leaves after it
        self.reverts(self.bm.functions.transfer(bid, D, ONE), B, "balance")
        self.assertEqual((pend(A), pend(B), pend(C), pend(D)), (3 * ONE, ONE, 0, 0))

        self.tx(self.bm.functions.payHolders(bid, 4 * ONE), self.settler)
        self.assertEqual((pend(A), pend(B), pend(C), pend(D)), (5 * ONE, ONE, ONE, ONE))

        for who, want in [(A, 5 * ONE), (B, ONE), (C, ONE), (D, ONE)]:
            before = self.bal(who)
            self.tx(self.bm.functions.withdraw(bid), who)
            self.assertEqual(self.bal(who) - before, want)
            self.assertEqual(pend(who), 0)
            self.tx(self.bm.functions.withdraw(bid), who)                           # second withdraw pays nothing
            self.assertEqual(self.bal(who) - before, want)
        self.assertEqual(self.bal(self.bm.address), 0)

    def test_transfer_to_zero_address_rejected(self):
        bid = self.post()
        self.buy(bid, self.A, ONE)
        self.reverts(self.bm.functions.transfer(bid, "0x" + "00" * 20, ONE), self.A, "zero address")

    def test_expire_after_deadline_and_redeem_by_what_each_put_in(self):
        A, B, C = self.A, self.B, self.C
        other = self.post()
        self.buy(other, C, 2 * ONE)                                   # a second bounty's pool must stay untouched
        bid = self.post(epochs_open=1)
        ca = self.buy(bid, A, 2 * ONE)                                # A's coins are cheaper than B's
        self.buy(bid, B, 5 * ONE + 1)
        pool = self.b(bid)["pool"]
        self.reverts(self.bm.functions.expire(bid), A, "not expirable")
        self.reverts(self.bm.functions.redeem(bid), A, "not expired")
        self.travel(DAY + 1)
        self.reverts(self.bm.functions.buy(bid, ONE, 10**12), C, "not open")
        self.tx(self.bm.functions.expire(bid), C)                     # anyone can expire
        self.assertEqual(self.b(bid)["status"], 2)
        self.reverts(self.bm.functions.solve(bid, b"\x09" * 32), self.settler, "cannot solve")
        a0, b0 = self.bal(A), self.bal(B)
        self.tx(self.bm.functions.redeem(bid), A)
        self.assertEqual(self.bal(A) - a0, ca)                        # what it put in, not a share by coin count
        self.reverts(self.bm.functions.redeem(bid), A, "nothing to redeem")
        self.tx(self.bm.functions.redeem(bid), B)
        self.assertEqual((self.bal(A) - a0) + (self.bal(B) - b0), pool)  # last redeemer takes the remainder
        self.assertEqual(self.b(bid)["pool"], 0)
        self.assertEqual(self.bal(self.bm.address), self.b(other)["pool"])


# ======================================================================================================================
class RegistryTest(EVMCase):
    def setUp(self):
        super().setUp()
        self.reg = self.deploy("Registry")

    def test_traces_learnings_and_transfer(self):
        P, Q, R = self.acct[1], self.acct[2], self.acct[3]
        t = [bytes([i]) * 32 for i in range(1, 5)]
        lot = b"\x77" * 32
        rcpt = self.tx(self.reg.functions.registerTraces(t[:3], lot), P)
        self.assertEqual(len(self.reg.events.TraceRegistered().process_receipt(rcpt)), 3)
        rcpt = self.tx(self.reg.functions.registerTraces(t, lot), Q)                 # first three already owned
        self.assertEqual(len(self.reg.events.TraceRegistered().process_receipt(rcpt)), 1)
        for i, want in enumerate([P, P, P, Q]):
            owner, at, kind = self.reg.functions.entries(t[i]).call()
            self.assertEqual((owner, kind), (want, 1))
            self.assertGreater(at, 0)
        self.reverts(self.reg.functions.registerTrace(t[0], lot), Q, "exists")

        L = b"\xaa" * 32
        att = b"\xbb" * 32
        self.reverts(self.reg.functions.registerLearning(L, [(t[0], 5000), (t[1], 4000)], att), R, "100%")
        self.reverts(self.reg.functions.registerLearning(L, [(t[0], 5000), (b"\xee" * 32, 5000)], att), R,
                     "unknown parent")
        self.reverts(self.reg.functions.registerLearning(L, [], att), R, "100%")
        self.tx(self.reg.functions.registerLearning(L, [(t[0], 6000), (t[1], 2500), (t[3], 1500)], att), R)
        self.assertEqual(self.reg.functions.entries(L).call()[2], 2)
        self.assertEqual(self.reg.functions.attestation(L).call(), att)
        self.assertEqual([tuple(p) for p in self.reg.functions.parents(L).call()],
                         [(t[0], 6000), (t[1], 2500), (t[3], 1500)])
        self.reverts(self.reg.functions.registerLearning(L, [(t[0], 10000)], att), R, "exists")

        L2 = b"\xab" * 32                                              # a learning can descend from a learning
        self.tx(self.reg.functions.registerLearning(L2, [(L, 7000), (t[2], 3000)], att), Q)

        self.reverts(self.reg.functions.transfer(t[0], Q, ), Q, "not owner")
        self.tx(self.reg.functions.transfer(t[0], Q), P)
        self.assertEqual(self.reg.functions.entries(t[0]).call()[0], Q)
        self.reverts(self.reg.functions.transfer(t[0], R), P, "not owner")
        self.tx(self.reg.functions.transfer(L, P), R)
        self.assertEqual(self.reg.functions.entries(L).call()[0], P)

    def test_transfer_to_zero_cannot_reopen_the_id(self):
        P, X = self.acct[1], self.acct[2]
        t = b"\x01" * 32
        self.tx(self.reg.functions.registerTrace(t, b"\x00" * 32), P)
        self.reverts(self.reg.functions.transfer(t, "0x" + "00" * 20), P, "zero address")
        self.reverts(self.reg.functions.registerTrace(t, b"\x00" * 32), X, "exists")


if __name__ == "__main__":
    unittest.main()
