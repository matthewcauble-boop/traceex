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
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python")]

from traceex import merkle  # noqa: E402

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
ONE = 10**6          # USDC micros per dollar
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
