"""Merkle payouts (spec section 5): one root per epoch on-chain, everyone claims with a proof.

Leaf = sha256(epoch as uint64 big-endian || 20-byte address || amount as uint256 big-endian), the same bytes as
abi.encodePacked(uint64, address, uint256) in PayoutDistributor.sol. Pairs are hashed sorted, so a proof is just the
sibling list.
"""
import hashlib


def _addr(a: str) -> bytes:
    a = a.lower().removeprefix("0x")
    b = bytes.fromhex(a)
    if len(b) != 20:
        raise ValueError(f"address must be 20 bytes: {a}")
    return b


def leaf(epoch: int, account: str, amount: int) -> bytes:
    return hashlib.sha256(epoch.to_bytes(8, "big") + _addr(account) + amount.to_bytes(32, "big")).digest()


def _pair(a: bytes, b: bytes) -> bytes:
    return hashlib.sha256(min(a, b) + max(a, b)).digest()


def build_tree(leaves):
    """Returns the list of levels, leaves first, root last."""
    if not leaves:
        return [[hashlib.sha256(b"").digest()]]
    levels = [sorted(leaves)]
    while len(levels[-1]) > 1:
        lv = levels[-1]
        levels.append([_pair(lv[i], lv[i + 1]) if i + 1 < len(lv) else lv[i] for i in range(0, len(lv), 2)])
    return levels


def proof(levels, target: bytes):
    path, idx = [], levels[0].index(target)
    for lv in levels[:-1]:
        sib = idx ^ 1
        if sib < len(lv):
            path.append(lv[sib])
        idx //= 2
    return path


def verify(root: bytes, target: bytes, path) -> bool:
    h = target
    for sib in path:
        h = _pair(h, sib)
    return h == root
