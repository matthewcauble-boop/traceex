"""Signed messages and optional AgentID sign-in for traceX (SPEC: identity). Standard library only.

Two things live here, and the node (node/identity.py) imports both, so client and node can never disagree:

1. **Address keys.** A traceX address is an Ethereum-style 0x address: the last 20 bytes of keccak-256 of a secp256k1
   public key. `AddressKey` makes one, and signs text the way every Ethereum wallet's `personal_sign` does (EIP-191),
   so a node recovers the signer's address from the signature alone, and a hardware wallet or MetaMask can sign the
   same messages. A signed action is a request body plus `_sig: {address, nonce, ts, signature}`; the signature
   covers the node, the route, the body's sha256, a one-time nonce and the time, so it can't be replayed elsewhere.
   With signatures, a validator's commit and reveal, a judge's confirmation, a poster's measurement and claim, a
   stake and a direct licence can reach the node from the agent itself instead of being relayed by the operator.

2. **AgentID (optional).** AgentID (AgentMail, https://www.agentid.com) is an OpenID Connect provider for AI agents.
   A node that turns it on lets an agent bind its AgentID subject to its traceX address: the agent proves it controls
   the AgentID (an ID token the node verified, carrying a nonce the node issued for that address) and the address
   (an EIP-191 signature over the binding). AgentID only offers the authorization-code flow, so some browser (a real
   one, a headless one, or the AgentMail console) must complete one redirect; `sign_in_with_agentid` drives everything
   else. Identity never replaces a sats bond: logins are cheap.

The elliptic-curve arithmetic below is plain Python (affine coordinates, RFC 6979 deterministic nonces). It is a
reference implementation sized for signing a few messages: fine for an agent's own requests, not a constant-time
library for a high-volume signer that an attacker can time. Keys are secrets: keep them out of repos and logs.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import urllib.parse

# --- keccak-256 (Ethereum's, not NIST SHA3-256: the padding differs) -------------------------------------------------
_RC = (0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
       0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
       0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
       0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
       0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008)
_ROT = ((0, 36, 3, 41, 18), (1, 44, 10, 45, 2), (62, 6, 43, 15, 61), (28, 55, 25, 21, 56), (27, 20, 39, 8, 14))
_M64 = (1 << 64) - 1


def _rol(v, n):
    return ((v << n) | (v >> (64 - n))) & _M64 if n else v


def _keccak_f(a):
    for rc in _RC:
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rol(c[(x + 1) % 5], 1) for x in range(5)]
        a = [[a[x][y] ^ d[x] for y in range(5)] for x in range(5)]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rol(a[x][y], _ROT[x][y])
        a = [[b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]
        a[0][0] ^= rc
    return a


def keccak256(data: bytes) -> bytes:
    rate = 136
    msg = bytearray(data) + b"\x01"
    msg += b"\x00" * (-len(msg) % rate)
    msg[-1] |= 0x80
    a = [[0] * 5 for _ in range(5)]
    for off in range(0, len(msg), rate):
        for i in range(rate // 8):
            a[i % 5][i // 5] ^= int.from_bytes(msg[off + 8 * i: off + 8 * i + 8], "little")
        a = _keccak_f(a)
    return b"".join(a[i % 5][i // 5].to_bytes(8, "little") for i in range(4))


# --- short Weierstrass curves: secp256k1 (addresses) and P-256 (AgentID's ES256) ---------------------------------------
class Curve:
    def __init__(self, name, p, a, b, n, gx, gy):
        self.name, self.p, self.a, self.b, self.n, self.g = name, p, a, b, n, (gx, gy)
        assert self.on_curve(self.g), name

    def on_curve(self, pt):
        if pt is None:
            return False
        x, y = pt
        return 0 <= x < self.p and 0 <= y < self.p and (y * y - x * x * x - self.a * x - self.b) % self.p == 0

    def add(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        p = self.p
        if P[0] == Q[0]:
            if (P[1] + Q[1]) % p == 0:
                return None
            lam = (3 * P[0] * P[0] + self.a) * pow(2 * P[1], -1, p) % p
        else:
            lam = (Q[1] - P[1]) * pow(Q[0] - P[0], -1, p) % p
        x = (lam * lam - P[0] - Q[0]) % p
        return x, (lam * (P[0] - x) - P[1]) % p

    def mul(self, k, P):
        R = None
        for bit in bin(k % self.n)[2:]:
            R = self.add(R, R)
            if bit == "1":
                R = self.add(R, P)
        return R


SECP256K1 = Curve("secp256k1", 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F, 0, 7,
                  0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141,
                  0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
                  0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
_P256_P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
P256 = Curve("P-256", _P256_P, _P256_P - 3, 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B,
             0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551,
             0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296,
             0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5)


def _rfc6979_k(curve, d, digest):
    """RFC 6979 deterministic nonce (HMAC-SHA256), for 256-bit curves and 32-byte digests."""
    x, h = d.to_bytes(32, "big"), (int.from_bytes(digest, "big") % curve.n).to_bytes(32, "big")
    v, k = b"\x01" * 32, b"\x00" * 32
    k = hmac.new(k, v + b"\x00" + x + h, hashlib.sha256).digest()
    v = hmac.new(k, v, hashlib.sha256).digest()
    k = hmac.new(k, v + b"\x01" + x + h, hashlib.sha256).digest()
    v = hmac.new(k, v, hashlib.sha256).digest()
    while True:
        v = hmac.new(k, v, hashlib.sha256).digest()
        cand = int.from_bytes(v, "big")
        if 1 <= cand < curve.n:
            return cand
        k = hmac.new(k, v + b"\x00", hashlib.sha256).digest()
        v = hmac.new(k, v, hashlib.sha256).digest()


def ecdsa_sign(curve, d, digest: bytes):
    """(r, s, recovery id) with low s. digest: 32 bytes, already hashed."""
    n, z = curve.n, int.from_bytes(digest, "big")
    k = _rfc6979_k(curve, d, digest)
    R = curve.mul(k, curve.g)
    r = R[0] % n
    s = pow(k, -1, n) * (z + r * d) % n
    if r == 0 or s == 0:
        raise ValueError("bad nonce")
    rec = (R[1] & 1) | (2 if R[0] >= n else 0)
    if s > n // 2:
        s, rec = n - s, rec ^ 1
    return r, s, rec


def ecdsa_verify(curve, pub, digest: bytes, r: int, s: int) -> bool:
    n = curve.n
    if not (1 <= r < n and 1 <= s < n) or not curve.on_curve(pub):
        return False
    w = pow(s, -1, n)
    z = int.from_bytes(digest, "big")
    X = curve.add(curve.mul(z * w % n, curve.g), curve.mul(r * w % n, pub))
    return X is not None and X[0] % n == r


def ecdsa_recover(digest: bytes, r: int, s: int, rec: int):
    """The secp256k1 public key that made (r, s) over digest, or None."""
    c = SECP256K1
    if not (1 <= r < c.n and 1 <= s < c.n) or rec not in (0, 1, 2, 3):
        return None
    x = r + (rec >> 1) * c.n
    if x >= c.p:
        return None
    beta = pow((x * x * x + 7) % c.p, (c.p + 1) // 4, c.p)
    y = beta if beta % 2 == rec & 1 else c.p - beta
    R = (x, y)
    if not c.on_curve(R):
        return None
    z = int.from_bytes(digest, "big")
    rinv = pow(r, -1, c.n)
    Q = c.add(c.mul(s * rinv % c.n, R), c.mul((-z * rinv) % c.n, c.g))
    return Q if Q is not None and c.on_curve(Q) else None


# --- addresses and EIP-191 messages ----------------------------------------------------------------------------------
ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}")
NONCE = re.compile(r"[A-Za-z0-9_-]{16,64}")
SIGNATURE = re.compile(r"0x[0-9a-fA-F]{130}")


def address_of(pub) -> str:
    """The 0x address (lowercase) of a secp256k1 public key (x, y)."""
    return "0x" + keccak256(pub[0].to_bytes(32, "big") + pub[1].to_bytes(32, "big"))[-20:].hex()


def checksum(address: str) -> str:
    """EIP-55 mixed-case form, for display."""
    a = address.lower().removeprefix("0x")
    h = keccak256(a.encode()).hex()
    return "0x" + "".join(c.upper() if c.isalpha() and int(h[i], 16) >= 8 else c for i, c in enumerate(a))


def eip191_digest(text: str) -> bytes:
    data = text.encode()
    return keccak256(b"\x19Ethereum Signed Message:\n" + str(len(data)).encode() + data)


def recover_address(text: str, signature: str):
    """The address that personal_sign'ed text, or None. signature: 0x + r(32) s(32) v(1), v in {0,1,27,28}."""
    if not isinstance(signature, str) or not SIGNATURE.fullmatch(signature):
        return None
    raw = bytes.fromhex(signature[2:])
    r, s, v = int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:64], "big"), raw[64]
    if v >= 27:
        v -= 27
    if v not in (0, 1) or s > SECP256K1.n // 2:          # high-s signatures are malleable copies: refuse them
        return None
    pub = ecdsa_recover(eip191_digest(text), r, s, v)
    return address_of(pub) if pub else None


def same_address(a, b) -> bool:
    return bool(a) and bool(b) and str(a).lower() == str(b).lower()


class AddressKey:
    """A secp256k1 key for a traceX address. The private key never leaves this object unless you call to_hex()."""

    def __init__(self, secret: int):
        if not 1 <= secret < SECP256K1.n:
            raise ValueError("private key out of range")
        self._d = secret
        self.public = SECP256K1.mul(secret, SECP256K1.g)
        self.address = address_of(self.public)

    @classmethod
    def generate(cls):
        return cls(secrets.randbelow(SECP256K1.n - 1) + 1)

    @classmethod
    def from_hex(cls, h: str):
        return cls(int(h.removeprefix("0x"), 16))

    @classmethod
    def from_env(cls, var="TRACEX_ADDRESS_KEY"):
        v = os.environ.get(var)
        return cls.from_hex(v) if v else None

    def to_hex(self) -> str:
        return "0x" + self._d.to_bytes(32, "big").hex()

    def sign_text(self, text: str) -> str:
        r, s, rec = ecdsa_sign(SECP256K1, self._d, eip191_digest(text))
        return "0x" + r.to_bytes(32, "big").hex() + s.to_bytes(32, "big").hex() + bytes([27 + rec]).hex()

    def __repr__(self):
        return f"AddressKey({checksum(self.address)})"


# --- what gets signed --------------------------------------------------------------------------------------------------
def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def action_message(node: str, path: str, body: dict, nonce: str, ts: int) -> str:
    """The text an address signs to send `body` to `path` on `node`. Everything that could be replayed elsewhere is
    in it: the node, the route, the body's hash, a one-time nonce and the time."""
    b = {k: v for k, v in body.items() if k != "_sig"}
    return ("traceX signed action\n"
            f"node: {node}\npath: {path}\nnonce: {nonce}\ntime: {int(ts)}\n"
            f"body-sha256: {hashlib.sha256(canonical(b)).hexdigest()}")


def sign_action(key: AddressKey, node: str, path: str, body: dict, nonce: str = None, ts: int = None) -> dict:
    """body plus `_sig`, ready to POST to path."""
    nonce = nonce or secrets.token_urlsafe(18)
    ts = int(time.time() if ts is None else ts)
    out = {k: v for k, v in body.items() if k != "_sig"}
    out["_sig"] = {"address": key.address, "nonce": nonce, "ts": ts,
                   "signature": key.sign_text(action_message(node, path, out, nonce, ts))}
    return out


def binding_message(node: str, issuer: str, subject: str, address: str, nonce: str) -> str:
    """What an address signs to bind an identity-provider subject to itself on one node."""
    return ("traceX identity binding\n"
            f"node: {node}\nissuer: {issuer}\nsubject: {subject}\naddress: {address.lower()}\nnonce: {nonce}")


# --- client helpers (use with traceex.Client; Client.sign_in_with_agentid delegates here) -----------------------------
def node_identity(client) -> dict:
    """GET /v0/identity: the node's id (what signed actions name), and which providers it accepts."""
    cached = getattr(client, "_identity_info", None)
    if cached is None:
        cached = client._call("GET", "/v0/identity")
        try:
            client._identity_info = cached
        except AttributeError:
            pass
    return cached


def signed_call(client, key: AddressKey, path: str, body: dict):
    """POST body to path, signed by key. The route's actor field (validator, judge, backer...) must be key's address."""
    return client._call("POST", path, sign_action(key, node_identity(client)["node"], path, body))


def bind_id_token(client, key: AddressKey, id_token: str, nonce: str, issuer: str, subject: str):
    """Bind a verified ID token's subject to key's address. The token must carry `nonce`, which the node issued for
    this address (POST /v0/identity/challenges). issuer/subject: the token's iss and sub (the node re-checks them)."""
    node = node_identity(client)["node"]
    sig = key.sign_text(binding_message(node, issuer, subject, key.address, nonce))
    return client._call("POST", "/v0/identity/bindings",
                        {"id_token": id_token, "address": key.address, "signature": sig})


def unbind(client, key: AddressKey, provider="agentid"):
    return signed_call(client, key, "/v0/identity/unbind", {"address": key.address, "provider": provider})


def binding(client, address: str):
    return client._call("GET", f"/v0/identity/bindings/{address}")


def sign_in_with_agentid(client, key: AddressKey, approve=None, poll=2.0, timeout=300.0, sleep=time.sleep):
    """Bind this address to the agent's AgentID, end to end, except the one browser step AgentID requires.

    1. The node starts an authorization-code + PKCE flow for key's address and hands back authorize_url.
    2. approve(authorize_url) gets the sign-in done: open it in a (headless) browser where the agent's AgentID is
       enrolled, or in a person's browser; an agent approves the waiting page with AgentMail's
       POST /v0/inboxes/{inbox_id}/authorize. Default: print the URL and wait.
    3. AgentID redirects to the node, which swaps the code for an ID token, verifies it and keeps only the subject
       and a hash of the email (never the token). This function polls until it has.
    4. key signs the binding (node, issuer, subject, address, nonce) and the node records it.
    Returns the binding."""
    started = client._call("POST", "/v0/identity/agentid/start", {"address": key.address})
    if approve is None:
        print(f"Open this to sign in with AgentID (expires in {started.get('expires_in', 300)} s):\n"
              f"{started['authorize_url']}", flush=True)
    else:
        approve(started["authorize_url"])
    deadline, state = time.monotonic() + timeout, started["state"]
    while True:
        p = client._call("GET", f"/v0/identity/agentid/pending/{urllib.parse.quote(state)}")
        if p.get("status") == "verified":
            break
        if p.get("status") in ("failed", "expired"):
            raise RuntimeError(f"AgentID sign-in {p['status']}: {p.get('error', '')}")
        if time.monotonic() > deadline:
            raise TimeoutError("AgentID sign-in not completed in time")
        sleep(poll)
    node = node_identity(client)["node"]
    sig = key.sign_text(binding_message(node, p["issuer"], p["subject"], key.address, p["nonce"]))
    return client._call("POST", "/v0/identity/bindings", {"pending": state, "address": key.address, "signature": sig})
