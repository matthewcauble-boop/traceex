"""Identity: signed actions and optional AgentID bindings (node/identity.py, sdk/python/traceex/identity.py).
python -m pytest -q tests/test_identity.py

Everything runs against a local fake OpenID provider: its keys are generated when the tests start and never written
anywhere. No real AgentID token or key is used. The attack rows at the bottom (stolen-token replay, binding hijack,
signed-message replay...) are the identity counterparts of examples/farming/attacks.py."""
import base64
import hashlib
import json
import os
import re
import secrets
import sys
import threading
import time
import unittest
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "sdk", "python"), os.path.join(ROOT, "node")]

from traceex import Client  # noqa: E402
from traceex import identity as tid  # noqa: E402
from traceex.identity import (AddressKey, P256, SECP256K1, ecdsa_sign, keccak256, recover_address,  # noqa: E402
                              sign_action, binding_message, checksum)
import identity as nid  # noqa: E402
from identity import (Identity, IdentityError, Conflict, OIDCConfig, OIDCVerifier, JWKS, agentid_config,  # noqa: E402
                      b64url, AGENTID)

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa, utils
    HAVE_CRYPTO = True
except ImportError:                                       # the module itself never needs it; only the cross-checks do
    HAVE_CRYPTO = False

ISSUER, CLIENT = "https://issuer.test", "https://node.test"


class FakeIssuer:
    """A tiny OpenID provider: P-256 keys made now, ES256 ID tokens, an authorization-code endpoint with PKCE."""

    def __init__(self, issuer=ISSUER):
        self.issuer, self.keys, self.codes = issuer, {}, {}
        self.rotate()

    def rotate(self):
        self.kid = secrets.token_hex(4)
        d = secrets.randbelow(P256.n - 1) + 1
        x, y = P256.mul(d, P256.g)
        self.keys[self.kid] = (d, {"kty": "EC", "crv": "P-256", "alg": "ES256", "use": "sig", "kid": self.kid,
                                   "x": b64url(x.to_bytes(32, "big")), "y": b64url(y.to_bytes(32, "big"))})

    def jwks(self):
        return {"keys": [k for _, k in self.keys.values()]}

    def token(self, sub="agent-sub-1", aud=CLIENT, nonce=None, exp_in=600, iat=None, kid=None, alg="ES256",
              email="helper@acme.agentmail.to", owner="owner-1", **extra):
        now = int(time.time() if iat is None else iat)
        claims = dict(iss=self.issuer, sub=sub, aud=aud, iat=now, exp=now + exp_in, jti=secrets.token_hex(8),
                      actor_type="agent", email=email, email_verified=True, owner_sub=owner)
        if nonce is not None:
            claims["nonce"] = nonce
        claims.update(extra)
        claims = {k: v for k, v in claims.items() if v is not None}
        kid = kid or self.kid
        h = b64url(json.dumps({"alg": alg, "kid": kid, "typ": "JWT"}).encode())
        p = b64url(json.dumps(claims).encode())
        r, s, _ = ecdsa_sign(P256, self.keys[kid][0], hashlib.sha256(f"{h}.{p}".encode()).digest())
        return f"{h}.{p}.{b64url(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))}"

    # the browser part of an authorization-code flow
    def authorize(self, url, sub="agent-sub-1"):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        code = secrets.token_urlsafe(16)
        self.codes[code] = dict(q, sub=sub)
        return {"code": code, "state": q["state"], "iss": self.issuer}, q

    def token_endpoint(self, url, form, auth):
        c = self.codes.pop(form.get("code"), None)
        if not c or form.get("grant_type") != "authorization_code" or form.get("redirect_uri") != c["redirect_uri"]:
            raise IdentityError("token_endpoint", "invalid_grant")
        if b64url(hashlib.sha256(form["code_verifier"].encode()).digest()) != c["code_challenge"]:
            raise IdentityError("token_endpoint", "PKCE verifier mismatch")
        return {"id_token": self.token(sub=c["sub"], aud=c["client_id"], nonce=c["nonce"]), "token_type": "Bearer",
                "access_token": "opaque-access", "expires_in": 600}


def config(**kw):
    return OIDCConfig(**dict(AGENTID, issuer=ISSUER, jwks_uri=ISSUER + "/jwks.json",
                             authorization_endpoint=ISSUER + "/authorize", token_endpoint=ISSUER + "/token",
                             client_id=CLIENT, redirect_uri=CLIENT + "/v0/identity/agentid/callback", **kw))


def make(issuer=None, db=None, **kw):
    issuer = issuer or FakeIssuer()
    jwks = JWKS(ISSUER + "/jwks.json", fetch=lambda uri: issuer.jwks())
    ident = Identity(db, {"agentid": OIDCVerifier(config(), jwks)}, node_id=CLIENT,
                     http_post=issuer.token_endpoint, **kw)
    return ident, issuer


def bind_direct(ident, issuer, key, sub="agent-sub-1", **tok):
    ch = ident.challenge(key.address)
    t = issuer.token(sub=sub, nonce=ch["nonce"], **tok)
    sig = key.sign_text(binding_message(ident.node_id, ISSUER, sub, key.address, ch["nonce"]))
    return ident.bind({"id_token": t, "address": key.address, "signature": sig}), t


# --- the crypto, against published vectors and an independent library ------------------------------------------------
class Crypto(unittest.TestCase):
    def test_keccak_and_address_vectors(self):
        self.assertEqual(keccak256(b"").hex(), "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
        self.assertEqual(keccak256(b"abc").hex(), "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45")
        self.assertEqual(checksum(AddressKey(1).address), "0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf")

    def test_sign_and_recover(self):
        k = AddressKey.generate()
        sig = k.sign_text("hello traceX")
        self.assertEqual(recover_address("hello traceX", sig), k.address)
        self.assertNotEqual(recover_address("hello traceX!", sig), k.address)
        raw = bytearray(bytes.fromhex(sig[2:]))                   # the malleable twin (n - s) is refused
        s = int.from_bytes(raw[32:64], "big")
        raw[32:64] = (SECP256K1.n - s).to_bytes(32, "big")
        raw[64] ^= 1
        self.assertIsNone(recover_address("hello traceX", "0x" + raw.hex()))
        self.assertIsNone(recover_address("hello", "0x1234"))

    @unittest.skipUnless(HAVE_CRYPTO, "cryptography not installed")
    def test_cross_check_with_cryptography(self):
        k = AddressKey.generate()
        digest = tid.eip191_digest("cross check")
        r, s, _ = ecdsa_sign(SECP256K1, int(k.to_hex(), 16), digest)
        pub = ec.EllipticCurvePublicNumbers(*k.public, ec.SECP256K1()).public_key()
        pub.verify(utils.encode_dss_signature(r, s), digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
        # and a P-256 token signed by cryptography verifies with ours
        priv = ec.generate_private_key(ec.SECP256R1())
        n = priv.public_key().public_numbers()
        jwk = {"kty": "EC", "crv": "P-256", "kid": "x", "x": b64url(n.x.to_bytes(32, "big")),
               "y": b64url(n.y.to_bytes(32, "big"))}
        h, p = b64url(b'{"alg":"ES256","kid":"x"}'), b64url(b'{"sub":"s"}')
        rr, ss = utils.decode_dss_signature(priv.sign(f"{h}.{p}".encode(), ec.ECDSA(hashes.SHA256())))
        _, claims = nid.verify_jws(f"{h}.{p}.{b64url(rr.to_bytes(32, 'big') + ss.to_bytes(32, 'big'))}",
                                   {"x": jwk}.get)
        self.assertEqual(claims, {"sub": "s"})

    @unittest.skipUnless(HAVE_CRYPTO, "cryptography not installed")
    def test_rs256_for_other_providers(self):
        priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        n = priv.public_key().public_numbers()
        jwk = {"kty": "RSA", "kid": "r", "n": b64url(n.n.to_bytes(256, "big")), "e": b64url(n.e.to_bytes(3, "big"))}
        h, p = b64url(b'{"alg":"RS256","kid":"r"}'), b64url(b'{"sub":"s"}')
        sig = priv.sign(f"{h}.{p}".encode(), padding.PKCS1v15(), hashes.SHA256())
        self.assertEqual(nid.verify_jws(f"{h}.{p}.{b64url(sig)}", {"r": jwk}.get, ("RS256",))[1], {"sub": "s"})
        with self.assertRaises(IdentityError):
            nid.verify_jws(f"{h}.{p}.{b64url(sig[:-1] + bytes([sig[-1] ^ 1]))}", {"r": jwk}.get, ("RS256",))


# --- the OIDC verifier ------------------------------------------------------------------------------------------------
class Verify(unittest.TestCase):
    def setUp(self):
        self.iss = FakeIssuer()
        self.v = OIDCVerifier(config(), JWKS(ISSUER + "/jwks.json", fetch=lambda u: self.iss.jwks()))

    def code(self, token, **kw):
        with self.assertRaises(IdentityError) as cm:
            self.v.verify(token, **kw)
        return cm.exception.code

    def test_good_token(self):
        c = self.v.verify(self.iss.token(nonce="n" * 20), nonce="n" * 20)
        self.assertEqual((c["sub"], c["actor_type"]), ("agent-sub-1", "agent"))

    def test_wrong_audience(self):
        self.assertEqual(self.code(self.iss.token(aud="https://other.app")), "wrong_audience")
        self.assertEqual(self.code(self.iss.token(aud=[CLIENT, "https://other.app"])), "wrong_audience")  # no azp

    def test_expired_and_future(self):
        self.assertEqual(self.code(self.iss.token(iat=time.time() - 2000, exp_in=600)), "expired")
        self.assertEqual(self.code(self.iss.token(iat=time.time() + 3600)), "stale")

    def test_bad_signature(self):
        h, p, s = self.iss.token().split(".")
        claims = json.loads(base64.urlsafe_b64decode(p + "=="))
        claims["sub"] = "someone-else"
        self.assertEqual(self.code(f"{h}.{b64url(json.dumps(claims).encode())}.{s}"), "bad_signature")
        other = FakeIssuer()                                       # right kid, wrong key
        other.keys = {self.iss.kid: other.keys[other.kid]}
        other.kid = self.iss.kid
        self.assertEqual(self.code(other.token()), "bad_signature")

    def test_wrong_issuer_alg_none_and_hmac(self):
        self.assertEqual(self.code(self.iss.token(iss="https://evil.test")), "wrong_issuer")   # right key, wrong iss
        self.assertEqual(self.code(FakeIssuer("https://evil.test").token()), "unknown_key")
        h = b64url(b'{"alg":"none"}')
        p = b64url(json.dumps({"iss": ISSUER, "aud": CLIENT, "sub": "x"}).encode())
        self.assertEqual(self.code(f"{h}.{p}."), "bad_alg")
        h = b64url(json.dumps({"alg": "HS256", "kid": self.iss.kid}).encode())
        self.assertEqual(self.code(f"{h}.{p}.{b64url(b'x' * 32)}"), "bad_alg")

    def test_nonce_actor_and_jti(self):
        self.assertEqual(self.code(self.iss.token(nonce="a" * 20), nonce="b" * 20), "wrong_nonce")
        self.assertEqual(self.code(self.iss.token(), nonce="b" * 20), "wrong_nonce")
        self.assertEqual(self.code(self.iss.token(actor_type="user")), "wrong_actor")
        self.assertEqual(self.code(self.iss.token(jti=None)), "malformed")

    def test_key_rotation_refetches_jwks(self):
        self.v.verify(self.iss.token())
        self.iss.rotate()                                          # a new kid the cache hasn't seen
        self.v.jwks.fetched -= 120                                 # past the one-refetch-a-minute guard
        self.assertEqual(self.v.verify(self.iss.token())["sub"], "agent-sub-1")

    def test_agentid_preset(self):
        self.assertIsNone(agentid_config(env={}))                 # off until the operator sets a client id
        c = agentid_config(env={"TRACEX_AGENTID_CLIENT_ID": "https://tracex.example"})
        self.assertEqual((c.issuer, c.jwks_uri, c.algs), ("https://auth.agentid.com",
                                                          "https://auth.agentid.com/v0/jwks.json", ("ES256",)))
        self.assertEqual(c.redirect_uri, "https://tracex.example/v0/identity/agentid/callback")
        self.assertNotIn("owner_email", c.scopes)
        with self.assertRaises(ValueError):                        # an opaque id needs its secret (registered client)
            agentid_config(env={"TRACEX_AGENTID_CLIENT_ID": "b7d41e0a"})


# --- bindings ----------------------------------------------------------------------------------------------------------
class Bindings(unittest.TestCase):
    def test_bind_and_lookup(self):
        ident, iss = make()
        k = AddressKey.generate()
        b, _ = bind_direct(ident, iss, k)
        self.assertEqual((b["subject"], b["address"]), ("agent-sub-1", k.address))
        self.assertEqual(ident.bindings_of(k.address.upper().replace("0X", "0x"))[0]["subject"], "agent-sub-1")

    def test_unbound_subject(self):
        ident, iss = make()
        who = ident.whoami({"id_token": iss.token(sub="nobody-yet")})
        self.assertEqual((who["bound"], who["address"]), (False, None))
        k = AddressKey.generate()
        bind_direct(ident, iss, k, sub="nobody-yet")
        self.assertEqual(ident.whoami({"id_token": iss.token(sub="nobody-yet")})["address"], k.address)

    def test_signature_must_be_the_address(self):
        ident, iss = make()
        k, other = AddressKey.generate(), AddressKey.generate()
        ch = ident.challenge(k.address)
        sig = other.sign_text(binding_message(ident.node_id, ISSUER, "agent-sub-1", k.address, ch["nonce"]))
        with self.assertRaises(IdentityError) as cm:
            ident.bind({"id_token": iss.token(nonce=ch["nonce"]), "address": k.address, "signature": sig})
        self.assertEqual(cm.exception.code, "bad_signature")

    def test_token_needs_a_nonce_this_node_issued(self):
        ident, iss = make()
        k = AddressKey.generate()
        for nonce in (None, "x" * 24):
            sig = k.sign_text(binding_message(ident.node_id, ISSUER, "agent-sub-1", k.address, nonce or ""))
            with self.assertRaises(IdentityError) as cm:
                ident.bind({"id_token": iss.token(nonce=nonce), "address": k.address, "signature": sig})
            self.assertEqual(cm.exception.code, "wrong_nonce")

    def test_challenge_expires(self):
        ident, iss = make()
        k = AddressKey.generate()
        ch = ident.challenge(k.address, now=time.time() - nid.CHALLENGE_TTL - 1)
        sig = k.sign_text(binding_message(ident.node_id, ISSUER, "agent-sub-1", k.address, ch["nonce"]))
        with self.assertRaises(IdentityError):
            ident.bind({"id_token": iss.token(nonce=ch["nonce"]), "address": k.address, "signature": sig})

    def test_one_subject_one_address_and_unbind(self):
        ident, iss = make()
        a, b = AddressKey.generate(), AddressKey.generate()
        bind_direct(ident, iss, a)
        with self.assertRaises(Conflict):
            bind_direct(ident, iss, b)                              # same subject, another address
        with self.assertRaises(Conflict):
            bind_direct(ident, iss, a, sub="agent-sub-2")           # same address, another subject
        bind_direct(ident, iss, a)                                  # re-binding the same pair refreshes it
        ident.unbind(ident.verify_action("/v0/identity/unbind", {"address": a.address},
                                         sign_action(a, ident.node_id, "/v0/identity/unbind",
                                                     {"address": a.address})["_sig"]), {"address": a.address})
        self.assertEqual(ident.bindings_of(a.address), [])
        self.assertEqual(bind_direct(ident, iss, b)[0]["address"], b.address)

    def test_privacy_only_subject_and_hashes(self):
        ident, iss = make()
        k = AddressKey.generate()
        _, token = bind_direct(ident, iss, k, email="secret.agent@acme.agentmail.to", owner="owner-xyz")
        dump = "\n".join(ident.db.conn.iterdump())
        for leak in ("secret.agent", "agentmail", "owner-xyz", token, token.split(".")[2]):
            self.assertNotIn(leak, dump)
        self.assertIn("agent-sub-1", dump)

    def test_owner_groups_for_sybil_analysis(self):
        ident, iss = make()
        for i in range(3):
            bind_direct(ident, iss, AddressKey.generate(), sub=f"s{i}", owner="same-owner")
        bind_direct(ident, iss, AddressKey.generate(), sub="s9", owner="lonely")
        self.assertEqual([g["addresses"] for g in ident.owner_groups()], [3])

    def test_read_only_node_refuses_writes(self):
        ident, iss = make(read_only=True)
        with self.assertRaises(PermissionError):
            ident.challenge(AddressKey.generate().address)


# --- the browser flow (authorization code + PKCE) ----------------------------------------------------------------------
class Browser(unittest.TestCase):
    def test_full_flow(self):
        ident, iss = make()
        k = AddressKey.generate()
        s = ident.start("agentid", k.address)
        back, q = iss.authorize(s["authorize_url"])
        self.assertEqual((q["code_challenge_method"], q["client_id"], q["response_type"]), ("S256", CLIENT, "code"))
        self.assertNotIn("owner", q["scope"])
        self.assertEqual(ident.callback("agentid", back)["status"], "verified")
        p = ident.pending(s["state"])
        self.assertEqual((p["status"], p["subject"]), ("verified", "agent-sub-1"))
        b = ident.bind({"pending": s["state"], "address": k.address, "signature": k.sign_text(p["sign"])})
        self.assertEqual(b["address"], k.address)
        with self.assertRaises(IdentityError):                     # the sign-in is spent
            ident.bind({"pending": s["state"], "address": k.address, "signature": k.sign_text(p["sign"])})
        with self.assertRaises(IdentityError):
            ident.callback("agentid", back)

    def test_mixup_and_denied(self):
        ident, iss = make()
        k = AddressKey.generate()
        s = ident.start("agentid", k.address)
        back, _ = iss.authorize(s["authorize_url"])
        with self.assertRaises(IdentityError) as cm:
            ident.callback("agentid", dict(back, iss="https://evil.test"))
        self.assertEqual(cm.exception.code, "wrong_issuer")
        self.assertEqual(ident.pending(s["state"])["status"], "failed")
        s2 = ident.start("agentid", k.address)
        with self.assertRaises(IdentityError):
            ident.callback("agentid", {"state": s2["state"], "error": "access_denied", "iss": ISSUER})

    def test_pkce_verifier_never_leaves_the_node(self):
        ident, iss = make()
        s = ident.start("agentid", AddressKey.generate().address)
        verifier = ident.db.execute("SELECT verifier FROM ident_pending").fetchone()[0]
        self.assertNotIn(verifier, s["authorize_url"])
        ident.callback("agentid", iss.authorize(s["authorize_url"])[0])
        self.assertIsNone(ident.db.execute("SELECT verifier FROM ident_pending").fetchone()[0])


# --- signed actions ----------------------------------------------------------------------------------------------------
class SignedActions(unittest.TestCase):
    PATH = "/v0/fixes/TXX-1/commits"

    def test_good_and_replay(self):
        ident, _ = make()
        k = AddressKey.generate()
        body = sign_action(k, ident.node_id, self.PATH, {"validator": k.address, "digest": "ab" * 32})
        self.assertEqual(ident.take_signature(None, self.PATH, dict(body)), k.address)
        with self.assertRaises(IdentityError) as cm:
            ident.take_signature(None, self.PATH, dict(body))
        self.assertEqual(cm.exception.code, "replay")

    def test_tamper_path_node_time(self):
        ident, _ = make()
        k = AddressKey.generate()
        mk = lambda **kw: sign_action(k, kw.pop("node", ident.node_id), kw.pop("path", self.PATH),
                                      {"validator": k.address, "digest": "ab" * 32}, **kw)
        b = mk()
        b["digest"] = "cd" * 32
        cases = {"bad_signature": [(self.PATH, b), (self.PATH, mk(path="/v0/fixes/TXX-2/commits")),
                                   (self.PATH, mk(node="https://another.node"))],
                 "expired": [(self.PATH, mk(ts=int(time.time()) - 3600))]}
        for code, items in cases.items():
            for path, body in items:
                with self.assertRaises(IdentityError) as cm:
                    ident.take_signature(None, path, body)
                self.assertEqual(cm.exception.code, code)

    def test_signer_must_be_the_actor(self):
        ident, _ = make()
        k, victim = AddressKey.generate(), AddressKey.generate()
        body = sign_action(k, ident.node_id, self.PATH, {"validator": victim.address, "digest": "ab" * 32})
        with self.assertRaises(IdentityError) as cm:
            ident.take_signature(None, self.PATH, body)
        self.assertEqual(cm.exception.code, "not_actor")

    def test_relay_and_policies(self):
        ident, iss = make()
        k = AddressKey.generate()
        self.assertTrue(ident.relay_ok(self.PATH, k.address))
        self.assertFalse(ident.relay_ok("/v0/epochs/settle", k.address))     # settlement stays with the operator
        self.assertFalse(ident.relay_ok(self.PATH, None))
        strict, iss2 = make(signers_must_bind=True)
        with self.assertRaises(IdentityError):
            strict.relay_ok(self.PATH, k.address)
        bind_direct(strict, iss2, k)
        self.assertTrue(strict.relay_ok(self.PATH, k.address))
        req, _ = make(require_signatures=True)
        with self.assertRaises(IdentityError):
            req.take_signature(None, "/v0/bounties/1/pledges", {"backer": k.address, "sats": 10})
        self.assertIsNone(req.take_signature(None, "/v0/bounties/1/pledges", {"backer": k.address}, admin=True))
        self.assertIsNone(req.take_signature(None, "/v0/search", {}))


# --- over HTTP, on a real sats node, through the exchange's own handler -----------------------------------------------
class OverHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exchange import make_handler, RateLimit
        from sats import SatsExchange
        cls.ex = SatsExchange(":memory:", test_credits=30_000_000)
        cls.ident, cls.iss = make(db=cls.ex.db)                    # the node's own database, as identity.of() does
        cls.ex._identity = cls.ident
        H = nid.wrap(make_handler(cls.ex, public=True, admin_token="op-token",
                                  limiter=RateLimit(reads=10_000, writes=10_000, all_writes=10_000)), cls.ex)
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def post(self, path, body, token=None):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), method="POST",
                                     headers=dict({"Content-Type": "application/json"},
                                                  **({"Authorization": f"Bearer {token}"} if token else {})))
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_describe(self):
        d = Client(self.url)._call("GET", "/v0/identity")
        self.assertEqual(d["node"], CLIENT)
        self.assertTrue(d["providers"]["agentid"]["browser_flow"])

    def test_relayed_route_accepts_its_actor_signature(self):
        k, other = AddressKey.generate(), AddressKey.generate()
        path, body = "/v0/fixes/TXF-NOPE/commits", {"validator": k.address, "digest": "ab" * 32}
        code, r = self.post(path, body)
        self.assertEqual(code, 403)                                  # unsigned: still the operator's
        signed = self.post(path, sign_action(k, CLIENT, path, body))
        self.assertNotEqual(signed[0], 403, signed)                  # signed by the validator: reaches the exchange
        code, r = self.post(path, sign_action(other, CLIENT, path, body))
        self.assertEqual((code, r.get("error", "")[:9]), (403, "signed by"))
        relayed = self.post(path, body, token="op-token")            # the operator's relay still works, and the
        self.assertEqual(signed, relayed)                            # exchange answers both the same way
        code, r = self.post("/v0/epochs/settle", sign_action(k, CLIENT, "/v0/epochs/settle", {}))
        self.assertEqual(code, 403)                                  # settlement is not a signable action

    def test_open_route_refuses_a_forged_actor(self):
        k, victim = AddressKey.generate(), AddressKey.generate()
        path = "/v0/bounties/1/pledges"
        code, r = self.post(path, sign_action(k, CLIENT, path, {"backer": victim.address, "sats": 5}))
        self.assertEqual(code, 403)

    def test_sdk_bind_and_sign_in_with_agentid(self):
        c = Client(self.url)
        k = AddressKey.generate()
        ch = c._call("POST", "/v0/identity/challenges", {"address": k.address})
        tok = self.iss.token(sub="http-agent", nonce=ch["nonce"])
        b = tid.bind_id_token(c, k, tok, ch["nonce"], ISSUER, "http-agent")
        self.assertEqual(b["address"], k.address)
        self.assertEqual(tid.binding(c, k.address)["bindings"][0]["subject"], "http-agent")
        code, r = self.post("/v0/identity/bindings", {"id_token": tok, "address": k.address,
                                                      "signature": k.sign_text("x")})
        self.assertEqual(code, 403)

        k2 = AddressKey.generate()

        def browser(authorize_url):                                  # what a (headless) browser would do
            back, _ = self.iss.authorize(authorize_url, sub="browser-agent")
            with urllib.request.urlopen(self.url + "/v0/identity/agentid/callback?" + urllib.parse.urlencode(back),
                                        timeout=10) as r:
                self.assertIn(b"Signed in with AgentID", r.read())
        b2 = tid.sign_in_with_agentid(c, k2, approve=browser, sleep=lambda s: None)
        self.assertEqual((b2["subject"], b2["address"]), ("browser-agent", k2.address))
        who = c._call("POST", "/v0/identity/sessions", {"id_token": self.iss.token(sub="browser-agent")})
        self.assertEqual(who["address"], k2.address)
        r = tid.unbind(c, k2)
        self.assertEqual(r["unbound"], 1)

    def test_browser_start_redirects(self):
        k = AddressKey.generate()

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **kw):
                return None
        opener = urllib.request.build_opener(NoRedirect)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            opener.open(f"{self.url}/v0/identity/agentid/start?address={k.address}", timeout=10)
        self.assertEqual(cm.exception.code, 302)
        self.assertTrue(cm.exception.headers["Location"].startswith(ISSUER + "/authorize?"))


# --- attack rows (the identity counterparts of examples/farming/attacks.py) --------------------------------------------
class Attacks(unittest.TestCase):
    def test_stolen_token_replay(self):
        """An attacker who copies a victim's ID token (logs, a proxy) replays it: refused after its one use, refused
        at any other node (audience), and useless without the victim's address key in the 10 minutes before."""
        ident, iss = make()
        victim, attacker = AddressKey.generate(), AddressKey.generate()
        ch = ident.challenge(victim.address)
        tok = iss.token(sub="victim-sub", nonce=ch["nonce"])
        bad = attacker.sign_text(binding_message(ident.node_id, ISSUER, "victim-sub", attacker.address, ch["nonce"]))
        with self.assertRaises(IdentityError) as cm:                 # before the victim uses it: wrong address
            ident.bind({"id_token": tok, "address": attacker.address, "signature": bad})
        self.assertEqual(cm.exception.code, "wrong_address")
        good = victim.sign_text(binding_message(ident.node_id, ISSUER, "victim-sub", victim.address, ch["nonce"]))
        ident.bind({"id_token": tok, "address": victim.address, "signature": good})   # the failed try burned nothing
        with self.assertRaises(IdentityError) as cm:                 # after: spent
            ident.bind({"id_token": tok, "address": victim.address, "signature": good})
        self.assertIn(cm.exception.code, ("wrong_nonce", "replay"))
        other, _ = make(iss)                                         # another node: different audience
        other.providers["agentid"].config.client_id = "https://other.node"
        with self.assertRaises(IdentityError) as cm:
            other.whoami({"id_token": tok})
        self.assertEqual(cm.exception.code, "wrong_audience")

    def test_binding_hijack(self):
        """An attacker with its own AgentID asks for a challenge for the victim's address, hoping to bind its subject
        to the victim's wallet (and act as it), or to bind the victim's subject to its own wallet. Both need a
        signature from the address being claimed."""
        ident, iss = make()
        victim, attacker = AddressKey.generate(), AddressKey.generate()
        ch = ident.challenge(victim.address)
        tok = iss.token(sub="attacker-sub", nonce=ch["nonce"])
        sig = attacker.sign_text(binding_message(ident.node_id, ISSUER, "attacker-sub", victim.address, ch["nonce"]))
        with self.assertRaises(IdentityError) as cm:
            ident.bind({"id_token": tok, "address": victim.address, "signature": sig})
        self.assertEqual(cm.exception.code, "bad_signature")
        bind_direct(ident, iss, victim, sub="victim-sub")
        with self.assertRaises(Conflict):                            # a bound subject can't be moved without its address
            bind_direct(ident, iss, attacker, sub="victim-sub")
        with self.assertRaises(IdentityError):                       # nor unbound by anyone else
            ident.take_signature(None, "/v0/identity/unbind",
                                 sign_action(attacker, ident.node_id, "/v0/identity/unbind",
                                             {"address": victim.address}))

    def test_browser_state_hijack(self):
        """An attacker finishes a victim's browser sign-in (or the other way round): the pending sign-in belongs to
        the address it was started for, and needs that address's signature."""
        ident, iss = make()
        victim, attacker = AddressKey.generate(), AddressKey.generate()
        s = ident.start("agentid", victim.address)
        ident.callback("agentid", iss.authorize(s["authorize_url"], sub="victim-sub")[0])
        p = ident.pending(s["state"])
        for who, addr in ((attacker, attacker.address), (attacker, victim.address)):
            with self.assertRaises(IdentityError):
                ident.bind({"pending": s["state"], "address": addr,
                            "signature": who.sign_text(binding_message(ident.node_id, ISSUER, "victim-sub", addr,
                                                                       p["nonce"]))})

    def test_signed_message_replay_and_cross_node(self):
        """A validator's signed reveal, captured in transit, replayed on the same node or sent to another."""
        a, _ = make()
        b, _ = make()
        b.node_id = "https://other.node"
        k = AddressKey.generate()
        path = "/v0/learnings/L1/reveals"
        msg = sign_action(k, a.node_id, path, {"validator": k.address, "attestation": {}, "salt": "s"})
        self.assertEqual(a.take_signature(None, path, dict(msg)), k.address)
        for node in (a, b):
            with self.assertRaises(IdentityError):
                node.take_signature(None, path, dict(msg))

    def test_identity_is_not_a_bond(self):
        """Logins are cheap: forty AgentID subjects under one owner bind forty addresses. Identity grants no
        reporter or validator standing; the owner hash only lets the operator see the cluster."""
        ident, iss = make()
        for i in range(40):
            bind_direct(ident, iss, AddressKey.generate(), sub=f"farm-{i}", owner="one-farmer")
        self.assertEqual(ident.owner_groups(), [{"owner": ident.owner_groups()[0]["owner"], "addresses": 40}])
        self.assertFalse(hasattr(ident, "bond"))


if __name__ == "__main__":
    unittest.main()
