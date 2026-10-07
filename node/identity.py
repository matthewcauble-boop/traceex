"""Identity on a traceX node: signed messages, and optional sign-in with AgentID (or any OpenID Connect provider).

What it adds (SPEC: identity; docs/identity.md):

* **Signed actions.** A request body may carry `_sig: {address, nonce, ts, signature}`, an EIP-191 signature by the
  address's secp256k1 key over the node, the route, the body's hash, a one-time nonce and the time
  (traceex.identity.action_message). The node recovers the signer, refuses replays (nonce per address, 5-minute
  window) and checks that the signer is the route's actor (the `validator` of a commit, the `judge` of a
  confirmation, the bounty's `poster` for a measurement or claim, the `backer` of a pledge...). Validator, judge and
  poster messages that a public node used to take only from its operator are accepted when signed by their actor:
  that closes "operator-relayed until signed" for those routes. Signatures need no identity provider at all.
* **Bindings (optional).** An agent that signs in with AgentID can bind its AgentID subject to its traceX address. The
  agent proves both halves: an ID token the node verified (ES256 against the issuer's JWKS, issuer, audience = this
  node's client id, expiry, single-use jti, and a nonce the node issued for that address), and an EIP-191 signature by
  the address over (node, issuer, subject, address, nonce). The node keeps the subject and keyed hashes of the email
  and owner id, never the email, the owner's name or any token. A node may require that signers of validator
  messages be bound (TRACEX_SIGNERS_MUST_BIND=1); it never requires AgentID for anything else.
* **Not a bond.** Logins are cheap: one owner can hold many AgentMail inboxes. Reporter and validator bonds stay
  exactly as they are, and the sybil analysis is unchanged. A binding only adds a stable pseudonymous owner hash an
  operator can group by.

Environment (all optional; with none set, signed actions work and AgentID is off):
  TRACEX_AGENTID_CLIENT_ID      this node's AgentID client id. Open client (no registration): an https URL the
                                operator controls that serves this node, e.g. https://tracex.example. Registered
                                client: the opaque id from the AgentID console.
  TRACEX_AGENTID_CLIENT_SECRET  registered clients only (client_secret_basic); never commit it.
  TRACEX_AGENTID_REDIRECT_URI   default: <client id>/v0/identity/agentid/callback (must be on the client id's site)
  TRACEX_NODE_ID                what signatures name as the node (default: TRACEX_PUBLIC_URL, the AgentID client
                                id, or a random id kept in the database)
  TRACEX_IDENTITY_SECRET        key for the email/owner hashes (default: random, kept in the database)
  TRACEX_REQUIRE_SIGNATURES=1   open routes with an actor (traces, pledges, bonds...) must be signed by it
  TRACEX_SIGNERS_MUST_BIND=1    signed validator/judge/poster messages are accepted only from bound addresses
"""
import base64
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import urlparse

from traceex.identity import (ADDRESS, NONCE, P256, action_message, binding_message, ecdsa_verify, recover_address,
                              same_address)

PROVIDER = re.compile(r"[a-z][a-z0-9_-]{0,31}")
STATE = re.compile(r"[A-Za-z0-9_-]{20,64}")
ACTION_WINDOW = 300            # a signed action is good for 5 minutes either side of its timestamp
CHALLENGE_TTL = 600            # a binding nonce, and a browser sign-in, must finish within 10 minutes
MAX_OPEN_CHALLENGES = 5        # per address
DIGESTINFO_SHA256 = bytes.fromhex("3031300d060960864801650304020105000420")


class IdentityError(PermissionError):
    """A signature, token or binding was refused. `code` says which check failed."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Conflict(IdentityError):
    pass


# --- JWS / JWT ---------------------------------------------------------------------------------------------------------
def b64url_decode(s: str) -> bytes:
    if not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9_-]*", s):
        raise IdentityError("malformed", "not base64url")
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _jwk_int(k, name):
    return int.from_bytes(b64url_decode(k.get(name, "")), "big")


def verify_jws(token: str, key_for, algs=("ES256",)):
    """(header, claims) of a compact JWS whose signature checks, else IdentityError. key_for(kid) -> JWK or None.
    Only asymmetric algorithms are ever accepted: 'none' and HMAC algorithms are refused whatever `algs` says."""
    if not isinstance(token, str) or len(token) > 16_384 or token.count(".") != 2:
        raise IdentityError("malformed", "not a compact JWT")
    h64, p64, s64 = token.split(".")
    try:
        header, claims = json.loads(b64url_decode(h64)), json.loads(b64url_decode(p64))
    except (ValueError, UnicodeDecodeError):
        raise IdentityError("malformed", "JWT header or payload is not JSON")
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise IdentityError("malformed", "JWT header or payload is not an object")
    alg = header.get("alg")
    if alg not in algs or alg not in ("ES256", "RS256"):
        raise IdentityError("bad_alg", f"algorithm {alg!r} is not accepted")
    key = key_for(header.get("kid"))
    if not key:
        raise IdentityError("unknown_key", "no issuer key with that kid")
    if key.get("alg") not in (None, alg) or key.get("use") not in (None, "sig"):
        raise IdentityError("bad_key", "the issuer key is not for this algorithm")
    signing_input, sig = f"{h64}.{p64}".encode(), b64url_decode(s64)
    if alg == "ES256":
        if key.get("kty") != "EC" or key.get("crv") != "P-256" or len(sig) != 64:
            raise IdentityError("bad_signature", "bad ES256 signature or key")
        pub = (_jwk_int(key, "x"), _jwk_int(key, "y"))
        ok = ecdsa_verify(P256, pub, hashlib.sha256(signing_input).digest(),
                          int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big"))
    else:
        if key.get("kty") != "RSA":
            raise IdentityError("bad_signature", "bad RS256 key")
        n, e = _jwk_int(key, "n"), _jwk_int(key, "e")
        k = (n.bit_length() + 7) // 8
        if n.bit_length() < 2048 or len(sig) != k:
            raise IdentityError("bad_signature", "bad RS256 signature or key")
        em = pow(int.from_bytes(sig, "big"), e, n).to_bytes(k, "big")
        t = DIGESTINFO_SHA256 + hashlib.sha256(signing_input).digest()
        ok = hmac.compare_digest(em, b"\x00\x01" + b"\xff" * (k - len(t) - 3) + b"\x00" + t)
    if not ok:
        raise IdentityError("bad_signature", "the token's signature does not verify")
    return header, claims


def http_get_json(url, timeout=10):
    if not url.startswith("https://"):
        raise ValueError("identity endpoints must be https")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}),
                                timeout=timeout) as r:
        return json.loads(r.read(1_000_000))


def http_post_form(url, form, auth=None, timeout=10):
    if not url.startswith("https://"):
        raise ValueError("identity endpoints must be https")
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    if auth:
        headers["Authorization"] = "Basic " + base64.b64encode(
            f"{urllib.parse.quote(auth[0], safe='')}:{urllib.parse.quote(auth[1], safe='')}".encode()).decode()
    req = urllib.request.Request(url, data=urllib.parse.urlencode(form).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read(1_000_000))
    except urllib.error.HTTPError as e:
        raise IdentityError("token_endpoint", f"token endpoint said {e.code}")


class JWKS:
    """An issuer's signing keys by kid, refetched when a token names a kid we don't have (key rotation), at most once
    a minute so a flood of junk kids can't turn the node into a JWKS hammer."""

    def __init__(self, uri=None, keys=None, fetch=None, ttl=3600, min_refresh=60, clock=time.time):
        self.uri, self.fetch, self.ttl, self.min_refresh, self.clock = uri, fetch or http_get_json, ttl, min_refresh, clock
        self.keys, self.fetched, self.lock = {}, None, threading.Lock()
        if keys is not None:
            self._load({"keys": keys})
            self.fetched = clock()

    def _load(self, doc):
        self.keys = {k.get("kid"): k for k in (doc or {}).get("keys", []) if isinstance(k, dict)}

    def __call__(self, kid):
        now = self.clock()
        with self.lock:
            stale = self.fetched is None or now - self.fetched > self.ttl
            if self.uri and (stale or (kid not in self.keys and now - self.fetched > self.min_refresh)):
                try:
                    self._load(self.fetch(self.uri))
                    self.fetched = now
                except Exception:          # keep the keys we have; a token with a new kid fails until the next try
                    if self.fetched is None:
                        self.fetched = now - self.ttl + self.min_refresh
            return self.keys.get(kid)


# --- OpenID Connect --------------------------------------------------------------------------------------------------
class OIDCConfig:
    """One relying-party configuration. Provider-agnostic: AgentID is a preset (agentid_config)."""

    def __init__(self, provider, issuer, jwks_uri, client_id, *, authorization_endpoint=None, token_endpoint=None,
                 client_secret=None, redirect_uri=None, scopes=("openid",), algs=("ES256", "RS256"),
                 required_claims=None, leeway=60, max_token_age=900, iss_param=False):
        if not PROVIDER.fullmatch(provider or ""):
            raise ValueError("provider name: lowercase letters, digits, _ or -")
        self.provider, self.issuer, self.jwks_uri, self.client_id = provider, issuer, jwks_uri, client_id
        self.authorization_endpoint, self.token_endpoint = authorization_endpoint, token_endpoint
        self.client_secret, self.redirect_uri, self.scopes, self.algs = client_secret, redirect_uri, tuple(scopes), tuple(algs)
        self.required_claims, self.leeway, self.max_token_age = dict(required_claims or {}), leeway, max_token_age
        self.iss_param = iss_param

    @property
    def browser_flow(self):
        return bool(self.authorization_endpoint and self.token_endpoint and self.redirect_uri)

    @classmethod
    def from_discovery(cls, provider, issuer, client_id, fetch=None, **kw):
        """Read issuer/.well-known/openid-configuration (any standard OIDC provider)."""
        d = (fetch or http_get_json)(issuer.rstrip("/") + "/.well-known/openid-configuration")
        if d.get("issuer") != issuer:
            raise ValueError("discovery document names a different issuer")
        kw.setdefault("algs", tuple(a for a in d.get("id_token_signing_alg_values_supported", ["RS256"])
                                    if a in ("ES256", "RS256")))
        return cls(provider, issuer, d["jwks_uri"], client_id, authorization_endpoint=d.get("authorization_endpoint"),
                   token_endpoint=d.get("token_endpoint"),
                   iss_param=bool(d.get("authorization_response_iss_parameter_supported")), **kw)


# AgentID (AgentMail). The endpoint values are real: read from https://auth.agentid.com/.well-known/openid-configuration
# on 2026-10-06 (brain/originals/agentid-research-2026-10-06.md in the project vault). Only the client id is the
# operator's to fill: TRACEX_AGENTID_CLIENT_ID (see the module docstring and docs/identity.md).
AGENTID = dict(provider="agentid", issuer="https://auth.agentid.com",
               jwks_uri="https://auth.agentid.com/v0/jwks.json",
               authorization_endpoint="https://auth.agentid.com/v0/authorize",
               token_endpoint="https://auth.agentid.com/v0/token",
               scopes=("openid", "email", "profile"),        # never owner_profile/owner_email: we don't keep them
               algs=("ES256",), required_claims={"actor_type": "agent"}, iss_param=True)


def agentid_config(client_id=None, client_secret=None, redirect_uri=None, env=None):
    """The AgentID preset, or None while the operator hasn't set a client id (AgentID stays off)."""
    env = os.environ if env is None else env
    client_id = client_id or env.get("TRACEX_AGENTID_CLIENT_ID", "").strip()     # TODO(owner): set on the live node
    if not client_id:
        return None
    client_secret = client_secret or env.get("TRACEX_AGENTID_CLIENT_SECRET") or None
    if not client_secret and not client_id.startswith("https://"):
        raise ValueError("an AgentID open client's id must be an https URL the node is served from "
                         "(or set TRACEX_AGENTID_CLIENT_SECRET for a registered client)")
    redirect_uri = redirect_uri or env.get("TRACEX_AGENTID_REDIRECT_URI") or (
        client_id.rstrip("/") + "/v0/identity/agentid/callback" if client_id.startswith("https://") else None)
    return OIDCConfig(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, **AGENTID)


class OIDCVerifier:
    def __init__(self, config: OIDCConfig, jwks=None, clock=time.time):
        self.config, self.clock = config, clock
        self.jwks = jwks if jwks is not None else JWKS(config.jwks_uri, clock=clock)

    def verify(self, token, nonce=None):
        """The claims of a valid ID token for this node, else IdentityError. nonce: required value of the nonce claim."""
        c = self.config
        _, claims = verify_jws(token, self.jwks, c.algs)
        now = self.clock()
        if claims.get("iss") != c.issuer:
            raise IdentityError("wrong_issuer", "the token is from another issuer")
        aud = claims.get("aud")
        auds = aud if isinstance(aud, list) else [aud]
        if c.client_id not in auds or (len(auds) > 1 and claims.get("azp") != c.client_id):
            raise IdentityError("wrong_audience", "the token was issued to another app")
        for k in ("exp", "iat"):
            if not isinstance(claims.get(k), (int, float)) or isinstance(claims.get(k), bool):
                raise IdentityError("malformed", f"the token has no numeric {k}")
        if now >= claims["exp"] + c.leeway:
            raise IdentityError("expired", "the token has expired")
        if claims["iat"] > now + c.leeway or (c.max_token_age and now - claims["iat"] > c.max_token_age + c.leeway):
            raise IdentityError("stale", "the token's issue time is out of range")
        sub = claims.get("sub")
        if not isinstance(sub, str) or not 1 <= len(sub) <= 255:
            raise IdentityError("malformed", "the token has no subject")
        if not isinstance(claims.get("jti"), str) or not claims["jti"]:
            raise IdentityError("malformed", "the token has no jti, so its replay can't be refused")
        for k, v in c.required_claims.items():
            if claims.get(k) != v:
                raise IdentityError("wrong_actor", f"the token's {k} is not {v!r}")
        if nonce is not None and not (isinstance(claims.get("nonce"), str)
                                      and hmac.compare_digest(claims["nonce"].encode(), str(nonce).encode())):
            raise IdentityError("wrong_nonce", "the token's nonce is not the one this node issued")
        return claims


# --- which routes have an actor, and which used to be operator-relayed -------------------------------------------------
BOUNTY_POSTER = "bounty poster"
RELAYED = [(re.compile(p), a) for p, a in (
    (r"/v0/fixes/[A-Za-z0-9-]+/(commits|reveals)", "validator"),
    (r"/v0/failures/[A-Za-z0-9-]+/repro", "validator"),
    (r"/v0/learnings/[^/]+/(commits|reveals)", "validator"),
    (r"/v0/challenges/prior-art/\d+/(commits|reveals)", "validator"),
    (r"/v0/challenges/submissions/\d+/(commits|reveals)", "validator"),
    (r"/v0/challenges/submissions/\d+/confirmations", "judge"),
    (r"/v0/bounties/(\d+)/(claims|measurements)", BOUNTY_POSTER),
    (r"/v0/validators", "address"),
    (r"/v0/licences/direct", "buyer"))]
OPEN = [(re.compile(p), a) for p, a in (
    (r"/v0/traces", "producer"), (r"/v0/bids", "bidder"), (r"/v0/usage", "consumer"),
    (r"/v0/bounties", "poster"), (r"/v0/bounties/\d+/pledges", "backer"),
    (r"/v0/challenges", "poster"), (r"/v0/challenges/\d+/pledges", "backer"),
    (r"/v0/challenges/\d+/submissions", "submitter"), (r"/v0/fixes", "claimant"),
    (r"/v0/reporters", "address"), (r"/v0/reporters/withdraw", "address"),
    (r"/v0/learnings/[^/]+/challenges", "challenger"), (r"/v0/identity/unbind", "address"))]


def _route(path, table):
    for rx, actor in table:
        m = rx.fullmatch(path)
        if m:
            return m, actor
    return None, None


SCHEMA = """
CREATE TABLE IF NOT EXISTS ident_meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS ident_challenges (nonce TEXT PRIMARY KEY, address TEXT NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS ident_jti (iss TEXT, jti TEXT, expires REAL NOT NULL, PRIMARY KEY (iss, jti));
CREATE TABLE IF NOT EXISTS ident_action_nonces (address TEXT, nonce TEXT, expires REAL NOT NULL,
                                                PRIMARY KEY (address, nonce));
CREATE TABLE IF NOT EXISTS ident_pending (state TEXT PRIMARY KEY, provider TEXT, address TEXT, nonce TEXT,
    verifier TEXT, expires REAL, status TEXT, error TEXT, issuer TEXT, subject TEXT, email_hash TEXT, owner_hash TEXT);
CREATE TABLE IF NOT EXISTS ident_bindings (provider TEXT, subject TEXT, address TEXT, issuer TEXT, email_hash TEXT,
    owner_hash TEXT, bound_at REAL, revoked_at REAL);
CREATE UNIQUE INDEX IF NOT EXISTS ident_live_subject ON ident_bindings (provider, subject) WHERE revoked_at IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ident_live_address ON ident_bindings (provider, address) WHERE revoked_at IS NULL;
"""


class _DB:
    """What exchange.SafeDB offers, for an Identity with no exchange (tests, tools)."""

    def __init__(self, path=":memory:"):
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)

    def execute(self, sql, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            return cur

    def executescript(self, sql):
        with self.lock:
            return self.conn.executescript(sql)

    def commit(self):
        with self.lock:
            self.conn.commit()


READ_ONLY = "this is the read-only preview: identity sign-in and bindings open on the live node"


class Identity:
    def __init__(self, db=None, providers=None, *, node_id=None, secret=None, http_post=http_post_form,
                 require_signatures=False, signers_must_bind=False, read_only=False, clock=time.time):
        """providers: {name: OIDCVerifier}. Nothing here is required: with no providers, signed actions still work."""
        self.db = db if db is not None else _DB()
        self.lock = getattr(self.db, "lock", None) or threading.RLock()
        self.db.executescript(SCHEMA)
        self.providers, self.http_post, self.clock = dict(providers or {}), http_post, clock
        self.require_signatures, self.signers_must_bind, self.read_only = require_signatures, signers_must_bind, read_only
        self.node_id = node_id or self._meta("node_id", lambda: "tracex-node-" + secrets.token_hex(8))
        self._secret = (secret or self._meta("hash_secret", lambda: secrets.token_hex(32))).encode()

    @classmethod
    def from_env(cls, db=None, env=None, read_only=False, **kw):
        env = os.environ if env is None else env
        providers = {}
        cfg = agentid_config(env=env)
        if cfg:
            providers["agentid"] = OIDCVerifier(cfg)
        node_id = env.get("TRACEX_NODE_ID") or env.get("TRACEX_PUBLIC_URL") or (cfg.client_id if cfg and
                                                                                cfg.client_id.startswith("https://") else None)
        return cls(db, providers, node_id=node_id, secret=env.get("TRACEX_IDENTITY_SECRET"),
                   require_signatures=env.get("TRACEX_REQUIRE_SIGNATURES") == "1",
                   signers_must_bind=env.get("TRACEX_SIGNERS_MUST_BIND") == "1", read_only=read_only, **kw)

    # --- small helpers ---------------------------------------------------------------------------------------------
    def _meta(self, k, make):
        with self.lock:
            row = self.db.execute("SELECT v FROM ident_meta WHERE k=?", (k,)).fetchone()
            if row:
                return row[0]
            v = make()
            self.db.execute("INSERT INTO ident_meta (k, v) VALUES (?, ?)", (k, v))
            self.db.commit()
            return v

    def _hash(self, value):
        """Keyed hash of an email or owner id: lets an operator see that two bindings share one, never what it is."""
        if not isinstance(value, str) or not value:
            return None
        return hmac.new(self._secret, value.strip().lower().encode(), hashlib.sha256).hexdigest()

    def _write(self):
        if self.read_only:
            raise PermissionError(READ_ONLY)

    def _provider(self, name):
        v = self.providers.get(name or "agentid")
        if not v:
            raise IdentityError("no_provider", f"this node doesn't accept {name or 'agentid'} sign-in")
        return v

    def _prune(self, now):
        for t in ("ident_challenges", "ident_jti", "ident_action_nonces", "ident_pending"):
            self.db.execute(f"DELETE FROM {t} WHERE expires < ?", (now - ACTION_WINDOW,))

    def describe(self):
        return {"node": self.node_id, "read_only": self.read_only,
                "providers": {n: {"issuer": v.config.issuer, "client_id": v.config.client_id,
                                  "browser_flow": v.config.browser_flow, "scopes": list(v.config.scopes)}
                              for n, v in self.providers.items()},
                "require_signatures": self.require_signatures, "signers_must_bind": self.signers_must_bind,
                "signed_routes": [rx.pattern for rx, _ in RELAYED + OPEN],
                "note": "Signed actions: body + _sig {address, nonce, ts, signature} (EIP-191 over "
                        "traceex.identity.action_message). An identity is optional and never replaces a sats bond."}

    # --- signed actions --------------------------------------------------------------------------------------------
    def verify_action(self, path, body, sig):
        """The address that signed this action, else IdentityError. Consumes the nonce."""
        if not isinstance(sig, dict):
            raise IdentityError("malformed", "_sig must be an object")
        addr, nonce, ts, signature = sig.get("address"), sig.get("nonce"), sig.get("ts"), sig.get("signature")
        if not (isinstance(addr, str) and ADDRESS.fullmatch(addr) and isinstance(nonce, str) and NONCE.fullmatch(nonce)
                and isinstance(ts, int) and not isinstance(ts, bool)):
            raise IdentityError("malformed", "_sig needs address (0x...), nonce (16-64 url-safe chars), ts (seconds)")
        now = self.clock()
        if abs(now - ts) > ACTION_WINDOW:
            raise IdentityError("expired", "the signed action's time is more than 5 minutes off")
        signer = recover_address(action_message(self.node_id, path, body, nonce, ts), signature)
        if not same_address(signer, addr):
            raise IdentityError("bad_signature", "the signature is not by _sig.address over this node, route and body")
        with self.lock:
            self._prune(now)
            try:
                self.db.execute("INSERT INTO ident_action_nonces (address, nonce, expires) VALUES (?, ?, ?)",
                                (addr.lower(), nonce, ts + ACTION_WINDOW))
                self.db.commit()
            except sqlite3.IntegrityError:
                raise IdentityError("replay", "that signed action was already used")
        return addr.lower()

    def _actor(self, ex, path, body, field, m):
        if field == BOUNTY_POSTER:
            row = ex.db.execute("SELECT poster FROM bounties WHERE id=?", (int(m[1]),)).fetchone() if ex else None
            return row[0] if row else None
        return body.get(field)

    def take_signature(self, ex, path, body, admin=False):
        """Strip `_sig` from a request body and return the verified signer (or None if unsigned). A signature that
        doesn't verify, or whose signer isn't the route's actor, is refused outright; so is an unsigned open-route
        write when the node requires signatures (the operator, with its admin token, is exempt)."""
        if not isinstance(body, dict):
            return None
        sig = body.pop("_sig", None)
        m, field = _route(path, RELAYED)
        if not m:
            m, field = _route(path, OPEN)
            relayed = False
        else:
            relayed = True
        if sig is None:
            if self.require_signatures and field and not relayed and not admin:
                raise IdentityError("unsigned", f"this node requires {path} to be signed by its {field}")
            return None
        signer = self.verify_action(path, body, sig)
        if field:
            actor = self._actor(ex, path, body, field, m)
            if not same_address(actor, signer):
                raise IdentityError("not_actor", f"signed by {signer}, but this call's {field} is {actor}")
        return signer

    def relay_ok(self, path, signer):
        """May a signed message stand in for the operator's relay on this route?"""
        m, _ = _route(path, RELAYED)
        if not m or not signer:
            return False
        if self.signers_must_bind and not self.bindings_of(signer):
            raise IdentityError("unbound", "this node accepts signed validator messages only from addresses bound to "
                                           "an identity (POST /v0/identity/bindings)")
        return True

    # --- bindings ----------------------------------------------------------------------------------------------------
    def challenge(self, address, now=None):
        """A one-time nonce for binding an identity to `address`: put it in the ID token's nonce."""
        self._write()
        if not isinstance(address, str) or not ADDRESS.fullmatch(address):
            raise ValueError("address must be a 0x address (40 hex characters)")
        now = self.clock() if now is None else now
        nonce = secrets.token_urlsafe(24)
        with self.lock:
            self._prune(now)
            old = self.db.execute("SELECT nonce FROM ident_challenges WHERE address=? ORDER BY expires DESC",
                                  (address.lower(),)).fetchall()
            for (n,) in old[MAX_OPEN_CHALLENGES - 1:]:
                self.db.execute("DELETE FROM ident_challenges WHERE nonce=?", (n,))
            self.db.execute("INSERT INTO ident_challenges (nonce, address, expires) VALUES (?, ?, ?)",
                            (nonce, address.lower(), now + CHALLENGE_TTL))
            self.db.commit()
        return {"nonce": nonce, "address": address.lower(), "expires_in": CHALLENGE_TTL, "node": self.node_id}

    def _record(self, provider, issuer, subject, address, email_hash, owner_hash, consume):
        """Write the binding, after every check passed. consume(): the single-use deletes/inserts, run under the
        same lock so two racing requests can't both win."""
        address = address.lower()
        with self.lock:
            live = self.db.execute("SELECT subject, address FROM ident_bindings WHERE provider=? AND revoked_at IS NULL "
                                   "AND (subject=? OR address=?)", (provider, subject, address)).fetchall()
            for s, a in live:
                if s == subject and a != address:
                    raise Conflict("subject_bound", f"this {provider} identity is bound to {a}; unbind it there first "
                                                    "(POST /v0/identity/unbind, signed by that address)")
                if a == address and s != subject:
                    raise Conflict("address_bound", f"{address} is bound to another {provider} identity; unbind first")
            consume()
            now = self.clock()
            if live:
                self.db.execute("UPDATE ident_bindings SET bound_at=?, email_hash=?, owner_hash=? WHERE provider=? "
                                "AND subject=? AND revoked_at IS NULL", (now, email_hash, owner_hash, provider, subject))
            else:
                self.db.execute("INSERT INTO ident_bindings (provider, subject, address, issuer, email_hash, owner_hash, "
                                "bound_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (provider, subject, address, issuer, email_hash, owner_hash, now))
            self.db.commit()
        return {"provider": provider, "subject": subject, "address": address, "bound_at": now, "node": self.node_id}

    def _consume_jti(self, issuer, claims):
        try:
            self.db.execute("INSERT INTO ident_jti (iss, jti, expires) VALUES (?, ?, ?)",
                            (issuer, claims["jti"], float(claims["exp"]) + 120))
        except sqlite3.IntegrityError:
            raise IdentityError("replay", "that ID token was already used")

    def bind(self, body):
        """POST /v0/identity/bindings. Either {id_token, address, signature, provider?} (the token's nonce must be one
        this node issued for address) or {pending: state, address, signature} after a browser sign-in."""
        self._write()
        address, signature = body.get("address"), body.get("signature")
        if not isinstance(address, str) or not ADDRESS.fullmatch(address):
            raise ValueError("address must be a 0x address (40 hex characters)")
        if body.get("pending"):
            return self._bind_pending(str(body["pending"]), address, signature)
        provider = body.get("provider") or "agentid"
        v = self._provider(provider)
        claims = v.verify(body.get("id_token"))
        nonce = claims.get("nonce")
        if not isinstance(nonce, str):
            raise IdentityError("wrong_nonce", "the token carries no nonce: get one from POST /v0/identity/challenges")
        row = self.db.execute("SELECT address, expires FROM ident_challenges WHERE nonce=?", (nonce,)).fetchone()
        if not row or row[1] < self.clock():
            raise IdentityError("wrong_nonce", "the token's nonce is unknown, used or expired")
        if not same_address(row[0], address):
            raise IdentityError("wrong_address", "the token's nonce was issued for a different address")
        msg = binding_message(self.node_id, v.config.issuer, claims["sub"], address, nonce)
        if not same_address(recover_address(msg, signature), address):
            raise IdentityError("bad_signature", "the binding is not signed by the address")

        def consume():
            if self.db.execute("DELETE FROM ident_challenges WHERE nonce=?", (nonce,)).rowcount != 1:
                raise IdentityError("replay", "that nonce was already used")
            self._consume_jti(v.config.issuer, claims)
        return self._record(provider, v.config.issuer, claims["sub"], address, self._hash(claims.get("email")),
                            self._hash(claims.get("owner_sub")), consume)

    def _bind_pending(self, state, address, signature):
        row = self.db.execute("SELECT provider, address, nonce, status, issuer, subject, email_hash, owner_hash, "
                              "expires FROM ident_pending WHERE state=?", (state,)).fetchone()
        if not row or row[3] != "verified" or row[8] < self.clock():
            raise IdentityError("not_verified", "no verified sign-in with that state (or it expired)")
        provider, want, nonce, _, issuer, subject, eh, oh, _ = row
        if not same_address(want, address):
            raise IdentityError("wrong_address", "that sign-in was started for a different address")
        if not same_address(recover_address(binding_message(self.node_id, issuer, subject, address, nonce), signature),
                            address):
            raise IdentityError("bad_signature", "the binding is not signed by the address")

        def consume():
            if self.db.execute("DELETE FROM ident_pending WHERE state=? AND status='verified'", (state,)).rowcount != 1:
                raise IdentityError("replay", "that sign-in was already used")
            self.db.execute("DELETE FROM ident_challenges WHERE nonce=?", (nonce,))
        return self._record(provider, issuer, subject, address, eh, oh, consume)

    def unbind(self, signer, body):
        """POST /v0/identity/unbind, signed by the bound address: {address, provider?}."""
        self._write()
        if not signer or not same_address(signer, body.get("address")):
            raise IdentityError("unsigned", "unbinding must be signed by the bound address")
        provider = body.get("provider") or "agentid"
        with self.lock:
            n = self.db.execute("UPDATE ident_bindings SET revoked_at=? WHERE provider=? AND address=? AND "
                                "revoked_at IS NULL", (self.clock(), provider, signer.lower())).rowcount
            self.db.commit()
        return {"unbound": n, "address": signer.lower(), "provider": provider}

    def bindings_of(self, address):
        rows = self.db.execute("SELECT provider, subject, issuer, bound_at FROM ident_bindings WHERE address=? AND "
                               "revoked_at IS NULL", (str(address or "").lower(),)).fetchall()
        return [{"provider": p, "subject": s, "issuer": i, "bound_at": t} for p, s, i, t in rows]

    def owner_groups(self, min_size=2):
        """Operator view for sybil analysis: how many bound addresses share one (hashed) owner. Hashes only."""
        rows = self.db.execute("SELECT owner_hash, COUNT(*) FROM ident_bindings WHERE revoked_at IS NULL AND owner_hash "
                               "IS NOT NULL GROUP BY owner_hash HAVING COUNT(*) >= ?", (min_size,)).fetchall()
        return sorted(({"owner": h[:16], "addresses": n} for h, n in rows), key=lambda r: -r["addresses"])

    def whoami(self, body):
        """POST /v0/identity/sessions {id_token, provider?}: who the token's subject is bound to here (or nobody)."""
        provider = body.get("provider") or "agentid"
        v = self._provider(provider)
        claims = v.verify(body.get("id_token"))
        row = self.db.execute("SELECT address, bound_at FROM ident_bindings WHERE provider=? AND subject=? AND "
                              "revoked_at IS NULL", (provider, claims["sub"])).fetchone()
        return {"provider": provider, "subject": claims["sub"], "bound": bool(row),
                "address": row[0] if row else None,
                "next": None if row else "POST /v0/identity/challenges {address}, sign in again with that nonce, "
                                         "then POST /v0/identity/bindings"}

    # --- browser (authorization code + PKCE) --------------------------------------------------------------------------
    def start(self, provider, address):
        self._write()
        v = self._provider(provider)
        c = v.config
        if not c.browser_flow:
            raise IdentityError("no_browser_flow", f"{provider} has no authorization endpoint configured here")
        ch = self.challenge(address)
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        with self.lock:
            self.db.execute("INSERT INTO ident_pending (state, provider, address, nonce, verifier, expires, status) "
                            "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
                            (state, provider, ch["address"], ch["nonce"], verifier, self.clock() + CHALLENGE_TTL))
            self.db.commit()
        q = {"response_type": "code", "client_id": c.client_id, "redirect_uri": c.redirect_uri,
             "scope": " ".join(c.scopes), "state": state, "nonce": ch["nonce"],
             "code_challenge": b64url(hashlib.sha256(verifier.encode()).digest()), "code_challenge_method": "S256"}
        return {"authorize_url": c.authorization_endpoint + "?" + urllib.parse.urlencode(q), "state": state,
                "address": ch["address"], "expires_in": CHALLENGE_TTL}

    def callback(self, provider, query):
        """The provider redirected the browser back: swap the code for tokens, verify the ID token, keep the subject
        and hashes, throw the tokens away."""
        self._write()
        state = query.get("state", "")
        if not STATE.fullmatch(state):
            raise ValueError("missing or malformed state")
        row = self.db.execute("SELECT provider, nonce, verifier, status, expires FROM ident_pending WHERE state=?",
                              (state,)).fetchone()
        if not row or row[0] != provider or row[3] != "pending" or row[4] < self.clock():
            raise IdentityError("unknown_state", "this sign-in is unknown, finished or expired; start again")
        v = self._provider(provider)
        c, nonce, verifier = v.config, row[1], row[2]

        def fail(code, msg):
            self.db.execute("UPDATE ident_pending SET status='failed', error=?, verifier=NULL WHERE state=?",
                            (code, state))
            self.db.commit()
            raise IdentityError(code, msg)
        if query.get("error"):
            fail("denied", f"the provider returned {query['error'][:60]}")
        if c.iss_param and query.get("iss") != c.issuer:          # RFC 9207: the code must come from our issuer
            fail("wrong_issuer", "the authorization response is not from the configured issuer")
        if not query.get("code"):
            fail("malformed", "no authorization code")
        form = {"grant_type": "authorization_code", "code": query["code"], "redirect_uri": c.redirect_uri,
                "code_verifier": verifier}
        auth = (c.client_id, c.client_secret) if c.client_secret else None
        if not auth:
            form["client_id"] = c.client_id
        try:
            tokens = self.http_post(c.token_endpoint, form, auth)
        except Exception as e:                    # network, HTTP or JSON trouble: this sign-in is over, the node is fine
            fail("token_endpoint", f"could not redeem the code ({type(e).__name__})")
        try:
            claims = v.verify(tokens.get("id_token") if isinstance(tokens, dict) else None, nonce=nonce)
        except IdentityError as e:
            fail(e.code, str(e))
        with self.lock:
            self._consume_jti(c.issuer, claims)
            n = self.db.execute("UPDATE ident_pending SET status='verified', verifier=NULL, issuer=?, subject=?, "
                                "email_hash=?, owner_hash=?, expires=? WHERE state=? AND status='pending'",
                                (c.issuer, claims["sub"], self._hash(claims.get("email")),
                                 self._hash(claims.get("owner_sub")), self.clock() + CHALLENGE_TTL, state)).rowcount
            self.db.commit()
        if n != 1:
            raise IdentityError("replay", "this sign-in already finished")
        return {"status": "verified", "subject": claims["sub"], "state": state}

    def pending(self, state):
        row = self.db.execute("SELECT status, error, issuer, subject, nonce, address, expires FROM ident_pending "
                              "WHERE state=?", (state,)).fetchone()
        if not row:
            return {"status": "unknown"}
        status = "expired" if row[6] < self.clock() and row[0] != "failed" else row[0]
        out = {"status": status, "address": row[5]}
        if status == "verified":
            out.update(issuer=row[2], subject=row[3], nonce=row[4],
                       sign=binding_message(self.node_id, row[2], row[3], row[5], row[4]))
        if status == "failed":
            out["error"] = row[1]
        return out


# --- one Identity per exchange, and the HTTP routes ------------------------------------------------------------------
_OF = threading.Lock()


def of(ex):
    ident = getattr(ex, "_identity", None)
    if ident is None:
        with _OF:
            ident = getattr(ex, "_identity", None)
            if ident is None:
                ident = Identity.from_env(ex.db, read_only=bool(getattr(ex, "identity_read_only", False)))
                ex._identity = ident
    return ident


def wrap(H, ex):
    """make_handler's class, plus identity: /v0/identity routes, `_sig` taken off every POST body and verified, and
    the operator check (`_admin`) also passing a relayed-route message signed by its own actor. exchange.make_handler
    ends with `return wrap(H, ex)`; nothing else in the exchange changes."""
    if getattr(H, "_identity_wrapped", False):            # already wrapped (make_handler does it once)
        return H

    class IdentityHandler(H):
        def _reset(self):                                   # one handler object serves every request on a keep-alive
            self._ident_body, self._signer = None, None

        def _body(self):
            if getattr(self, "_ident_body", None) is not None:
                return self._ident_body
            b = super()._body()
            self._signer = None
            if isinstance(b, dict) and self.command == "POST":
                self._signer = of(ex).take_signature(ex, urlparse(self.path).path, b, admin=super()._admin())
            self._ident_body = b
            return b

        def _admin(self):
            if super()._admin():
                return True
            if self.command != "POST":
                return False
            self._body()
            return of(ex).relay_ok(urlparse(self.path).path, self._signer)

        def _identity_route(self, method):
            u = urlparse(self.path)
            if not (u.path == "/v0/identity" or u.path.startswith("/v0/identity/")):
                return False
            if not self._limited("read" if method == "GET" else "write"):
                route(self, ex, method, u.path, {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()})
            return True

        def do_GET(self):
            self._reset()
            if not self._identity_route("GET"):
                super().do_GET()

        def do_POST(self):
            self._reset()
            if not self._identity_route("POST"):
                super().do_POST()

    IdentityHandler.__name__ = H.__name__
    IdentityHandler._identity_wrapped = True
    return IdentityHandler


def _page(h, code, title, body_html):
    doc = (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><body style='font:16px/1.5 system-ui,sans-serif;max-width:640px;"
           f"margin:48px auto;padding:0 16px'><h1 style='font-size:22px'>{html.escape(title)}</h1>{body_html}</body>")
    raw = doc.encode()
    h.send_response(code)
    h._headers("text/html; charset=utf-8", len(raw), {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})
    h.wfile.write(raw)


def route(h, ex, method, path, query=None):
    """Serve /v0/identity/... on an exchange handler `h` (make_handler's class). Returns True when it answered."""
    if not (path == "/v0/identity" or path.startswith("/v0/identity/")):
        return False
    query = query or {}
    try:
        ident = of(ex)
        if method == "GET":
            if path == "/v0/identity":
                return h._send(200, ident.describe()) or True
            m = re.fullmatch(r"/v0/identity/bindings/(0x[0-9a-fA-F]{40})", path)
            if m:
                return h._send(200, {"address": m[1].lower(), "bindings": ident.bindings_of(m[1])}) or True
            m = re.fullmatch(r"/v0/identity/([a-z0-9_-]+)/pending/([A-Za-z0-9_-]{20,64})", path)
            if m:
                return h._send(200, ident.pending(m[2])) or True
            m = re.fullmatch(r"/v0/identity/([a-z0-9_-]+)/start", path)
            if m:                                           # a browser: straight to the provider
                s = ident.start(m[1], query.get("address", ""))
                h.send_response(302)
                h.send_header("Location", s["authorize_url"])
                h._headers("text/plain", 0, {"Cache-Control": "no-store"})
                return True
            m = re.fullmatch(r"/v0/identity/([a-z0-9_-]+)/callback", path)
            if m:
                try:
                    r = ident.callback(m[1], query)
                except (PermissionError, ValueError) as e:
                    _page(h, 403 if isinstance(e, PermissionError) else 400, "Sign-in not completed",
                          f"<p>{html.escape(str(e))}</p><p>Start again from your agent or the traceX site.</p>")
                    return True
                _page(h, 200, "Signed in with AgentID" if m[1] == "agentid" else "Signed in",
                      "<p>The node verified the sign-in and kept only the agent's subject id (no email, no token).</p>"
                      "<p>Last step, on the agent's side: its address key signs the binding. The traceX SDK "
                      "(<code>sign_in_with_agentid</code>) does that by itself; you can close this window.</p>"
                      f"<p style='color:#666;font-size:14px'>Sign-in reference: <code>{html.escape(r['state'])}</code>"
                      "</p>")
                return True
            return h._send(404, {"error": "not found"}) or True
        body = h._body()
        if path == "/v0/identity/challenges":
            return h._send(200, ident.challenge(body.get("address"))) or True
        if path == "/v0/identity/bindings":
            return h._send(200, ident.bind(body)) or True
        if path == "/v0/identity/sessions":
            return h._send(200, ident.whoami(body)) or True
        if path == "/v0/identity/unbind":
            signer = getattr(h, "_signer", None)
            if signer is None:                               # the exchange's _body didn't take the signature: do it here
                signer = ident.take_signature(ex, path, body)
            return h._send(200, ident.unbind(signer, body)) or True
        m = re.fullmatch(r"/v0/identity/([a-z0-9_-]+)/start", path)
        if m:
            return h._send(200, ident.start(m[1], body.get("address", ""))) or True
        return h._send(404, {"error": "not found"}) or True
    except Conflict as e:
        h._send(409, {"error": str(e), "code": e.code})
    except IdentityError as e:
        h._send(403, {"error": str(e), "code": e.code})
    except PermissionError as e:
        h._send(403, {"error": str(e)})
    except (ValueError, KeyError, TypeError) as e:
        h._send(400, {"error": str(e)})
    return True
