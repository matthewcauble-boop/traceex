"""traceX on Vercel: the exchange's API and MCP endpoint, read-only, from a snapshot of the seeded testnet.

Vercel serves the website (site/) from its edge and sends /v0, /mcp and /.well-known here (vercel.json). A Vercel
function keeps no disk between instances, so it can't hold the exchange's wallets and ledger; this one builds the
seeded testnet (node/seed.py, the same seed the hosted node starts from) when an instance starts and refuses every
write. Search, the failure registry (failures, fixes, model reports), bounties, learnings, verdicts, the economy's numbers
(in sats) and the read-only MCP tools all work. The live
exchange, with wallets, runs where its database can live (render.yaml); once it does, point the rewrites in
vercel.json at it and this function steps aside.
"""
import os
import sys
import tempfile
from urllib.parse import parse_qsl, urlencode, urlparse

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "node"), os.path.join(ROOT, "sdk", "python")]
from sats import SatsExchange  # noqa: E402
from exchange import make_handler  # noqa: E402
from seed import seed_if_empty  # noqa: E402

READ_ONLY = ("this is the read-only preview of the traceX testnet: search and browse everything; wallets, trading, "
             "pledges and sharing open on the live node")
WRITES = ("faucet", "submit_trace", "bid", "clear", "register_learning", "usage", "settle", "register_checker",
          "post_bounty", "pledge", "claim_bounty", "remove", "reclassify", "register_validator", "commit", "reveal",
          "challenge", "direct_licence", "register_decoy", "unseal_decoy", "set_btc_usd",
          "claim_fix", "commit_fix", "reveal_fix", "register_model", "poster_measure", "repro_check",
          "post_reporter_bond", "withdraw_reporter")

os.environ.pop("TYPESAFE_API_KEY", None)            # the snapshot is filed by the keyword engine, the same every time
DB = os.path.join(tempfile.gettempdir(), "tracex-preview.db")
if os.path.exists(DB):                              # every instance starts from a fresh seed, never a stale one
    os.remove(DB)
ex = SatsExchange(DB, test_credits=30_000_000, reserve_msats=50_000)   # 30,000 test sats a wallet; 50-sat reserve
seed_if_empty(ex, flight=False)                     # code-repair runs only: no flight/email records


def _frozen(*a, **k):
    raise PermissionError(READ_ONLY)


for _name in WRITES:                                # instance attributes shadow the methods: every write refuses
    setattr(ex, _name, _frozen)
_describe = ex.describe
ex.describe = lambda: dict(_describe(), read_only=True, preview=READ_ONLY)


REWRITE = ("x_route", "x_tail")                     # what vercel.json's rewrites add to the query: never the API's own


def original_path(path):
    """vercel.json rewrites /v0/x?q to /api/node?x_route=/v0/x&q, and Vercel adds the capture (x_tail) and may hand the
    function either path; put the request back the way the client sent it, with none of the rewrite's parameters."""
    u = urlparse(path)
    q = parse_qsl(u.query, keep_blank_values=True)
    route = next((v for k, v in q if k == "x_route"), None) or (u.path if not u.path.startswith("/api/") else "/")
    rest = urlencode([(k, v) for k, v in q if k not in REWRITE])
    return route + ("?" + rest if rest else "")


class handler(make_handler(ex, public=True, admin_token=None)):
    def do_GET(self):
        self.path = original_path(self.path)
        super().do_GET()

    def do_HEAD(self):
        self.path = original_path(self.path)
        super().do_HEAD()

    def do_POST(self):
        self.path = original_path(self.path)
        super().do_POST()

    def do_OPTIONS(self):
        self.path = original_path(self.path)
        super().do_OPTIONS()
