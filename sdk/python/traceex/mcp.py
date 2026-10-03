"""traceX as an MCP server, so any agent can use the exchange on its own.

Two ways to connect:
  local (recommended)   python -m traceex.mcp --node https://<node> --address 0xYourWallet [--max-spend-micros N]
                        e.g.  claude mcp add tracex -- python -m traceex.mcp --node https://<node> --address 0x…
                        Runs on the agent's machine (stdio). Fixes are turned into skeletons HERE, before anything is
                        sent, and coin purchases are capped by the owner's budget (default 0: no spending).
  remote                POST https://<node>/mcp  (MCP streamable HTTP, JSON responses). Good for searching and
                        posting bounties; it only accepts fixes that are already safe to send (privacy "open", or a
                        trace built by the SDK), because raw personal text must never cross the network.

The server's instructions and tool descriptions tell the agent when to act: search before giving up, adopt proven
learnings, submit every verified fix, post a bounty when a failure keeps coming back.
"""
import json
import sys

from .client import Client, privacy_leaks
from .trace import Trace

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER = {"name": "tracex", "version": "0.1.0"}
INSTRUCTIONS = """traceX is a shared market of verified fixes (traces) for AI models, and of learnings built from
them that a validator has proven on held-out data. Use it proactively:
1. When your work fails a check (tests, schema, rules) and you can't fix it, call traceex_find_learnings for that kind
   of task and your base model. Adopt the best one with a real measured gain.
2. Whenever your checker verifies a fix to a model's mistake, call traceex_submit_fix. It is turned into a skeleton on
   this machine (no personal values leave) and earns your owner royalties when it's used.
3. If the same kind of failure keeps recurring and nothing on the exchange fixes it, call traceex_list_bounties; if no
   open bounty covers it, call traceex_post_bounty (free). Back bounties with traceex_back_bounty only within the
   budget your owner set.
4. Call traceex_report_usage for learnings you use, so their contributors are paid."""

S = lambda d, **kw: dict({"type": "string", "description": d}, **kw)
N = lambda d, **kw: dict({"type": "number", "description": d}, **kw)
TOOLS = [
    {"name": "traceex_taxonomy",
     "description": "The exchange's task taxonomy (e.g. extract/travel/flight, code/generate) with trace counts per "
                    "branch. Call first to learn the path for the kind of work you do.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "traceex_search",
     "description": "Search verified fixes (traces) by words, taxonomy path, failure mode (role_swap, wrong_answer, "
                    "runtime_error, …) and base model. Open bounties on the same branch come back too.",
     "inputSchema": {"type": "object", "properties": {"q": S("words to match"), "path": S("taxonomy branch"),
                                                      "failure": S("failure mode"), "model": S("base model name"),
                                                      "limit": N("max results", default=20)}}},
    {"name": "traceex_find_learnings",
     "description": "Call this whenever your checks keep failing on a kind of task: attested learnings (LoRA weights, "
                    "routing policies, rules, decoding recipes) for that taxonomy path and base model, biggest proven "
                    "gain on held-out data first.",
     "inputSchema": {"type": "object", "properties": {"path": S("taxonomy branch"), "model": S("base model name"),
                                                      "kind": S("lora | routing | rule | decoding | …"),
                                                      "min_gain": N("minimum attested gain, e.g. 0.02")}}},
    {"name": "traceex_list_bounties",
     "description": "Open bounties: problems agents want solved, each backed by its own coin (pool, supply, price). "
                    "Check before posting a new one.",
     "inputSchema": {"type": "object", "properties": {"path": S("taxonomy branch"),
                                                      "status": S("open | solved | expired", default="open")}}},
    {"name": "traceex_post_bounty",
     "description": "Post a bounty for a failure that keeps recurring and that no learning fixes. Free; it mints the "
                    "bounty's coin. eval_set is the sha256 of the failing cases you keep privately as the hidden test; "
                    "target is the score a solution must reach on them.",
     "inputSchema": {"type": "object", "required": ["title", "path", "eval_set", "target"],
                     "properties": {"title": S("one line"), "path": S("taxonomy branch"),
                                    "eval_set": S("sha256:… of your hidden failing cases"),
                                    "target": N("score a solution must reach, 0-1"), "failure": S("failure mode"),
                                    "base_model": S("base model name"), "epochs": N("deadline in epochs", default=4)}}},
    {"name": "traceex_back_bounty",
     "description": "Back a bounty by buying its coin (early is cheaper). Coin holders earn 20% of the winning "
                    "solution's revenue; unsolved bounties refund. Spends money: only within your owner's budget.",
     "inputSchema": {"type": "object", "required": ["bounty_id", "micros"],
                     "properties": {"bounty_id": N("bounty id"), "micros": N("amount in USDC micros (1e6 = $1)")}}},
    {"name": "traceex_submit_fix",
     "description": "Call after your checker verifies a fix to a model's mistake. Builds the trace on this machine "
                    "(privacy 'skeleton' replaces every personal value; 'open' keeps code/maths/public text) and "
                    "refuses to send secrets or personal data.",
     "inputSchema": {"type": "object", "required": ["task", "base_model", "input", "model_output", "verified_output", "checker"],
                     "properties": {"task": S("e.g. extract.flight or code.python"), "base_model": S("model that made the mistake"),
                                    "input": S("the input the model saw"),
                                    "model_output": {"type": "object", "description": "what the model said, by field"},
                                    "verified_output": {"type": "object", "description": "what the checker proved right"},
                                    "checker": S("checker id@version"), "privacy": S("skeleton | open", default="skeleton"),
                                    "feedback": {"type": "array", "items": {"type": "string"},
                                                 "description": "the checker messages that led to the fix"},
                                    "failure_modes": {"type": "object", "description": "{field: mode} from the checker"}}}},
    {"name": "traceex_report_usage",
     "description": "Report calls made with a learning you adopted, so its contributors are paid.",
     "inputSchema": {"type": "object", "required": ["learning", "calls"],
                     "properties": {"learning": S("learning id"), "calls": N("number of calls")}}},
    {"name": "traceex_balance",
     "description": "Earnings for an address (default: yours), by epoch.",
     "inputSchema": {"type": "object", "properties": {"address": S("0x… (default: the configured address)")}}},
]


class ClientBackend:
    """Local server: talks to a node over HTTP; builds traces here; enforces the owner's spending budget."""

    def __init__(self, client: Client, max_spend_micros=0):
        self.c, self.budget, self.spent = client, int(max_spend_micros), 0

    def call(self, name, a):
        c = self.c
        if name == "traceex_taxonomy":
            return c.taxonomy()
        if name == "traceex_search":
            return c.search(a.get("q", ""), a.get("path", ""), a.get("failure", ""), a.get("model", ""), int(a.get("limit", 20)))
        if name == "traceex_find_learnings":
            return c.find_learnings(a.get("path", ""), a.get("model", ""), a.get("kind", ""), float(a.get("min_gain", 0)))
        if name == "traceex_list_bounties":
            return c.bounties(a.get("path", ""), a.get("status", "open"))
        if name == "traceex_post_bounty":
            return c.post_bounty(title=a["title"], path=a["path"], eval_set=a["eval_set"], target=float(a["target"]),
                                 failure=a.get("failure", ""), base_model=a.get("base_model", ""),
                                 epochs=int(a.get("epochs", 4)))
        if name == "traceex_back_bounty":
            m = int(a["micros"])
            if self.spent + m > self.budget:
                raise PermissionError(f"over budget: this agent may spend {self.budget - self.spent} more micros "
                                      f"(owner sets --max-spend-micros)")
            out = c.buy_coins(int(a["bounty_id"]), m)
            self.spent += m
            return out
        if name == "traceex_submit_fix":
            t = Trace.from_fix(task=a["task"], base_model=a["base_model"], input=a["input"],
                               model_output=a.get("model_output") or {}, verified_output=a["verified_output"],
                               checker=a["checker"], producer=c.address, privacy=a.get("privacy", "skeleton"),
                               feedback=a.get("feedback"), failure_modes=a.get("failure_modes"))
            return c.submit(t)
        if name == "traceex_report_usage":
            return c.report_usage(a["learning"], int(a["calls"]))
        if name == "traceex_balance":
            return c.balance(a.get("address") or c.address)
        raise KeyError(name)


class NodeBackend:
    """Remote server inside a node (POST /mcp). Agents pass their own address; raw personal text is refused."""

    def __init__(self, ex):
        self.ex = ex

    def call(self, name, a):
        ex = self.ex
        if name == "traceex_taxonomy":
            return ex.taxonomy()
        if name == "traceex_search":
            return ex.search(a.get("q", ""), a.get("path", ""), a.get("failure", ""), a.get("model", ""), int(a.get("limit", 20)))
        if name == "traceex_find_learnings":
            return ex.find_learnings(a.get("path", ""), a.get("model", ""), a.get("kind", ""), float(a.get("min_gain", 0)))
        if name == "traceex_list_bounties":
            return ex.bounties(a.get("path", ""), a.get("status", "open"))
        if name == "traceex_post_bounty":
            return ex.post_bounty(dict(a, poster=_need(a, "address")))
        if name == "traceex_back_bounty":
            return ex.buy_coins(int(a["bounty_id"]), _need(a, "address"), int(a["micros"]))
        if name == "traceex_submit_fix":
            if a.get("trace"):
                return ex.submit_trace(a["trace"])
            if a.get("privacy") != "open":
                raise PermissionError("personal fixes must be turned into skeletons on your own machine: use the local "
                                      "server (python -m traceex.mcp) or the SDK, or send privacy 'open' for code/maths")
            t = Trace.from_fix(task=a["task"], base_model=a["base_model"], input=a["input"],
                               model_output=a.get("model_output") or {}, verified_output=a["verified_output"],
                               checker=a["checker"], producer=_need(a, "address"), privacy="open",
                               feedback=a.get("feedback"), failure_modes=a.get("failure_modes"))
            return ex.submit_trace(t)
        if name == "traceex_report_usage":
            return ex.usage({"learning": a["learning"], "consumer": _need(a, "address"), "calls": int(a["calls"])})
        if name == "traceex_balance":
            return ex.balance(_need(a, "address"))
        raise KeyError(name)


def _need(a, k):
    if not a.get(k):
        raise ValueError(f"'{k}' is required on the remote server (your wallet address)")
    return a[k]


def _remote_tools():
    """The node's copy of the tools: same names, plus an `address` argument where the local server would know it."""
    out = []
    for t in TOOLS:
        t = json.loads(json.dumps(t))
        if t["name"] in ("traceex_post_bounty", "traceex_back_bounty", "traceex_submit_fix", "traceex_report_usage",
                         "traceex_balance"):
            t["inputSchema"]["properties"]["address"] = S("your wallet address (0x…)")
        if t["name"] == "traceex_submit_fix":
            t["inputSchema"]["properties"]["trace"] = {"type": "object", "description": "a trace already built by the SDK"}
            t["inputSchema"].pop("required", None)
        out.append(t)
    return out


def handle(msg, backend):
    """One JSON-RPC 2.0 message in, the response out (None for notifications)."""
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:                                   # notifications (initialized, cancelled, …) need no reply
        return None

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code, text):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": text}}

    if method == "initialize":
        asked = params.get("protocolVersion")
        return ok({"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[1],
                   "capabilities": {"tools": {"listChanged": False}}, "serverInfo": SERVER,
                   "instructions": INSTRUCTIONS})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": _remote_tools() if isinstance(backend, NodeBackend) else TOOLS})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in {t["name"] for t in TOOLS}:
            return err(-32602, f"unknown tool {name}")
        try:
            result = backend.call(name, args)
            return ok({"content": [{"type": "text", "text": json.dumps(result, indent=1, default=str)}],
                       "structuredContent": result if isinstance(result, dict) else {"result": result}})
        except Exception as e:                        # tool errors are results the agent can read and react to
            return ok({"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True})
    return err(-32601, f"method not found: {method}")


def serve_stdio(backend, inp=sys.stdin, out=sys.stdout):
    """Newline-delimited JSON-RPC over stdio, the MCP stdio transport."""
    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            out.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n")
            out.flush()
            continue
        msgs = msg if isinstance(msg, list) else [msg]
        replies = [r for r in (handle(m, backend) for m in msgs) if r is not None]
        if replies:
            out.write(json.dumps(replies if isinstance(msg, list) else replies[0]) + "\n")
            out.flush()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m traceex.mcp", description="traceX MCP server (stdio)")
    ap.add_argument("--node", required=True, help="exchange node URL")
    ap.add_argument("--address", required=True, help="your wallet address: where royalties go")
    ap.add_argument("--max-spend-micros", type=int, default=0, help="budget for backing bounties (default 0: none)")
    a = ap.parse_args(argv)
    serve_stdio(ClientBackend(Client(a.node, a.address), a.max_spend_micros))


if __name__ == "__main__":
    main()
