"""traceex — share verified fixes (traces) from any model on traceX.

    from traceex import Trace, Client
    trace = Trace.from_fix(task="extract.flight", base_model="needle3", input=email,
                           model_output=pred, verified_output=fixed, checker="flight-rules@1",
                           producer="0xYourAddress")
    Client("http://localhost:8787").submit(trace)      # skeletonised on your machine before it leaves
"""
from .canon import canonical, object_id
from .skeleton import skeletonize, find_pii, find_secrets, find_open_risks
from .trace import Trace, Learning
from .loop import extract_checked
from .auction import clear_shared, clear_exclusive, Bid
from .royalty import split_trace_sale, split_usage, TRACE_SALE_SPLIT
from .merkle import leaf, build_tree, proof, verify
from .client import Client
from .adapt import routing_from_traces, apply_routing, first_pass_score, attest, AdaptiveAgent

__all__ = ["find_secrets", "find_open_risks", "routing_from_traces", "apply_routing", "first_pass_score", "attest", "AdaptiveAgent","canonical", "object_id", "skeletonize", "find_pii", "Trace", "Learning", "extract_checked",
           "clear_shared", "clear_exclusive", "Bid", "split_trace_sale", "split_usage", "TRACE_SALE_SPLIT",
           "leaf", "build_tree", "proof", "verify", "Client"]
__version__ = "0.1.0"
