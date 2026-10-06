"""traceX from OpenTelemetry (ingestion path B, SPEC 4i): a SpanExporter that turns "the model answered, a checker
failed it, the model retried, the checker passed it" into a skeleton trace.

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from traceex.otel import TraceexSpanExporter
    provider.add_span_processor(SimpleSpanProcessor(TraceexSpanExporter()))          # dry run: .traceex/outbox

It reads GenAI semantic-convention spans, so it works with any OTel-instrumented agent, and with the tools that export
OTel (Laminar, LangSmith, OpenLLMetry, OpenInference: add the processor to the provider they set up, next to theirs):
  * a model call: gen_ai.operation.name chat / text_completion / generate_content (or any span with
    gen_ai.request.model), its prompt from gen_ai.input.messages (or gen_ai.prompt.N.content, gen_ai.prompt, or a
    gen_ai.user.message event) and its answer from gen_ai.output.messages (or gen_ai.completion.N.content,
    gen_ai.completion, or a gen_ai.choice event);
  * a check of that answer: an execute_tool span, an evaluation (a gen_ai.evaluation.result event, or
    gen_ai.evaluation.* attributes), or any span whose name says check / eval / validate / verify / test, or that sets
    traceex.check.passed. It failed if traceex.check.passed is false, the evaluation's label says fail, its score is
    under 0.5, or the span ended in error (status ERROR, error.type, an exception event); otherwise it passed.
  * the pattern: model call A, a check that fails it, a later model call B (the retry), a check that passes B. A and B
    pair up as the model's output and the verified output; the failed check's message is the feedback; the failure mode
    comes from the exception type (wrong_answer when the check just said no).
Optional span attributes: traceex.task (the task name; default "agent.<agent name>" or "agent.output"),
traceex.input / traceex.output (explicit text, when the instrumentation records none).

What leaves the machine: nothing, by default (dry run: traces go to .traceex/outbox for review; send them with
`python -m traceex.outbox send`). With submit=True, only the trace JSON: the task name, the model name, the checker's
name, the producer address, a timestamp, and skeletons (traceex.skeleton: every name, email, phone number, URL, date,
time, amount, code and number replaced by a typed placeholder, consistently across the prompt, both answers and the
feedback) of the prompt's last user message (at most max_input characters), the failed and the passing answers, and
the checker's message. Never: system prompts, other messages, tool arguments, span attributes, resource attributes,
trace or span ids, timings. A trace that still holds personal data or a secret after skeletonizing is dropped.
"""
import datetime as dt
import json
import os
import re
import threading

from . import outbox as _outbox
from .client import Client, privacy_leaks
from .trace import Trace

try:                                       # the real base class when the OpenTelemetry SDK is installed
    from opentelemetry.sdk.trace.export import SpanExporter as _Base, SpanExportResult as _Result
    _OK, _FAIL = _Result.SUCCESS, _Result.FAILURE
except Exception:                          # duck-typed without it: export(spans) / shutdown() / force_flush()
    _Base, _OK, _FAIL = object, 0, 1

LLM_OPS = {"chat", "text_completion", "generate_content", "completion", "generate"}
CHECK_WORDS = re.compile(r"check|eval|validat|verif|test|assert|grade|judge", re.I)
PASS = {"pass", "passed", "true", "ok", "success", "correct", "valid", "good", "yes"}
FAIL = {"fail", "failed", "false", "incorrect", "invalid", "error", "bad", "no", "wrong"}
NO_ADDRESS = "0x" + "0" * 40


def _attrs(span):
    return dict(getattr(span, "attributes", None) or {})


def _events(span):
    return list(getattr(span, "events", None) or [])


def _sid(span):
    ctx = span.get_span_context() if hasattr(span, "get_span_context") else getattr(span, "context", None)
    return getattr(ctx, "span_id", None), getattr(ctx, "trace_id", None)


def _parent(span):
    p = getattr(span, "parent", None)
    return getattr(p, "span_id", None) if p is not None else None


def _messages(raw):
    """GenAI semconv messages (a JSON string or a list): [(role, text)]."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return [("", raw)]
    out = []
    for m in raw if isinstance(raw, list) else [raw]:
        if not isinstance(m, dict):
            out.append(("", str(m)))
            continue
        parts = m.get("parts")
        if isinstance(parts, list):
            text = "\n".join(str(p.get("content", "")) for p in parts if isinstance(p, dict) and p.get("type", "text") == "text")
        else:
            c = m.get("content", m.get("text", ""))
            text = c if isinstance(c, str) else json.dumps(c)
        out.append((m.get("role", ""), text))
    return out


def _indexed(attrs, prefix):
    """Legacy flattened attributes: prefix.N.role / prefix.N.content -> [(role, text)] in N order."""
    rows = {}
    for k, v in attrs.items():
        m = re.fullmatch(re.escape(prefix) + r"\.(\d+)\.(role|content)", k)
        if m:
            rows.setdefault(int(m[1]), {})[m[2]] = v
    return [(r.get("role", ""), str(r.get("content", ""))) for _, r in sorted(rows.items())]


def prompt_of(span):
    a = _attrs(span)
    if a.get("traceex.input"):
        return str(a["traceex.input"])
    msgs = []
    if a.get("gen_ai.input.messages"):
        msgs = _messages(a["gen_ai.input.messages"])
    msgs = msgs or _indexed(a, "gen_ai.prompt")
    if not msgs and a.get("gen_ai.prompt"):
        msgs = _messages(a["gen_ai.prompt"])
    if not msgs:
        for e in _events(span):
            if e.name in ("gen_ai.user.message", "gen_ai.content.prompt"):
                ea = dict(e.attributes or {})
                msgs.append(("user", str(ea.get("content") or ea.get("gen_ai.prompt") or "")))
    users = [t for r, t in msgs if r in ("user", "")]
    return users[-1] if users else ""


def output_of(span):
    a = _attrs(span)
    if a.get("traceex.output"):
        return str(a["traceex.output"])
    msgs = []
    if a.get("gen_ai.output.messages"):
        msgs = _messages(a["gen_ai.output.messages"])
    msgs = msgs or _indexed(a, "gen_ai.completion")
    if not msgs and a.get("gen_ai.completion"):
        msgs = _messages(a["gen_ai.completion"])
    if not msgs:
        for e in _events(span):
            if e.name in ("gen_ai.choice", "gen_ai.content.completion"):
                ea = dict(e.attributes or {})
                m = ea.get("message") or ea.get("content") or ea.get("gen_ai.completion") or ""
                msgs += _messages(m) if isinstance(m, str) and m.startswith(("{", "[")) else [("assistant", str(m))]
    texts = [t for r, t in msgs if r in ("assistant", "")]
    return texts[-1] if texts else ""


def is_llm(span):
    a = _attrs(span)
    op = a.get("gen_ai.operation.name")
    return op in LLM_OPS or (op is None and bool(a.get("gen_ai.request.model") or a.get("gen_ai.response.model")))


def _evaluation(span):
    """(label, score, explanation, name) from gen_ai.evaluation.* attributes or a gen_ai.evaluation.result event."""
    srcs = [_attrs(span)] + [dict(e.attributes or {}) for e in _events(span) if e.name == "gen_ai.evaluation.result"]
    for s in srcs:
        if any(k.startswith("gen_ai.evaluation.") for k in s):
            return (s.get("gen_ai.evaluation.score.label"), s.get("gen_ai.evaluation.score.value"),
                    s.get("gen_ai.evaluation.explanation"), s.get("gen_ai.evaluation.name"))
    return None


def is_check(span):
    if is_llm(span):
        return False
    a = _attrs(span)
    return (a.get("gen_ai.operation.name") == "execute_tool" or "traceex.check.passed" in a
            or _evaluation(span) is not None or bool(CHECK_WORDS.search(getattr(span, "name", "") or "")))


def _status_error(span):
    st = getattr(span, "status", None)
    code = getattr(st, "status_code", None)
    return getattr(code, "name", str(code)) == "ERROR"


def verdict(span):
    """(passed, feedback, exception type) of a check span."""
    a = _attrs(span)
    exc_type, exc_msg = a.get("error.type"), None
    for e in _events(span):
        if e.name == "exception":
            ea = dict(e.attributes or {})
            exc_type = ea.get("exception.type") or exc_type
            exc_msg = ea.get("exception.message") or exc_msg
    feedback = a.get("traceex.check.feedback") or exc_msg or a.get("gen_ai.tool.call.result") or \
        getattr(getattr(span, "status", None), "description", None) or ""
    if "traceex.check.passed" in a:
        return bool(a["traceex.check.passed"]), str(feedback), exc_type
    ev = _evaluation(span)
    if ev:
        label, score, expl, _ = ev
        feedback = expl or feedback
        if label is not None and str(label).strip().lower() in PASS | FAIL:
            return str(label).strip().lower() in PASS, str(feedback), exc_type
        if score is not None:
            return float(score) >= 0.5, str(feedback), exc_type
    failed = _status_error(span) or bool(exc_type)
    return not failed, str(feedback), exc_type


def failure_mode(exc_type):
    """A checker that raises is saying no: AssertionError and ValueError (or no exception) mean a wrong answer; a parse
    or schema error, an answer in the wrong format; anything else, that the answer broke something when it ran."""
    name = str(exc_type or "").rsplit(".", 1)[-1]
    if not name or name in ("AssertionError", "ValueError"):
        return "wrong_answer"
    if name in ("NameError", "AttributeError", "ImportError", "ModuleNotFoundError", "KeyError"):
        return "wrong_name"
    if name in ("SyntaxError", "IndentationError"):
        return "syntax_error"
    if name in ("JSONDecodeError", "ValidationError", "SchemaError"):
        return "invalid_format"
    if "Timeout" in name:
        return "timeout"
    return "runtime_error"


def _checker_name(span):
    a = _attrs(span)
    ev = _evaluation(span)
    name = a.get("gen_ai.tool.name") or (ev[3] if ev else None) or getattr(span, "name", "") or "check"
    return re.sub(r"[^a-z0-9_.-]+", "-", str(name).lower()).strip("-")[:60] or "check"


def find_fixes(spans):
    """The pattern, in one trace's spans: [(failed model span, failed check, retry model span, passing check)]."""
    spans = sorted(spans, key=lambda s: (getattr(s, "start_time", 0) or 0, getattr(s, "end_time", 0) or 0))
    by_id = {_sid(s)[0]: s for s in spans}
    llms = [s for s in spans if is_llm(s)]

    def subject(check):
        """The model call a check judged: its parent if that is a model call, else the last call that ended before it."""
        p = by_id.get(_parent(check))
        if p is not None and is_llm(p):
            return p
        start = getattr(check, "start_time", 0) or 0
        before = [s for s in llms if (getattr(s, "end_time", 0) or 0) <= start]
        return before[-1] if before else None
    judged = []                                    # (model span, passed, feedback, exc, check span)
    for c in spans:
        if is_check(c):
            m = subject(c)
            if m is not None:
                passed, fb, exc = verdict(c)
                judged.append((m, passed, fb, exc, c))
    out, used = [], set()
    for i, (m1, ok1, fb, exc, c1) in enumerate(judged):
        if ok1 or id(m1) in used:
            continue
        for m2, ok2, _, _, c2 in judged[i + 1:]:
            if m2 is m1:
                continue
            if ok2:
                out.append((m1, c1, m2, c2, fb, exc))
                used |= {id(m1), id(m2)}
            break                                  # only the very next attempt counts as the retry
    return out


class TraceexSpanExporter(_Base):
    """A SpanExporter (OpenTelemetry SDK) that writes a skeleton trace for every failed-then-fixed model answer.

    outbox: where traces wait for review (default .traceex/outbox). submit=True sends them instead (needs `client`, or
    `node` and `producer`). privacy: "skeleton" (default: everything personal replaced on this machine) or "open" (code
    and other non-personal text kept, still scanned for secrets, emails and phone numbers). max_input: characters of the
    prompt kept before skeletonizing. Spans are held per trace until its root span ends (or force_flush)."""

    def __init__(self, outbox=None, submit=False, client=None, node=None, producer=None, privacy="skeleton",
                 max_input=4000, task=None, max_buffered=10_000):
        self.outbox = outbox or os.path.join(".traceex", "outbox")
        self.submit, self.privacy, self.max_input, self.task = submit, privacy, int(max_input), task
        self.producer = producer or (client.address if client else None) or os.environ.get("TRACEX_ADDRESS") or NO_ADDRESS
        self.client = client or (Client(node, self.producer) if submit and node else None)
        self.buffer, self.max_buffered, self.lock = {}, int(max_buffered), threading.Lock()
        self.written, self.sent, self.refused, self.errors = [], [], 0, []

    # --- SpanExporter --------------------------------------------------------------------------------------------------
    def export(self, spans):
        try:
            done = []
            with self.lock:
                for s in spans:
                    sid, tid = _sid(s)
                    self.buffer.setdefault(tid, []).append(s)
                    if _parent(s) is None:              # the root ended: the whole trace is here
                        done.append(tid)
                while sum(len(v) for v in self.buffer.values()) > self.max_buffered and self.buffer:
                    done.append(next(iter(self.buffer)))
                batches = [self.buffer.pop(t) for t in dict.fromkeys(done) if t in self.buffer]
            for b in batches:
                self.process(b)
            return _OK
        except Exception as e:                          # an exporter must never break the app it observes
            self.errors.append(f"{type(e).__name__}: {e}")
            return _FAIL

    def force_flush(self, timeout_millis=30_000):
        with self.lock:
            batches, self.buffer = list(self.buffer.values()), {}
        for b in batches:
            self.process(b)
        return True

    def shutdown(self):
        self.force_flush()

    # --- spans -> traces ------------------------------------------------------------------------------------------------
    def build(self, m1, c1, m2, c2, feedback, exc, agent=None):
        a1 = _attrs(m1)
        bad, good = output_of(m1), output_of(m2)
        if not bad or not good or bad.strip() == good.strip():
            return None
        prompt = prompt_of(m1)[:self.max_input]
        model = a1.get("gen_ai.response.model") or a1.get("gen_ai.request.model") or "unknown"
        agent = a1.get("gen_ai.agent.name") or _attrs(c1).get("gen_ai.agent.name") or agent
        task = self.task or a1.get("traceex.task") or (f"agent.{re.sub(r'[^a-z0-9_]+', '_', str(agent).lower())}"
                                                       if agent else "agent.output")
        return Trace.from_fix(task=str(task)[:80], base_model=str(model)[:120], input=prompt or "(no prompt recorded)",
                              model_output={"output": bad}, verified_output={"output": good},
                              checker=f"{_checker_name(c1)}@1", producer=self.producer, privacy=self.privacy,
                              feedback=[str(feedback)[:1000]] if feedback else None,
                              failure_modes={"output": failure_mode(exc)}, fixed_by={"output": "model"},
                              created=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def process(self, spans):
        out = []
        agent = next((_attrs(s)["gen_ai.agent.name"] for s in spans if _attrs(s).get("gen_ai.agent.name")), None)
        for m1, c1, m2, c2, fb, exc in find_fixes(spans):
            t = self.build(m1, c1, m2, c2, fb, exc, agent)
            if t is None:
                continue
            if privacy_leaks(t):
                self.refused += 1
                continue
            if self.submit and self.client:
                try:
                    self.sent.append(self.client.submit(t))
                    out.append(t)
                    continue
                except Exception as e:                  # the node unreachable: the trace waits in the outbox
                    self.errors.append(f"{type(e).__name__}: {str(e)[:80]}")
            self.written.append(_outbox.write(t, self.outbox, {"source": "opentelemetry", "note": "local only"}))
            out.append(t)
        return out
