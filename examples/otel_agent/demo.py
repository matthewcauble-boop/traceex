"""Ingestion path B, runnable: a fake agent instrumented with OpenTelemetry GenAI spans, and traceX's exporter turning
its "answer, checker fails it, retry, checker passes" into a skeleton trace. No network, no model, no key.

    pip install opentelemetry-sdk          # the exporter also works without it; this demo uses the real SDK
    python examples/otel_agent/demo.py      # writes to a temporary outbox and prints what would be sent

The agent extracts the departure date from a booking email. Its first answer copies the wrong date (the booking date,
in the email's own format); a date-rules tool span fails it with the reason; the agent retries with that feedback and
the tool passes the answer. Every span uses the GenAI semantic conventions any OTel-instrumented agent emits (and that
Laminar, LangSmith, OpenLLMetry or OpenInference export), so the same exporter works on a real agent: add
SimpleSpanProcessor(TraceexSpanExporter()) to its TracerProvider.
"""
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sdk", "python"))
from traceex.otel import TraceexSpanExporter  # noqa: E402
from traceex import outbox  # noqa: E402

EMAIL = """Hi Jordan Parker,
Thanks for booking with Skyward on 3/02/2026 (confirmation QX7PLM, card ending 4417).
Your flight SK 1123 departs Austin (AUS) on Thursday, October 15, 2026 at 8:05 AM.
Questions? jordan.parker@example.com or (512) 555-0142"""
MODEL = "qwen2.5-0.5b-instruct"


def fake_model(prompt, feedback=None):
    """A deterministic stand-in for a model: wrong (the booking date) the first time, right once told why."""
    if feedback:
        return "2026-10-15"
    return "3/02/2026"


def date_rules(answer, email):
    """The checker: an ISO date that is the departure date in the email."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", answer):
        raise ValueError(f"'{answer}' is not an ISO date (YYYY-MM-DD); the departure date follows 'departs'")
    if answer != "2026-10-15" or "October 15, 2026" not in email:
        raise ValueError("that date is not the departure date")
    return True


def run_agent(tracer):
    from opentelemetry.trace import Status, StatusCode
    with tracer.start_as_current_span("invoke_agent flight-extractor") as root:
        root.set_attribute("gen_ai.operation.name", "invoke_agent")
        root.set_attribute("gen_ai.agent.name", "flight-extractor")
        feedback = None
        for attempt in range(2):
            prompt = f"Extract the departure date (YYYY-MM-DD) from this email:\n{EMAIL}" + \
                     (f"\nYour last answer was rejected: {feedback}" if feedback else "")
            with tracer.start_as_current_span(f"chat {MODEL}") as llm:
                llm.set_attribute("gen_ai.operation.name", "chat")
                llm.set_attribute("gen_ai.request.model", MODEL)
                llm.set_attribute("gen_ai.agent.name", "flight-extractor")
                llm.set_attribute("gen_ai.input.messages", json.dumps(
                    [{"role": "system", "parts": [{"type": "text", "content": "You extract travel data."}]},
                     {"role": "user", "parts": [{"type": "text", "content": prompt}]}]))
                answer = fake_model(prompt, feedback)
                llm.set_attribute("gen_ai.output.messages", json.dumps(
                    [{"role": "assistant", "parts": [{"type": "text", "content": answer}]}]))
            with tracer.start_as_current_span("execute_tool date_rules") as tool:
                tool.set_attribute("gen_ai.operation.name", "execute_tool")
                tool.set_attribute("gen_ai.tool.name", "date_rules")
                try:
                    date_rules(answer, EMAIL)
                    tool.set_status(Status(StatusCode.OK))
                    return answer
                except ValueError as e:
                    tool.record_exception(e)
                    tool.set_status(Status(StatusCode.ERROR, str(e)))
                    feedback = str(e)
    return None


def main(argv=None):
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except ImportError:
        print("this demo needs the OpenTelemetry SDK: pip install opentelemetry-sdk")
        return 1
    box = tempfile.mkdtemp(prefix="traceex-otel-")
    exporter = TraceexSpanExporter(outbox=box)                    # dry run: nothing is sent
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    answer = run_agent(provider.get_tracer("demo-agent"))
    provider.shutdown()
    print(f"agent's final answer: {answer}")
    items = outbox.items(box)
    print(f"{len(items)} trace written to {box} (dry run: nothing sent)\n")
    for _, item in items:
        t = item["trace"]
        print(json.dumps({k: t[k] for k in ("task", "base_model", "input", "model_output", "verified_output",
                                            "feedback", "failure_modes", "checker", "privacy")}, indent=1))
        sent = json.dumps(t)
        leaked = [v for v in ("Jordan", "Parker", "QX7PLM", "4417", "jordan.parker@example.com", "555-0142", "Austin",
                              "October 15", "2026-10-15", "3/02/2026") if v in sent]
        print("\npersonal values in what would be sent:", leaked or "none")
    shutil.rmtree(box, ignore_errors=True)                        # a demo: leave nothing behind
    return 0 if items else 1


if __name__ == "__main__":
    sys.exit(main())
