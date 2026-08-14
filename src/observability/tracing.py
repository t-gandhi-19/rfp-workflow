"""The §19 span topology: rfp.run -> rfp.stage.* -> rfp.question.

WHY CONTEXTVARS AND NOT A PARAMETER. The run id and the question id have to
appear on spans emitted three layers below the controller — inside an agent,
inside a tool, inside the gateway client — and threading them through every
signature would put observability plumbing in the contract of every function it
passes. `contextvars` also survives the per-question fan-out correctly: each
`asyncio` task inherits a COPY of the context at creation, so forty concurrent
questions each carry their own id without any of them seeing another's.

WHY THE EXPORTER IS OPTIONAL AND SAYS SO. With no endpoint configured, this
installs a real tracer that records spans and exports nothing. That is a
deliberate third state, distinct both from "tracing is on" and from "tracing is
broken": the unit tests need spans to inspect without a collector, and a run on
a laptop with no Langfuse should not fail. `tracing_status()` reports which
state is in force, and the run report prints it — a trace url nobody can open
is worse than an honest "not exported".

THE JOIN IS THE TRACEPARENT. Our spans and LiteLLM's are two instrumentations of
one call; they become one trace because httpx is instrumented and injects
`traceparent` into every outbound request — to the gateway, to write-api, to
mcp-server. Nothing correlates by timestamp or by run id after the fact.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

#: Span names, §19. Namespaced so a trace UI groups them and so a query for
#: "every question span" is one prefix rather than a list of names.
RUN_SPAN = "rfp.run"
STAGE_SPAN = "rfp.stage"
QUESTION_SPAN = "rfp.question"
LLM_SPAN = "rfp.llm"
TOOL_SPAN = "rfp.tool"

#: Attribute keys. Constants because they are asserted in tests and read in a
#: trace UI, and a typo in either place is invisible.
ATTR_RUN_ID = "rfp.run_id"
ATTR_RFP_ID = "rfp.rfp_id"
ATTR_QUESTION_ID = "rfp.question_id"
ATTR_STAGE = "rfp.stage"
ATTR_PROMPT_VERSION = "rfp.prompt_version"
ATTR_MODEL_ALIAS = "rfp.model_alias"
ATTR_SUBJECT = "rfp.jwt_subject"
ATTR_TOOL = "rfp.tool"
ATTR_TOKENS = "rfp.tokens"
ATTR_EVAL_METRIC = "rfp.eval.metric"
ATTR_EVAL_VALUE = "rfp.eval.value"
ATTR_EVAL_PASSED = "rfp.eval.passed"

_run_id: ContextVar[str | None] = ContextVar("rfp_run_id", default=None)
_question_id: ContextVar[str | None] = ContextVar("rfp_question_id", default=None)

_configured = False
_exporting = False


@dataclass(frozen=True)
class TracingStatus:
    """Which of the three states tracing is in."""

    configured: bool
    exporting: bool
    endpoint: str | None

    @property
    def summary(self) -> str:
        if not self.configured:
            return "tracing not configured"
        if not self.exporting:
            return "tracing on, no exporter (set OTEL_EXPORTER_OTLP_ENDPOINT to export)"
        return f"tracing on, exporting to {self.endpoint}"


def otlp_endpoint() -> str | None:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    return endpoint or None


def configure_tracing(service_name: str = "rfp-workflow") -> TracingStatus:
    """Install the tracer provider. Idempotent.

    Called once per process — by the run script, and by each service's lifespan.
    Calling it twice would stack exporters and double every span, so the second
    call is a no-op that returns the state the first one established.
    """
    global _configured, _exporting
    if _configured:
        return tracing_status()

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": service_name,
                "service.version": os.environ.get("GIT_SHA", "local-dev"),
            }
        )
    )
    endpoint = otlp_endpoint()
    if endpoint:
        # Imported here rather than at module scope: the exporter pulls in the
        # protobuf stack, and a process that exports nothing should not pay for
        # it or fail on it.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        _exporting = True

    trace.set_tracer_provider(provider)
    _configured = True
    return tracing_status()


def tracing_status() -> TracingStatus:
    return TracingStatus(configured=_configured, exporting=_exporting, endpoint=otlp_endpoint())


def tracer() -> trace.Tracer:
    return trace.get_tracer("rfp.workflow")


def current_run_id() -> str | None:
    return _run_id.get()


def current_question_id() -> str | None:
    return _question_id.get()


def _stamp(span: Span) -> None:
    """Put the ambient ids on a span.

    Every span carries them, not only the ones that opened them, because the
    question a trace is read with is "what was this run doing" — and a gateway
    span with no run id cannot answer it.
    """
    if (run_id := _run_id.get()) is not None:
        span.set_attribute(ATTR_RUN_ID, run_id)
    if (question_id := _question_id.get()) is not None:
        span.set_attribute(ATTR_QUESTION_ID, question_id)


@contextlib.contextmanager
def run_span(*, run_id: str, rfp_id: str) -> Iterator[Span]:
    """The root span. Everything else in the run descends from it."""
    token = _run_id.set(run_id)
    try:
        with tracer().start_as_current_span(RUN_SPAN) as span:
            span.set_attribute(ATTR_RUN_ID, run_id)
            span.set_attribute(ATTR_RFP_ID, rfp_id)
            yield span
    finally:
        _run_id.reset(token)


@contextlib.contextmanager
def stage_span(stage: str) -> Iterator[Span]:
    """One controller stage. Named `rfp.stage.drafting`, not `rfp.stage`."""
    with tracer().start_as_current_span(f"{STAGE_SPAN}.{stage}") as span:
        span.set_attribute(ATTR_STAGE, stage)
        _stamp(span)
        yield span


@contextlib.contextmanager
def question_span(question_id: str) -> Iterator[Span]:
    """One question's whole pipeline.

    Sets the contextvar so every span below — the rerank, the draft, the
    critique, the tool calls — carries the question id without being told.
    """
    token = _question_id.set(question_id)
    try:
        with tracer().start_as_current_span(QUESTION_SPAN) as span:
            span.set_attribute(ATTR_QUESTION_ID, question_id)
            _stamp(span)
            yield span
    finally:
        _question_id.reset(token)


@contextlib.contextmanager
def llm_span(*, step: str, prompt_version: str, model_alias: str) -> Iterator[Span]:
    """One model call.

    `prompt_version` is REQUIRED, with no default (rule 17): a span that cannot
    say which text produced an answer is a span that cannot explain a quality
    regression, and a default would let one exist.
    """
    with tracer().start_as_current_span(f"{LLM_SPAN}.{step}") as span:
        span.set_attribute(ATTR_PROMPT_VERSION, prompt_version)
        span.set_attribute(ATTR_MODEL_ALIAS, model_alias)
        _stamp(span)
        yield span


@contextlib.contextmanager
def tool_span(tool: str) -> Iterator[Span]:
    with tracer().start_as_current_span(f"{TOOL_SPAN}.{tool}") as span:
        span.set_attribute(ATTR_TOOL, tool)
        _stamp(span)
        yield span


def attach_subject(subject: str) -> None:
    """Put the JWT subject on the current span.

    This is what joins a Postgres row's `written_by` to the trace that produced
    it: the row records the subject, and so does the span, and the traceparent
    the caller sent is what puts them in the same trace.
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(ATTR_SUBJECT, subject)


def attach_eval_score(*, metric: str, value: float, passed: bool) -> None:
    """Record one eval metric on the current span (§19: scores attach to traces)."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(ATTR_EVAL_METRIC, metric)
        span.set_attribute(ATTR_EVAL_VALUE, value)
        span.set_attribute(ATTR_EVAL_PASSED, passed)


def record_failure(exc: BaseException) -> None:
    """Mark the current span failed, with the exception's type and message."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
        span.record_exception(exc)


def instrument_httpx() -> None:
    """Inject `traceparent` into every outbound HTTP call.

    THIS IS THE JOIN. Our spans and LiteLLM's proxy-side spans are two
    instrumentations of the same call; they land in one trace because the
    request carries a traceparent, not because anything correlates them
    afterwards by run id or timestamp. It also carries the trace into write-api
    and mcp-server, which is why their spans are children rather than roots.
    """
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument()


def instrument_fastapi(app: Any) -> None:
    """Continue an inbound traceparent into a service's own spans."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)


def trace_url(run_id: str) -> str | None:
    """Where a human can look at this run, or None if nothing was exported.

    None rather than a plausible-looking URL: a link into a collector that never
    received the trace is worse than no link, because the reader concludes the
    run was not traced rather than that it was not exported.
    """
    base = os.environ.get("TRACE_UI_BASE", "").strip()
    if not base or not _exporting:
        return None
    return f"{base.rstrip('/')}/traces?q={run_id}"
