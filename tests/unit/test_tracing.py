"""The §19 span topology, asserted against spans actually emitted.

An in-memory exporter, not a mock: what is under test is the SHAPE of the tree —
which span is whose parent, and which attributes reached which span — and a mock
would assert that functions were called, which is a different claim.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.observability import tracing
from src.observability.tracing import (
    ATTR_PROMPT_VERSION,
    ATTR_QUESTION_ID,
    ATTR_RUN_ID,
    ATTR_SUBJECT,
    attach_eval_score,
    attach_subject,
    current_question_id,
    current_run_id,
    llm_span,
    question_span,
    run_span,
    stage_span,
    tool_span,
)


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    """A real tracer provider, exporting into memory for the test's duration.

    The RAW module global is saved and restored, not `get_tracer_provider()`.
    That call returns the SDK's proxy when nothing is installed, and putting the
    proxy back as the global makes it delegate to itself — a `RecursionError` in
    the next test that does not use this fixture, which is exactly how it was
    found.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = trace._TRACER_PROVIDER
    trace._TRACER_PROVIDER = provider
    try:
        yield exporter
    finally:
        trace._TRACER_PROVIDER = previous


def names(exporter: InMemorySpanExporter) -> list[str]:
    return [span.name for span in exporter.get_finished_spans()]


def by_name(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    return next(span for span in exporter.get_finished_spans() if span.name == name)


def attrs(span: ReadableSpan) -> dict[str, object]:
    """A span's attributes, asserting there are some.

    `ReadableSpan.attributes` is Optional, and every assertion below indexes it.
    Failing here with "span carries no attributes" says what went wrong; a
    TypeError from indexing None does not.
    """
    assert span.attributes is not None, f"span {span.name} carries no attributes"
    return dict(span.attributes)


class TestTheTopology:
    def test_the_run_span_is_the_root(self, spans: InMemorySpanExporter) -> None:
        with run_span(run_id="run-1", rfp_id="rfp-1"):
            pass
        assert by_name(spans, "rfp.run").parent is None

    def test_a_stage_span_is_a_child_of_the_run(self, spans: InMemorySpanExporter) -> None:
        with run_span(run_id="run-1", rfp_id="rfp-1"), stage_span("drafting"):
            pass
        run = by_name(spans, "rfp.run")
        stage = by_name(spans, "rfp.stage.drafting")
        assert stage.parent is not None
        assert stage.parent.span_id == run.context.span_id

    def test_a_question_span_is_a_child_of_the_stage(self, spans: InMemorySpanExporter) -> None:
        with (
            run_span(run_id="run-1", rfp_id="rfp-1"),
            stage_span("drafting"),
            question_span("GQ-1"),
        ):
            pass
        stage = by_name(spans, "rfp.stage.drafting")
        question = by_name(spans, "rfp.question")
        assert question.parent is not None
        assert question.parent.span_id == stage.context.span_id

    def test_the_stage_is_named_not_merely_labelled(self, spans: InMemorySpanExporter) -> None:
        """`rfp.stage.drafting`, so a trace UI groups by stage without needing
        to read an attribute off every span first."""
        with run_span(run_id="run-1", rfp_id="rfp-1"), stage_span("compliance"):
            pass
        assert "rfp.stage.compliance" in names(spans)


class TestTheAmbientIdsReachEverySpan:
    def test_a_deep_span_carries_the_run_id_without_being_told(
        self, spans: InMemorySpanExporter
    ) -> None:
        """The question a trace is read with is "what was this run doing", and a
        gateway span with no run id cannot answer it."""
        with (
            run_span(run_id="run-7", rfp_id="rfp-1"),
            stage_span("drafting"),
            question_span("GQ-9"),
            tool_span("get_evidence"),
        ):
            pass
        tool = attrs(by_name(spans, "rfp.tool.get_evidence"))
        assert tool[ATTR_RUN_ID] == "run-7"
        assert tool[ATTR_QUESTION_ID] == "GQ-9"

    def test_the_contextvars_are_reset_on_exit(self) -> None:
        with run_span(run_id="run-1", rfp_id="rfp-1"):
            assert current_run_id() == "run-1"
            with question_span("GQ-1"):
                assert current_question_id() == "GQ-1"
            assert current_question_id() is None
        assert current_run_id() is None

    def test_a_span_outside_a_run_carries_no_run_id(self, spans: InMemorySpanExporter) -> None:
        """Better an absent attribute than a stale one from the previous run."""
        with tool_span("entity_exists"):
            pass
        assert ATTR_RUN_ID not in attrs(by_name(spans, "rfp.tool.entity_exists"))


class TestThePromptVersionIsOnEveryLlmSpan:
    def test_it_is_recorded(self, spans: InMemorySpanExporter) -> None:
        with llm_span(step="drafter", prompt_version="drafter@v1", model_alias="drafter-model"):
            pass
        assert attrs(by_name(spans, "rfp.llm.drafter"))[ATTR_PROMPT_VERSION] == "drafter@v1"

    def test_it_has_no_default(self) -> None:
        """Rule 17. A span that cannot say which text produced an answer cannot
        explain a quality regression, and a default would let one exist."""
        with pytest.raises(TypeError), llm_span(step="drafter", model_alias="drafter-model"):  # type: ignore[call-arg]
            pass

    def test_the_alias_is_recorded_too(self, spans: InMemorySpanExporter) -> None:
        with llm_span(step="critic", prompt_version="critic@v1", model_alias="critic-model"):
            pass
        assert attrs(by_name(spans, "rfp.llm.critic"))["rfp.model_alias"] == "critic-model"


class TestTheSubjectJoinsTheRowToTheTrace:
    def test_it_lands_on_the_current_span(self, spans: InMemorySpanExporter) -> None:
        with tool_span("save_draft"):
            attach_subject("service-account-drafter-sa")
        subject = attrs(by_name(spans, "rfp.tool.save_draft"))[ATTR_SUBJECT]
        assert subject == "service-account-drafter-sa"

    def test_it_is_a_no_op_outside_a_span(self) -> None:
        """Called from a request that arrived with tracing off; it must not
        raise on the way to a perfectly good 200."""
        attach_subject("nobody")


class TestEvalScoresAttachToTraces:
    def test_a_metric_lands_on_the_span(self, spans: InMemorySpanExporter) -> None:
        with run_span(run_id="run-1", rfp_id="rfp-1"):
            attach_eval_score(metric="grounding.traceability", value=0.97, passed=True)
        attributes = attrs(by_name(spans, "rfp.run"))
        assert attributes["rfp.eval.metric"] == "grounding.traceability"
        assert attributes["rfp.eval.value"] == 0.97
        assert attributes["rfp.eval.passed"] is True


class TestTheThirdState:
    def test_unconfigured_tracing_says_so(self) -> None:
        assert (
            "not configured"
            in tracing.TracingStatus(configured=False, exporting=False, endpoint=None).summary
        )

    def test_recording_without_an_exporter_is_distinct_from_exporting(self) -> None:
        """A real third state: spans are recorded and nothing is sent. The unit
        tests need it, and a laptop with no collector should not fail."""
        summary = tracing.TracingStatus(configured=True, exporting=False, endpoint=None).summary
        assert "no exporter" in summary

    def test_exporting_names_the_endpoint(self) -> None:
        summary = tracing.TracingStatus(
            configured=True, exporting=True, endpoint="http://collector:4318"
        ).summary
        assert "http://collector:4318" in summary

    def test_no_trace_url_when_nothing_was_exported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A link into a collector that never received the trace is worse than
        no link: the reader concludes the run was not traced."""
        monkeypatch.setattr(tracing, "_exporting", False)
        monkeypatch.setenv("TRACE_UI_BASE", "http://langfuse:3000")
        assert tracing.trace_url("run-1") is None

    def test_a_trace_url_when_it_was(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tracing, "_exporting", True)
        monkeypatch.setenv("TRACE_UI_BASE", "http://langfuse:3000")
        url = tracing.trace_url("run-1")
        assert url is not None
        assert "run-1" in url
