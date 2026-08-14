"""The HTML report — a record, not a dashboard.

The assertions are about what the report must not lose: the threshold each
number was judged against, the verdict reached at the time, which SHA it was
compared to, and the fact that a stand-in-embedding run says nothing about
retrieval quality. An HTML file outlives the terminal it was produced in, so
anything stated only on the console is effectively unstated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evals.contracts import (
    CategoryResult,
    CategoryStatus,
    EvalMetric,
    MetricDirection,
    Violation,
)
from src.evals.registry import all_placeholders
from src.evals.report import ReportContext, render, write
from src.evals.store import Delta


def context(**overrides: object) -> ReportContext:
    base: dict[str, object] = {
        "git_sha": "abc123def456",
        "run_id": "evals-testrun0001",
        "rerank_enabled": True,
        "embedder": "embed-model",
        "fake_embeddings": False,
        "previous_sha": None,
        "deltas": [],
        "rows_written": 7,
    }
    base.update(overrides)
    return ReportContext(**base)  # type: ignore[arg-type]


def passing_category() -> CategoryResult:
    return CategoryResult(
        key="extraction",
        label="Extraction — against the manual key",
        status=CategoryStatus.PASS,
        metrics=[
            EvalMetric(
                key="extraction_recall",
                label="Recall against the manual key",
                value=1.0,
                direction=MetricDirection.HIGHER_IS_BETTER,
                threshold=0.95,
                passed=True,
                detail="20/20 found",
            ),
            EvalMetric(
                key="mrr",
                label="Mean reciprocal rank",
                value=0.9333,
                direction=MetricDirection.HIGHER_IS_BETTER,
            ),
            EvalMetric(
                key="pdf_docx_parity",
                label="PDF/DOCX differences",
                value=0.0,
                direction=MetricDirection.MUST_BE_ZERO,
                threshold=0.0,
                passed=True,
            ),
        ],
    )


class TestItIsSelfContained:
    """A report that fetches a stylesheet renders wrong in six months."""

    def test_there_are_no_external_requests(self) -> None:
        html = render([passing_category(), *all_placeholders()], context())
        for marker in ("http://", "https://", "<script", "src=", "@import"):
            assert marker not in html, marker

    def test_the_css_is_inline(self) -> None:
        assert "<style>" in render([passing_category(), *all_placeholders()], context())


class TestEveryNumberCarriesItsJudgement:
    def test_a_gated_metric_shows_its_threshold(self) -> None:
        """Recomputing a verdict at render time would let a threshold that moved
        after the fact silently rewrite history."""
        html = render([passing_category(), *all_placeholders()], context())
        assert "0.95" in html

    def test_a_gated_metric_shows_its_verdict(self) -> None:
        assert "PASS" in render([passing_category(), *all_placeholders()], context())

    def test_an_ungated_metric_is_marked_reported_not_passed(self) -> None:
        """MRR describes the shape of a result; nothing should fail on it, and
        the report must not imply otherwise."""
        assert "reported" in render([passing_category(), *all_placeholders()], context())

    def test_a_must_be_zero_metric_renders_as_a_count(self) -> None:
        """'0.0000 violations' reads as a rate that happens to be zero."""
        html = render([passing_category(), *all_placeholders()], context())
        assert ">0<" in html.replace(" ", "")


class TestUnimplementedCategories:
    def test_every_remaining_placeholder_is_named(self) -> None:
        """Derived from the registry rather than hand-listed.

        This test used to name four categories. Phase 4 implemented three of
        them, and a hand-written list would have gone on asserting that
        "Grounding" appears as a PLACEHOLDER long after it had become a
        measurement — passing for the wrong reason, since the word appears in
        the report either way.
        """
        html = render([passing_category(), *all_placeholders()], context())
        for result in all_placeholders():
            assert result.label.split(" —")[0] in html

    def test_they_are_marked_not_implemented(self) -> None:
        html = render([passing_category(), *all_placeholders()], context())
        assert "NOT IMPLEMENTED" in html

    def test_they_are_greyed(self) -> None:
        html = render([passing_category(), *all_placeholders()], context())
        assert 'class="greyed"' in html

    def test_each_states_the_input_it_waits_on(self) -> None:
        html = render([passing_category(), *all_placeholders()], context())
        assert "judge-model" in html
        assert "Phase 4" in html

    def test_they_do_not_change_the_overall_verdict(self) -> None:
        """A greyed category is not a failing one."""
        html = render([passing_category(), *all_placeholders()], context())
        assert '<h1>Eval report <span class="badge pass">PASS</span></h1>' in html


class TestTheOverallVerdict:
    def test_one_failing_category_fails_the_report(self) -> None:
        failing = passing_category().model_copy(update={"status": CategoryStatus.FAIL})
        html = render([failing, *all_placeholders()], context())
        assert '<h1>Eval report <span class="badge fail">FAIL</span></h1>' in html


class TestProvenanceIsOnTheReport:
    def test_the_commit_is_stated(self) -> None:
        assert "abc123def456" in render([passing_category(), *all_placeholders()], context())

    def test_the_run_id_is_stated(self) -> None:
        html = render([passing_category(), *all_placeholders()], context())
        assert "evals-testrun0001" in html

    def test_a_missing_baseline_says_so_rather_than_showing_nothing(self) -> None:
        html = render([passing_category(), *all_placeholders()], context())
        assert "no earlier SHA" in html

    def test_a_baseline_names_the_sha_it_compared_against(self) -> None:
        """The table has no notion of git ancestry, so the reader needs the SHA
        to judge whether the comparison means anything."""
        html = render(
            [passing_category(), *all_placeholders()],
            context(previous_sha="deadbeef1234", deltas=[]),
        )
        assert "deadbeef1234" in html

    def test_rerank_off_is_labelled_as_an_ablation_arm(self) -> None:
        html = render([passing_category(), *all_placeholders()], context(rerank_enabled=False))
        assert "not the shipped configuration" in html

    def test_skipped_persistence_is_stated(self) -> None:
        html = render([passing_category(), *all_placeholders()], context(rows_written=None))
        assert "skipped" in html


class TestTheStandInEmbeddingWarning:
    """The console warning does not survive into the artifact people keep."""

    def test_a_stand_in_run_is_marked_on_the_report_itself(self) -> None:
        html = render([passing_category(), *all_placeholders()], context(fake_embeddings=True))
        assert "STAND-IN" in html
        assert "NOTHING about retrieval quality" in html

    def test_a_real_run_names_the_embedder_instead(self) -> None:
        html = render([passing_category(), *all_placeholders()], context())
        assert "STAND-IN" not in html
        assert "embed-model" in html


class TestDeltas:
    def test_an_improvement_is_marked_up(self) -> None:
        html = render(
            [passing_category(), *all_placeholders()],
            context(
                previous_sha="old",
                deltas=[Delta(metric="extraction.extraction_recall", previous=0.9, current=1.0)],
            ),
        )
        assert 'class="up"' in html
        assert "+0.1000" in html

    def test_a_regression_is_marked_down(self) -> None:
        html = render(
            [passing_category(), *all_placeholders()],
            context(
                previous_sha="old",
                deltas=[Delta(metric="extraction.extraction_recall", previous=1.0, current=0.9)],
            ),
        )
        assert 'class="down"' in html

    def test_direction_decides_the_colour_not_the_sign(self) -> None:
        """A MUST_BE_ZERO count going UP is a regression, not an improvement."""
        html = render(
            [passing_category(), *all_placeholders()],
            context(
                previous_sha="old",
                deltas=[Delta(metric="extraction.pdf_docx_parity", previous=0.0, current=3.0)],
            ),
        )
        assert 'class="down">+3.0000' in html

    def test_a_metric_with_no_baseline_shows_a_dash(self) -> None:
        """An implied zero would render every new metric as a large improvement."""
        html = render([passing_category(), *all_placeholders()], context())
        assert "—" in html


class TestViolationsAreRendered:
    def test_a_violation_is_shown_with_its_rule_and_detail(self) -> None:
        category = passing_category().model_copy(
            update={
                "status": CategoryStatus.FAIL,
                "violations": [
                    Violation(
                        rule="extraction_recall",
                        question_number="2.3",
                        detail="in the key, not extracted",
                    )
                ],
            }
        )
        html = render([category, *all_placeholders()], context())
        assert "extraction_recall" in html
        assert "2.3" in html
        assert "in the key, not extracted" in html


class TestEscaping:
    def test_detail_text_is_escaped(self) -> None:
        """Violation details quote document content, which is untrusted."""
        category = passing_category().model_copy(
            update={
                "status": CategoryStatus.FAIL,
                "violations": [
                    Violation(rule="r", detail="<script>alert(1)</script>"),
                ],
            }
        )
        html = render([category, *all_placeholders()], context())
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html


class TestWriting:
    def test_it_creates_the_output_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "evals.html"
        written = write([passing_category(), *all_placeholders()], context(), target)
        assert written.is_file()

    def test_it_writes_utf8_with_unix_newlines(self, tmp_path: Path) -> None:
        """Δ is in the header row; a cp1252 default would fail on Windows."""
        target = tmp_path / "evals.html"
        write([passing_category(), *all_placeholders()], context(), target)
        raw = target.read_bytes()
        assert "Δ".encode() in raw
        assert b"\r\n" not in raw

    def test_an_unregistered_category_is_refused_at_render(self) -> None:
        rogue = CategoryResult(key="vibes", label="v", status=CategoryStatus.PASS)
        with pytest.raises(ValueError, match="not in the registry"):
            render([rogue], context())
