"""Run-level invariants: halt consistency, clock sanity, totals arithmetic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.contracts import (
    ArtifactPaths,
    EvalScore,
    HaltReason,
    QuestionStatus,
    RunStage,
    RunState,
    RunTotals,
)

START = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
LATER = START + timedelta(minutes=7)


def _state(**overrides: object) -> RunState:
    base: dict[str, object] = {
        "run_id": "run-001",
        "rfp_id": "rfp-001",
        "stage": RunStage.DRAFTING,
        "started_at": START,
        "updated_at": LATER,
    }
    base.update(overrides)
    return RunState.model_validate(base)


class TestHaltConsistency:
    """A halted run always says why; a live run never claims a reason."""

    def test_accepts_a_running_stage_without_a_reason(self) -> None:
        assert _state().halted_reason is None

    def test_rejects_halted_without_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="requires halted_reason"):
            _state(stage=RunStage.HALTED)

    @pytest.mark.parametrize(
        "reason", [HaltReason.DOMAIN_MISMATCH, HaltReason.BUDGET, HaltReason.INFRA]
    )
    def test_accepts_each_permitted_halt_reason(self, reason: HaltReason) -> None:
        state = _state(stage=RunStage.HALTED, halted_reason=reason)
        assert state.halted_reason is reason

    def test_rejects_a_reason_on_a_live_run(self) -> None:
        with pytest.raises(ValidationError, match="not HALTED"):
            _state(stage=RunStage.DRAFTING, halted_reason=HaltReason.BUDGET)


class TestClock:
    def test_rejects_updated_before_started(self) -> None:
        with pytest.raises(ValidationError, match="precedes"):
            _state(updated_at=START - timedelta(seconds=1))

    def test_accepts_equal_timestamps(self) -> None:
        """A freshly created run has not been updated yet."""
        assert _state(stage=RunStage.CREATED, updated_at=START).updated_at == START


class TestPerQuestionStatus:
    def test_carries_a_typed_status_map(self) -> None:
        state = _state(
            per_question_status={
                "q-001": QuestionStatus.COMPLETE,
                "q-002": QuestionStatus.ESCALATED,
                "q-003": QuestionStatus.PENDING,
            }
        )
        assert state.per_question_status["q-002"] is QuestionStatus.ESCALATED

    def test_rejects_an_unknown_status_value(self) -> None:
        with pytest.raises(ValidationError):
            _state(per_question_status={"q-001": "almost-done"})

    def test_rejects_negative_counters(self) -> None:
        with pytest.raises(ValidationError):
            _state(tokens_used=-1)
        with pytest.raises(ValidationError):
            _state(cost_usd=-0.01)


class TestRunTotals:
    def test_accepts_consistent_totals(self) -> None:
        totals = RunTotals(
            questions=20, answered=14, escalated=5, failed=1, tokens_used=120_000, cost_usd=0.42
        )
        assert totals.answered + totals.escalated + totals.failed == totals.questions

    def test_accepts_partial_totals_for_a_halted_run(self) -> None:
        """A budget halt leaves questions unaccounted for; that is legitimate."""
        totals = RunTotals(
            questions=20, answered=6, escalated=2, failed=0, tokens_used=90_000, cost_usd=2.0
        )
        assert totals.answered + totals.escalated + totals.failed < totals.questions

    def test_rejects_more_outcomes_than_questions(self) -> None:
        with pytest.raises(ValidationError, match="exceeds"):
            RunTotals(questions=10, answered=8, escalated=4, failed=0, tokens_used=1, cost_usd=0.1)


class TestEvalScore:
    def test_carries_direction_with_the_metric(self) -> None:
        """`passed` is supplied, since some metrics pass by staying low."""
        recall = EvalScore(
            git_sha="abc123",
            run_id="run-001",
            metric="retrieval.recall_at_5",
            value=0.86,
            threshold=0.8,
            passed=True,
        )
        violations = EvalScore(
            git_sha="abc123",
            run_id="run-001",
            metric="compliance.word_limit_violations",
            value=0.0,
            threshold=0.0,
            passed=True,
        )
        assert recall.passed and violations.passed

    def test_rejects_an_empty_metric_name(self) -> None:
        with pytest.raises(ValidationError):
            EvalScore(
                git_sha="abc123",
                run_id="run-001",
                metric="",
                value=1.0,
                threshold=0.5,
                passed=True,
            )


class TestArtifactPaths:
    def test_defaults_to_nothing_written_yet(self) -> None:
        paths = ArtifactPaths()
        assert (paths.response_docx, paths.escalations_json, paths.eval_report_html) == (
            None,
            None,
            None,
        )
