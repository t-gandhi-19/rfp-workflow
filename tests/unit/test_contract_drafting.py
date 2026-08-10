"""The drafting invariants — the ones that keep an ungrounded answer off the page.

Each validator is tested from both sides: the shape it must accept, and the
shape it must refuse. A validator only tested on its happy path is not a
guarantee.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts import CritiqueResult, DraftedAnswer, confidence_escalation_threshold


def _answer(**overrides: object) -> DraftedAnswer:
    base: dict[str, object] = {
        "question_id": "q-001",
        "answer_text": "We run a six-week assessment before any workload moves.",
        "source_ids": ["ans-014"],
        "confidence": 0.82,
        "needs_sme_review": False,
    }
    base.update(overrides)
    return DraftedAnswer.model_validate(base)


class TestGroundedAnswersOnly:
    """needs_sme_review=False => cited, and free of unsupported claims."""

    def test_accepts_a_cited_answer(self) -> None:
        answer = _answer()
        assert answer.needs_sme_review is False
        assert answer.source_ids == ["ans-014"]

    def test_rejects_an_uncited_unescalated_answer(self) -> None:
        with pytest.raises(ValidationError, match="at least one source id"):
            _answer(source_ids=[])

    def test_rejects_unsupported_claims_on_an_unescalated_answer(self) -> None:
        with pytest.raises(ValidationError, match="unsupported"):
            _answer(unsupported_claims=["We are the market leader."])

    def test_allows_uncited_when_escalated(self) -> None:
        """Escalation is the pressure valve: no source is fine if a human is told."""
        answer = _answer(
            source_ids=[],
            confidence=0.0,
            needs_sme_review=True,
            escalation_reason="NO_MATCH: corpus does not cover mainframe modernization",
        )
        assert answer.source_ids == []

    def test_allows_unsupported_claims_when_escalated(self) -> None:
        answer = _answer(
            confidence=0.3,
            needs_sme_review=True,
            unsupported_claims=["99.99% uptime across all regions"],
            escalation_reason="unsupported numeric claim",
        )
        assert answer.unsupported_claims


class TestLowConfidenceEscalates:
    """Below the configured floor, review is not optional."""

    def test_rejects_low_confidence_without_escalation(self) -> None:
        threshold = confidence_escalation_threshold()
        with pytest.raises(ValidationError, match="below the escalation threshold"):
            _answer(confidence=threshold - 0.01)

    def test_accepts_low_confidence_when_escalated(self) -> None:
        threshold = confidence_escalation_threshold()
        answer = _answer(
            confidence=threshold - 0.01,
            needs_sme_review=True,
            escalation_reason="thin evidence",
        )
        assert answer.needs_sme_review is True

    def test_accepts_confidence_exactly_at_threshold(self) -> None:
        """The floor is inclusive — at the threshold is not below it."""
        answer = _answer(confidence=confidence_escalation_threshold())
        assert answer.needs_sme_review is False

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
    def test_rejects_confidence_outside_unit_interval(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            _answer(confidence=bad)


class TestEscalationsCarryAReason:
    """The assembler writes an SME-TODO block per escalation; it is never blank."""

    def test_rejects_escalation_without_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="escalation_reason"):
            _answer(confidence=0.2, needs_sme_review=True)

    def test_rejects_whitespace_only_reason(self) -> None:
        with pytest.raises(ValidationError, match="escalation_reason"):
            _answer(confidence=0.2, needs_sme_review=True, escalation_reason="   ")


class TestCriticCannotInflate:
    """CLAUDE.md rule 14: the critic lowers confidence or adds flags. Never raises."""

    def test_accepts_a_negative_delta(self) -> None:
        critique = CritiqueResult(question_id="q-001", confidence_delta=-0.15)
        assert critique.confidence_delta == pytest.approx(-0.15)

    def test_accepts_a_zero_delta(self) -> None:
        assert CritiqueResult(question_id="q-001", confidence_delta=0.0).issues == []

    @pytest.mark.parametrize("bad", [0.01, 0.5, 1.0])
    def test_rejects_a_positive_delta(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            CritiqueResult(question_id="q-001", confidence_delta=bad)

    def test_rejects_a_delta_below_negative_one(self) -> None:
        with pytest.raises(ValidationError):
            CritiqueResult(question_id="q-001", confidence_delta=-1.5)


class TestBoundaryHygiene:
    def test_rejects_unknown_fields(self) -> None:
        """extra='forbid' — a typo at a boundary fails loudly, not silently."""
        with pytest.raises(ValidationError):
            _answer(confidenc=0.9)

    def test_rejects_empty_question_id(self) -> None:
        with pytest.raises(ValidationError):
            _answer(question_id="")
