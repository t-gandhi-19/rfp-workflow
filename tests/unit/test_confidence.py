"""Computed confidence — hand-worked cases.

Confidence is the number that decides whether a human looks at an answer, so the
arithmetic is asserted against values computed by hand rather than by re-running
the formula under test.
"""

from __future__ import annotations

import pytest

from src.contracts import DraftedAnswer, confidence_escalation_threshold
from src.contracts.thresholds import scoring_config
from src.retrieval.confidence import compute_confidence, coverage_ratio, requires_sme_review

THRESHOLD = 0.6


class TestConfigIsWhatTheseTestsAssume:
    def test_threshold(self) -> None:
        assert scoring_config().confidence.escalation_threshold == THRESHOLD
        assert scoring_config().confidence.clamp_min == 0.0
        assert scoring_config().confidence.clamp_max == 1.0


class TestCoverageRatio:
    def test_fully_sourced(self) -> None:
        assert coverage_ratio(claims_with_sources=4, total_claims=4) == pytest.approx(1.0)

    def test_half_sourced(self) -> None:
        assert coverage_ratio(claims_with_sources=2, total_claims=4) == pytest.approx(0.5)

    def test_nothing_sourced(self) -> None:
        assert coverage_ratio(claims_with_sources=0, total_claims=3) == pytest.approx(0.0)

    def test_no_claims_scores_zero(self) -> None:
        """'Nothing to check' must not read as 'fully checked'."""
        assert coverage_ratio(claims_with_sources=0, total_claims=0) == pytest.approx(0.0)

    def test_more_sourced_than_total_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            coverage_ratio(claims_with_sources=5, total_claims=4)

    def test_negative_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            coverage_ratio(claims_with_sources=-1, total_claims=4)


class TestHandComputedConfidence:
    def test_strong_and_fully_sourced(self) -> None:
        """0.90 * 1.0 + 0.0 = 0.90"""
        assert compute_confidence(
            primary_final_score=0.90, claims_with_sources=5, total_claims=5
        ) == pytest.approx(0.90)

    def test_coverage_halves_it(self) -> None:
        """0.90 * 0.5 = 0.45 — a partly ungrounded answer is not slightly worse."""
        assert compute_confidence(
            primary_final_score=0.90, claims_with_sources=2, total_claims=4
        ) == pytest.approx(0.45)

    def test_a_critique_subtracts(self) -> None:
        """0.80 * 1.0 - 0.15 = 0.65"""
        assert compute_confidence(
            primary_final_score=0.80,
            claims_with_sources=3,
            total_claims=3,
            critique_delta=-0.15,
        ) == pytest.approx(0.65)

    def test_coverage_and_critique_together(self) -> None:
        """0.72 * (3/4) - 0.10 = 0.54 - 0.10 = 0.44"""
        assert compute_confidence(
            primary_final_score=0.72,
            claims_with_sources=3,
            total_claims=4,
            critique_delta=-0.10,
        ) == pytest.approx(0.44)

    def test_clamped_at_zero(self) -> None:
        """0.30 * 0.5 - 0.9 = -0.75, clamped to 0."""
        assert compute_confidence(
            primary_final_score=0.30,
            claims_with_sources=1,
            total_claims=2,
            critique_delta=-0.9,
        ) == pytest.approx(0.0)

    def test_clamped_at_one(self) -> None:
        assert compute_confidence(
            primary_final_score=1.0, claims_with_sources=3, total_claims=3
        ) == pytest.approx(1.0)


class TestTheCriticCannotInflate:
    def test_a_positive_delta_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be positive"):
            compute_confidence(
                primary_final_score=0.5,
                claims_with_sources=1,
                total_claims=1,
                critique_delta=0.2,
            )

    def test_a_score_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            compute_confidence(primary_final_score=1.4, claims_with_sources=1, total_claims=1)


class TestEscalationThreshold:
    def test_below_the_floor_requires_review(self) -> None:
        assert requires_sme_review(THRESHOLD - 0.001) is True

    def test_exactly_at_the_floor_does_not(self) -> None:
        assert requires_sme_review(THRESHOLD) is False

    def test_it_agrees_with_the_contract(self) -> None:
        """Both read the same config value, so they cannot drift apart."""
        assert confidence_escalation_threshold() == scoring_config().confidence.escalation_threshold

    def test_a_computed_low_confidence_forces_the_contract_to_escalate(self) -> None:
        """End to end: the arithmetic makes the contract refuse an unescalated answer."""
        low = compute_confidence(primary_final_score=0.80, claims_with_sources=1, total_claims=4)
        assert low == pytest.approx(0.20)
        assert requires_sme_review(low) is True

        with pytest.raises(ValueError, match="below the escalation threshold"):
            DraftedAnswer(
                question_id="q-1",
                answer_text="an answer",
                source_ids=["ans-1"],
                confidence=low,
                needs_sme_review=False,
            )

    def test_a_computed_high_confidence_is_accepted_unescalated(self) -> None:
        high = compute_confidence(primary_final_score=0.88, claims_with_sources=4, total_claims=4)
        answer = DraftedAnswer(
            question_id="q-1",
            answer_text="an answer",
            source_ids=["ans-1"],
            confidence=high,
            needs_sme_review=False,
        )
        assert answer.confidence == pytest.approx(0.88)
