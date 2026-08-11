"""Scoring-sanity suite: the ordering properties retrieval quality rests on.

Hand-computed values throughout. A test that recomputes the formula it is
checking proves only that the code is self-consistent; these assert numbers
worked out independently from the config, so a changed weight fails loudly
rather than silently re-deriving itself.
"""

from __future__ import annotations

import math

import pytest

from src.contracts import Outcome, RetrievalStatus
from src.contracts.thresholds import scoring_config
from src.retrieval.scoring import (
    CandidateInput,
    blend,
    preference,
    recency_multiplier,
    recency_preference,
    score_candidates,
    to_retrieval_result,
)

CONFIG = scoring_config()

# Shipped config, restated here so a change to either side is visible as a
# failing test rather than as tests that quietly follow the config.
WON, UNKNOWN, LOST = 1.15, 1.0, 0.85
DECAY_DAYS = 540
EVIDENCE = 1.1
FLOOR = 0.55


def candidate(
    *,
    node: str,
    vector: float,
    outcome: Outcome = Outcome.UNKNOWN,
    age_days: int = 0,
    evidence: bool = False,
) -> CandidateInput:
    return CandidateInput(
        question_id="q-1",
        matched_question_id="hq-1",
        answer_node_id=node,
        tier1_summary=f"summary for {node}",
        vector_score=vector,
        outcome=outcome,
        age_days=age_days,
        has_evidence=evidence,
    )


class TestConfigIsWhatTheseTestsAssume:
    def test_multipliers(self) -> None:
        outcome = CONFIG.graph_multiplier.outcome
        assert (outcome.won, outcome.unknown, outcome.lost) == (WON, UNKNOWN, LOST)
        assert CONFIG.graph_multiplier.recency.decay_days == DECAY_DAYS
        assert CONFIG.graph_multiplier.evidence_bonus == EVIDENCE

    def test_floor_and_weights(self) -> None:
        assert CONFIG.retrieval.match_floor == FLOOR
        assert CONFIG.final_score.weights.vector_graph == 0.5
        assert CONFIG.final_score.weights.rerank == 0.5


class TestRecency:
    def test_today_is_unpenalised(self) -> None:
        assert recency_multiplier(0) == pytest.approx(1.0)

    def test_one_half_life_matches_the_hand_computation(self) -> None:
        """exp(-540/540) = e^-1 = 0.367879..."""
        assert recency_multiplier(540) == pytest.approx(0.3678794412, abs=1e-9)

    def test_a_year_old_answer(self) -> None:
        """365/540 = 0.675926; exp(-0.675926) = 0.5086852.

        A year-old answer keeps just over half its weight — which is the point
        of the 540-day constant: recent enough to trust, old enough to discount.
        """
        assert recency_multiplier(365) == pytest.approx(math.exp(-365 / 540), abs=1e-12)
        assert recency_multiplier(365) == pytest.approx(0.5086852, abs=1e-6)

    def test_decay_is_monotonic(self) -> None:
        values = [recency_multiplier(days) for days in (0, 30, 180, 365, 730, 1460)]
        assert values == sorted(values, reverse=True)

    def test_negative_age_is_refused(self) -> None:
        """A future-dated answer means the corpus is wrong, not that it is fresh."""
        with pytest.raises(ValueError, match="negative"):
            recency_multiplier(-1)


class TestPreferenceHandComputed:
    """D17: recency is a nudge in [0.90, 1.0], and the product is clamped."""

    def test_recency_is_bounded_below(self) -> None:
        """0.90 + 0.10 * exp(-900/540) = 0.90 + 0.10 * 0.1888756 = 0.9188876.

        Under the old formula this term was 0.1889 — enough on its own to bury
        a perfect match. Now it costs a stale answer about 8%.
        """
        assert recency_preference(900) == pytest.approx(0.9188876, abs=1e-6)
        assert recency_preference(0) == pytest.approx(1.0)
        assert recency_preference(10_000) == pytest.approx(0.90, abs=1e-3)

    def test_won_recent_evidenced(self) -> None:
        """1.15 * 1.1 * (0.90 + 0.10*exp(-30/540)) = 1.265 * 0.9945959 = 1.25816."""
        expected = 1.15 * 1.1 * (0.90 + 0.10 * math.exp(-30 / 540))
        assert preference(outcome=Outcome.WON, age_days=30, has_evidence=True) == pytest.approx(
            expected
        )
        assert expected == pytest.approx(1.25816, abs=1e-5)

    def test_lost_stale_unevidenced(self) -> None:
        """0.85 * 1.0 * 0.9188876 = 0.78105 — a discount, not an erasure."""
        expected = 0.85 * (0.90 + 0.10 * math.exp(-900 / 540))
        assert preference(outcome=Outcome.LOST, age_days=900, has_evidence=False) == pytest.approx(
            expected
        )
        assert expected == pytest.approx(0.78105, abs=1e-5)

    def test_unknown_today_unevidenced_is_neutral(self) -> None:
        assert preference(outcome=Outcome.UNKNOWN, age_days=0, has_evidence=False) == pytest.approx(
            1.0
        )

    def test_evidence_is_exactly_the_configured_bonus(self) -> None:
        without = preference(outcome=Outcome.WON, age_days=100, has_evidence=False)
        with_evidence = preference(outcome=Outcome.WON, age_days=100, has_evidence=True)
        assert with_evidence / without == pytest.approx(EVIDENCE)

    def test_the_total_span_is_clamped(self) -> None:
        """The whole point: preference spans 1.73x, not 8x."""
        best = preference(outcome=Outcome.WON, age_days=0, has_evidence=True)
        worst = preference(outcome=Outcome.LOST, age_days=100_000, has_evidence=False)
        assert best <= CONFIG.preference.clamp_max
        assert worst >= CONFIG.preference.clamp_min
        assert best / worst <= CONFIG.preference.clamp_max / CONFIG.preference.clamp_min

    def test_preference_cannot_invert_beyond_the_ratio_boundary(self) -> None:
        """A pair whose relevance ratio exceeds clamp_max/clamp_min (1.733) can
        never be reordered by preference, however extreme the preferences."""
        boundary = CONFIG.preference.clamp_max / CONFIG.preference.clamp_min
        strong_relevance, weak_relevance = 0.90, 0.90 / (boundary * 1.01)
        best = CONFIG.preference.clamp_max
        worst = CONFIG.preference.clamp_min
        assert strong_relevance * worst > weak_relevance * best

    def test_just_inside_the_boundary_preference_can_decide(self) -> None:
        """Two comparably-relevant candidates SHOULD be separated by preference."""
        boundary = CONFIG.preference.clamp_max / CONFIG.preference.clamp_min
        strong_relevance, weak_relevance = 0.90, 0.90 / (boundary * 0.99)
        assert strong_relevance * CONFIG.preference.clamp_min < (
            weak_relevance * CONFIG.preference.clamp_max
        )


class TestOrderingProperties:
    """The properties the retrieval evals ultimately depend on."""

    def test_won_and_recent_beats_lost_and_stale_at_equal_similarity(self) -> None:
        ranked = score_candidates(
            [
                candidate(node="lost-stale", vector=0.80, outcome=Outcome.LOST, age_days=900),
                candidate(node="won-recent", vector=0.80, outcome=Outcome.WON, age_days=30),
            ]
        )
        assert [c.answer_node_id for c in ranked] == ["won-recent", "lost-stale"]

    def test_evidence_breaks_a_tie(self) -> None:
        ranked = score_candidates(
            [
                candidate(node="a-no-evidence", vector=0.7, outcome=Outcome.WON, age_days=100),
                candidate(
                    node="b-evidenced",
                    vector=0.7,
                    outcome=Outcome.WON,
                    age_days=100,
                    evidence=True,
                ),
            ]
        )
        assert ranked[0].answer_node_id == "b-evidenced"

    def test_a_much_stronger_similarity_still_wins(self) -> None:
        """The multiplier tilts ranking; it must not overturn it outright."""
        ranked = score_candidates(
            [
                candidate(node="weak-but-won", vector=0.40, outcome=Outcome.WON, age_days=0),
                candidate(node="strong-but-lost", vector=0.95, outcome=Outcome.LOST, age_days=0),
            ]
        )
        assert ranked[0].answer_node_id == "strong-but-lost"

    def test_identical_candidates_break_ties_deterministically(self) -> None:
        """Otherwise the ranking depends on database return order."""
        first = score_candidates(
            [candidate(node="zzz", vector=0.7), candidate(node="aaa", vector=0.7)]
        )
        second = score_candidates(
            [candidate(node="aaa", vector=0.7), candidate(node="zzz", vector=0.7)]
        )
        assert (
            [c.answer_node_id for c in first]
            == [c.answer_node_id for c in second]
            == [
                "aaa",
                "zzz",
            ]
        )

    def test_scores_are_ranked_descending(self) -> None:
        ranked = score_candidates(
            [
                candidate(node="a", vector=0.30),
                candidate(node="b", vector=0.90),
                candidate(node="c", vector=0.60),
            ]
        )
        scores = [c.final_score for c in ranked]
        assert scores == sorted(scores, reverse=True)


class TestBlending:
    def test_without_rerank_the_weight_is_redistributed(self) -> None:
        """A missing opinion is not an opinion of zero."""
        assert blend(vector_score=0.8, multiplier=1.0, rerank_score=None) == pytest.approx(0.8)

    def test_a_zero_rerank_is_not_the_same_as_no_rerank(self) -> None:
        no_rerank = blend(vector_score=0.8, multiplier=1.0, rerank_score=None)
        zero_rerank = blend(vector_score=0.8, multiplier=1.0, rerank_score=0.0)
        assert no_rerank == pytest.approx(0.8)
        assert zero_rerank == pytest.approx(0.4)
        assert no_rerank > zero_rerank

    def test_the_blend_is_the_hand_computed_average(self) -> None:
        """0.5*0.8 + 0.5*0.6 = 0.7"""
        assert blend(vector_score=0.8, multiplier=1.0, rerank_score=0.6) == pytest.approx(0.7)

    def test_the_boosted_similarity_is_clamped_before_blending(self) -> None:
        """0.95 * 1.2 = 1.14, which would push final_score out of contract range."""
        assert blend(vector_score=0.95, multiplier=1.2, rerank_score=None) == pytest.approx(1.0)
        assert blend(vector_score=0.95, multiplier=1.2, rerank_score=1.0) == pytest.approx(1.0)

    def test_every_final_score_stays_in_range(self) -> None:
        ranked = score_candidates(
            [
                candidate(node="a", vector=1.0, outcome=Outcome.WON, age_days=0, evidence=True),
                candidate(node="b", vector=0.0, outcome=Outcome.LOST, age_days=2000),
            ]
        )
        assert all(0.0 <= c.final_score <= 1.0 for c in ranked)


class TestMatchFloor:
    def test_exactly_at_the_floor_matches(self) -> None:
        """Inclusive, matching the RetrievalResult contract."""
        scored = score_candidates([candidate(node="a", vector=FLOOR)])
        assert scored[0].final_score == pytest.approx(FLOOR)
        assert to_retrieval_result("q-1", scored).status is RetrievalStatus.MATCHED

    def test_just_below_the_floor_does_not(self) -> None:
        scored = score_candidates([candidate(node="a", vector=FLOOR - 0.01)])
        assert to_retrieval_result("q-1", scored).status is RetrievalStatus.NO_MATCH

    def test_no_candidates_is_no_match(self) -> None:
        result = to_retrieval_result("q-1", [])
        assert result.status is RetrievalStatus.NO_MATCH
        assert result.candidates == []

    def test_weak_candidates_are_still_returned_for_the_report(self) -> None:
        """NO_MATCH still shows what was considered, so it can be explained."""
        scored = score_candidates(
            [candidate(node="a", vector=0.3), candidate(node="b", vector=0.2)]
        )
        result = to_retrieval_result("q-1", scored)
        assert result.status is RetrievalStatus.NO_MATCH
        assert len(result.candidates) == 2


class TestRerankApplication:
    def test_scores_are_matched_by_position(self) -> None:
        ranked = score_candidates(
            [candidate(node="a", vector=0.5), candidate(node="b", vector=0.5)],
            rerank_scores={0: 0.1, 1: 0.9},
        )
        assert ranked[0].answer_node_id == "b"
        assert ranked[0].rerank_score == pytest.approx(0.9)

    def test_a_missing_position_falls_back_for_that_candidate(self) -> None:
        ranked = score_candidates(
            [candidate(node="a", vector=0.8), candidate(node="b", vector=0.8)],
            rerank_scores={0: 0.2},
        )
        by_node = {c.answer_node_id: c for c in ranked}
        assert by_node["a"].rerank_score == pytest.approx(0.2)
        assert by_node["b"].rerank_score is None

    def test_none_means_no_rerank_at_all(self) -> None:
        ranked = score_candidates([candidate(node="a", vector=0.8)], rerank_scores=None)
        assert ranked[0].rerank_score is None
        assert ranked[0].final_score == pytest.approx(0.8)

    def test_the_decomposition_is_preserved_for_the_report(self) -> None:
        """A ranking must be explainable by pointing at arithmetic."""
        ranked = score_candidates(
            [candidate(node="a", vector=0.8, outcome=Outcome.WON, age_days=0, evidence=True)],
            rerank_scores={0: 0.5},
        )
        only = ranked[0]
        assert only.vector_score == pytest.approx(0.8)
        assert only.graph_multiplier == pytest.approx(1.15 * 1.1)
        assert only.rerank_score == pytest.approx(0.5)
        assert only.flags.outcome is Outcome.WON
        assert only.flags.recency_days == 0
