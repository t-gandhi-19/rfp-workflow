"""Scoring-sanity suite: the ordering properties retrieval quality rests on.

Hand-computed values throughout. A test that recomputes the formula it is
checking proves only that the code is self-consistent; these assert numbers
worked out independently from the config, so a changed weight fails loudly
rather than silently re-deriving itself.
"""

from __future__ import annotations

import math

import pytest

from src.contracts import Outcome, RetrievalStatus, ScoredCandidate
from src.contracts.thresholds import scoring_config
from src.retrieval.calibration import GEOMETRY, POPULATION, CalibrationArtifact
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

# The floor is DERIVED and lives in the artifact (amendment O), so it is stated
# here as the arithmetic that produces it rather than copied from config:
#   calibrated(bg_p99 0.75)         = (0.75 - 0.60) / 0.30 = 0.50
#   calibrated(same_topic_p05 0.84) = (0.84 - 0.60) / 0.30 = 0.80
#   midpoint (weight 0.5)           = 0.65
#
# The anchors are chosen to divide EXACTLY. An earlier set gave a floor of
# 0.6666..., and storing it rounded to 0.666667 made a candidate whose relevance
# was exactly 0.6666... fall below its own floor — a rounding artifact that
# would read as a scoring bug. Real artifacts carry full float precision; the
# test fixture should not introduce a precision problem the real one lacks.
FLOOR = 0.65

# A calibration artifact with anchors in a realistic band for a prefixed
# nomic-embed-text corpus, chosen to divide cleanly so every value below can be
# worked out by hand rather than recomputed from the code under test.
#
#   span            = same_topic_p50 - bg_p50 = 0.90 - 0.60 = 0.30
#   calibrated(x)   = (x - 0.60) / 0.30, clamped to [0, 1]
#
#   raw 0.60 -> 0.0000     raw 0.80 -> 0.6667
#   raw 0.70 -> 0.3333     raw 0.85 -> 0.8333
#   raw 0.75 -> 0.5000     raw 0.90 -> 1.0000
#
# The pair counts are the ones the real corpus produces (40 questions + 72
# paraphrases over 36 families), so a test artifact cannot quietly describe a
# corpus shape that would fail the guards.
CALIBRATION = CalibrationArtifact(
    geometry=GEOMETRY,
    population=POPULATION,
    embed_model_tag="test-model:v1",
    corpus_hash="0000000000000000",
    computed_at="2026-08-13T00:00:00+00:00",
    background_pair_count=12192,
    same_topic_pair_count=240,
    bg_p50=0.60,
    bg_p95=0.72,
    bg_p99=0.75,
    same_topic_p05=0.84,
    same_topic_p50=0.90,
    derived_floor=FLOOR,
)


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


def rank(
    candidates: list[CandidateInput], *, rerank_scores: dict[int, float] | None = None
) -> list[ScoredCandidate]:
    """Score against the test artifact. Calibration is required, never defaulted."""
    return score_candidates(candidates, calibration=CALIBRATION, rerank_scores=rerank_scores)


class TestConfigIsWhatTheseTestsAssume:
    def test_multipliers(self) -> None:
        outcome = CONFIG.graph_multiplier.outcome
        assert (outcome.won, outcome.unknown, outcome.lost) == (WON, UNKNOWN, LOST)
        assert CONFIG.graph_multiplier.recency.decay_days == DECAY_DAYS
        assert CONFIG.graph_multiplier.evidence_bonus == EVIDENCE

    def test_weights(self) -> None:
        assert CONFIG.final_score.weights.vector_graph == 0.5
        assert CONFIG.final_score.weights.rerank == 0.5

    def test_config_holds_no_floor_value_at_all(self) -> None:
        """Amendments L and O, together.

        L removed `match_floor` rather than aliasing it, because an alias keeps
        every stale reference working while resolving to the wrong units. O then
        removed `match_floor_calibrated` too: right units, but a derived value
        living in config is a placeholder waiting to be forgotten, and it
        shipped once as a known-wrong 0.50 that retrieval consumed at runtime.
        """
        assert not hasattr(CONFIG.retrieval, "match_floor")
        assert not hasattr(CONFIG.retrieval, "match_floor_calibrated")

    def test_config_holds_the_derivation_rule_instead(self) -> None:
        rule = CONFIG.calibration.floor_derivation
        assert rule.background_anchor == "bg_p99"
        assert rule.same_topic_anchor == "same_topic_p05"
        assert rule.midpoint_weight == 0.5


class TestCalibrationIsRequired:
    """Amendment J. The scorer used to accept `calibration=None` and fall back to
    raw cosine, which made the floor fail OPEN: with no artifact the configured
    value was compared against unmapped cosine, where every real score exceeded
    it, so retrieval matched everything while reporting that a floor applied.
    """

    def test_score_candidates_refuses_without_an_artifact(self) -> None:
        with pytest.raises(TypeError):
            score_candidates([candidate(node="a", vector=0.8)])  # type: ignore[call-arg]

    def test_there_is_no_raw_cosine_fallback(self) -> None:
        """Passing None is a type error, not a request for the old behaviour."""
        with pytest.raises((TypeError, AttributeError)):
            score_candidates([candidate(node="a", vector=0.8)], calibration=None)  # type: ignore[arg-type]

    def test_calibration_maps_raw_onto_the_measured_band(self) -> None:
        """0.75 sits halfway between bg_p50 (0.60) and same_topic_p50 (0.90)."""
        scored = rank([candidate(node="a", vector=0.75)])
        assert scored[0].vector_score == pytest.approx(0.75)
        assert scored[0].calibrated_similarity == pytest.approx(0.50)

    def test_a_background_level_score_calibrates_to_zero(self) -> None:
        """Raw 0.60 is the median unrelated question. It means nothing, so: 0."""
        assert rank([candidate(node="a", vector=0.60)])[0].calibrated_similarity == pytest.approx(
            0.0
        )

    def test_a_paraphrase_level_score_calibrates_to_one(self) -> None:
        assert rank([candidate(node="a", vector=0.90)])[0].calibrated_similarity == pytest.approx(
            1.0
        )


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
        ranked = rank(
            [
                candidate(node="lost-stale", vector=0.80, outcome=Outcome.LOST, age_days=900),
                candidate(node="won-recent", vector=0.80, outcome=Outcome.WON, age_days=30),
            ]
        )
        assert [c.answer_node_id for c in ranked] == ["won-recent", "lost-stale"]

    def test_evidence_breaks_a_tie(self) -> None:
        ranked = rank(
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
        """Preference tilts ranking; it must not overturn it outright.

        Both sit inside the calibrated band rather than against its clamps, so
        this tests the arithmetic and not the clamping:

            weak-but-won     raw 0.70 -> calibrated 0.3333
            strong-but-lost  raw 0.82 -> calibrated 0.7333

        The relevance ratio is 2.2, past the 1.733 boundary, so no preference
        combination can invert them.
        """
        ranked = rank(
            [
                candidate(node="weak-but-won", vector=0.70, outcome=Outcome.WON, age_days=0),
                candidate(node="strong-but-lost", vector=0.82, outcome=Outcome.LOST, age_days=0),
            ]
        )
        assert ranked[0].answer_node_id == "strong-but-lost"

    def test_identical_candidates_break_ties_deterministically(self) -> None:
        """Otherwise the ranking depends on database return order."""
        first = rank([candidate(node="zzz", vector=0.7), candidate(node="aaa", vector=0.7)])
        second = rank([candidate(node="aaa", vector=0.7), candidate(node="zzz", vector=0.7)])
        assert (
            [c.answer_node_id for c in first]
            == [c.answer_node_id for c in second]
            == [
                "aaa",
                "zzz",
            ]
        )

    def test_scores_are_ranked_descending(self) -> None:
        """All three inside the band, so none is flattened by a clamp."""
        ranked = rank(
            [
                candidate(node="a", vector=0.65),
                candidate(node="b", vector=0.88),
                candidate(node="c", vector=0.75),
            ]
        )
        scores = [c.final_score for c in ranked]
        assert scores == sorted(scores, reverse=True)
        assert [c.answer_node_id for c in ranked] == ["b", "c", "a"]


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
        """Both calibration clamps exercised: raw 1.0 saturates, raw 0.0 floors."""
        ranked = rank(
            [
                candidate(node="a", vector=1.0, outcome=Outcome.WON, age_days=0, evidence=True),
                candidate(node="b", vector=0.0, outcome=Outcome.LOST, age_days=2000),
            ]
        )
        assert all(0.0 <= c.final_score <= 1.0 for c in ranked)
        assert [c.calibrated_similarity for c in ranked] == [1.0, 0.0]

    def test_the_full_chain_is_hand_computed(self) -> None:
        """raw -> calibrated -> relevance -> preference -> final, end to end.

        raw          0.85
        calibrated   (0.85 - 0.60) / 0.30            = 0.833333
        relevance    0.5 * 0.833333 + 0.5 * 0.60     = 0.716667
        preference   1.15 * 1.1 * (0.90 + 0.10 * e^(-30/540))
                     = 1.265 * 0.9945959             = 1.258164
        final        0.716667 * 1.258164             = 0.901684
        """
        only = rank(
            [candidate(node="a", vector=0.85, outcome=Outcome.WON, age_days=30, evidence=True)],
            rerank_scores={0: 0.60},
        )[0]
        assert only.calibrated_similarity == pytest.approx(0.833333, abs=1e-6)
        assert only.relevance == pytest.approx(0.716667, abs=1e-6)
        assert only.preference == pytest.approx(1.258164, abs=1e-6)
        assert only.final_score == pytest.approx(0.901684, abs=1e-6)


class TestMatchFloor:
    """The floor is judged on RELEVANCE, in CALIBRATED space.

    Raw 0.75 calibrates to exactly 0.50, the configured floor. Under the old
    raw-units comparison that same candidate would have cleared a 0.55 floor on
    its raw score alone — which is the bug amendment L names: a floor that every
    real cosine exceeds is not a floor.
    """

    #: The raw cosine that calibrates to exactly the derived floor:
    #: 0.60 + 0.65 * 0.30 = 0.795.
    AT_FLOOR = 0.795

    def test_exactly_at_the_floor_matches(self) -> None:
        """Inclusive, matching the RetrievalResult contract."""
        scored = rank([candidate(node="a", vector=self.AT_FLOOR)])
        assert scored[0].relevance == pytest.approx(FLOOR)
        assert (
            to_retrieval_result("q-1", scored, calibration=CALIBRATION).status
            is RetrievalStatus.MATCHED
        )

    def test_just_below_the_floor_does_not(self) -> None:
        """Raw 0.7935 -> calibrated 0.645, just under 0.65."""
        scored = rank([candidate(node="a", vector=0.7935)])
        assert scored[0].relevance == pytest.approx(0.645, abs=1e-9)
        result = to_retrieval_result("q-1", scored, calibration=CALIBRATION)
        assert result.status is RetrievalStatus.NO_MATCH

    def test_preference_cannot_rescue_a_below_floor_candidate(self) -> None:
        """D17's invariant, at the floor rather than in the abstract.

        Raw 0.7935 calibrates to 0.645, just under the 0.65 floor. A won, fresh,
        evidenced answer carries preference 1.265, so final_score reaches
        0.645 * 1.265 = 0.8159 — comfortably ABOVE the floor. It still does not
        qualify, because the verdict reads relevance and nothing else.
        """
        scored = rank(
            [candidate(node="a", vector=0.7935, outcome=Outcome.WON, age_days=0, evidence=True)]
        )
        assert scored[0].relevance == pytest.approx(0.645, abs=1e-9)
        assert scored[0].preference == pytest.approx(1.265)
        assert scored[0].final_score == pytest.approx(0.815925, abs=1e-6)
        assert scored[0].final_score > FLOOR
        assert (
            to_retrieval_result("q-1", scored, calibration=CALIBRATION).status
            is RetrievalStatus.NO_MATCH
        )

    def test_preference_cannot_doom_an_above_floor_candidate(self) -> None:
        scored = rank(
            [candidate(node="a", vector=self.AT_FLOOR, outcome=Outcome.LOST, age_days=100_000)]
        )
        assert scored[0].preference < 1.0
        assert (
            to_retrieval_result("q-1", scored, calibration=CALIBRATION).status
            is RetrievalStatus.MATCHED
        )

    def test_no_candidates_is_no_match(self) -> None:
        result = to_retrieval_result("q-1", [], calibration=CALIBRATION)
        assert result.status is RetrievalStatus.NO_MATCH
        assert result.candidates == []

    def test_weak_candidates_are_still_returned_for_the_report(self) -> None:
        """NO_MATCH still shows what was considered, so it can be explained."""
        scored = rank([candidate(node="a", vector=0.65), candidate(node="b", vector=0.62)])
        result = to_retrieval_result("q-1", scored, calibration=CALIBRATION)
        assert result.status is RetrievalStatus.NO_MATCH
        assert len(result.candidates) == 2

    def test_the_floor_recorded_on_the_result_comes_from_the_artifact(self) -> None:
        """`floor_used` travels with the result so a verdict can be re-checked.

        It is the ARTIFACT's floor. Amendment O leaves config with no floor
        value to disagree with it.
        """
        result = to_retrieval_result(
            "q-1", rank([candidate(node="a", vector=0.85)]), calibration=CALIBRATION
        )
        assert result.floor_used == pytest.approx(CALIBRATION.derived_floor)

    def test_an_override_is_available_for_the_eval_sweep(self) -> None:
        """The harness plots floor sensitivity; production never passes this."""
        result = to_retrieval_result(
            "q-1",
            rank([candidate(node="a", vector=0.85)]),
            calibration=CALIBRATION,
            floor_override=0.10,
        )
        assert result.floor_used == pytest.approx(0.10)
        assert result.status is RetrievalStatus.MATCHED


class TestRerankApplication:
    def test_scores_are_matched_by_position(self) -> None:
        """Both raw 0.75 -> calibrated 0.50, so the rerank alone separates them:
        a = 0.5*0.50 + 0.5*0.1 = 0.30, b = 0.5*0.50 + 0.5*0.9 = 0.70."""
        ranked = rank(
            [candidate(node="a", vector=0.75), candidate(node="b", vector=0.75)],
            rerank_scores={0: 0.1, 1: 0.9},
        )
        assert ranked[0].answer_node_id == "b"
        assert ranked[0].rerank_score == pytest.approx(0.9)
        assert ranked[0].relevance == pytest.approx(0.70)
        assert ranked[1].relevance == pytest.approx(0.30)

    def test_a_missing_position_falls_back_for_that_candidate(self) -> None:
        ranked = rank(
            [candidate(node="a", vector=0.8), candidate(node="b", vector=0.8)],
            rerank_scores={0: 0.2},
        )
        by_node = {c.answer_node_id: c for c in ranked}
        assert by_node["a"].rerank_score == pytest.approx(0.2)
        assert by_node["b"].rerank_score is None
        # b keeps the full calibrated value; a is dragged down by its low rerank.
        assert by_node["b"].relevance == pytest.approx(0.666667, abs=1e-6)
        assert by_node["a"].relevance == pytest.approx(0.433333, abs=1e-6)

    def test_none_means_no_rerank_at_all(self) -> None:
        """Raw 0.8 -> calibrated 0.6667; the rerank weight is redistributed."""
        ranked = rank([candidate(node="a", vector=0.8)], rerank_scores=None)
        assert ranked[0].rerank_score is None
        assert ranked[0].final_score == pytest.approx(0.666667, abs=1e-6)

    def test_the_decomposition_is_preserved_for_the_report(self) -> None:
        """A ranking must be explainable by pointing at arithmetic."""
        ranked = rank(
            [candidate(node="a", vector=0.8, outcome=Outcome.WON, age_days=0, evidence=True)],
            rerank_scores={0: 0.5},
        )
        only = ranked[0]
        assert only.vector_score == pytest.approx(0.8)
        assert only.calibrated_similarity == pytest.approx(0.666667, abs=1e-6)
        assert only.graph_multiplier == pytest.approx(1.15 * 1.1)
        assert only.preference == pytest.approx(1.15 * 1.1)
        assert only.rerank_score == pytest.approx(0.5)
        assert only.flags.outcome is Outcome.WON
        assert only.flags.recency_days == 0
