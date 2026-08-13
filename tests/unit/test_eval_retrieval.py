"""The retrieval eval's judgement, tested without a database.

The eval decides whether Phase 3 shipped. Everything it concludes is computed by
the pure functions here — the metrics, the refusal gate, the preference-decisive
diagnostic — so they are driven directly with constructed outcomes rather than
only through a live run. A gate that has never been shown to FAIL is not a gate,
and a live run is the one place it is expensive to arrange a failure.
"""

from __future__ import annotations

import pytest

from src.contracts import RetrievalStatus
from src.evals.contracts import CategoryStatus, Violation
from src.evals.retrieval import (
    CHAIN_HEADS,
    CHAIN_SUPERSEDED,
    COMMISSIONED_RANK1_ACCURACY,
    CONFIDENTIAL_ANSWER_ID,
    RECALL_AT_5_THRESHOLD,
    CandidateAttribution,
    QuestionOutcome,
    RetrievalRun,
    _check_chain_heads,
    _check_no_match_set,
    classify,
    golden_questions,
    to_category_result,
)
from src.retrieval.calibration import BAITS, UNANSWERABLE

FLOOR = 0.5975344852115505


def attribution(answer_id: str, rank: int = 1, *, cleared: bool = True) -> CandidateAttribution:
    return CandidateAttribution(
        rank=rank,
        answer_id=answer_id,
        raw_cosine=0.72,
        calibrated=0.82,
        preference=1.05,
        final=0.86,
        relevance=0.82,
        cleared_floor=cleared,
    )


def outcome(
    number: str,
    kind: str,
    *,
    expected: str | None = None,
    status: RetrievalStatus = RetrievalStatus.MATCHED,
    rank: int | None = 1,
    candidates: list[CandidateAttribution] | None = None,
    rank1: str | None = "ANS-0001",
    rank1_no_pref: str | None = "ANS-0001",
) -> QuestionOutcome:
    return QuestionOutcome(
        number=number,
        kind=kind,
        expected_answer_id=expected,
        status=status,
        rank_of_expected=rank,
        candidates=candidates if candidates is not None else [attribution(expected or "ANS-0001")],
        floor_used=FLOOR,
        rank1_with_preference=rank1,
        rank1_without_preference=rank1_no_pref,
    )


def healthy_run() -> RetrievalRun:
    """Fifteen answerables at rank 1, three unanswerables refusing, two baits."""
    outcomes = [
        outcome(f"a{index}", "answerable", expected=f"ANS-{index:04d}", rank=1)
        for index in range(1, 16)
    ]
    outcomes += [
        outcome(number, "unanswerable", status=RetrievalStatus.NO_MATCH, rank=None)
        for number in UNANSWERABLE
    ]
    outcomes += [
        outcome(number, "bait", status=RetrievalStatus.NO_MATCH, rank=None) for number in BAITS
    ]
    # Every chain head seen somewhere, so the head check is satisfied.
    outcomes[0] = outcome(
        "a1",
        "answerable",
        expected="ANS-0001",
        rank=1,
        candidates=[attribution(head, index + 1) for index, head in enumerate(CHAIN_HEADS)]
        + [attribution("ANS-0001", len(CHAIN_HEADS) + 1)],
    )
    outcomes[0] = outcome(
        "a1",
        "answerable",
        expected=CHAIN_HEADS[0],
        rank=1,
        candidates=[attribution(head, index + 1) for index, head in enumerate(CHAIN_HEADS)],
    )
    return RetrievalRun(outcomes=outcomes)


class TestTheGoldenKeyIsReadNotDerived:
    def test_it_reads_twenty_questions(self) -> None:
        assert len(golden_questions()) == 20

    def test_the_three_unanswerables_carry_no_expected_answer(self) -> None:
        by_number = {q["number"]: q for q in golden_questions()}
        for number in UNANSWERABLE:
            assert not by_number[number]["expected_best_match_answer_id"]

    def test_there_are_fifteen_expected_matches(self) -> None:
        """The denominator of Recall@5, taken from the key rather than assumed."""
        answerable = [
            q
            for q in golden_questions()
            if classify(q["number"], q["expected_best_match_answer_id"]) == "answerable"
        ]
        assert len(answerable) == 15


class TestClassification:
    @pytest.mark.parametrize("number", UNANSWERABLE)
    def test_unanswerables(self, number: str) -> None:
        assert classify(number, None) == "unanswerable"

    @pytest.mark.parametrize("number", BAITS)
    def test_baits_are_baits_even_with_an_expected_answer(self, number: str) -> None:
        """Bait classification wins, so a key edit cannot quietly gate one."""
        assert classify(number, "ANS-0001") == "bait"

    def test_answerable(self) -> None:
        assert classify("1.1", "ANS-0037") == "answerable"

    def test_an_answerless_ordinary_question_is_neither(self) -> None:
        assert classify("1.1", None) == "other"


class TestMetrics:
    def test_recall_counts_only_the_top_five(self) -> None:
        run = RetrievalRun(
            outcomes=[
                outcome("a", "answerable", expected="ANS-1", rank=5),
                outcome("b", "answerable", expected="ANS-2", rank=6),
            ]
        )
        assert run.recall_at_5 == pytest.approx(0.5)

    def test_a_never_retrieved_answer_counts_as_a_miss(self) -> None:
        run = RetrievalRun(outcomes=[outcome("a", "answerable", expected="ANS-1", rank=None)])
        assert run.recall_at_5 == 0.0
        assert run.mrr == 0.0

    def test_mrr_is_the_mean_reciprocal_rank(self) -> None:
        run = RetrievalRun(
            outcomes=[
                outcome("a", "answerable", expected="ANS-1", rank=1),
                outcome("b", "answerable", expected="ANS-2", rank=2),
                outcome("c", "answerable", expected="ANS-3", rank=4),
            ]
        )
        assert run.mrr == pytest.approx((1.0 + 0.5 + 0.25) / 3)

    def test_unanswerables_are_not_in_the_recall_denominator(self) -> None:
        """They have no expected answer, so counting them would cap recall below 1."""
        run = healthy_run()
        assert len(run.expected_matches) == 15
        assert run.recall_at_5 == 1.0

    def test_preference_decisive_is_measured_over_answerables_only(self) -> None:
        run = RetrievalRun(
            outcomes=[
                outcome("a", "answerable", expected="ANS-1", rank1="X", rank1_no_pref="Y"),
                outcome("b", "answerable", expected="ANS-2", rank1="X", rank1_no_pref="X"),
                outcome("c", "unanswerable", rank1="X", rank1_no_pref="Y"),
            ]
        )
        assert run.preference_decisive_rate == pytest.approx(0.5)

    def test_a_question_with_no_candidates_is_not_preference_decisive(self) -> None:
        subject = outcome("a", "answerable", candidates=[], rank1=None, rank1_no_pref=None)
        assert subject.preference_decisive is False


class TestTheRefusalGate:
    def test_a_healthy_run_has_no_violations(self) -> None:
        run = healthy_run()
        _check_no_match_set(run)
        assert run.violations == []

    def test_an_unanswerable_that_matched_is_a_violation(self) -> None:
        run = healthy_run()
        run.outcomes = [
            outcome(o.number, o.kind, status=RetrievalStatus.MATCHED, rank=None)
            if o.number == "2.7"
            else o
            for o in run.outcomes
        ]
        _check_no_match_set(run)
        assert [v.question_number for v in run.violations] == ["2.7"]
        assert "instead of escalating" in run.violations[0].detail

    def test_an_answerable_that_refused_is_a_violation(self) -> None:
        run = healthy_run()
        run.outcomes.append(
            outcome(
                "9.9",
                "answerable",
                expected="ANS-9999",
                status=RetrievalStatus.NO_MATCH,
                rank=None,
            )
        )
        _check_no_match_set(run)
        assert [v.question_number for v in run.violations] == ["9.9"]
        assert "ANS-9999" in run.violations[0].detail

    @pytest.mark.parametrize("number", BAITS)
    def test_a_refused_bait_is_never_a_violation(self, number: str) -> None:
        """The reconciliation. Baits are commissioned BELOW the floor, so
        refusing them is the expected landing, and three other places in the
        codebase already record them as advisory."""
        run = healthy_run()
        _check_no_match_set(run)
        assert not any(v.question_number == number for v in run.violations)

    @pytest.mark.parametrize("number", BAITS)
    def test_a_matched_bait_is_also_not_a_violation(self, number: str) -> None:
        """Advisory means advisory in both directions, or it is a gate."""
        run = healthy_run()
        run.outcomes = [
            outcome(o.number, "bait", status=RetrievalStatus.MATCHED, rank=None)
            if o.number == number
            else o
            for o in run.outcomes
        ]
        _check_no_match_set(run)
        assert not any(v.question_number == number for v in run.violations)

    def test_a_missing_unanswerable_probe_is_a_violation(self) -> None:
        """Silence is not a pass: a gate cannot judge a probe that is absent."""
        run = RetrievalRun(outcomes=[outcome("1.1", "answerable", expected="ANS-1")])
        _check_no_match_set(run)
        assert {v.question_number for v in run.violations} == set(UNANSWERABLE)


class TestChainHeads:
    def test_every_head_seen_passes(self) -> None:
        run = healthy_run()
        _check_chain_heads(run)
        assert run.violations == []

    def test_a_lost_head_is_a_violation(self) -> None:
        """Suppressing the stale answer is only half the requirement."""
        run = RetrievalRun(outcomes=[outcome("a", "answerable", candidates=[])])
        _check_chain_heads(run)
        assert len(run.violations) == len(CHAIN_HEADS)
        assert all(v.rule == "staleness" for v in run.violations)
        assert "the current answer was lost" in run.violations[0].detail

    def test_the_head_and_superseded_lists_are_disjoint_and_paired(self) -> None:
        assert len(CHAIN_HEADS) == len(CHAIN_SUPERSEDED) == 4
        assert not set(CHAIN_HEADS) & set(CHAIN_SUPERSEDED)


class TestCategoryResult:
    def test_a_healthy_run_passes(self) -> None:
        run = healthy_run()
        _check_no_match_set(run)
        _check_chain_heads(run)
        result = to_category_result(run)
        assert result.status is CategoryStatus.PASS
        assert result.key == "retrieval"

    def test_recall_below_the_threshold_fails(self) -> None:
        run = healthy_run()
        run.outcomes = [
            outcome(o.number, o.kind, expected=o.expected_answer_id, rank=None)
            if o.kind == "answerable"
            else o
            for o in run.outcomes
        ]
        result = to_category_result(run)
        recall = next(m for m in result.metrics if m.key == "recall_at_5")
        assert recall.value == 0.0
        assert recall.passed is False
        assert result.status is CategoryStatus.FAIL

    def test_the_recall_threshold_is_the_documented_one(self) -> None:
        result = to_category_result(healthy_run())
        recall = next(m for m in result.metrics if m.key == "recall_at_5")
        assert recall.threshold == RECALL_AT_5_THRESHOLD == 0.8

    def test_mrr_is_reported_and_not_gated(self) -> None:
        result = to_category_result(healthy_run())
        mrr = next(m for m in result.metrics if m.key == "mrr")
        assert mrr.passed is None and mrr.threshold is None

    def test_the_bait_row_is_reported_and_not_gated(self) -> None:
        result = to_category_result(healthy_run())
        baits = next(m for m in result.metrics if m.key == "baits_refused")
        assert baits.passed is None and baits.threshold is None
        assert baits.value == 2.0

    def test_the_preference_diagnostic_is_reported_and_not_gated(self) -> None:
        result = to_category_result(healthy_run())
        metric = next(m for m in result.metrics if m.key == "preference_decisive_rate")
        assert metric.passed is None and metric.threshold is None

    def test_a_confidentiality_violation_fails_the_category(self) -> None:
        run = healthy_run()
        run.violations.append(
            Violation(
                rule="confidentiality",
                question_number="1.1",
                detail=f"{CONFIDENTIAL_ANSWER_ID} reached a Meridian candidate list",
            )
        )
        result = to_category_result(run)
        assert result.status is CategoryStatus.FAIL
        metric = next(m for m in result.metrics if m.key == "confidentiality")
        assert metric.passed is False

    def test_every_zero_tolerance_metric_is_actually_gated(self) -> None:
        """A metric with no threshold cannot fail, whatever its label says."""
        result = to_category_result(healthy_run())
        gated = {m.key for m in result.metrics if m.passed is not None}
        assert gated == {
            "recall_at_5",
            "rank1_accuracy",
            "no_match_gate",
            "staleness",
            "confidentiality",
            "paraphrase_exclusion",
        }


class TestRank1PrimarySourceAccuracy:
    """The gate that would have caught golden 1.1 without needing luck.

    Recall@5 is blind to a wrong rank 1 whenever the right answer is anywhere in
    the top five. Rank 1 is the drafter's primary source — cited, and the input
    to the confidence formula — so it gets its own gate.
    """

    def test_a_perfect_run_scores_one(self) -> None:
        assert healthy_run().rank1_accuracy == 1.0

    def test_the_golden_1_1_shape_is_caught(self) -> None:
        """Expected answer present at rank 2. Recall passes; this must not.

        The exact shape observed with rerank off: 14 of 15 at rank 1, the
        fifteenth at rank 2 behind a candidate preference promoted.
        """
        run = healthy_run()
        run.outcomes = [
            outcome(
                o.number,
                o.kind,
                expected=o.expected_answer_id,
                rank=2,
                rank1="ANS-0032",
                rank1_no_pref=o.expected_answer_id,
            )
            if o.number == "a2"
            else o
            for o in run.outcomes
        ]
        assert run.recall_at_5 == 1.0, "Recall@5 is blind to this, which is the point"
        assert run.rank1_accuracy == pytest.approx(14 / 15)

        result = to_category_result(run)
        metric = next(m for m in result.metrics if m.key == "rank1_accuracy")
        assert metric.passed is False
        assert result.status is CategoryStatus.FAIL

    def test_the_failure_names_the_question_and_both_candidates(self) -> None:
        """A gate that says only "0.9333" obliges someone to reproduce it."""
        run = healthy_run()
        run.outcomes = [
            outcome(
                o.number,
                o.kind,
                expected="ANS-0037",
                rank=2,
                rank1="ANS-0032",
                rank1_no_pref="ANS-0037",
            )
            if o.number == "a2"
            else o
            for o in run.outcomes
        ]
        metric = next(m for m in to_category_result(run).metrics if m.key == "rank1_accuracy")
        assert metric.detail is not None
        assert "a2" in metric.detail
        assert "ANS-0037" in metric.detail
        assert "ANS-0032" in metric.detail

    def test_it_is_commissioned_at_the_shipped_value(self) -> None:
        """Ratchet, not a chosen threshold: any regression from 15/15 fails."""
        assert COMMISSIONED_RANK1_ACCURACY == 1.0
        metric = next(
            m for m in to_category_result(healthy_run()).metrics if m.key == "rank1_accuracy"
        )
        assert metric.threshold == COMMISSIONED_RANK1_ACCURACY

    def test_misses_are_listed_for_action(self) -> None:
        run = healthy_run()
        run.outcomes = [
            outcome(o.number, o.kind, expected=o.expected_answer_id, rank=3)
            if o.number in ("a2", "a3")
            else o
            for o in run.outcomes
        ]
        assert {o.number for o in run.rank1_misses} == {"a2", "a3"}

    def test_a_never_retrieved_answer_is_also_a_rank1_miss(self) -> None:
        run = RetrievalRun(outcomes=[outcome("a", "answerable", expected="ANS-1", rank=None)])
        assert run.rank1_accuracy == 0.0
