"""Retrieval invariants: ranking order, ownership, and status/score agreement."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts import (
    CandidateFlags,
    Outcome,
    RetrievalResult,
    RetrievalStatus,
    ScoredCandidate,
)

FLOOR = 0.55


def _candidate(
    final_score: float, *, question_id: str = "q-001", node: str = "ans-1"
) -> ScoredCandidate:
    return ScoredCandidate(
        question_id=question_id,
        answer_node_id=node,
        tier1_summary="Six-week assessment using MigrateHub.",
        vector_score=0.8,
        graph_multiplier=1.0,
        rerank_score=0.7,
        final_score=final_score,
        flags=CandidateFlags(superseded=False, outcome=Outcome.WON, recency_days=90),
    )


class TestRanking:
    def test_accepts_descending_candidates(self) -> None:
        result = RetrievalResult(
            question_id="q-001",
            candidates=[_candidate(0.91, node="a"), _candidate(0.72, node="b")],
            status=RetrievalStatus.MATCHED,
            floor_used=FLOOR,
        )
        assert [c.final_score for c in result.candidates] == [0.91, 0.72]

    def test_accepts_ties(self) -> None:
        result = RetrievalResult(
            question_id="q-001",
            candidates=[_candidate(0.7, node="a"), _candidate(0.7, node="b")],
            status=RetrievalStatus.MATCHED,
            floor_used=FLOOR,
        )
        assert len(result.candidates) == 2

    def test_rejects_unranked_candidates(self) -> None:
        with pytest.raises(ValidationError, match="ranked by final_score"):
            RetrievalResult(
                question_id="q-001",
                candidates=[_candidate(0.60, node="a"), _candidate(0.95, node="b")],
                status=RetrievalStatus.MATCHED,
                floor_used=FLOOR,
            )


class TestOwnership:
    def test_rejects_a_candidate_scored_for_another_question(self) -> None:
        """Guards against a fan-out bug pasting one question's results onto another."""
        with pytest.raises(ValidationError, match="different question"):
            RetrievalResult(
                question_id="q-001",
                candidates=[_candidate(0.9, question_id="q-002", node="stray")],
                status=RetrievalStatus.MATCHED,
                floor_used=FLOOR,
            )


class TestStatusAgreesWithScores:
    """`status` must be derivable from the scores, never merely asserted."""

    def test_accepts_no_match_when_everything_is_below_floor(self) -> None:
        result = RetrievalResult(
            question_id="q-001",
            candidates=[_candidate(0.41, node="a"), _candidate(0.22, node="b")],
            status=RetrievalStatus.NO_MATCH,
            floor_used=FLOOR,
        )
        assert result.status is RetrievalStatus.NO_MATCH

    def test_accepts_no_match_with_no_candidates_at_all(self) -> None:
        result = RetrievalResult(
            question_id="q-001", candidates=[], status=RetrievalStatus.NO_MATCH, floor_used=FLOOR
        )
        assert result.candidates == []

    def test_rejects_no_match_hiding_a_good_candidate(self) -> None:
        with pytest.raises(ValidationError, match="NO_MATCH"):
            RetrievalResult(
                question_id="q-001",
                candidates=[_candidate(0.88, node="a")],
                status=RetrievalStatus.NO_MATCH,
                floor_used=FLOOR,
            )

    def test_rejects_matched_with_nothing_reaching_the_floor(self) -> None:
        with pytest.raises(ValidationError, match="MATCHED"):
            RetrievalResult(
                question_id="q-001",
                candidates=[_candidate(0.30, node="a")],
                status=RetrievalStatus.MATCHED,
                floor_used=FLOOR,
            )

    def test_floor_is_inclusive(self) -> None:
        """A candidate exactly at the floor counts as a match."""
        result = RetrievalResult(
            question_id="q-001",
            candidates=[_candidate(FLOOR, node="a")],
            status=RetrievalStatus.MATCHED,
            floor_used=FLOOR,
        )
        assert result.status is RetrievalStatus.MATCHED


class TestCandidateBounds:
    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_rejects_vector_score_outside_unit_interval(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            ScoredCandidate.model_validate(_candidate(0.5).model_dump() | {"vector_score": bad})

    def test_rejects_non_positive_graph_multiplier(self) -> None:
        """A zero multiplier would silently erase a candidate rather than rank it."""
        with pytest.raises(ValidationError):
            ScoredCandidate.model_validate(_candidate(0.5).model_dump() | {"graph_multiplier": 0.0})

    def test_rerank_score_may_be_absent(self) -> None:
        """Only the top N are reranked; the rest carry no rerank score."""
        candidate = ScoredCandidate.model_validate(
            _candidate(0.5).model_dump() | {"rerank_score": None}
        )
        assert candidate.rerank_score is None

    def test_rejects_negative_recency(self) -> None:
        with pytest.raises(ValidationError):
            CandidateFlags(recency_days=-1)
