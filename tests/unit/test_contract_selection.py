"""The retriever's selection object: one question per test.

These assert the INVARIANTS, not the happy path. Every validator here exists
because the field it guards is one a downstream reader trusts without being able
to check it — the drafter reads `status` to decide whether to answer or
escalate, and an auditor reads the selections to decide whether the citation was
earned.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts import (
    CandidateSelection,
    RankingBasis,
    RetrievalStatus,
    RetrieverSelection,
    SelectionOutcome,
)


def selection(
    *,
    node: str = "answer-1",
    rank: int = 0,
    outcome: SelectionOutcome = SelectionOutcome.SELECTED,
    relevance: float = 0.80,
    floor: float = 0.60,
    preference: float = 1.0,
    changed_rank: bool = False,
) -> CandidateSelection:
    return CandidateSelection(
        answer_node_id=node,
        rank=rank,
        outcome=outcome,
        relevance=relevance,
        floor_used=floor,
        preference=preference,
        preference_changed_rank=changed_rank,
    )


def wrap(
    selections: list[CandidateSelection],
    *,
    status: RetrievalStatus = RetrievalStatus.MATCHED,
    basis: RankingBasis = RankingBasis.SIMILARITY_AND_RERANK,
    skip: str | None = None,
) -> RetrieverSelection:
    return RetrieverSelection(
        question_id="Q-1",
        requesting_customer="Meridian Freight",
        status=status,
        ranking_basis=basis,
        rerank_skip_reason=skip,
        candidates_considered=len(selections),
        selections=selections,
    )


class TestOutcomeFollowsFromTheComparison:
    def test_a_candidate_above_the_floor_may_be_selected(self) -> None:
        assert selection(relevance=0.80, floor=0.60).outcome is SelectionOutcome.SELECTED

    def test_a_candidate_exactly_on_the_floor_clears_it(self) -> None:
        """The floor is inclusive, matching `RetrievalResult`. Asserted because
        a boundary that disagrees between the two layers would select a
        candidate at one and reject it at the other."""
        assert selection(relevance=0.60, floor=0.60).outcome is SelectionOutcome.SELECTED

    def test_selected_below_the_floor_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="below the floor"):
            selection(outcome=SelectionOutcome.SELECTED, relevance=0.40, floor=0.60)

    def test_rejected_above_the_floor_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="reaches the floor"):
            selection(outcome=SelectionOutcome.BELOW_RELEVANCE_FLOOR, relevance=0.80, floor=0.60)

    def test_a_large_preference_cannot_rescue_a_below_floor_candidate(self) -> None:
        """D17 stated as a test. Preference reorders comparable candidates; it
        is not a second chance at the floor."""
        with pytest.raises(ValidationError, match="below the floor"):
            selection(
                outcome=SelectionOutcome.SELECTED,
                relevance=0.40,
                floor=0.60,
                preference=1.30,
            )


class TestRankingBasisAndSkipReason:
    def test_a_reranked_ordering_carries_no_skip_reason(self) -> None:
        assert (
            wrap([selection()], basis=RankingBasis.SIMILARITY_AND_RERANK).rerank_skip_reason is None
        )

    def test_a_degraded_ordering_must_say_why(self) -> None:
        with pytest.raises(ValidationError, match="requires rerank_skip_reason"):
            wrap([selection()], basis=RankingBasis.CALIBRATED_SIMILARITY_ONLY)

    def test_a_degraded_ordering_with_a_reason_is_accepted(self) -> None:
        result = wrap(
            [selection()],
            basis=RankingBasis.CALIBRATED_SIMILARITY_ONLY,
            skip="rerank timed out after 180s",
        )
        assert result.rerank_skip_reason == "rerank timed out after 180s"

    def test_claiming_a_rerank_while_carrying_a_skip_reason_is_refused(self) -> None:
        """The half that overstates the ranking, which is the dangerous one."""
        with pytest.raises(ValidationError, match="contradicts rerank_skip_reason"):
            wrap(
                [selection()],
                basis=RankingBasis.SIMILARITY_AND_RERANK,
                skip="gateway error: 503",
            )


class TestRanksAreAnOrdering:
    def test_contiguous_ranks_are_accepted(self) -> None:
        result = wrap([selection(node="a", rank=0), selection(node="b", rank=1)])
        assert [s.rank for s in result.selections] == [0, 1]

    def test_a_gap_in_the_ranks_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="ranks must be"):
            wrap([selection(node="a", rank=0), selection(node="b", rank=2)])

    def test_ranks_out_of_order_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="ranks must be"):
            wrap([selection(node="a", rank=1), selection(node="b", rank=0)])


class TestStatusAgreesWithTheSelections:
    def test_no_match_with_no_selection_is_accepted(self) -> None:
        result = wrap(
            [selection(outcome=SelectionOutcome.BELOW_RELEVANCE_FLOOR, relevance=0.10)],
            status=RetrievalStatus.NO_MATCH,
        )
        assert result.status is RetrievalStatus.NO_MATCH

    def test_no_match_with_an_empty_list_is_accepted(self) -> None:
        """The corpus covering nothing is a normal, tested outcome."""
        assert wrap([], status=RetrievalStatus.NO_MATCH).selections == []

    def test_no_match_carrying_a_selected_candidate_is_refused(self) -> None:
        """The failure this exists for: a NO_MATCH that still ships a perfectly
        good candidate, leaving the drafter's escalate-or-answer decision
        resting on a field nothing verifies."""
        with pytest.raises(ValidationError, match="marked SELECTED"):
            wrap([selection()], status=RetrievalStatus.NO_MATCH)

    def test_matched_with_nothing_selected_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no candidate is marked SELECTED"):
            wrap(
                [selection(outcome=SelectionOutcome.BELOW_RELEVANCE_FLOOR, relevance=0.10)],
                status=RetrievalStatus.MATCHED,
            )


class TestRejectsSortBelowSelections:
    def test_a_reject_after_every_selection_is_accepted(self) -> None:
        result = wrap(
            [
                selection(node="a", rank=0),
                selection(
                    node="b",
                    rank=1,
                    outcome=SelectionOutcome.BELOW_RELEVANCE_FLOOR,
                    relevance=0.10,
                ),
            ]
        )
        assert len(result.selections) == 2

    def test_a_selection_below_a_reject_is_refused(self) -> None:
        """The drafter reads this list in order and cites the top of it."""
        with pytest.raises(ValidationError, match="sort below a rejected"):
            wrap(
                [
                    selection(
                        node="a",
                        rank=0,
                        outcome=SelectionOutcome.BELOW_RELEVANCE_FLOOR,
                        relevance=0.10,
                    ),
                    selection(node="b", rank=1),
                ]
            )


class TestTheRequestingCustomerIsRecorded:
    def test_it_is_required(self) -> None:
        """Threaded through retrieval already; recorded here because
        cross-customer confidentiality is a zero-tolerance eval category and the
        audit trail could not previously show who asked."""
        with pytest.raises(ValidationError):
            RetrieverSelection(
                question_id="Q-1",
                status=RetrievalStatus.NO_MATCH,
                ranking_basis=RankingBasis.SIMILARITY_AND_RERANK,
                candidates_considered=0,
            )  # type: ignore[call-arg]

    def test_it_may_not_be_blank(self) -> None:
        with pytest.raises(ValidationError):
            RetrieverSelection(
                question_id="Q-1",
                requesting_customer="",
                status=RetrievalStatus.NO_MATCH,
                ranking_basis=RankingBasis.SIMILARITY_AND_RERANK,
                candidates_considered=0,
            )
