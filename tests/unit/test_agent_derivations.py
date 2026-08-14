"""The two things the agents return that a model did not decide.

`build_selection` turns scored candidates into selection reasons, and
`to_drafted_answer` turns a drafter payload into a confidence. Both are rule 3:
the model supplies prose, citations and a rerank score; the arithmetic and the
reasons are ours. These assert that they stay ours.
"""

from __future__ import annotations

import pytest

from src.agents.layer import build_selection, to_drafted_answer
from src.contracts import (
    CandidateFlags,
    DraftClaim,
    DraftPayload,
    Outcome,
    RankingBasis,
    RetrievalResult,
    RetrievalStatus,
    ScoredCandidate,
    SelectionOutcome,
)
from src.contracts.thresholds import confidence_escalation_threshold
from tests.stubs import make_questions, make_selection


def candidate(
    node: str, *, relevance: float, final: float, preference: float = 1.0
) -> ScoredCandidate:
    return ScoredCandidate(
        question_id="GQ-001",
        answer_node_id=node,
        tier1_summary="summary",
        vector_score=relevance,
        calibrated_similarity=relevance,
        relevance=relevance,
        preference=preference,
        graph_multiplier=preference,
        final_score=final,
        flags=CandidateFlags(recency_days=30, outcome=Outcome.WON),
    )


def result(candidates: list[ScoredCandidate], *, floor: float = 0.60) -> RetrievalResult:
    cleared = any(c.relevance >= floor for c in candidates)
    return RetrievalResult(
        question_id="GQ-001",
        candidates=candidates,
        status=RetrievalStatus.MATCHED if cleared else RetrievalStatus.NO_MATCH,
        floor_used=floor,
    )


class TestSelectionReasonsAreDerived:
    def test_a_candidate_over_the_floor_is_selected(self) -> None:
        selection = build_selection(
            result=result([candidate("a", relevance=0.80, final=0.80)]),
            trace_reranked=True,
            rerank_skip_reason=None,
            candidates_considered=1,
            requesting_customer="Meridian Freight",
        )
        assert selection.selections[0].outcome is SelectionOutcome.SELECTED

    def test_a_candidate_under_the_floor_is_rejected(self) -> None:
        selection = build_selection(
            result=result([candidate("a", relevance=0.20, final=0.20)]),
            trace_reranked=True,
            rerank_skip_reason=None,
            candidates_considered=1,
            requesting_customer="Meridian Freight",
        )
        assert selection.selections[0].outcome is SelectionOutcome.BELOW_RELEVANCE_FLOOR
        assert selection.status is RetrievalStatus.NO_MATCH

    def test_preference_changing_the_order_is_recorded(self) -> None:
        """Computed by re-sorting on relevance alone and comparing positions —
        what preference DID, not what it was configured to be able to do."""
        selection = build_selection(
            # Ranked by final_score: 'b' first. By relevance alone: 'a' first.
            result=result(
                [
                    candidate("b", relevance=0.70, final=0.90, preference=1.30),
                    candidate("a", relevance=0.85, final=0.85, preference=1.00),
                ]
            ),
            trace_reranked=True,
            rerank_skip_reason=None,
            candidates_considered=2,
            requesting_customer="Meridian Freight",
        )
        assert [s.preference_changed_rank for s in selection.selections] == [True, True]

    def test_preference_leaving_the_order_alone_is_recorded_too(self) -> None:
        selection = build_selection(
            result=result(
                [
                    candidate("a", relevance=0.85, final=0.90),
                    candidate("b", relevance=0.70, final=0.70),
                ]
            ),
            trace_reranked=True,
            rerank_skip_reason=None,
            candidates_considered=2,
            requesting_customer="Meridian Freight",
        )
        assert [s.preference_changed_rank for s in selection.selections] == [False, False]

    def test_a_reranked_ordering_says_so(self) -> None:
        selection = build_selection(
            result=result([candidate("a", relevance=0.80, final=0.80)]),
            trace_reranked=True,
            rerank_skip_reason=None,
            candidates_considered=1,
            requesting_customer="Meridian Freight",
        )
        assert selection.ranking_basis is RankingBasis.SIMILARITY_AND_RERANK
        assert selection.rerank_skip_reason is None

    def test_a_degraded_ordering_carries_the_reason(self) -> None:
        """A rerank failure degrades one question, not the run — but the
        citation it produces must be distinguishable from a reranked one."""
        selection = build_selection(
            result=result([candidate("a", relevance=0.80, final=0.80)]),
            trace_reranked=False,
            rerank_skip_reason="rerank timed out after 180s",
            candidates_considered=1,
            requesting_customer="Meridian Freight",
        )
        assert selection.ranking_basis is RankingBasis.CALIBRATED_SIMILARITY_ONLY
        assert selection.rerank_skip_reason == "rerank timed out after 180s"

    def test_a_degraded_ordering_with_no_stated_reason_still_gets_one(self) -> None:
        """The contract refuses a degraded basis with no reason, so a missing
        one is filled rather than allowed to raise deep in a fan-out."""
        selection = build_selection(
            result=result([candidate("a", relevance=0.80, final=0.80)]),
            trace_reranked=False,
            rerank_skip_reason=None,
            candidates_considered=1,
            requesting_customer="Meridian Freight",
        )
        assert selection.rerank_skip_reason == "rerank not applied"

    def test_the_requesting_customer_is_carried_onto_the_object(self) -> None:
        selection = build_selection(
            result=result([candidate("a", relevance=0.80, final=0.80)]),
            trace_reranked=True,
            rerank_skip_reason=None,
            candidates_considered=1,
            requesting_customer="Bluepine Logistics",
        )
        assert selection.requesting_customer == "Bluepine Logistics"


class TestConfidenceIsComputedFromTheClaims:
    def test_a_fully_cited_answer_scores_the_primary_relevance(self) -> None:
        """coverage 1.0, so confidence is the primary source's relevance."""
        payload = DraftPayload(
            question_id="GQ-001",
            answer_text="We migrate stateful workloads in waves.",
            claims=[DraftClaim(text="We migrate in waves.", source_ids=["answer-GQ-001"])],
        )
        answer = to_drafted_answer(
            payload, selection=make_selection("GQ-001"), question=make_questions(1)[0]
        )
        assert answer.confidence == pytest.approx(0.82)

    def test_half_uncited_claims_halve_the_confidence(self) -> None:
        """Coverage is MULTIPLICATIVE. A partly ungrounded answer is not a
        slightly worse answer; its unsupported half could be anything."""
        payload = DraftPayload(
            question_id="GQ-001",
            answer_text="Two claims, one sourced.",
            claims=[
                DraftClaim(text="Sourced.", source_ids=["answer-GQ-001"]),
                DraftClaim(text="Unsourced."),
            ],
        )
        answer = to_drafted_answer(
            payload, selection=make_selection("GQ-001"), question=make_questions(1)[0]
        )
        assert answer.confidence == pytest.approx(0.41)

    def test_an_uncited_claim_becomes_an_unsupported_claim(self) -> None:
        payload = DraftPayload(
            question_id="GQ-001",
            answer_text="Two claims, one sourced.",
            claims=[
                DraftClaim(text="Sourced.", source_ids=["answer-GQ-001"]),
                DraftClaim(text="Unsourced."),
            ],
        )
        answer = to_drafted_answer(
            payload, selection=make_selection("GQ-001"), question=make_questions(1)[0]
        )
        assert answer.unsupported_claims == ["Unsourced."]
        assert answer.needs_sme_review is True

    def test_an_answer_with_no_claims_scores_zero_and_escalates(self) -> None:
        """An answer the decomposition found nothing in is one nothing can be
        verified about, and "nothing to check" is not "fully checked"."""
        payload = DraftPayload(
            question_id="GQ-001",
            answer_text="",
            claims=[],
            escalate=True,
            escalation_reason="nothing in the sources answers this",
        )
        answer = to_drafted_answer(
            payload, selection=make_selection("GQ-001"), question=make_questions(1)[0]
        )
        assert answer.confidence == 0.0
        assert answer.needs_sme_review is True

    def test_the_drafters_own_escalation_is_honoured(self) -> None:
        """The controller can only ever ADD an escalation, never remove one."""
        payload = DraftPayload(
            question_id="GQ-001",
            answer_text="An answer the drafter was unsure of.",
            claims=[DraftClaim(text="Sourced.", source_ids=["answer-GQ-001"])],
            escalate=True,
            escalation_reason="the sources cover a different cloud provider",
        )
        answer = to_drafted_answer(
            payload, selection=make_selection("GQ-001"), question=make_questions(1)[0]
        )
        assert answer.needs_sme_review is True
        assert answer.escalation_reason == "the sources cover a different cloud provider"

    def test_a_confidence_below_the_floor_forces_review(self) -> None:
        payload = DraftPayload(
            question_id="GQ-001",
            answer_text="Sourced but weakly matched.",
            claims=[DraftClaim(text="Sourced.", source_ids=["answer-GQ-001"])],
        )
        weak = make_selection("GQ-001")
        weak.selections[0].relevance = confidence_escalation_threshold() - 0.05
        answer = to_drafted_answer(payload, selection=weak, question=make_questions(1)[0])
        assert answer.needs_sme_review is True

    def test_the_inputs_travel_with_the_answer(self) -> None:
        """So the controller can RE-DERIVE once the critic's delta is known,
        rather than folding it into an already clamped number."""
        payload = DraftPayload(
            question_id="GQ-001",
            answer_text="An answer.",
            claims=[DraftClaim(text="Sourced.", source_ids=["answer-GQ-001"])],
        )
        answer = to_drafted_answer(
            payload, selection=make_selection("GQ-001"), question=make_questions(1)[0]
        )
        assert answer.confidence_inputs is not None
        assert answer.confidence_inputs.total_claims == 1
        assert answer.confidence_inputs.claims_with_sources == 1

    def test_source_ids_are_collected_from_the_claims_in_order(self) -> None:
        payload = DraftPayload(
            question_id="GQ-001",
            answer_text="Three claims, two sources.",
            claims=[
                DraftClaim(text="One.", source_ids=["ANS-2", "ANS-1"]),
                DraftClaim(text="Two.", source_ids=["ANS-1"]),
            ],
        )
        assert payload.source_ids == ["ANS-2", "ANS-1"]


class TestTheDrafterCannotStateItsOwnConfidence:
    def test_the_payload_has_no_confidence_field(self) -> None:
        assert "confidence" not in DraftPayload.model_fields

    def test_an_extra_confidence_field_is_refused(self) -> None:
        """`extra="forbid"`, so a model that returns one fails validation and
        earns the controller's retry rather than having the number ignored."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DraftPayload.model_validate(
                {"question_id": "GQ-001", "answer_text": "x", "claims": [], "confidence": 0.99}
            )
