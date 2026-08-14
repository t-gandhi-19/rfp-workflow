"""The retriever's typed output: what it chose, and why.

WHY THIS EXISTS SEPARATELY FROM `RetrievalResult`. That model carries the score
DECOMPOSITION — what each number came out as. This one carries the DECISION —
what the numbers were taken to mean. They are different claims, and only the
second is what a reviewer is actually auditing when they ask why an answer cites
the source it cites.

WHY THE REASONS ARE NOT WRITTEN BY THE MODEL. CLAUDE.md rule 3: the LLM reranks,
and nothing else here. Every field below is DERIVED by
`src/retrieval/` arithmetic from the scored candidates — a free-text
justification authored by the retriever agent would be a plausible story about a
ranking it did not perform, and it would be the story that reached the audit
trail. The closed vocabularies in `SelectionOutcome` and `RankingBasis` exist so
this object cannot carry prose at all.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.contracts.enums import RankingBasis, RetrievalStatus, SelectionOutcome


class CandidateSelection(BaseModel):
    """One candidate's fate, with the comparison that decided it.

    `relevance` and `floor_used` are repeated here rather than referenced back
    into `RetrievalResult` because this object is written to the audit trail on
    its own. A selection reason that can only be checked by holding a second
    document next to it is not a reason anybody will check.
    """

    model_config = ConfigDict(extra="forbid")

    answer_node_id: str = Field(min_length=1)
    #: Position in the final ordering, zero-based.
    rank: int = Field(ge=0)
    outcome: SelectionOutcome
    #: The number the floor was applied to. D17: relevance, never final_score.
    relevance: float = Field(ge=0.0, le=1.0)
    floor_used: float = Field(ge=0.0, le=1.0)
    #: The clamped preference multiplier that was applied.
    preference: float = Field(gt=0.0)
    #: Whether preference actually moved this candidate relative to the order
    #: relevance alone would have produced. Recorded because preference is the
    #: part of the score most able to surprise a reader: it may reorder
    #: comparable candidates and must never rescue a below-floor one, and this
    #: is the field that lets an audit confirm it did not.
    preference_changed_rank: bool

    @model_validator(mode="after")
    def _outcome_follows_from_the_comparison(self) -> CandidateSelection:
        """The outcome is DERIVABLE, so it is checked rather than trusted.

        Exactly the invariant `RetrievalResult` enforces at the result level,
        applied per candidate: a selection that says SELECTED while sitting
        below the floor it reports would be an audit trail that disagrees with
        itself, and the disagreement would be invisible — both fields read
        plausibly on their own.
        """
        cleared = self.relevance >= self.floor_used
        if self.outcome is SelectionOutcome.SELECTED and not cleared:
            raise ValueError(
                f"outcome=SELECTED but relevance {self.relevance} is below the floor "
                f"{self.floor_used}; preference may reorder qualifying candidates and "
                f"must never rescue a below-floor one"
            )
        if self.outcome is SelectionOutcome.BELOW_RELEVANCE_FLOOR and cleared:
            raise ValueError(
                f"outcome=BELOW_RELEVANCE_FLOOR but relevance {self.relevance} reaches "
                f"the floor {self.floor_used}"
            )
        return self


class RetrieverSelection(BaseModel):
    """What the retriever hands the drafter for one question.

    This is the retriever task's `output_pydantic`. It is the whole of what the
    drafter is entitled to know about where its evidence came from.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    #: Recorded, not merely passed. `find_similar_questions` requires it and so
    #: does `retrieve`, so there is no path that retrieves without stating who
    #: is asking — but the audit trail could not previously SHOW who asked, and
    #: cross-customer confidentiality is a zero-tolerance eval category.
    requesting_customer: str = Field(min_length=1)
    status: RetrievalStatus
    ranking_basis: RankingBasis
    #: Present exactly when the rerank did not inform the ordering.
    rerank_skip_reason: str | None = None
    #: How many candidates the graph returned before the rerank shortlist.
    candidates_considered: int = Field(ge=0)
    selections: list[CandidateSelection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ranking_basis_and_skip_reason_agree(self) -> RetrieverSelection:
        """A biconditional, because either half alone can lie quietly.

        A trace claiming SIMILARITY_AND_RERANK while carrying "rerank timed
        out" overstates the ranking; one claiming CALIBRATED_SIMILARITY_ONLY
        with no reason understates it and leaves nobody able to say which
        failure occurred. Both are readable, and neither is checkable without
        this.
        """
        degraded = self.ranking_basis is RankingBasis.CALIBRATED_SIMILARITY_ONLY
        has_reason = self.rerank_skip_reason is not None
        if degraded and not has_reason:
            raise ValueError(
                "ranking_basis=CALIBRATED_SIMILARITY_ONLY requires rerank_skip_reason; "
                "a degraded ranking that cannot say why it degraded is not auditable"
            )
        if not degraded and has_reason:
            raise ValueError(
                f"ranking_basis={self.ranking_basis} contradicts rerank_skip_reason "
                f"{self.rerank_skip_reason!r}"
            )
        return self

    @model_validator(mode="after")
    def _ranks_are_a_contiguous_ordering(self) -> RetrieverSelection:
        ranks = [selection.rank for selection in self.selections]
        if ranks != list(range(len(ranks))):
            raise ValueError(f"ranks must be 0..n-1 in order, got {ranks}")
        return self

    @model_validator(mode="after")
    def _status_agrees_with_the_selections(self) -> RetrieverSelection:
        """NO_MATCH means nothing was selected, in both directions.

        The drafter's escalate-or-answer decision reads `status`. If that field
        could disagree with the selection list, the decision would rest on
        something nothing verifies — the same gap `RetrievalResult` closes one
        layer down, and worth closing again here because THIS is the object the
        drafter actually receives.
        """
        selected = [s for s in self.selections if s.outcome is SelectionOutcome.SELECTED]
        if self.status is RetrievalStatus.NO_MATCH and selected:
            raise ValueError(
                f"status=NO_MATCH but {len(selected)} candidate(s) are marked SELECTED"
            )
        if self.status is RetrievalStatus.MATCHED and not selected:
            raise ValueError("status=MATCHED but no candidate is marked SELECTED")
        return self

    @model_validator(mode="after")
    def _selected_candidates_outrank_rejected_ones(self) -> RetrieverSelection:
        """Rejects sort below every selection.

        Without this the object could rank a below-floor candidate first while
        still reporting MATCHED, and the drafter — which reads the list in order
        — would cite it. The floor decides membership; rank decides order; a
        rejected candidate is never at the top of either.
        """
        first_reject = next(
            (
                s.rank
                for s in self.selections
                if s.outcome is SelectionOutcome.BELOW_RELEVANCE_FLOOR
            ),
            None,
        )
        if first_reject is None:
            return self
        misplaced = [
            s.rank
            for s in self.selections
            if s.outcome is SelectionOutcome.SELECTED and s.rank > first_reject
        ]
        if misplaced:
            raise ValueError(
                f"selected candidate(s) at rank {misplaced} sort below a rejected "
                f"candidate at rank {first_reject}"
            )
        return self
