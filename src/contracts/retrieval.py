"""Retrieval contracts: scored candidates and the ranked result.

The scoring arithmetic itself lives in `src/retrieval/` (Phase 3) and is
deterministic. These models carry the *result* of that arithmetic and enforce
the invariants a downstream agent is entitled to assume — above all that
`status` and the candidate scores cannot disagree about whether anything
actually matched.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.contracts.enums import Outcome, RetrievalStatus


class CandidateFlags(BaseModel):
    """Graph facts that shaped the score, carried forward for audit.

    These are the inputs to the multiplier and the reason a candidate was kept
    or dropped, so they travel with the candidate rather than being recomputed.
    """

    model_config = ConfigDict(extra="forbid")

    superseded: bool = False
    outcome: Outcome = Outcome.UNKNOWN
    confidential: bool = False
    recency_days: int = Field(ge=0)
    sme_id: str | None = None


class ScoredCandidate(BaseModel):
    """One candidate answer with its full score decomposition.

    `tier1_summary` is all the retriever sees at first — full answer text is a
    separate, explicit Tier 2 fetch (progressive disclosure, build prompt §10).
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    answer_node_id: str = Field(min_length=1)
    tier1_summary: str
    #: Raw cosine, kept for the audit trail. Not comparable across models.
    vector_score: float = Field(ge=0.0, le=1.0)
    #: Raw cosine mapped onto measured corpus anchors (D17). 0 = indistinguishable
    #: from an unrelated question, 1 = as close as a genuine paraphrase.
    calibrated_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    #: "Is this the right answer?" — calibrated similarity blended with rerank.
    #: The NO_MATCH decision uses THIS and nothing else.
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    #: "Among relevant candidates, which should win?" — outcome x evidence x
    #: recency nudge, clamped so it can only reorder comparable candidates.
    preference: float = Field(default=1.0, gt=0.0)
    #: Retained name for `preference`; the report and older readers use it.
    graph_multiplier: float = Field(gt=0.0)
    rerank_score: float | None = Field(default=None, ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    flags: CandidateFlags


class RetrievalResult(BaseModel):
    """Ranked candidates for one question, plus the floor that judged them."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    candidates: list[ScoredCandidate] = Field(default_factory=list)
    status: RetrievalStatus
    floor_used: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _candidates_belong_to_question(self) -> RetrievalResult:
        mismatched = {
            c.answer_node_id for c in self.candidates if c.question_id != self.question_id
        }
        if mismatched:
            raise ValueError(f"candidates scored for a different question: {sorted(mismatched)}")
        return self

    @model_validator(mode="after")
    def _candidates_ranked(self) -> RetrievalResult:
        scores = [c.final_score for c in self.candidates]
        if scores != sorted(scores, reverse=True):
            raise ValueError("candidates must be ranked by final_score, descending")
        return self

    @model_validator(mode="after")
    def _status_is_decided_by_relevance_alone(self) -> RetrievalResult:
        """`status` must be derivable from RELEVANCE, not asserted alongside it.

        D17: the floor is judged on relevance, never on final_score. Preference
        may reorder qualifying candidates; it must never rescue a below-floor
        one or doom an above-floor one. Encoding it here rather than leaving it
        to the scorer means a future caller cannot reintroduce the coupling that
        let an 8x preference span overturn a 1.5x similarity difference.

        Without this a NO_MATCH could also ship with a perfectly good candidate
        attached, and the drafter's escalate-or-answer decision would rest on a
        field nothing verifies.
        """
        cleared = [c for c in self.candidates if c.relevance >= self.floor_used]
        if self.status is RetrievalStatus.NO_MATCH and cleared:
            raise ValueError(
                f"status=NO_MATCH but {len(cleared)} candidate(s) reach the relevance "
                f"floor {self.floor_used}"
            )
        if self.status is RetrievalStatus.MATCHED and not cleared:
            raise ValueError(
                f"status=MATCHED but no candidate reaches the relevance floor {self.floor_used}"
            )
        return self
