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
    vector_score: float = Field(ge=0.0, le=1.0)
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
    def _status_matches_scores(self) -> RetrievalResult:
        """`status` must be derivable from the scores, not asserted alongside them.

        Without this, a NO_MATCH could ship with a perfectly good candidate
        attached (or the reverse), and the drafter's escalate-or-answer decision
        would rest on a field nothing checks.
        """
        cleared = [c for c in self.candidates if c.final_score >= self.floor_used]
        if self.status is RetrievalStatus.NO_MATCH and cleared:
            raise ValueError(
                f"status=NO_MATCH but {len(cleared)} candidate(s) reach floor {self.floor_used}"
            )
        if self.status is RetrievalStatus.MATCHED and not cleared:
            raise ValueError(f"status=MATCHED but no candidate reaches floor {self.floor_used}")
        return self
