"""Drafting contracts: the answer, and the critique that may only weaken it."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.contracts.thresholds import confidence_escalation_threshold


class DraftedAnswer(BaseModel):
    """One drafted answer with its citations and computed confidence.

    `confidence` is arithmetic from `src/retrieval/confidence.py` — the primary
    source's final_score scaled by claim coverage, plus the critic's delta. It
    is never the model's self-report (build prompt §10).

    The validators below are the load-bearing part of this contract: they make
    "an unescalated answer is always cited and fully supported" an invariant the
    assembler can rely on, rather than a property the drafter is asked to
    remember.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    answer_text: str
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_sme_review: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None

    @model_validator(mode="after")
    def _unescalated_answers_are_grounded(self) -> DraftedAnswer:
        """needs_sme_review=False => cited and free of unsupported claims."""
        if self.needs_sme_review:
            return self
        if not self.source_ids:
            raise ValueError(
                "needs_sme_review=False requires at least one source id; "
                "an uncited answer must be escalated, never emitted"
            )
        if self.unsupported_claims:
            raise ValueError(
                f"needs_sme_review=False but {len(self.unsupported_claims)} unsupported "
                "claim(s) present; unsupported claims must escalate"
            )
        return self

    @model_validator(mode="after")
    def _low_confidence_escalates(self) -> DraftedAnswer:
        """confidence below the configured floor => needs_sme_review=True."""
        threshold = confidence_escalation_threshold()
        if self.confidence < threshold and not self.needs_sme_review:
            raise ValueError(
                f"confidence {self.confidence} is below the escalation threshold "
                f"{threshold}; needs_sme_review must be True"
            )
        return self

    @model_validator(mode="after")
    def _escalations_carry_a_reason(self) -> DraftedAnswer:
        """Every escalation names why.

        The assembler writes an SME-TODO block for each escalation and those
        blocks are never blank (build prompt §12), so the reason has to exist by
        the time the answer is constructed — which is always, since the trigger
        is known at escalation time.
        """
        if self.needs_sme_review and not (self.escalation_reason or "").strip():
            raise ValueError("needs_sme_review=True requires a non-empty escalation_reason")
        return self


class CritiqueResult(BaseModel):
    """The critic's verdict on one drafted answer.

    The critic can only lower confidence or add flags. It cannot trigger a
    redraft and it cannot raise confidence — enforced here by the field bound,
    not by prompt wording (CLAUDE.md rule 14).
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    issues: list[str] = Field(default_factory=list)
    confidence_delta: float = Field(le=0.0, ge=-1.0)
    added_unsupported_claims: list[str] = Field(default_factory=list)
