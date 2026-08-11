"""Intake contracts: the document, the triage verdict, the extracted questions."""

from __future__ import annotations

from datetime import date

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from src.contracts.enums import DocumentFormat, HaltReason, QuestionType


class DocumentSection(BaseModel):
    """One section of the source document.

    Sections are a typed list rather than a bare mapping so section identity
    survives the boundary intact (CLAUDE.md rule 13).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    order: int = Field(ge=0)
    text: str


class RFPDocument(BaseModel):
    """A single ingested RFP, before any question extraction.

    `raw_text` is untrusted data and never an instruction: it passes the
    injection sanitizer before reaching any prompt (CLAUDE.md rule 15).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    format: DocumentFormat
    raw_text: str
    sections: list[DocumentSection] = Field(default_factory=list)
    detected_deadline: date | None = None
    customer_name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    intake_timestamp: AwareDatetime

    @model_validator(mode="after")
    def _sections_ordered(self) -> RFPDocument:
        orders = [section.order for section in self.sections]
        if orders != sorted(orders):
            raise ValueError("sections must be in ascending `order`")
        if len(set(orders)) != len(orders):
            raise ValueError("section `order` values must be unique")
        return self


class TriageResult(BaseModel):
    """Verdict of the triage stage.

    A domain mismatch halts the whole run with a readable reason — one of only
    three run-level halts (build prompt §12).
    """

    model_config = ConfigDict(extra="forbid")

    rfp_id: str = Field(min_length=1)
    domain_match: bool
    detected_domain: str = Field(min_length=1)
    is_rfi: bool = False
    deadline: date | None = None
    halt_reason: HaltReason | None = None

    @model_validator(mode="after")
    def _mismatch_must_halt(self) -> TriageResult:
        if not self.domain_match and self.halt_reason is None:
            raise ValueError(
                "domain_match=False requires halt_reason (expected HaltReason.DOMAIN_MISMATCH)"
            )
        if self.domain_match and self.halt_reason is HaltReason.DOMAIN_MISMATCH:
            raise ValueError("halt_reason=DOMAIN_MISMATCH contradicts domain_match=True")
        return self


class ExtractedQuestion(BaseModel):
    """One question pulled from the RFP by the deterministic extractor.

    `text` keeps the document's wording; `normalized_text` is the stripped form
    (numbering and boilerplate removed) used for embedding and matching. Both
    are kept so a citation can always be traced back to what the document
    actually said.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    rfp_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    section: str = Field(min_length=1)
    question_type: QuestionType
    word_limit: int | None = Field(default=None, gt=0)
    mandatory: bool = False
    order: int = Field(ge=0)
    #: The number as printed in the document — "1.1", "3.4". Distinct from
    #: `order`, which is our own zero-based position.
    #:
    #: None when a document numbers nothing; the extractor still assigns `order`.
    #:
    #: Phase 4 note: `escalations.json` and the assembler's SME-TODO blocks must
    #: reference this, not `order`. A reviewer opening the source document looks
    #: for "Question 3.4"; an internal index means nothing to them and cannot be
    #: cross-checked against the RFP they were sent.
    printed_number: str | None = Field(default=None, min_length=1)
