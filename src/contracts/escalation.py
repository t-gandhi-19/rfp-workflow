"""`escalations.json`, and the SME-TODO blocks the assembler writes from it.

ONE RECORD, TWO RENDERINGS. The escalation list a reviewer opens and the TODO
block embedded in the draft response are the same facts twice. They were always
going to be, and the risk is that they drift into disagreeing about which
question is which — so `reference` below is computed here and used by both,
rather than each rendering formatting a question handle its own way.

AMENDMENT I COMES DUE HERE. `ExtractedQuestion.printed_number` carries this note:

    `escalations.json` and the assembler's SME-TODO blocks must reference this,
    not `order`. A reviewer opening the source document looks for "Question
    3.4"; an internal index means nothing to them and cannot be cross-checked
    against the RFP they were sent.

`order` is still carried, because `printed_number` is None for a document that
numbers nothing and a record with no handle at all is unusable. But `reference`
prefers the printed number, and the assembler has no other way to name a
question — which is what makes the note structural instead of remembered.
"""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from src.contracts.enums import EscalationTrigger


class EscalationRecord(BaseModel):
    """One question routed to a human, with everything they need to act.

    Deliberately self-contained. An SME receives this list without the run's
    internal state, so a record that required joining against something else to
    be understood would be a record that gets ignored.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    #: As printed in the source document — "3.4". None only when the document
    #: numbers nothing at all.
    printed_number: str | None = Field(default=None, min_length=1)
    #: Our own zero-based position. Always present, so `reference` always
    #: resolves to something.
    order: int = Field(ge=0)
    question_text: str = Field(min_length=1)
    trigger: EscalationTrigger
    #: Human-readable, and required. The blocks are never blank (build prompt
    #: §12), and the trigger alone does not say WHAT was found.
    reason: str = Field(min_length=1)
    #: The specific finding, where one exists: the matched forbidden term, the
    #: unresolved entity, the injection PATTERN NAME (trap GQ-004 asserts on
    #: this being named, not merely on the escalation happening).
    detail: str | None = None
    #: Resolved through `get_sme_for_capability` where the question's capability
    #: maps. Both or neither — a name with no id cannot be looked up, an id with
    #: no name cannot be addressed.
    sme_id: str | None = None
    sme_name: str | None = None
    capability_id: str | None = None
    #: The computed confidence, when a draft got far enough to have one.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    #: What was drafted before the escalation, if anything. An SME starting from
    #: a flagged draft is doing review; one starting from nothing is doing the
    #: whole question. Never emitted into the response document.
    draft_text: str | None = None

    @model_validator(mode="after")
    def _sme_identity_is_whole(self) -> EscalationRecord:
        if (self.sme_id is None) != (self.sme_name is None):
            raise ValueError(
                f"sme_id and sme_name travel together; got id={self.sme_id!r}, "
                f"name={self.sme_name!r}"
            )
        return self

    @property
    def reference(self) -> str:
        """How a human is told which question this is.

        The printed number when the document had one, and an explicitly marked
        internal index when it did not — marked, so that a reviewer who cannot
        find "Q7" in their document knows immediately that the document did not
        number its questions, rather than hunting for a number that was never
        printed.
        """
        if self.printed_number is not None:
            return self.printed_number
        return f"[unnumbered #{self.order + 1}]"


class EscalationsRecord(BaseModel):
    """The whole `escalations.json` artifact for one run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    rfp_id: str = Field(min_length=1)
    generated_at: AwareDatetime
    escalations: list[EscalationRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_record_per_question(self) -> EscalationsRecord:
        """A question escalates once.

        Several guardrails can fire on one draft, and the natural way to write
        that loop appends a record per hit — which hands the SME the same
        question three times and inflates the escalation count the operational
        report publishes. The controller records the FIRST trigger; this makes
        that a property of the artifact rather than of the loop that built it.
        """
        seen = [record.question_id for record in self.escalations]
        duplicated = sorted({qid for qid in seen if seen.count(qid) > 1})
        if duplicated:
            raise ValueError(f"question(s) escalated more than once: {duplicated}")
        return self

    @model_validator(mode="after")
    def _ordered_as_the_document_reads(self) -> EscalationsRecord:
        """Sorted by `order`, so the list walks the RFP front to back.

        An SME works through this next to the source document. Any other order
        makes them search for each entry.
        """
        orders = [record.order for record in self.escalations]
        if orders != sorted(orders):
            raise ValueError(f"escalations must be in ascending question order, got {orders}")
        return self
