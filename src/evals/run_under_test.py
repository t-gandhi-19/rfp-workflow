"""What the answer-side eval categories are handed.

ONE OBJECT, ASSEMBLED ONCE. Grounding, compliance, quality, adversarial and
operational all read the same run from different angles, and each of them
needing to reconstruct it from the artifact directory would be five chances to
reconstruct it differently. The harness builds this once and passes it down.

IT IS BUILT FROM THE RUN, NOT FROM THE DOCUMENT. A category that re-parsed
`response.txt` would be grading the assembler's formatting as much as the
pipeline's behaviour — and would silently start passing if the assembler ever
stopped writing an answer it was given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.contracts import (
    ComplianceResult,
    DraftedAnswer,
    EscalationRecord,
    EscalationsRecord,
    ExtractedQuestion,
    RetrieverSelection,
    RunTotals,
)


@dataclass(frozen=True)
class StageTiming:
    """How long one stage took. Read off the spans, or measured by the runner."""

    stage: str
    seconds: float


@dataclass
class RunUnderTest:
    """One completed run, in the shape the categories read it."""

    run_id: str
    questions: list[ExtractedQuestion]
    #: Accepted answers only — a question that escalated is NOT here, and that
    #: asymmetry is the point: the grounding category grades what would have been
    #: sent, and an escalated draft is not going to be sent.
    answers: dict[str, DraftedAnswer]
    escalations: EscalationsRecord
    compliance: list[ComplianceResult]
    totals: RunTotals
    #: What the retriever selected, per question. Grounding checks citations
    #: against THIS rather than against the graph: the claim being tested is
    #: "the drafter cited what it was given", and re-querying would also pass an
    #: answer that cited something real it never saw.
    selections: dict[str, RetrieverSelection] = field(default_factory=dict)
    #: Wall-clock per stage, for the operational category.
    timings: list[StageTiming] = field(default_factory=list)
    #: Present when the run's cost was fully priced by the gateway. False means
    #: `totals.cost_usd` is a floor, and the operational category says so rather
    #: than publishing an under-count as a total.
    cost_is_complete: bool = True

    @property
    def accepted(self) -> int:
        return len(self.answers)

    @property
    def escalated(self) -> int:
        return len(self.escalations.escalations)

    def question(self, question_id: str) -> ExtractedQuestion | None:
        return next((q for q in self.questions if q.id == question_id), None)

    def escalation_for(self, question_id: str) -> EscalationRecord | None:
        return next((e for e in self.escalations.escalations if e.question_id == question_id), None)

    def reference(self, question_id: str) -> str:
        """How a report names a question: its printed number where it has one."""
        question = self.question(question_id)
        if question is None:
            return question_id
        return question.printed_number or f"[unnumbered #{question.order + 1}]"

    def sources_for(self, question_id: str) -> set[str]:
        """The answer ids the retriever SELECTED for this question."""
        selection = self.selections.get(question_id)
        if selection is None:
            return set()
        from src.contracts.enums import SelectionOutcome

        return {
            candidate.answer_node_id
            for candidate in selection.selections
            if candidate.outcome is SelectionOutcome.SELECTED
        }
