"""A stub agent layer, so the controller can be tested before 1c exists.

THIS IS THE POINT OF BUILDING THE SPINE FIRST. The controller's job is stage
order, budget enforcement, blast radius and checkpointing — none of which needs
a model to be true or false. Stubbing the five agents makes every one of those
assertions fast, deterministic and free, and it means the agents arrive into a
controller that is already known to work rather than into one whose behaviour is
first observed at the same moment theirs is.

The stubs are DELIBERATELY OBEDIENT about the budget: they call
`budget.authorize` exactly where a real agent would, because the controller's
enforcement is only meaningful if the thing being enforced against actually
asks. A stub that skipped authorization would make the budget tests pass by
never spending anything.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from src.contracts import (
    ComplianceResult,
    ConfidenceInputs,
    CritiqueResult,
    DraftedAnswer,
    EscalationsRecord,
    ExtractedQuestion,
    GuardrailVerdict,
    HaltReason,
    QuestionStatus,
    QuestionType,
    RankingBasis,
    RetrievalStatus,
    RetrieverSelection,
    RFPDocument,
    RunState,
    TriageResult,
)
from src.contracts.enums import DocumentFormat, SelectionOutcome
from src.contracts.selection import CandidateSelection
from src.controller.budget import BudgetLedger, CallKind, CallUsage
from src.controller.protocols import AgentValidationError


class SimulatedKill(BaseException):
    """A kill, not a failure.

    Inherits BaseException rather than Exception ON PURPOSE: the controller's
    per-question blast radius is `except Exception`, so an ordinary error
    escalates one question and the run carries on. A kill has to go straight
    past that and unwind the run, which is what makes "what survived in
    Postgres" a real question.

    Not `KeyboardInterrupt`, which would be the most faithful signal, because
    pytest intercepts that at session level and aborts the run before an
    assertion can be made about it.
    """


def make_document(
    *,
    doc_id: str = "rfp-golden",
    customer: str = "Meridian Freight",
    domain: str = "cloud_migration",
) -> RFPDocument:
    return RFPDocument(
        id=doc_id,
        source_filename="golden_rfp.pdf",
        format=DocumentFormat.PDF,
        raw_text="Synthetic cloud-migration RFP body.",
        customer_name=customer,
        domain=domain,
        intake_timestamp=datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
    )


def make_questions(count: int = 3) -> list[ExtractedQuestion]:
    return [
        ExtractedQuestion(
            id=f"GQ-{index + 1:03d}",
            rfp_id="rfp-golden",
            text=f"Question {index + 1}: describe your migration approach.",
            normalized_text=f"describe your migration approach {index + 1}",
            section="Technical",
            question_type=QuestionType.TECHNICAL,
            order=index,
            printed_number=f"{index + 1}.0",
        )
        for index in range(count)
    ]


def make_selection(
    question_id: str, *, status: RetrievalStatus = RetrievalStatus.MATCHED
) -> RetrieverSelection:
    selections = (
        [
            CandidateSelection(
                answer_node_id=f"answer-{question_id}",
                rank=0,
                outcome=SelectionOutcome.SELECTED,
                relevance=0.82,
                floor_used=0.60,
                preference=1.0,
                preference_changed_rank=False,
            )
        ]
        if status is RetrievalStatus.MATCHED
        else []
    )
    return RetrieverSelection(
        question_id=question_id,
        requesting_customer="Meridian Freight",
        status=status,
        ranking_basis=RankingBasis.SIMILARITY_AND_RERANK,
        candidates_considered=len(selections),
        selections=selections,
    )


@dataclass
class StubAgentLayer:
    """The five agents, scripted.

    Each knob names ONE failure the controller has to handle, so a test turns on
    exactly the condition it is about.
    """

    questions: list[ExtractedQuestion] = field(default_factory=lambda: make_questions(3))
    domain_match: bool = True
    detected_domain: str = "cloud_migration"
    #: question_id -> RetrievalStatus. Absent means MATCHED.
    retrieval_status: dict[str, RetrievalStatus] = field(default_factory=dict)
    #: question_ids whose FIRST draft fails validation. The retry then succeeds.
    fail_draft_validation_once: set[str] = field(default_factory=set)
    #: question_ids whose every draft fails validation.
    fail_draft_validation_always: set[str] = field(default_factory=set)
    #: question_ids that raise a plain error during retrieval.
    raise_on_retrieve: set[str] = field(default_factory=set)
    #: question_id -> confidence the drafter's arithmetic lands on.
    confidence: dict[str, float] = field(default_factory=dict)
    #: question_id -> the critic's delta.
    critique_delta: dict[str, float] = field(default_factory=dict)
    #: Tokens booked per call, so budget ceilings can be reached deterministically.
    tokens_per_call: int = 0
    cost_per_call: float | None = 0.0
    triage_raises: bool = False
    extract_raises: bool = False
    #: question_ids whose critique kills the run. `except Exception` is the
    #: controller's per-question blast radius; `SimulatedKill` is a
    #: BaseException and goes straight past it, unwinding the run with whatever
    #: was already checkpointed.
    #:
    #: THE KILL LATCHES. Once it fires, every later agent call raises too,
    #: because a killed process does not go on answering questions. Without the
    #: latch the controller's `return_exceptions=True` — which exists so a halt
    #: cannot cancel other questions' checkpoints — let the NEXT question run to
    #: completion, and a "killed mid-fan-out" run finished more work than a
    #: killed run can.
    kill_on_critique: set[str] = field(default_factory=set)
    _killed: bool = False

    calls: Counter[str] = field(default_factory=Counter)
    draft_retry_contexts: dict[str, str] = field(default_factory=dict)

    def _spend(self, budget: BudgetLedger) -> None:
        budget.record(CallUsage(tokens=self.tokens_per_call, cost_usd=self.cost_per_call))

    def _refuse_if_killed(self) -> None:
        if self._killed:
            raise SimulatedKill("the run was killed; this process is gone")

    async def triage(self, document: RFPDocument, *, budget: BudgetLedger) -> TriageResult:
        self.calls["triage"] += 1
        if self.triage_raises:
            raise RuntimeError("ollama unreachable")
        budget.authorize(CallKind.TRIAGE)
        self._spend(budget)
        return TriageResult(
            rfp_id=document.id,
            domain_match=self.domain_match,
            detected_domain=self.detected_domain,
            halt_reason=None if self.domain_match else HaltReason.DOMAIN_MISMATCH,
        )

    async def extract(
        self, document: RFPDocument, *, budget: BudgetLedger
    ) -> list[ExtractedQuestion]:
        self.calls["extract"] += 1
        if self.extract_raises:
            raise RuntimeError("pdf is unreadable")
        # Deterministic parsing needs no model; the assist is conditional, which
        # is exactly why the ledger is asked rather than told.
        return list(self.questions)

    async def retrieve(
        self,
        question: ExtractedQuestion,
        *,
        requesting_customer: str,
        budget: BudgetLedger,
    ) -> RetrieverSelection:
        self._refuse_if_killed()
        self.calls["retrieve"] += 1
        if question.id in self.raise_on_retrieve:
            raise RuntimeError("graph session died")
        budget.authorize(CallKind.RERANK, question_id=question.id)
        self._spend(budget)
        return make_selection(
            question.id, status=self.retrieval_status.get(question.id, RetrievalStatus.MATCHED)
        )

    async def draft(
        self,
        question: ExtractedQuestion,
        selection: RetrieverSelection,
        *,
        budget: BudgetLedger,
        retry_context: str | None = None,
    ) -> DraftedAnswer:
        self._refuse_if_killed()
        self.calls["draft"] += 1
        if retry_context is not None:
            self.draft_retry_contexts[question.id] = retry_context
        else:
            budget.authorize(CallKind.DRAFT, question_id=question.id)
        self._spend(budget)

        if question.id in self.fail_draft_validation_always or (
            question.id in self.fail_draft_validation_once and retry_context is None
        ):
            raise AgentValidationError(
                f"source_ids must be a list of strings, got null (question {question.id})"
            )

        score = self.confidence.get(question.id, 0.9)
        return DraftedAnswer(
            question_id=question.id,
            answer_text=f"Synthetic answer for {question.id}.",
            source_ids=[f"answer-{question.id}"],
            confidence=score,
            needs_sme_review=score < 0.6,
            escalation_reason="below the floor" if score < 0.6 else None,
            confidence_inputs=ConfidenceInputs(
                primary_final_score=score,
                claims_with_sources=1,
                total_claims=1,
            ),
        )

    async def critique(
        self,
        question: ExtractedQuestion,
        answer: DraftedAnswer,
        *,
        budget: BudgetLedger,
    ) -> CritiqueResult:
        self._refuse_if_killed()
        self.calls["critique"] += 1
        if question.id in self.kill_on_critique:
            self._killed = True
            raise SimulatedKill(f"killed while critiquing {question.id}")
        budget.authorize(CallKind.CRITIQUE, question_id=question.id)
        self._spend(budget)
        delta = self.critique_delta.get(question.id, 0.0)
        return CritiqueResult(
            question_id=question.id,
            issues=["phrasing is broad"] if delta else [],
            confidence_delta=delta,
        )


@dataclass
class StubGuardrails:
    """Passes everything unless told otherwise."""

    #: question_id -> the verdict to return.
    verdicts: dict[str, GuardrailVerdict] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list)

    async def apply(
        self,
        *,
        question: ExtractedQuestion,
        answer: DraftedAnswer,
        selection: RetrieverSelection,
        document: RFPDocument,
    ) -> GuardrailVerdict:
        self.seen.append(question.id)
        return self.verdicts.get(question.id, GuardrailVerdict(passed=True))


@dataclass
class StubCompliance:
    checked: list[str] = field(default_factory=list)

    def check(self, question: ExtractedQuestion, answer: DraftedAnswer | None) -> ComplianceResult:
        self.checked.append(question.id)
        return ComplianceResult(
            question_id=question.id,
            mandatory_answered=answer is not None or not question.mandatory,
            within_word_limit=True,
            template_slots_filled=answer is not None,
        )


@dataclass
class StubAssembler:
    """Records what it was handed, and renders a deterministic body.

    `rendered` is what the byte-identical resume assertion compares: a real
    assembler writes a .docx, and comparing text here isolates "did the resumed
    run reach assembly with identical inputs" from "does python-docx produce
    identical bytes".
    """

    calls: int = 0
    rendered: str = ""
    last_escalations: EscalationsRecord | None = None

    def assemble(
        self,
        *,
        run_id: str,
        document: RFPDocument,
        questions: list[ExtractedQuestion],
        answers: dict[str, DraftedAnswer],
        escalations: EscalationsRecord,
    ) -> dict[str, str]:
        self.calls += 1
        self.last_escalations = escalations
        lines: list[str] = []
        for question in questions:
            answer = answers.get(question.id)
            if answer is None:
                lines.append(f"{question.printed_number}\tESCALATED")
            else:
                lines.append(
                    f"{question.printed_number}\t{answer.answer_text}\t"
                    f"{answer.confidence:.4f}\t{','.join(answer.source_ids)}"
                )
        for record in escalations.escalations:
            lines.append(f"TODO {record.reference}\t{record.trigger}\t{record.reason}")
        self.rendered = "\n".join(lines)
        return {
            "response_docx": f"out/{run_id}/response.docx",
            "escalations_json": f"out/{run_id}/escalations.json",
        }


@dataclass
class RecordingCheckpointer:
    """Captures every checkpoint instead of writing one.

    The ORDER matters as much as the content: `save_run` at each transition and
    `save_question_status` on each change are what a resume reads back, so the
    tests assert the sequence rather than the final state.
    """

    timeout: float = 5.0
    runs: list[RunState] = field(default_factory=list)
    statuses: list[tuple[str, QuestionStatus]] = field(default_factory=list)
    drafts: list[DraftedAnswer] = field(default_factory=list)

    async def save_run(self, state: RunState, *, client: httpx.AsyncClient) -> None:
        self.runs.append(state.model_copy(deep=True))

    async def save_question_status(
        self,
        *,
        run_id: str,
        question_id: str,
        status: QuestionStatus,
        client: httpx.AsyncClient,
    ) -> None:
        self.statuses.append((question_id, status))

    async def save_draft(
        self, *, run_id: str, answer: DraftedAnswer, client: httpx.AsyncClient
    ) -> None:
        self.drafts.append(answer)

    @property
    def stages(self) -> list[str]:
        return [state.stage.value for state in self.runs]
