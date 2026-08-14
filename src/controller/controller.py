"""The deterministic execution controller — the spine of a run.

CLAUDE.md rule 7: there is no manager agent and no supervisor LLM. Stage order
is fixed and lives here, in Python, as a straight line:

    triage -> extract -> per-question fan-out -> compliance -> assemble

WHAT IS DELIBERATE ABOUT THE FAILURE HANDLING. There are exactly two blast
radii, and they are not interchangeable:

* **One question fails -> that question escalates and the run continues.** A
  retrieval error, a validation failure that survives its retry, a guardrail
  hit, a spent per-question allowance: all of them produce an escalation
  carrying the cause, and the other thirty-nine questions are unaffected. An
  RFP response missing one answer is a response; a run abandoned over one
  question is nothing.
* **Three things halt the whole run**, and only three: a triage domain
  mismatch, a run-level budget ceiling, and infrastructure the run cannot
  proceed without. Everything already produced is checkpointed before the halt,
  so `make resume` picks up rather than restarts.

WHY BUDGETS ARE ENFORCED HERE. Rule 14. The ledger is the controller's, every
call passes through it, and the agent framework is never asked to respect a
ceiling it could ignore. See `src/controller/budget.py`.

WHY CONFIDENCE IS RECOMPUTED HERE. Rule 3. The critic returns a delta; the
controller folds it into `compute_confidence` with the drafter's measured terms
and rebuilds the answer. The model never states the number, and the contract
refuses an answer whose confidence sits below the floor without an escalation —
so a lowered confidence FORCES the escalation rather than suggesting it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from src.contracts import (
    ArtifactPaths,
    ComplianceResult,
    CritiqueResult,
    DraftedAnswer,
    EscalationRecord,
    EscalationsRecord,
    EscalationTrigger,
    ExtractedQuestion,
    HaltReason,
    QuestionStatus,
    RetrievalStatus,
    RetrieverSelection,
    RFPDocument,
    RunResult,
    RunStage,
    RunState,
    RunTotals,
)
from src.controller.budget import (
    BudgetLedger,
    JudgeInProductionError,
    QuestionBudgetError,
    RunBudgetError,
)
from src.controller.checkpoint import ResumableRun
from src.controller.limits import LimitsConfig, limits_config
from src.controller.protocols import (
    AgentLayer,
    AgentValidationError,
    Assembler,
    ComplianceChecker,
    GuardrailSuite,
    RunCheckpointer,
)
from src.observability.tracing import (
    question_span,
    record_failure,
    run_span,
    stage_span,
    trace_url,
)
from src.retrieval.confidence import compute_confidence

logger = logging.getLogger("rfp.controller")


class RunHaltedError(RuntimeError):
    """Internal signal: unwind the fan-out and halt the run.

    Carries the reason so the halt is recorded with a cause rather than as a
    bare failure — `RunState` refuses a HALTED stage with no `halted_reason`.
    """

    def __init__(self, reason: HaltReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class QuestionOutcome:
    """What one question produced. Exactly one of `answer`/`escalation` is set."""

    question: ExtractedQuestion
    answer: DraftedAnswer | None = None
    escalation: EscalationRecord | None = None

    @property
    def escalated(self) -> bool:
        return self.escalation is not None


@dataclass
class RunController:
    """Executes one run. Every collaborator is required and injected.

    No collaborator has a default. A controller that silently constructed its
    own guardrail suite, or defaulted to one that passes everything, would run
    and produce plausible output while enforcing nothing — and that output is
    indistinguishable from a correct run's until somebody audits it.
    """

    agents: AgentLayer
    guardrails: GuardrailSuite
    compliance: ComplianceChecker
    assembler: Assembler
    checkpointer: RunCheckpointer
    ledger: BudgetLedger = field(default_factory=BudgetLedger)
    limits: LimitsConfig = field(default_factory=limits_config)
    #: The HTTP client for this run's checkpoints, opened in `_execute` and
    #: shared by every stage so one token is fetched per run rather than per
    #: write. Not part of the public surface.
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("no checkpoint client; controller methods run inside `run()`")
        return self._client

    # -- entry points -----------------------------------------------------

    async def run(self, document: RFPDocument, *, run_id: str) -> RunResult:
        """Execute a fresh run end to end."""
        return await self._execute(document, run_id=run_id, resumed=None)

    async def resume(self, document: RFPDocument, *, resumed: ResumableRun) -> RunResult:
        """Re-execute a run, skipping questions that already reached a terminal state.

        The skipped questions' answers come back from Postgres unchanged and the
        assembler receives the same inputs in the same order, which is what makes
        the resumed artifact byte-identical to the one an uninterrupted run would
        have produced.
        """
        return await self._execute(document, run_id=resumed.state.run_id, resumed=resumed)

    # -- the stage machine ------------------------------------------------

    async def _execute(
        self, document: RFPDocument, *, run_id: str, resumed: ResumableRun | None
    ) -> RunResult:
        started = resumed.state.started_at if resumed else _now()
        state = RunState(
            run_id=run_id,
            rfp_id=document.id,
            stage=RunStage.CREATED,
            per_question_status=dict(resumed.state.per_question_status) if resumed else {},
            tokens_used=resumed.state.tokens_used if resumed else 0,
            cost_usd=resumed.state.cost_usd if resumed else 0.0,
            started_at=started,
            updated_at=_now(),
        )
        # A resumed run keeps spending against what it already spent, or a run
        # killed at 90% of its ceiling would restart with a full budget and the
        # ceiling would bound nothing.
        if resumed:
            self.ledger.tokens_used = resumed.state.tokens_used
            self.ledger.cost_usd = resumed.state.cost_usd

        # `run_span` is a SYNC context manager and the client an async one, so
        # they cannot share one `async with` — mypy caught that here rather than
        # it surfacing as an AttributeError mid-run.
        with run_span(run_id=run_id, rfp_id=document.id):
            async with httpx.AsyncClient(timeout=self.checkpointer.timeout) as client:
                self._client = client
                return await self._run_stages(document, run_id, state, resumed)

    async def _run_stages(
        self,
        document: RFPDocument,
        run_id: str,
        state: RunState,
        resumed: ResumableRun | None,
    ) -> RunResult:
        """The stage sequence, with the run span and the client already open."""
        await self._checkpoint(state)

        try:
            questions = await self._parse_and_extract(document, state)
            outcomes = await self._fan_out(document, questions, state, resumed)
        except RunHaltedError as halt:
            record_failure(halt)
            return await self._halt(state, halt, document=document)

        compliance = await self._check_compliance(questions, outcomes, state)
        artifacts = await self._assemble(document, questions, outcomes, state)

        state = await self._advance(state, RunStage.COMPLETE)
        return RunResult(
            run_id=run_id,
            answers=[o.answer for o in outcomes if o.answer is not None],
            compliance=compliance,
            totals=self._totals(questions, outcomes),
            trace_url=trace_url(run_id),
            artifact_paths=artifacts,
        )

    async def _parse_and_extract(
        self, document: RFPDocument, state: RunState
    ) -> list[ExtractedQuestion]:
        state = await self._advance(state, RunStage.PARSING)
        with stage_span(RunStage.PARSING.value):
            try:
                triage = await self.agents.triage(document, budget=self.ledger)
            except (RunBudgetError, JudgeInProductionError) as exc:
                raise RunHaltedError(HaltReason.BUDGET, str(exc)) from exc
            except Exception as exc:
                # Triage is the run's first act. There is no partial result to
                # keep and no per-question blast radius to fall back to.
                raise RunHaltedError(HaltReason.INFRA, f"triage failed: {exc}") from exc

        if not triage.domain_match:
            raise RunHaltedError(
                HaltReason.DOMAIN_MISMATCH,
                f"this system answers cloud-migration RFPs; '{document.source_filename}' "
                f"was triaged as '{triage.detected_domain}'",
            )

        state = await self._advance(state, RunStage.EXTRACTING)
        with stage_span(RunStage.EXTRACTING.value):
            try:
                questions = await self.agents.extract(document, budget=self.ledger)
            except (RunBudgetError, JudgeInProductionError) as exc:
                raise RunHaltedError(HaltReason.BUDGET, str(exc)) from exc
            except Exception as exc:
                raise RunHaltedError(HaltReason.INFRA, f"extraction failed: {exc}") from exc

        if not questions:
            raise RunHaltedError(
                HaltReason.INFRA,
                f"no questions were extracted from '{document.source_filename}'; "
                "there is nothing to answer",
            )
        return questions

    async def _fan_out(
        self,
        document: RFPDocument,
        questions: list[ExtractedQuestion],
        state: RunState,
        resumed: ResumableRun | None,
    ) -> list[QuestionOutcome]:
        """Per-question work, concurrent under the configured semaphore.

        `return_exceptions=True` so one task raising cannot cancel the others
        mid-flight and lose their checkpoints. The run-level halt is then
        re-raised AFTER every task has settled, which is what "partials saved"
        actually requires.
        """
        await self._advance(state, RunStage.DRAFTING)
        semaphore = asyncio.Semaphore(self.limits.concurrency.question_fanout)

        with stage_span(RunStage.DRAFTING.value):
            results = await asyncio.gather(
                *(
                    self._run_question(document, question, state, semaphore, resumed)
                    for question in questions
                ),
                return_exceptions=True,
            )

        outcomes: list[QuestionOutcome] = []
        halt: RunHaltedError | None = None
        for question, result in zip(questions, results, strict=True):
            if isinstance(result, RunHaltedError):
                # Keep the FIRST halt: it is the one that describes why the
                # ceiling was crossed, and later tasks merely observed the
                # same exhausted budget.
                halt = halt or result
                outcomes.append(
                    QuestionOutcome(
                        question=question,
                        escalation=self._escalate(
                            question,
                            EscalationTrigger.STAGE_ERROR,
                            "the run halted before this question was answered",
                        ),
                    )
                )
            elif isinstance(result, BaseException):
                raise result
            else:
                outcomes.append(result)

        if halt is not None:
            raise halt
        return outcomes

    async def _run_question(
        self,
        document: RFPDocument,
        question: ExtractedQuestion,
        state: RunState,
        semaphore: asyncio.Semaphore,
        resumed: ResumableRun | None,
    ) -> QuestionOutcome:
        """One question, start to finish. Never raises except to halt the run."""
        if resumed is not None and question.id in resumed.answers:
            # Already terminal. Re-running it would spend the budget again to
            # produce what is already on disk.
            answer = resumed.answers[question.id]
            logger.info("run %s: question %s already complete, skipping", state.run_id, question.id)
            return self._outcome_from_existing(question, answer)

        # The span is INSIDE the semaphore, so its duration is the question's
        # work rather than its work plus however long it waited for a slot. A
        # span that included the wait would make every question after the first
        # five look slow in a way no code change could fix.
        async with semaphore:
            with question_span(question.id):
                return await self._guarded_pipeline(document, question, state)

    async def _guarded_pipeline(
        self, document: RFPDocument, question: ExtractedQuestion, state: RunState
    ) -> QuestionOutcome:
        """The per-question blast radius, with the span already open."""
        try:
            return await self._pipeline(document, question, state)
        except RunHaltedError:
            raise
        except (RunBudgetError, JudgeInProductionError) as exc:
            raise RunHaltedError(HaltReason.BUDGET, str(exc)) from exc
        except QuestionBudgetError as exc:
            record_failure(exc)
            return await self._escalated(
                question, state, EscalationTrigger.VALIDATION_FAILED, str(exc)
            )
        except Exception as exc:
            record_failure(exc)
            logger.warning(
                "run %s: question %s failed (%s: %s)",
                state.run_id,
                question.id,
                type(exc).__name__,
                exc,
            )
            return await self._escalated(
                question,
                state,
                EscalationTrigger.STAGE_ERROR,
                f"{type(exc).__name__}: {exc}",
            )

    async def _pipeline(
        self, document: RFPDocument, question: ExtractedQuestion, state: RunState
    ) -> QuestionOutcome:
        await self._set_status(state, question.id, QuestionStatus.PENDING)

        # Escalate-first. A question the sanitizer has flagged is going to a
        # human whatever the pipeline produces, so producing anything for it is
        # spending a rerank, a draft and a critique to reach a conclusion
        # already reached.
        screened = self.guardrails.screen(question)
        if not screened.passed:
            assert screened.trigger is not None  # noqa: S101 - the contract guarantees it
            return await self._escalated(
                question,
                state,
                screened.trigger,
                screened.reason or "",
                detail=screened.detail,
            )

        selection = await self.agents.retrieve(
            question,
            requesting_customer=document.customer_name,
            budget=self.ledger,
        )
        if selection.status is RetrievalStatus.NO_MATCH:
            # Correct behaviour, not a failure: the corpus does not cover this
            # question, and inventing an answer is the thing being prevented.
            return await self._escalated(
                question,
                state,
                EscalationTrigger.NO_MATCH,
                f"retrieval cleared no candidate over the relevance floor "
                f"({selection.candidates_considered} candidate(s) considered)",
            )

        answer = await self._draft_with_one_retry(question, selection)
        await self._save_draft(state, answer)
        await self._set_status(state, question.id, QuestionStatus.DRAFTED)

        critique = await self.agents.critique(question, answer, budget=self.ledger)
        await self._set_status(state, question.id, QuestionStatus.CRITIQUED)
        answer = self._apply_critique(answer, critique)
        await self._save_draft(state, answer)

        verdict = await self.guardrails.apply(
            question=question, answer=answer, selection=selection, document=document
        )
        if not verdict.passed:
            assert verdict.trigger is not None  # noqa: S101 - the contract guarantees it
            return await self._escalated(
                question,
                state,
                verdict.trigger,
                verdict.reason or "",
                detail=verdict.detail,
                answer=answer,
            )

        if answer.needs_sme_review:
            return await self._escalated(
                question,
                state,
                EscalationTrigger.LOW_CONFIDENCE,
                answer.escalation_reason or "confidence below the escalation threshold",
                answer=answer,
            )

        await self._set_status(state, question.id, QuestionStatus.COMPLETE)
        return QuestionOutcome(question=question, answer=answer)

    async def _draft_with_one_retry(
        self, question: ExtractedQuestion, selection: RetrieverSelection
    ) -> DraftedAnswer:
        """One draft call, plus at most one retry carrying the validation error.

        Only an `AgentValidationError` earns the retry. Anything else propagates
        and escalates the question — repeating a call that failed for reasons
        outside the model's control spends the budget to fail twice.
        """
        try:
            return await self.agents.draft(question, selection, budget=self.ledger)
        except AgentValidationError as first:
            self.ledger.authorize_retry(question.id)
            logger.info("question %s: retrying the draft with the validation error", question.id)
            return await self.agents.draft(
                question, selection, budget=self.ledger, retry_context=str(first)
            )

    def _apply_critique(self, answer: DraftedAnswer, critique: CritiqueResult) -> DraftedAnswer:
        """Fold the critic's delta into a recomputed confidence.

        The critic may only lower confidence or add flags (rule 14), and the
        bound lives on `CritiqueResult.confidence_delta` rather than in prose
        here. If the new confidence falls below the floor, `DraftedAnswer`
        refuses to validate without `needs_sme_review` — so this sets it, and the
        contract is what makes the escalation compulsory.
        """
        delta = critique.confidence_delta
        added = list(critique.added_unsupported_claims)
        issues = list(critique.issues)

        unsupported = [*answer.unsupported_claims, *added]
        inputs = answer.confidence_inputs
        if inputs is None:
            # Nothing measured to re-derive from. Folding the delta into the
            # clamped number is only equal to recomputing while the clamp bounds
            # are [0, 1], so this path states the assumption rather than hiding
            # it: it is reachable only for an answer read back from Postgres,
            # which is already terminal and never critiqued.
            confidence = max(0.0, min(1.0, answer.confidence + delta))
        else:
            confidence = compute_confidence(
                primary_final_score=inputs.primary_final_score,
                claims_with_sources=inputs.claims_with_sources,
                total_claims=inputs.total_claims,
                critique_delta=delta,
            )

        needs_review = (
            answer.needs_sme_review
            or bool(unsupported)
            or not answer.source_ids
            or _below_floor(confidence)
        )
        reason = answer.escalation_reason
        if needs_review and not reason:
            reason = _critique_reason(confidence, unsupported, issues)

        return DraftedAnswer(
            question_id=answer.question_id,
            answer_text=answer.answer_text,
            source_ids=list(answer.source_ids),
            confidence=confidence,
            needs_sme_review=needs_review,
            unsupported_claims=unsupported,
            escalation_reason=reason,
            confidence_inputs=inputs,
        )

    # -- the deterministic tail -------------------------------------------

    async def _check_compliance(
        self,
        questions: list[ExtractedQuestion],
        outcomes: list[QuestionOutcome],
        state: RunState,
    ) -> list[ComplianceResult]:
        await self._advance(state, RunStage.COMPLIANCE)
        with stage_span(RunStage.COMPLIANCE.value):
            by_id = {o.question.id: o for o in outcomes}
            return [
                self.compliance.check(question, _answer_for(by_id.get(question.id)))
                for question in questions
            ]

    async def _assemble(
        self,
        document: RFPDocument,
        questions: list[ExtractedQuestion],
        outcomes: list[QuestionOutcome],
        state: RunState,
    ) -> ArtifactPaths:
        await self._advance(state, RunStage.ASSEMBLING)
        with stage_span(RunStage.ASSEMBLING.value):
            answers = {o.question.id: o.answer for o in outcomes if o.answer is not None}
            written = self.assembler.assemble(
                run_id=state.run_id,
                document=document,
                questions=questions,
                answers=answers,
                escalations=self._escalations_record(state, document, outcomes),
            )
        return ArtifactPaths(
            response_docx=written.get("response_docx"),
            escalations_json=written.get("escalations_json"),
            eval_report_html=written.get("eval_report_html"),
        )

    def _escalations_record(
        self, state: RunState, document: RFPDocument, outcomes: list[QuestionOutcome]
    ) -> EscalationsRecord:
        records = [o.escalation for o in outcomes if o.escalation is not None]
        return EscalationsRecord(
            run_id=state.run_id,
            rfp_id=document.id,
            generated_at=_now(),
            escalations=sorted(records, key=lambda record: record.order),
        )

    # -- halting ----------------------------------------------------------

    async def _halt(
        self, state: RunState, halt: RunHaltedError, *, document: RFPDocument
    ) -> RunResult:
        """Record the halt with everything already produced still checkpointed."""
        logger.error("run %s halted (%s): %s", state.run_id, halt.reason, halt.detail)
        halted = state.model_copy(
            update={
                "stage": RunStage.HALTED,
                "halted_reason": halt.reason,
                "updated_at": _now(),
                "tokens_used": self.ledger.tokens_used,
                "cost_usd": self.ledger.cost_usd,
            }
        )
        await self._checkpoint(halted)
        return RunResult(
            run_id=state.run_id,
            answers=[],
            compliance=[],
            totals=RunTotals(
                questions=len(state.per_question_status),
                answered=0,
                escalated=0,
                failed=0,
                tokens_used=self.ledger.tokens_used,
                cost_usd=self.ledger.cost_usd,
            ),
            artifact_paths=ArtifactPaths(),
        )

    # -- checkpointing ----------------------------------------------------

    async def _advance(self, state: RunState, stage: RunStage) -> RunState:
        """Move to `stage` and checkpoint. Called at every transition (§13).

        The stage SPAN is opened separately, in `_staged`, because a span is a
        duration and this is an instant. Emitting one here would produce seven
        zero-length spans that record the transitions and none of the work.
        """
        state.stage = stage
        state.updated_at = _now()
        state.tokens_used = self.ledger.tokens_used
        state.cost_usd = self.ledger.cost_usd
        await self._checkpoint(state)
        return state

    async def _checkpoint(self, state: RunState) -> None:
        await self.checkpointer.save_run(state, client=self.client)

    async def _set_status(self, state: RunState, question_id: str, status: QuestionStatus) -> None:
        state.per_question_status[question_id] = status
        await self.checkpointer.save_question_status(
            run_id=state.run_id,
            question_id=question_id,
            status=status,
            client=self.client,
        )

    async def _save_draft(self, state: RunState, answer: DraftedAnswer) -> None:
        await self.checkpointer.save_draft(run_id=state.run_id, answer=answer, client=self.client)

    # -- escalation -------------------------------------------------------

    async def _escalated(
        self,
        question: ExtractedQuestion,
        state: RunState,
        trigger: EscalationTrigger,
        reason: str,
        *,
        detail: str | None = None,
        answer: DraftedAnswer | None = None,
    ) -> QuestionOutcome:
        await self._set_status(state, question.id, QuestionStatus.ESCALATED)
        return QuestionOutcome(
            question=question,
            escalation=self._escalate(question, trigger, reason, detail=detail, answer=answer),
        )

    def _escalate(
        self,
        question: ExtractedQuestion,
        trigger: EscalationTrigger,
        reason: str,
        *,
        detail: str | None = None,
        answer: DraftedAnswer | None = None,
    ) -> EscalationRecord:
        """Build the record. SME resolution happens at assembly (1d), where the
        graph session lives."""
        return EscalationRecord(
            question_id=question.id,
            printed_number=question.printed_number,
            order=question.order,
            question_text=question.text,
            trigger=trigger,
            reason=reason or f"escalated by {trigger}",
            detail=detail,
            confidence=answer.confidence if answer else None,
            draft_text=answer.answer_text if answer else None,
        )

    def _outcome_from_existing(
        self, question: ExtractedQuestion, answer: DraftedAnswer
    ) -> QuestionOutcome:
        """Rebuild an outcome from a checkpointed answer, without re-running it."""
        if not answer.needs_sme_review:
            return QuestionOutcome(question=question, answer=answer)
        return QuestionOutcome(
            question=question,
            escalation=self._escalate(
                question,
                EscalationTrigger.LOW_CONFIDENCE,
                answer.escalation_reason or "escalated in a previous attempt at this run",
                answer=answer,
            ),
        )

    # -- reporting --------------------------------------------------------

    def _totals(
        self, questions: list[ExtractedQuestion], outcomes: list[QuestionOutcome]
    ) -> RunTotals:
        answered = sum(1 for o in outcomes if o.answer is not None)
        escalated = sum(1 for o in outcomes if o.escalated)
        return RunTotals(
            questions=len(questions),
            answered=answered,
            escalated=escalated,
            failed=len(questions) - answered - escalated,
            tokens_used=self.ledger.tokens_used,
            cost_usd=self.ledger.cost_usd,
        )


def _now() -> datetime:
    return datetime.now(UTC)


def _below_floor(confidence: float) -> bool:
    from src.retrieval.confidence import requires_sme_review

    return requires_sme_review(confidence)


def _critique_reason(confidence: float, unsupported: list[str], issues: list[str]) -> str:
    if unsupported:
        return f"the critic found {len(unsupported)} unsupported claim(s): {unsupported[0]}"
    if _below_floor(confidence):
        return f"computed confidence {confidence:.2f} is below the escalation threshold"
    return issues[0] if issues else "flagged by the critic"


def _answer_for(outcome: QuestionOutcome | None) -> DraftedAnswer | None:
    return outcome.answer if outcome else None
