"""The controller's spine: stage order, blast radius, budgets, checkpointing.

Run against a stub agent layer, which is the point of building the controller
before the agents. Every assertion here is about the CONTROLLER's behaviour —
what it does when a question fails, when a ceiling is crossed, when a validation
error arrives — and none of it needs a model to be true.
"""

from __future__ import annotations

import pytest

from src.contracts import (
    EscalationRecord,
    EscalationTrigger,
    GuardrailVerdict,
    HaltReason,
    QuestionStatus,
    RetrievalStatus,
    RunStage,
)
from src.controller import BudgetLedger, RunController
from src.controller.limits import limits_config
from tests.stubs import (
    RecordingCheckpointer,
    StubAgentLayer,
    StubAssembler,
    StubCompliance,
    StubGuardrails,
    make_document,
    make_questions,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def escalations_of(assembler: StubAssembler) -> list[EscalationRecord]:
    """The escalations the assembler was handed, asserting it was called.

    The assertion is the point: `last_escalations` is None until `assemble`
    runs, so reading `.escalations` off it directly would raise an
    AttributeError on a run that halted — a confusing failure for a test whose
    actual finding is "assembly never happened".
    """
    assert assembler.last_escalations is not None, "the assembler was never called"
    return list(assembler.last_escalations.escalations)


def build(
    agents: StubAgentLayer | None = None,
    *,
    guardrails: StubGuardrails | None = None,
    ledger: BudgetLedger | None = None,
) -> tuple[RunController, RecordingCheckpointer, StubAssembler]:
    checkpointer = RecordingCheckpointer()
    assembler = StubAssembler()
    controller = RunController(
        agents=agents or StubAgentLayer(),
        guardrails=guardrails or StubGuardrails(),
        compliance=StubCompliance(),
        assembler=assembler,
        checkpointer=checkpointer,
        ledger=ledger or BudgetLedger(),
    )
    return controller, checkpointer, assembler


class TestStageOrder:
    async def test_the_stages_run_in_the_fixed_order(self) -> None:
        """Rule 7: the order is Python, not a supervisor's decision."""
        controller, checkpointer, _ = build()
        await controller.run(make_document(), run_id="run-1")
        assert checkpointer.stages == [
            RunStage.CREATED.value,
            RunStage.PARSING.value,
            RunStage.EXTRACTING.value,
            RunStage.DRAFTING.value,
            RunStage.COMPLIANCE.value,
            RunStage.ASSEMBLING.value,
            RunStage.COMPLETE.value,
        ]

    async def test_every_transition_is_checkpointed(self) -> None:
        """§13: a checkpoint at every stage transition, not only at the end."""
        controller, checkpointer, _ = build()
        await controller.run(make_document(), run_id="run-1")
        assert len(checkpointer.runs) == 7

    async def test_each_question_reaches_complete(self) -> None:
        controller, checkpointer, _ = build()
        result = await controller.run(make_document(), run_id="run-1")
        assert result.totals.answered == 3
        assert result.totals.escalated == 0
        final = dict(checkpointer.statuses)
        assert set(final.values()) == {QuestionStatus.COMPLETE}

    async def test_per_question_status_changes_are_checkpointed(self) -> None:
        """PENDING -> DRAFTED -> CRITIQUED -> COMPLETE, per question."""
        controller, checkpointer, _ = build(StubAgentLayer(questions=make_questions(1)))
        await controller.run(make_document(), run_id="run-1")
        assert [status for _, status in checkpointer.statuses] == [
            QuestionStatus.PENDING,
            QuestionStatus.DRAFTED,
            QuestionStatus.CRITIQUED,
            QuestionStatus.COMPLETE,
        ]


class TestTheRunHaltsOnlyForThreeReasons:
    async def test_a_domain_mismatch_halts_readably(self) -> None:
        agents = StubAgentLayer(domain_match=False, detected_domain="payroll_outsourcing")
        controller, checkpointer, _ = build(agents)
        await controller.run(make_document(), run_id="run-1")
        halted = checkpointer.runs[-1]
        assert halted.stage is RunStage.HALTED
        assert halted.halted_reason is HaltReason.DOMAIN_MISMATCH

    async def test_a_domain_mismatch_never_reaches_extraction(self) -> None:
        """Halting after doing the work is not halting."""
        agents = StubAgentLayer(domain_match=False, detected_domain="payroll_outsourcing")
        controller, _, assembler = build(agents)
        await controller.run(make_document(), run_id="run-1")
        assert agents.calls["extract"] == 0
        assert assembler.calls == 0

    async def test_triage_failing_halts_as_infra(self) -> None:
        """There is no partial result to keep and no per-question fallback."""
        controller, checkpointer, _ = build(StubAgentLayer(triage_raises=True))
        await controller.run(make_document(), run_id="run-1")
        assert checkpointer.runs[-1].halted_reason is HaltReason.INFRA

    async def test_extraction_failing_halts_as_infra(self) -> None:
        controller, checkpointer, _ = build(StubAgentLayer(extract_raises=True))
        await controller.run(make_document(), run_id="run-1")
        assert checkpointer.runs[-1].halted_reason is HaltReason.INFRA

    async def test_a_document_with_no_questions_halts(self) -> None:
        """Nothing to answer is a halt, not a run that succeeds vacuously."""
        controller, checkpointer, _ = build(StubAgentLayer(questions=[]))
        await controller.run(make_document(), run_id="run-1")
        assert checkpointer.runs[-1].halted_reason is HaltReason.INFRA


class TestOneQuestionFailingNeverHaltsTheRun:
    async def test_a_retrieval_error_escalates_only_that_question(self) -> None:
        agents = StubAgentLayer(raise_on_retrieve={"GQ-002"})
        controller, _, assembler = build(agents)
        result = await controller.run(make_document(), run_id="run-1")
        assert result.totals.answered == 2
        assert result.totals.escalated == 1
        assert assembler.calls == 1

    async def test_the_escalation_names_the_cause(self) -> None:
        agents = StubAgentLayer(raise_on_retrieve={"GQ-002"})
        controller, _, assembler = build(agents)
        await controller.run(make_document(), run_id="run-1")
        record = escalations_of(assembler)[0]
        assert record.trigger is EscalationTrigger.STAGE_ERROR
        assert "graph session died" in record.reason

    async def test_no_match_escalates_rather_than_inventing_an_answer(self) -> None:
        """NO_MATCH is the correct outcome for an uncovered question."""
        agents = StubAgentLayer(retrieval_status={"GQ-003": RetrievalStatus.NO_MATCH})
        controller, _, assembler = build(agents)
        result = await controller.run(make_document(), run_id="run-1")
        assert result.totals.escalated == 1
        assert escalations_of(assembler)[0].trigger is EscalationTrigger.NO_MATCH

    async def test_no_match_spends_no_draft_call(self) -> None:
        """Escalate-first: there is nothing to draft from."""
        agents = StubAgentLayer(retrieval_status={"GQ-003": RetrievalStatus.NO_MATCH})
        controller, _, _ = build(agents)
        await controller.run(make_document(), run_id="run-1")
        assert agents.calls["draft"] == 2

    async def test_a_guardrail_hit_escalates_with_its_trigger(self) -> None:
        guardrails = StubGuardrails(
            verdicts={
                "GQ-002": GuardrailVerdict(
                    passed=False,
                    trigger=EscalationTrigger.PRICING_BLOCKED,
                    reason="draft quoted a rate card",
                    detail="$1,200 per workload",
                )
            }
        )
        controller, _, assembler = build(guardrails=guardrails)
        result = await controller.run(make_document(), run_id="run-1")
        assert result.totals.escalated == 1
        record = escalations_of(assembler)[0]
        assert record.trigger is EscalationTrigger.PRICING_BLOCKED
        assert record.detail == "$1,200 per workload"

    async def test_an_escalated_question_keeps_its_draft_for_the_sme(self) -> None:
        """An SME starting from a flagged draft is doing review; one starting
        from nothing is doing the whole question."""
        guardrails = StubGuardrails(
            verdicts={
                "GQ-001": GuardrailVerdict(
                    passed=False,
                    trigger=EscalationTrigger.LEGAL_TERM,
                    reason="offered a warranty",
                )
            }
        )
        controller, _, assembler = build(guardrails=guardrails)
        await controller.run(make_document(), run_id="run-1")
        assert escalations_of(assembler)[0].draft_text is not None


class TestTheValidationRetry:
    async def test_one_retry_is_allowed_and_carries_the_error(self) -> None:
        agents = StubAgentLayer(fail_draft_validation_once={"GQ-002"})
        controller, _, _ = build(agents)
        result = await controller.run(make_document(), run_id="run-1")
        assert result.totals.answered == 3
        assert "source_ids must be a list" in agents.draft_retry_contexts["GQ-002"]

    async def test_a_second_failure_escalates_that_question(self) -> None:
        agents = StubAgentLayer(fail_draft_validation_always={"GQ-002"})
        controller, _, assembler = build(agents)
        result = await controller.run(make_document(), run_id="run-1")
        assert result.totals.answered == 2
        assert escalations_of(assembler)[0].question_id == "GQ-002"

    async def test_the_retry_is_never_taken_twice(self) -> None:
        """The budget permits one. Two drafts for the failing question, no more."""
        agents = StubAgentLayer(fail_draft_validation_always={"GQ-002"})
        ledger = BudgetLedger()
        controller, _, _ = build(agents, ledger=ledger)
        await controller.run(make_document(), run_id="run-1")
        assert ledger.retries_for("GQ-002") == 1


class TestTheCriticCanOnlyWeaken:
    async def test_a_negative_delta_lowers_the_computed_confidence(self) -> None:
        agents = StubAgentLayer(
            questions=make_questions(1),
            confidence={"GQ-001": 0.90},
            critique_delta={"GQ-001": -0.10},
        )
        controller, checkpointer, _ = build(agents)
        await controller.run(make_document(), run_id="run-1")
        assert checkpointer.drafts[-1].confidence == pytest.approx(0.80)

    async def test_a_delta_that_crosses_the_floor_forces_an_escalation(self) -> None:
        """The contract refuses a sub-floor answer without needs_sme_review, so
        the escalation is compulsory rather than advisory."""
        agents = StubAgentLayer(
            questions=make_questions(1),
            confidence={"GQ-001": 0.65},
            critique_delta={"GQ-001": -0.20},
        )
        controller, _, assembler = build(agents)
        result = await controller.run(make_document(), run_id="run-1")
        assert result.totals.escalated == 1
        assert escalations_of(assembler)[0].trigger is EscalationTrigger.LOW_CONFIDENCE

    async def test_confidence_is_recomputed_not_folded_into_a_clamped_number(self) -> None:
        """Recomputed from the drafter's measured terms, so the arithmetic is
        exact regardless of what the clamp bounds are set to."""
        agents = StubAgentLayer(
            questions=make_questions(1),
            confidence={"GQ-001": 0.50},
            critique_delta={"GQ-001": -0.30},
        )
        controller, checkpointer, _ = build(agents)
        await controller.run(make_document(), run_id="run-1")
        assert checkpointer.drafts[-1].confidence == pytest.approx(0.20)


class TestBudgetsHaltTheRun:
    async def test_a_token_ceiling_halts_with_partials_saved(self) -> None:
        limits = limits_config()
        per_call = limits.budget.per_run.max_tokens // 2
        agents = StubAgentLayer(questions=make_questions(6), tokens_per_call=per_call)
        controller, checkpointer, _ = build(agents)
        await controller.run(make_document(), run_id="run-1")
        halted = checkpointer.runs[-1]
        assert halted.stage is RunStage.HALTED
        assert halted.halted_reason is HaltReason.BUDGET

    async def test_the_halt_records_what_had_been_spent(self) -> None:
        limits = limits_config()
        agents = StubAgentLayer(
            questions=make_questions(6), tokens_per_call=limits.budget.per_run.max_tokens // 2
        )
        controller, checkpointer, _ = build(agents)
        await controller.run(make_document(), run_id="run-1")
        assert checkpointer.runs[-1].tokens_used > 0

    async def test_drafts_produced_before_the_ceiling_were_checkpointed(self) -> None:
        """'Partials saved' means saved, not merely counted."""
        limits = limits_config()
        agents = StubAgentLayer(
            questions=make_questions(6), tokens_per_call=limits.budget.per_run.max_tokens // 3
        )
        controller, checkpointer, _ = build(agents)
        await controller.run(make_document(), run_id="run-1")
        assert checkpointer.drafts

    async def test_a_halted_run_assembles_nothing(self) -> None:
        limits = limits_config()
        agents = StubAgentLayer(
            questions=make_questions(6), tokens_per_call=limits.budget.per_run.max_tokens // 2
        )
        controller, _, assembler = build(agents)
        await controller.run(make_document(), run_id="run-1")
        assert assembler.calls == 0


class TestTheFanOut:
    async def test_every_question_is_attempted(self) -> None:
        agents = StubAgentLayer(questions=make_questions(12))
        controller, _, _ = build(agents)
        result = await controller.run(make_document(), run_id="run-1")
        assert result.totals.questions == 12
        assert agents.calls["retrieve"] == 12

    async def test_compliance_covers_every_question_including_escalated_ones(self) -> None:
        """A mandatory question that escalated is still a compliance fact."""
        agents = StubAgentLayer(retrieval_status={"GQ-002": RetrievalStatus.NO_MATCH})
        compliance = StubCompliance()
        checkpointer = RecordingCheckpointer()
        controller = RunController(
            agents=agents,
            guardrails=StubGuardrails(),
            compliance=compliance,
            assembler=StubAssembler(),
            checkpointer=checkpointer,
        )
        await controller.run(make_document(), run_id="run-1")
        assert compliance.checked == ["GQ-001", "GQ-002", "GQ-003"]

    async def test_escalations_reach_the_assembler_in_document_order(self) -> None:
        agents = StubAgentLayer(
            questions=make_questions(4),
            retrieval_status={
                "GQ-004": RetrievalStatus.NO_MATCH,
                "GQ-002": RetrievalStatus.NO_MATCH,
            },
        )
        controller, _, assembler = build(agents)
        await controller.run(make_document(), run_id="run-1")
        assert [r.order for r in escalations_of(assembler)] == [1, 3]
