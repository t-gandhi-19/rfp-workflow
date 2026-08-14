"""Kill a run mid-fan-out, resume it, and prove the artifact is unchanged.

WHY THIS IS AN INTEGRATION TEST. Resume is a claim about POSTGRES: that what a
dying run left behind is enough to finish it. A stubbed checkpointer would
assert that the controller called `save_draft`, which is not the same claim and
is the one already covered in `tests/unit/test_controller.py`. This one writes
through the real write-api, reads back through the real reader identity, and
finishes the run from what actually survived.

THE KILL IS A REAL KILL. The stub raises a BaseException from inside the
fan-out, so it goes straight past the controller's per-question
`except Exception` and unwinds the run, leaving exactly what had been
checkpointed. Simulating the kill with a budget halt would have tested the halt
path instead, which is a different path with a different exit.

THE AGENT LAYER IS STUBBED, deliberately. What is under test is the controller
and the durable state, and a model in the loop would make the comparison
non-deterministic for reasons that have nothing to do with resume.
"""

from __future__ import annotations

import uuid

import pytest

from src.contracts import QuestionStatus, RunStage
from src.controller import BudgetLedger, Checkpointer, RunController, load_run
from src.controller.limits import limits_config
from tests import live
from tests.stubs import (
    SimulatedKill,
    StubAgentLayer,
    StubAssembler,
    StubCompliance,
    StubGuardrails,
    make_document,
    make_questions,
)

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

QUESTION_COUNT = 6
#: Where the kill lands. Questions before it complete and are checkpointed;
#: this one and everything after it are unfinished work for the resume.
KILLED_AT = "GQ-005"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _requires_stack() -> None:
    live.require_write_api_and_keycloak()


def sequential_limits():  # type: ignore[no-untyped-def]
    """Fan-out of one.

    Not a workaround for flakiness — a precondition of the assertion. With
    concurrency 5 the question the kill lands on depends on scheduling, so
    "questions 1-4 completed" would be true most of the time and the test would
    fail for reasons unrelated to resume. The controller's concurrency is
    covered in the unit suite; what this file is about is what survives.
    """
    limits = limits_config()
    return limits.model_copy(
        update={"concurrency": limits.concurrency.model_copy(update={"question_fanout": 1})}
    )


def build(agents: StubAgentLayer) -> tuple[RunController, StubAssembler]:
    assembler = StubAssembler()
    controller = RunController(
        agents=agents,
        guardrails=StubGuardrails(),
        compliance=StubCompliance(),
        assembler=assembler,
        checkpointer=Checkpointer(),
        ledger=BudgetLedger(),
        limits=sequential_limits(),
    )
    return controller, assembler


class TestKillAndResume:
    async def test_the_resumed_artifact_is_byte_identical(self) -> None:
        """The whole point of checkpointing, stated as one comparison."""
        document = make_document()
        questions = make_questions(QUESTION_COUNT)

        control, control_assembler = build(StubAgentLayer(questions=questions))
        await control.run(document, run_id=f"ctrl-{uuid.uuid4().hex[:12]}")

        run_id = f"kill-{uuid.uuid4().hex[:12]}"
        killed, _ = build(StubAgentLayer(questions=questions, kill_on_critique={KILLED_AT}))
        with pytest.raises(SimulatedKill):
            await killed.run(document, run_id=run_id)

        resumed_state = await load_run(run_id)
        resumer, resumed_assembler = build(StubAgentLayer(questions=questions))
        await resumer.resume(document, resumed=resumed_state)

        assert resumed_assembler.rendered == control_assembler.rendered

    async def test_the_kill_leaves_the_completed_questions_behind(self) -> None:
        document = make_document()
        questions = make_questions(QUESTION_COUNT)
        run_id = f"kill-{uuid.uuid4().hex[:12]}"

        killed, _ = build(StubAgentLayer(questions=questions, kill_on_critique={KILLED_AT}))
        with pytest.raises(SimulatedKill):
            await killed.run(document, run_id=run_id)

        state = await load_run(run_id)
        assert set(state.answers) == {"GQ-001", "GQ-002", "GQ-003", "GQ-004"}

    async def test_the_unfinished_question_is_not_treated_as_done(self) -> None:
        """It was checkpointed PENDING, which is mid-pipeline, not terminal.

        Reading it back as an answer would ship whatever the dying run happened
        to have written — a draft that was never critiqued and never passed a
        guardrail.
        """
        document = make_document()
        questions = make_questions(QUESTION_COUNT)
        run_id = f"kill-{uuid.uuid4().hex[:12]}"

        killed, _ = build(StubAgentLayer(questions=questions, kill_on_critique={KILLED_AT}))
        with pytest.raises(SimulatedKill):
            await killed.run(document, run_id=run_id)

        state = await load_run(run_id)
        assert state.state.per_question_status[KILLED_AT] is QuestionStatus.DRAFTED
        assert KILLED_AT not in state.answers

    async def test_the_resume_re_executes_only_the_unfinished_questions(self) -> None:
        """Re-running a completed question would spend the budget again to
        produce what is already on disk."""
        document = make_document()
        questions = make_questions(QUESTION_COUNT)
        run_id = f"kill-{uuid.uuid4().hex[:12]}"

        killed, _ = build(StubAgentLayer(questions=questions, kill_on_critique={KILLED_AT}))
        with pytest.raises(SimulatedKill):
            await killed.run(document, run_id=run_id)

        resumed_state = await load_run(run_id)
        agents = StubAgentLayer(questions=questions)
        resumer, _ = build(agents)
        await resumer.resume(document, resumed=resumed_state)

        # GQ-005 and GQ-006 only.
        assert agents.calls["draft"] == 2
        assert agents.calls["retrieve"] == 2

    async def test_the_resumed_run_reaches_complete(self) -> None:
        document = make_document()
        questions = make_questions(QUESTION_COUNT)
        run_id = f"kill-{uuid.uuid4().hex[:12]}"

        killed, _ = build(StubAgentLayer(questions=questions, kill_on_critique={KILLED_AT}))
        with pytest.raises(SimulatedKill):
            await killed.run(document, run_id=run_id)

        resumed_state = await load_run(run_id)
        resumer, _ = build(StubAgentLayer(questions=questions))
        result = await resumer.resume(document, resumed=resumed_state)

        assert result.totals.answered == QUESTION_COUNT
        assert (await load_run(run_id)).state.stage is RunStage.COMPLETE

    async def test_a_resumed_run_keeps_spending_against_what_it_spent(self) -> None:
        """A run killed at 90% of its ceiling must not restart with a full
        budget, or the ceiling bounds nothing."""
        document = make_document()
        questions = make_questions(QUESTION_COUNT)
        run_id = f"kill-{uuid.uuid4().hex[:12]}"

        killed, _ = build(
            StubAgentLayer(questions=questions, kill_on_critique={KILLED_AT}, tokens_per_call=1_000)
        )
        with pytest.raises(SimulatedKill):
            await killed.run(document, run_id=run_id)

        resumed_state = await load_run(run_id)
        ledger = BudgetLedger()
        resumer = RunController(
            agents=StubAgentLayer(questions=questions, tokens_per_call=1_000),
            guardrails=StubGuardrails(),
            compliance=StubCompliance(),
            assembler=StubAssembler(),
            checkpointer=Checkpointer(),
            ledger=ledger,
            limits=sequential_limits(),
        )
        await resumer.resume(document, resumed=resumed_state)
        assert ledger.tokens_used >= resumed_state.state.tokens_used


class TestUnknownRuns:
    async def test_resuming_an_unknown_run_says_so(self) -> None:
        """`make resume RUN=<id>` takes an id from a human, and the useful
        answer to a typo is an error naming it — not an empty run that appears
        to succeed instantly."""
        from src.controller import CheckpointError

        with pytest.raises(CheckpointError, match="no run"):
            await load_run("run-that-was-never-started")
