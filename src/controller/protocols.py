"""The seams the controller orchestrates across.

The controller is deterministic Python and knows the stage order. It does not
know how a draft is produced, how compliance is judged, or how a document is
assembled — those arrive in 1c and 1d. Stating the seams as Protocols lets the
spine be BUILT AND TESTED FIRST, against a stub, which is the point of building
it before the agents rather than after.

WHY THE LEDGER IS THREADED IN RATHER THAN HELD BY THE AGENT LAYER. Two of these
calls are conditional: `extract` reaches for the assist model only on ambiguous
segments, and `retrieve` skips the rerank when there is nothing to rerank. The
controller cannot authorize those eagerly without over-counting, and letting the
agent layer own its own allowance would put enforcement inside the thing being
limited (rule 14). So the ledger is the controller's, and the layer asks it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from src.contracts import (
    ComplianceResult,
    CritiqueResult,
    DraftedAnswer,
    EscalationsRecord,
    ExtractedQuestion,
    GuardrailVerdict,
    QuestionStatus,
    RetrieverSelection,
    RFPDocument,
    RunState,
    TriageResult,
)
from src.controller.budget import BudgetLedger


class AgentValidationError(RuntimeError):
    """A model's reply did not satisfy the task's `output_pydantic` contract.

    Distinct from every other failure because it is the ONE that earns a retry:
    the controller re-issues the call with this error in the context, and
    escalates the question if the second attempt fails too. A gateway timeout
    or a graph error gets no retry — repeating a call that failed for reasons
    outside the model's control is spending the budget to fail twice.
    """


@runtime_checkable
class AgentLayer(Protocol):
    """The five agents (1c), as the controller sees them.

    Every method is one budgeted step. None of them writes to Postgres, decides
    a stage transition, or knows about other questions — all three belong to the
    controller, and an agent layer that did any of them would be a supervisor by
    another name (rule 7).
    """

    async def triage(self, document: RFPDocument, *, budget: BudgetLedger) -> TriageResult:
        """Local model, no tools. A domain mismatch halts the run readably."""
        ...

    async def extract(
        self, document: RFPDocument, *, budget: BudgetLedger
    ) -> list[ExtractedQuestion]:
        """Deterministic parsing; the model assists only on ambiguous segments."""
        ...

    async def retrieve(
        self,
        question: ExtractedQuestion,
        *,
        requesting_customer: str,
        budget: BudgetLedger,
    ) -> RetrieverSelection:
        """Three-tier MCP retrieval. Selection reasons are derived, not authored."""
        ...

    async def draft(
        self,
        question: ExtractedQuestion,
        selection: RetrieverSelection,
        *,
        budget: BudgetLedger,
        retry_context: str | None = None,
    ) -> DraftedAnswer:
        """Groq 70B. Confidence on the returned answer is COMPUTED, not self-reported.

        `retry_context` carries the validation error from the previous attempt,
        so the one permitted retry is an informed second try rather than an
        identical call hoping for a different reply.
        """
        ...

    async def critique(
        self,
        question: ExtractedQuestion,
        answer: DraftedAnswer,
        *,
        budget: BudgetLedger,
    ) -> CritiqueResult:
        """Groq 70B. May only lower confidence or add flags (rule 14)."""
        ...


@runtime_checkable
class RunCheckpointer(Protocol):
    """Where run state is made durable.

    A Protocol rather than the concrete `Checkpointer` so the controller's own
    tests do not need Keycloak and write-api running to assert that a transition
    was checkpointed. The real implementation is the only one used in a run;
    this exists so that "was it checkpointed" is answerable without also asking
    "is the stack up".
    """

    timeout: float

    async def save_run(self, state: RunState, *, client: httpx.AsyncClient) -> None: ...

    async def save_question_status(
        self,
        *,
        run_id: str,
        question_id: str,
        status: QuestionStatus,
        client: httpx.AsyncClient,
    ) -> None: ...

    async def save_draft(
        self, *, run_id: str, answer: DraftedAnswer, client: httpx.AsyncClient
    ) -> None: ...


@runtime_checkable
class GuardrailSuite(Protocol):
    """The deterministic post-processors that run on every draft (1d).

    A REQUIRED collaborator with no default. A no-op default would make an
    unwired controller look like a working one — every draft passing every
    guardrail is exactly what a correct run and a completely unguarded run both
    look like from the outside.
    """

    async def apply(
        self,
        *,
        question: ExtractedQuestion,
        answer: DraftedAnswer,
        selection: RetrieverSelection,
        document: RFPDocument,
    ) -> GuardrailVerdict: ...


@runtime_checkable
class ComplianceChecker(Protocol):
    """Deterministic compliance (1d). Never a model — build prompt §12."""

    def check(
        self, question: ExtractedQuestion, answer: DraftedAnswer | None
    ) -> ComplianceResult: ...


@runtime_checkable
class Assembler(Protocol):
    """The fixed-template assembler (1d)."""

    def assemble(
        self,
        *,
        run_id: str,
        document: RFPDocument,
        questions: list[ExtractedQuestion],
        answers: dict[str, DraftedAnswer],
        escalations: EscalationsRecord,
    ) -> dict[str, str]:
        """Write the artifacts. Returns a mapping of artifact name to path."""
        ...
