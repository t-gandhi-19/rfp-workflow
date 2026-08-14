"""The call ledger. Every model call in a run is authorized here first.

WHY AUTHORIZATION IS SEPARATE FROM RECORDING. `authorize` runs BEFORE the call
and can refuse it; `record` runs after and books what it cost. Collapsing them
into one "spend" would mean the ceiling is only ever checked after the money is
gone, which is the difference between a budget and a receipt.

WHY THE CONTROLLER HOLDS THIS AND NOT THE AGENT FRAMEWORK. Rule 14. A budget
enforced by asking crewAI to please make one call is not enforced. Every call
site in `src/controller/` passes through here, and the counters are per-question
so one question exhausting its allowance cannot consume another's.

THE TWO FAILURE MODES ARE DIFFERENT AND THE EXCEPTIONS SAY SO.
`QuestionBudgetError` escalates ONE question and the run continues.
`RunBudgetError` halts the WHOLE run as halted(budget), with everything
already checkpointed. Catching the wrong one would either abandon a run over a
single question or keep spending past the ceiling.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from src.controller.limits import LimitsConfig, limits_config


class CallKind(StrEnum):
    """Every kind of model call a run can make.

    JUDGE is on this list precisely so it can be refused. It is an eval-only
    call (D16) and a production run that made one would be paying for a score
    nobody reads while breaching the per-question budget — enumerating it here
    turns that from an unwritten rule into a raised exception.
    """

    TRIAGE = "triage"
    EXTRACT_ASSIST = "extract_assist"
    RERANK = "rerank"
    DRAFT = "draft"
    CRITIQUE = "critique"
    JUDGE = "judge"


#: Calls counted once per RUN, and the limits.yaml field that bounds each.
RUN_SCOPED: dict[CallKind, str] = {
    CallKind.TRIAGE: "triage",
    CallKind.EXTRACT_ASSIST: "extract_assist",
}

#: Calls counted once per QUESTION, and the limits.yaml field that bounds each.
QUESTION_SCOPED: dict[CallKind, str] = {
    CallKind.RERANK: "rerank_calls",
    CallKind.DRAFT: "draft_calls",
    CallKind.CRITIQUE: "critique_calls",
}


class BudgetError(RuntimeError):
    """Base for the two, so a caller can log both and handle each."""


class QuestionBudgetError(BudgetError):
    """One question's allowance is spent. That question escalates."""


class RunBudgetError(BudgetError):
    """A run-level ceiling is spent. The run halts as halted(budget)."""


class JudgeInProductionError(BudgetError):
    """The judge was authorized in a run that is not an eval run.

    Not a budget overrun — a wiring bug. It is raised rather than counted
    because there is no allowance to spend: D16 puts the judge in eval runs and
    nowhere else, so the correct number of judge calls in a production run is
    not "few", it is none.
    """


@dataclass(frozen=True)
class CallUsage:
    """What one model call actually consumed.

    `cost_usd` is None when the gateway did not report a cost — which is a
    STATE, not a zero. Booking an unknown cost as free would let a run sail past
    a USD ceiling while the ledger insisted it had spent nothing, so the ledger
    tracks the unknowns separately and says so.
    """

    tokens: int = 0
    cost_usd: float | None = None

    @property
    def cost_is_known(self) -> bool:
        return self.cost_usd is not None


@dataclass
class BudgetLedger:
    """Per-run accounting, enforced before each call and updated after it."""

    limits: LimitsConfig = field(default_factory=limits_config)
    #: Eval runs set this. Production runs never do.
    allow_judge: bool = False

    tokens_used: int = 0
    cost_usd: float = 0.0
    #: Calls whose cost the gateway did not report. A USD ceiling cannot be
    #: honestly enforced while this is non-zero, and the run report says so
    #: rather than presenting an under-count as a total.
    calls_with_unknown_cost: int = 0

    _run_calls: Counter[CallKind] = field(default_factory=Counter)
    _question_calls: dict[str, Counter[CallKind]] = field(default_factory=dict)
    _question_retries: Counter[str] = field(default_factory=Counter)

    # -- ceilings ---------------------------------------------------------

    @property
    def max_tokens(self) -> int:
        return self.limits.budget.per_run.max_tokens

    @property
    def max_usd(self) -> float:
        return self.limits.budget.per_run.max_usd

    @property
    def cost_is_complete(self) -> bool:
        """Whether `cost_usd` accounts for every call made."""
        return self.calls_with_unknown_cost == 0

    def run_ceiling_breached(self) -> str | None:
        """Which run ceiling is spent, or None. Checked before every call.

        The token ceiling is always enforceable. The USD ceiling is enforced on
        what has been MEASURED: if some calls reported no cost, the total is a
        floor rather than a total, and crossing the ceiling on a floor is still
        a crossing — an under-count that exceeds the limit exceeded it.
        """
        if self.tokens_used >= self.max_tokens:
            return f"token ceiling reached: {self.tokens_used} >= {self.max_tokens}"
        if self.cost_usd >= self.max_usd:
            measured = "" if self.cost_is_complete else " (measured floor; some calls unpriced)"
            return f"cost ceiling reached: ${self.cost_usd:.4f} >= ${self.max_usd:.2f}{measured}"
        return None

    # -- authorization ----------------------------------------------------

    def authorize(self, kind: CallKind, *, question_id: str | None = None) -> None:
        """Permit one call of `kind`, or raise.

        Run ceilings are checked first and for every kind, so a run that has
        already spent its budget stops immediately rather than making one more
        call of whichever kind happened to have allowance left.
        """
        if kind is CallKind.JUDGE and not self.allow_judge:
            raise JudgeInProductionError(
                "judge-model was authorized in a run that is not an eval run. "
                "D16 places the judge in eval runs only; it never contributes to a "
                "drafted response."
            )

        if breached := self.run_ceiling_breached():
            raise RunBudgetError(breached)

        if kind in RUN_SCOPED:
            self._authorize_run_scoped(kind)
            return

        if kind in QUESTION_SCOPED:
            if question_id is None:
                raise ValueError(f"{kind} is budgeted per question and needs a question_id")
            self._authorize_question_scoped(kind, question_id)
            return

        # JUDGE in an eval run: allowed, counted, and bounded only by the run
        # ceilings checked above. Eval runs score every answer, so a per-question
        # allowance would be a limit on how much of the run can be measured.
        self._run_calls[kind] += 1

    def _authorize_run_scoped(self, kind: CallKind) -> None:
        allowance = getattr(self.limits.budget.per_run_calls, RUN_SCOPED[kind])
        spent = self._run_calls[kind]
        if spent >= allowance:
            raise RunBudgetError(
                f"{kind} is budgeted at {allowance} call(s) per run and {spent} have been made"
            )
        self._run_calls[kind] += 1

    def _authorize_question_scoped(self, kind: CallKind, question_id: str) -> None:
        allowance = getattr(self.limits.budget.per_question, QUESTION_SCOPED[kind])
        spent = self._question_calls.setdefault(question_id, Counter())[kind]
        if spent >= allowance:
            raise QuestionBudgetError(
                f"{kind} is budgeted at {allowance} call(s) per question and question "
                f"{question_id} has already made {spent}"
            )
        self._question_calls[question_id][kind] += 1

    def authorize_retry(self, question_id: str) -> None:
        """Permit one validation retry for `question_id`, or raise.

        Counted apart from the call kinds because the retry is a repeat of a
        call already authorized — charging it against `draft_calls` would make
        the single permitted retry impossible, and giving it no counter at all
        would make it unlimited.
        """
        allowance = self.limits.budget.per_question.validation_retries
        spent = self._question_retries[question_id]
        if spent >= allowance:
            raise QuestionBudgetError(
                f"question {question_id} has used its {allowance} validation retry/retries"
            )
        self._question_retries[question_id] += 1

    # -- recording --------------------------------------------------------

    def record(self, usage: CallUsage) -> None:
        """Book what a completed call consumed.

        Never raises. A call that has already happened has already been paid
        for, and refusing to record it would leave the ledger reporting less
        than was spent. The refusal belongs in the next `authorize`.
        """
        self.tokens_used += usage.tokens
        if usage.cost_usd is None:
            self.calls_with_unknown_cost += 1
        else:
            self.cost_usd += usage.cost_usd

    # -- reporting --------------------------------------------------------

    def calls_for(self, question_id: str) -> Counter[CallKind]:
        """What one question has spent. Copied, so a reader cannot mutate it."""
        return Counter(self._question_calls.get(question_id, Counter()))

    def run_calls(self) -> Counter[CallKind]:
        return Counter(self._run_calls)

    def retries_for(self, question_id: str) -> int:
        return self._question_retries[question_id]

    def total_calls(self) -> int:
        per_question = sum(sum(counter.values()) for counter in self._question_calls.values())
        return sum(self._run_calls.values()) + per_question + sum(self._question_retries.values())
