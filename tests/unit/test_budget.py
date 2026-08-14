"""The call ledger. One question per test.

The hard budget is CLAUDE.md rule 14, and it is the rule most likely to be
quietly broken by a later change — a retry loop added for robustness, a second
critique pass added for quality. These assert the ceilings against the shipped
`config/limits.yaml`, so a change to that file that widens them fails here.
"""

from __future__ import annotations

import pytest

from src.controller.budget import (
    BudgetLedger,
    CallKind,
    CallUsage,
    JudgeInProductionError,
    QuestionBudgetError,
    RunBudgetError,
)
from src.controller.limits import limits_config


class TestThePerQuestionBudget:
    def test_one_rerank_one_draft_one_critique_are_allowed(self) -> None:
        ledger = BudgetLedger()
        ledger.authorize(CallKind.RERANK, question_id="GQ-001")
        ledger.authorize(CallKind.DRAFT, question_id="GQ-001")
        ledger.authorize(CallKind.CRITIQUE, question_id="GQ-001")
        assert sum(ledger.calls_for("GQ-001").values()) == 3

    @pytest.mark.parametrize("kind", [CallKind.RERANK, CallKind.DRAFT, CallKind.CRITIQUE])
    def test_a_second_call_of_the_same_kind_is_refused(self, kind: CallKind) -> None:
        ledger = BudgetLedger()
        ledger.authorize(kind, question_id="GQ-001")
        with pytest.raises(QuestionBudgetError):
            ledger.authorize(kind, question_id="GQ-001")

    def test_one_question_exhausting_its_allowance_does_not_touch_another(self) -> None:
        """The counters are per question, which is what lets one question
        escalate while the other thirty-nine proceed."""
        ledger = BudgetLedger()
        ledger.authorize(CallKind.DRAFT, question_id="GQ-001")
        with pytest.raises(QuestionBudgetError):
            ledger.authorize(CallKind.DRAFT, question_id="GQ-001")
        ledger.authorize(CallKind.DRAFT, question_id="GQ-002")
        assert ledger.calls_for("GQ-002")[CallKind.DRAFT] == 1

    def test_a_question_scoped_call_without_a_question_id_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="budgeted per question"):
            BudgetLedger().authorize(CallKind.DRAFT)

    def test_the_shipped_config_permits_exactly_one_of_each(self) -> None:
        """Asserted against the file, so widening it fails here rather than
        silently doubling the per-question spend."""
        per_question = limits_config().budget.per_question
        assert per_question.rerank_calls == 1
        assert per_question.draft_calls == 1
        assert per_question.critique_calls == 1
        assert per_question.validation_retries == 1


class TestTheValidationRetry:
    def test_one_retry_is_permitted(self) -> None:
        ledger = BudgetLedger()
        ledger.authorize_retry("GQ-001")
        assert ledger.retries_for("GQ-001") == 1

    def test_a_second_retry_is_refused(self) -> None:
        ledger = BudgetLedger()
        ledger.authorize_retry("GQ-001")
        with pytest.raises(QuestionBudgetError, match="validation retry"):
            ledger.authorize_retry("GQ-001")

    def test_the_retry_is_not_charged_against_the_draft_allowance(self) -> None:
        """Charging it there would make the single permitted retry impossible,
        since the draft allowance is already spent by the attempt that failed."""
        ledger = BudgetLedger()
        ledger.authorize(CallKind.DRAFT, question_id="GQ-001")
        ledger.authorize_retry("GQ-001")
        assert ledger.calls_for("GQ-001")[CallKind.DRAFT] == 1


class TestThePerRunBudget:
    def test_one_triage_is_allowed(self) -> None:
        ledger = BudgetLedger()
        ledger.authorize(CallKind.TRIAGE)
        with pytest.raises(RunBudgetError, match="triage"):
            ledger.authorize(CallKind.TRIAGE)

    def test_one_extract_assist_is_allowed(self) -> None:
        ledger = BudgetLedger()
        ledger.authorize(CallKind.EXTRACT_ASSIST)
        with pytest.raises(RunBudgetError, match="extract_assist"):
            ledger.authorize(CallKind.EXTRACT_ASSIST)

    def test_the_extract_assist_is_optional(self) -> None:
        """`<= 1`, not `== 1`: deterministic parsing owns the fields and the
        assist is only consulted on ambiguous segments."""
        assert limits_config().budget.per_run_calls.extract_assist == 1
        assert BudgetLedger().run_calls()[CallKind.EXTRACT_ASSIST] == 0


class TestTheJudgeIsRefusedInProduction:
    def test_a_production_run_cannot_authorize_the_judge(self) -> None:
        """D16. Not a budget overrun — there is no allowance to spend."""
        with pytest.raises(JudgeInProductionError):
            BudgetLedger().authorize(CallKind.JUDGE, question_id="GQ-001")

    def test_an_eval_run_may(self) -> None:
        ledger = BudgetLedger(allow_judge=True)
        ledger.authorize(CallKind.JUDGE, question_id="GQ-001")
        assert ledger.run_calls()[CallKind.JUDGE] == 1

    def test_an_eval_run_may_judge_every_answer(self) -> None:
        """A per-question allowance would cap how much of a run can be measured."""
        ledger = BudgetLedger(allow_judge=True)
        for index in range(40):
            ledger.authorize(CallKind.JUDGE, question_id=f"GQ-{index:03d}")
        assert ledger.run_calls()[CallKind.JUDGE] == 40

    def test_the_refusal_precedes_the_ceiling_check(self) -> None:
        """A judge call in production is a wiring bug whether or not there is
        budget left, so it is refused as one rather than as an overrun."""
        ledger = BudgetLedger()
        ledger.record(CallUsage(tokens=ledger.max_tokens + 1))
        with pytest.raises(JudgeInProductionError):
            ledger.authorize(CallKind.JUDGE, question_id="GQ-001")


class TestTheRunCeilings:
    def test_a_run_under_both_ceilings_authorizes_normally(self) -> None:
        ledger = BudgetLedger()
        ledger.record(CallUsage(tokens=10, cost_usd=0.01))
        ledger.authorize(CallKind.DRAFT, question_id="GQ-001")
        assert ledger.run_ceiling_breached() is None

    def test_crossing_the_token_ceiling_halts_the_run(self) -> None:
        ledger = BudgetLedger()
        ledger.record(CallUsage(tokens=ledger.max_tokens))
        with pytest.raises(RunBudgetError, match="token ceiling"):
            ledger.authorize(CallKind.DRAFT, question_id="GQ-001")

    def test_crossing_the_cost_ceiling_halts_the_run(self) -> None:
        ledger = BudgetLedger()
        ledger.record(CallUsage(tokens=1, cost_usd=ledger.max_usd))
        with pytest.raises(RunBudgetError, match="cost ceiling"):
            ledger.authorize(CallKind.DRAFT, question_id="GQ-001")

    def test_the_ceiling_is_checked_for_every_kind_of_call(self) -> None:
        """Not only for the expensive ones: a run that has spent its budget
        stops, rather than making one more call of whichever kind had allowance."""
        ledger = BudgetLedger()
        ledger.record(CallUsage(tokens=ledger.max_tokens))
        with pytest.raises(RunBudgetError, match="token ceiling"):
            ledger.authorize(CallKind.TRIAGE)


class TestUnpricedCallsAreNotFreeCalls:
    def test_a_missing_cost_is_recorded_as_unknown_not_as_zero(self) -> None:
        ledger = BudgetLedger()
        ledger.record(CallUsage(tokens=100, cost_usd=None))
        assert ledger.cost_usd == 0.0
        assert ledger.calls_with_unknown_cost == 1
        assert ledger.cost_is_complete is False

    def test_a_fully_priced_run_reports_a_complete_cost(self) -> None:
        ledger = BudgetLedger()
        ledger.record(CallUsage(tokens=100, cost_usd=0.02))
        assert ledger.cost_is_complete is True

    def test_the_breach_message_says_the_total_is_a_floor(self) -> None:
        """A USD total that omits unpriced calls is a floor, and a report that
        presents it as a total understates what the run cost."""
        ledger = BudgetLedger()
        ledger.record(CallUsage(tokens=1, cost_usd=None))
        ledger.record(CallUsage(tokens=1, cost_usd=ledger.max_usd))
        breach = ledger.run_ceiling_breached()
        assert breach is not None
        assert "measured floor" in breach

    def test_an_underpriced_run_still_breaches_when_the_floor_crosses(self) -> None:
        """An under-count that exceeds the limit exceeded it."""
        ledger = BudgetLedger()
        ledger.record(CallUsage(tokens=1, cost_usd=None))
        ledger.record(CallUsage(tokens=1, cost_usd=ledger.max_usd))
        with pytest.raises(RunBudgetError):
            ledger.authorize(CallKind.DRAFT, question_id="GQ-001")


class TestRecordingNeverRefuses:
    def test_a_call_that_already_happened_is_always_booked(self) -> None:
        """Refusing to record would leave the ledger reporting less than was
        spent. The refusal belongs in the next authorize."""
        ledger = BudgetLedger()
        ledger.record(CallUsage(tokens=ledger.max_tokens * 10, cost_usd=999.0))
        assert ledger.tokens_used == ledger.max_tokens * 10
        assert ledger.cost_usd == 999.0
