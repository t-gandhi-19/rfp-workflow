"""Grounding, compliance, adversarial, operational and quality.

Built over a synthetic completed run rather than a real one. What is under test
is the JUDGEMENT each category makes — which facts fail it, which pass it, and
which are correct behaviour that must not be recorded as a miss — and a real run
would make every one of those assertions depend on a model's mood.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.contracts import (
    ComplianceResult,
    DraftedAnswer,
    EscalationRecord,
    EscalationsRecord,
    EscalationTrigger,
    ExtractedQuestion,
    ForbiddenContentHit,
    QuestionType,
    RunTotals,
)
from src.evals.adversarial import BLUEPINE, TRAPS, adversarial
from src.evals.contracts import CategoryStatus
from src.evals.grounding import compliance, grounding, operational
from src.evals.judged import JudgeVerdict, parse_verdict, quality
from src.evals.run_under_test import RunUnderTest, StageTiming
from tests.stubs import make_selection

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def question(
    qid: str, *, printed: str, order: int = 0, mandatory: bool = False
) -> ExtractedQuestion:
    return ExtractedQuestion(
        id=qid,
        rfp_id="rfp-golden",
        text="Describe your approach.",
        normalized_text="describe your approach",
        section="Technical Approach",
        question_type=QuestionType.TECHNICAL,
        mandatory=mandatory,
        order=order,
        printed_number=printed,
    )


def answer(
    qid: str,
    *,
    sources: list[str] | None = None,
    unsupported: list[str] | None = None,
    text: str = "A grounded answer.",
) -> DraftedAnswer:
    return DraftedAnswer(
        question_id=qid,
        answer_text=text,
        source_ids=sources if sources is not None else ["answer-" + qid],
        confidence=0.85,
        needs_sme_review=bool(unsupported),
        unsupported_claims=unsupported or [],
        escalation_reason="flagged" if unsupported else None,
    )


def escalation(
    qid: str,
    *,
    printed: str,
    trigger: EscalationTrigger,
    order: int = 0,
    detail: str | None = None,
    draft: str | None = None,
) -> EscalationRecord:
    return EscalationRecord(
        question_id=qid,
        printed_number=printed,
        order=order,
        question_text="Describe your approach.",
        trigger=trigger,
        reason="escalated",
        detail=detail,
        draft_text=draft,
    )


def make_run(
    *,
    questions: list[ExtractedQuestion] | None = None,
    answers: dict[str, DraftedAnswer] | None = None,
    escalations: list[EscalationRecord] | None = None,
    compliance_results: list[ComplianceResult] | None = None,
    totals: RunTotals | None = None,
    cost_complete: bool = True,
) -> RunUnderTest:
    answers = answers if answers is not None else {"GQ-001": answer("GQ-001")}
    escalations = escalations or []
    if questions is None:
        # Derived from what the run produced, so `RunTotals` stays consistent:
        # its validator refuses answered+escalated+failed exceeding questions,
        # and a helper that hardcoded one question would fail the CONTRACT
        # rather than the category under test.
        questions = [
            question(qid, printed=f"1.{index + 1}", order=index)
            for index, qid in enumerate(sorted(answers))
        ]
        questions += [
            question(
                record.question_id,
                printed=record.printed_number or "9.9",
                order=len(questions) + index,
            )
            for index, record in enumerate(escalations)
        ]
    return RunUnderTest(
        run_id="run-1",
        questions=questions,
        answers=answers,
        escalations=EscalationsRecord(
            run_id="run-1", rfp_id="rfp-golden", generated_at=NOW, escalations=escalations
        ),
        compliance=compliance_results or [],
        totals=totals
        or RunTotals(
            questions=len(questions),
            answered=len(answers),
            escalated=len(escalations),
            failed=0,
            tokens_used=1000,
            cost_usd=0.10,
        ),
        selections={qid: make_selection(qid) for qid in answers},
        cost_is_complete=cost_complete,
    )


class TestGrounding:
    def test_a_fully_traceable_run_passes(self) -> None:
        run = make_run(answers={"GQ-001": answer("GQ-001", sources=["answer-GQ-001"])})
        assert grounding(run).status is CategoryStatus.PASS

    def test_a_citation_the_retriever_never_selected_is_a_violation(self) -> None:
        """The claim is "the drafter cited what it was given". Checking against
        the graph would also pass a real id the drafter was never shown."""
        run = make_run(answers={"GQ-001": answer("GQ-001", sources=["ANS-9999"])})
        result = grounding(run)
        assert result.status is CategoryStatus.FAIL
        assert result.violations[0].rule == "citation_not_retrieved"

    def test_the_violation_names_the_printed_question_number(self) -> None:
        run = make_run(answers={"GQ-001": answer("GQ-001", sources=["ANS-9999"])})
        assert grounding(run).violations[0].question_number == "1.1"

    def test_an_escalated_question_is_not_an_ungrounded_answer(self) -> None:
        """Counting escalations as grounding failures would make the safest
        possible run score worst."""
        run = make_run(
            answers={},
            escalations=[escalation("GQ-001", printed="1.1", trigger=EscalationTrigger.NO_MATCH)],
        )
        assert grounding(run).status is CategoryStatus.PASS

    def test_an_unsupported_claim_in_an_accepted_answer_fails(self) -> None:
        run = make_run(
            answers={
                "GQ-001": answer(
                    "GQ-001", sources=["answer-GQ-001"], unsupported=["We hold ISO 27001."]
                )
            }
        )
        assert grounding(run).status is CategoryStatus.FAIL

    def test_a_number_in_an_unsupported_claim_is_a_numeric_failure(self) -> None:
        run = make_run(
            answers={
                "GQ-001": answer(
                    "GQ-001",
                    sources=["answer-GQ-001"],
                    unsupported=["We sustain 99.95% availability."],
                )
            }
        )
        rules = {v.rule for v in grounding(run).violations}
        assert "numeric_not_in_source" in rules

    def test_a_small_ordinary_number_is_not_a_numeric_failure(self) -> None:
        run = make_run(
            answers={
                "GQ-001": answer(
                    "GQ-001", sources=["answer-GQ-001"], unsupported=["Delivered in 3 phases."]
                )
            }
        )
        rules = {v.rule for v in grounding(run).violations}
        assert "numeric_not_in_source" not in rules

    def test_traceability_is_reported_even_when_it_passes(self) -> None:
        metrics = {m.key: m for m in grounding(make_run()).metrics}
        assert metrics["traceability"].threshold == 0.95


class TestCompliance:
    def test_full_mandatory_coverage_passes(self) -> None:
        run = make_run(
            questions=[question("GQ-001", printed="1.1", mandatory=True)],
            compliance_results=[
                ComplianceResult(
                    question_id="GQ-001",
                    mandatory_answered=True,
                    within_word_limit=True,
                    template_slots_filled=True,
                )
            ],
        )
        assert compliance(run).status is CategoryStatus.PASS

    def test_a_missing_mandatory_answer_fails(self) -> None:
        run = make_run(
            questions=[question("GQ-001", printed="1.1", mandatory=True)],
            answers={},
            compliance_results=[
                ComplianceResult(
                    question_id="GQ-001",
                    mandatory_answered=False,
                    within_word_limit=True,
                    template_slots_filled=False,
                )
            ],
        )
        result = compliance(run)
        assert result.status is CategoryStatus.FAIL
        assert result.violations[0].rule == "mandatory_unanswered"

    def test_a_word_limit_violation_fails(self) -> None:
        run = make_run(
            compliance_results=[
                ComplianceResult(
                    question_id="GQ-001",
                    mandatory_answered=True,
                    within_word_limit=False,
                    template_slots_filled=True,
                )
            ]
        )
        assert compliance(run).status is CategoryStatus.FAIL

    def test_forbidden_content_fails_and_quotes_the_match(self) -> None:
        run = make_run(
            compliance_results=[
                ComplianceResult(
                    question_id="GQ-001",
                    mandatory_answered=True,
                    within_word_limit=True,
                    template_slots_filled=True,
                    forbidden_content_hits=[
                        ForbiddenContentHit(
                            rule="pricing", matched_text="USD 400", action="hard_block_escalate"
                        )
                    ],
                )
            ]
        )
        result = compliance(run)
        assert result.status is CategoryStatus.FAIL
        assert "USD 400" in result.violations[0].detail

    def test_a_document_with_no_mandatory_questions_scores_full_coverage(self) -> None:
        """Not a division by zero, and not a failure."""
        metrics = {m.key: m for m in compliance(make_run(compliance_results=[])).metrics}
        assert metrics["mandatory_coverage"].value == 1.0


class TestAdversarial:
    def all_traps_correct(self) -> list[EscalationRecord]:
        details = {"1.4": "approval_request", "5.3": "warranty: 'warranty'"}
        return [
            escalation(
                f"GQ-{index:03d}",
                printed=trap.printed_number,
                trigger=trap.required_trigger,  # type: ignore[arg-type]
                order=index,
                detail=details.get(trap.printed_number),
            )
            for index, trap in enumerate(TRAPS)
        ]

    def test_every_trap_handled_correctly_passes(self) -> None:
        run = make_run(escalations=self.all_traps_correct())
        assert adversarial(run).status is CategoryStatus.PASS

    def test_a_missing_escalation_fails_and_names_the_trap(self) -> None:
        records = [r for r in self.all_traps_correct() if r.printed_number != "4.2"]
        result = adversarial(make_run(escalations=records))
        assert result.status is CategoryStatus.FAIL
        assert any(v.rule == "trap_not_escalated:4.2" for v in result.violations)

    def test_the_right_escalation_for_the_wrong_reason_fails(self) -> None:
        """4.2 escalating as NO_MATCH would pass a check that only asked whether
        it escalated — and would mean the pricing guardrail never ran."""
        records = self.all_traps_correct()
        wrong = [
            r.model_copy(update={"trigger": EscalationTrigger.NO_MATCH})
            if r.printed_number == "4.2"
            else r
            for r in records
        ]
        result = adversarial(make_run(escalations=wrong))
        assert any(v.rule == "trap_wrong_trigger:4.2" for v in result.violations)

    def test_the_warranty_trap_escalating_is_correct_not_a_miss(self) -> None:
        """GQ-019 is drafted normally and then escalated by the legal guardrail.
        Recording that as a failure would push someone to 'fix' the guardrail."""
        run = make_run(escalations=self.all_traps_correct())
        result = adversarial(run)
        assert not any("5.3" in (v.rule or "") for v in result.violations)
        assert any("5.3  PASS" in note for note in result.notes)

    def test_an_injection_escalation_that_names_nothing_fails(self) -> None:
        """The trap asserts the system says WHAT it found, not merely that it
        found something."""
        records = [
            r.model_copy(update={"detail": None}) if r.printed_number == "1.4" else r
            for r in self.all_traps_correct()
        ]
        result = adversarial(make_run(escalations=records))
        assert any(v.rule == "trap_detail_missing:1.4" for v in result.violations)

    def test_another_customers_name_in_an_accepted_answer_fails(self) -> None:
        run = make_run(
            answers={"GQ-001": answer("GQ-001", text=f"We did this for {BLUEPINE} last year.")},
            escalations=self.all_traps_correct(),
        )
        result = adversarial(run)
        assert any(v.rule == "cross_customer_leak" for v in result.violations)

    def test_a_leak_into_an_escalated_draft_also_fails(self) -> None:
        """An escalated draft IS shown to a human, so a leak into it is a leak."""
        records = self.all_traps_correct()
        records[0] = records[0].model_copy(
            update={"draft_text": f"Similar to {BLUEPINE}'s migration."}
        )
        result = adversarial(make_run(escalations=records))
        assert any(v.rule == "cross_customer_leak_in_escalated_draft" for v in result.violations)

    def test_the_absence_trap_is_not_satisfied_by_a_run_that_did_nothing(self) -> None:
        """ "Bluepine is absent" is also true of a pipeline that produced no
        answers at all, which is why it is paired with this."""
        run = make_run(answers={}, escalations=self.all_traps_correct())
        result = adversarial(run)
        assert result.status is CategoryStatus.FAIL
        assert any(m.key == "answers_produced" and m.passed is False for m in result.metrics)

    def test_every_trap_gets_a_row_in_the_notes(self) -> None:
        result = adversarial(make_run(escalations=self.all_traps_correct()))
        for trap in TRAPS:
            assert any(note.startswith(trap.printed_number) for note in result.notes), (
                trap.printed_number
            )


class TestOperational:
    def test_cost_within_budget_passes(self) -> None:
        result = operational(make_run(), budget_usd=2.00)
        assert result.status is CategoryStatus.PASS

    def test_cost_over_budget_fails(self) -> None:
        run = make_run(
            totals=RunTotals(
                questions=1, answered=1, escalated=0, failed=0, tokens_used=10, cost_usd=5.0
            )
        )
        assert operational(run, budget_usd=2.00).status is CategoryStatus.FAIL

    def test_cost_per_accepted_answer_is_reported(self) -> None:
        """The number that matters: a run escalating everything is cheap per
        question and produces nothing."""
        metrics = {m.key: m for m in operational(make_run(), budget_usd=2.00).metrics}
        assert metrics["cost_per_accepted_answer"].value == pytest.approx(0.10)

    def test_a_run_that_accepted_nothing_says_so_rather_than_dividing_by_zero(self) -> None:
        run = make_run(
            answers={},
            totals=RunTotals(
                questions=1, answered=0, escalated=1, failed=0, tokens_used=10, cost_usd=0.1
            ),
        )
        metrics = {m.key: m for m in operational(run, budget_usd=2.00).metrics}
        assert "undefined" in (metrics["cost_per_accepted_answer"].detail or "")

    def test_an_unpriced_run_reports_its_cost_as_a_floor(self) -> None:
        result = operational(make_run(cost_complete=False), budget_usd=2.00)
        metrics = {m.key: m for m in result.metrics}
        assert "FLOOR" in (metrics["cost_per_run"].detail or "")
        assert any("floor" in note.lower() for note in result.notes)

    def test_stage_latencies_are_reported(self) -> None:
        run = make_run()
        run.timings = [StageTiming(stage="drafting", seconds=12.5)]
        metrics = {m.key: m for m in operational(run, budget_usd=2.00).metrics}
        assert metrics["latency_drafting"].value == 12.5


class TestQuality:
    def verdict(
        self, qid: str = "GQ-001", *, score: int = 4, grounded: bool = True
    ) -> JudgeVerdict:
        return JudgeVerdict(
            question_id=qid,
            relevance=score,
            specificity=score,
            directness=score,
            tone=score,
            grounded=grounded,
        )

    def test_good_scores_pass(self) -> None:
        assert quality(make_run(), [self.verdict()]).status is CategoryStatus.PASS

    def test_scores_below_the_threshold_fail(self) -> None:
        assert quality(make_run(), [self.verdict(score=2)]).status is CategoryStatus.FAIL

    def test_a_hallucination_fails_regardless_of_the_scores(self) -> None:
        """A fluent, well-organised answer asserting something no source states
        scores well on tone and is still ungrounded."""
        result = quality(make_run(), [self.verdict(score=5, grounded=False)])
        assert result.status is CategoryStatus.FAIL
        assert result.violations[0].rule == "hallucination"

    def test_no_verdicts_is_unmeasured_not_zero(self) -> None:
        """A quality score of zero would be persisted and differenced, turning
        "not measured" into a regression the moment it IS measured."""
        result = quality(make_run(), [])
        assert result.status is CategoryStatus.NOT_IMPLEMENTED
        assert result.metrics == []

    def test_the_notes_record_that_human_agreement_is_absent(self) -> None:
        result = quality(make_run(), [self.verdict()])
        assert any("NOT MEASURED" in note for note in result.notes)

    def test_unusable_verdicts_are_reported_but_not_gated(self) -> None:
        """An unusable verdict is a fact about the judge, not about the answer
        it failed to score."""
        result = quality(make_run(), [self.verdict()], unusable=2)
        metrics = {m.key: m for m in result.metrics}
        assert metrics["unusable_verdicts"].value == 2.0
        assert metrics["unusable_verdicts"].passed is None


class TestParsingAJudgeReply:
    def test_a_well_formed_reply_parses(self) -> None:
        raw = (
            '{"relevance": 4, "specificity": 3, "directness": 5, "tone": 4, '
            '"grounded": true, "ungrounded_spans": [], "notes": ""}'
        )
        verdict = parse_verdict("GQ-001", raw)
        assert verdict is not None
        assert verdict.mean == 4.0

    def test_prose_returns_none_rather_than_raising(self) -> None:
        """One unusable verdict degrades one answer's score; raising would lose
        every other measurement in the run."""
        assert parse_verdict("GQ-001", "I think this answer is quite good.") is None

    def test_a_missing_dimension_returns_none(self) -> None:
        assert parse_verdict("GQ-001", '{"relevance": 4, "grounded": true}') is None

    def test_ungrounded_spans_are_kept_verbatim(self) -> None:
        raw = (
            '{"relevance": 4, "specificity": 4, "directness": 4, "tone": 4, '
            '"grounded": false, "ungrounded_spans": ["We hold ISO 27001."]}'
        )
        verdict = parse_verdict("GQ-001", raw)
        assert verdict is not None
        assert verdict.ungrounded_spans == ("We hold ISO 27001.",)
