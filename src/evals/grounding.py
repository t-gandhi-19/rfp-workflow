"""Grounding, compliance and operational — the categories that need no model.

All three are arithmetic over a completed run. Quality and the hallucination
half of grounding need the judge and live in `src/evals/judged.py`; keeping them
apart means these can run in CI, where no model does.

THE DENOMINATOR IS ACCEPTED ANSWERS, not questions. An escalated question is not
ungrounded — it is a question the system declined to answer, which is the
behaviour the whole design is for. Counting escalations as grounding failures
would make the safest possible run score worst.
"""

from __future__ import annotations

import re

from src.compliance.checker import summarise
from src.evals.contracts import (
    CategoryResult,
    CategoryStatus,
    EvalMetric,
    MetricDirection,
    Violation,
)
from src.evals.registry import Category
from src.evals.run_under_test import RunUnderTest

#: §3: every claim traceable to a retrieved answer, at or above this.
TRACEABILITY_THRESHOLD = 0.95

#: Numbers worth checking. Bare small integers carry no factual claim on their
#: own and would make this fire on "three phases".
_NUMERIC = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
_TRIVIAL = {str(n) for n in range(11)}


def _status(metrics: list[EvalMetric], violations: list[Violation]) -> CategoryStatus:
    if violations or any(m.passed is False for m in metrics):
        return CategoryStatus.FAIL
    return CategoryStatus.PASS


def grounding(run: RunUnderTest) -> CategoryResult:
    """Traceability, entity failures, numeric failures. Zero-tolerance on two.

    TRACEABILITY IS CHECKED AGAINST WHAT THE RETRIEVER SELECTED, not against the
    graph. The claim being tested is "the drafter cited what it was given"; a
    check against the graph would also pass an answer that cited a real id it
    was never shown, which is a fabricated citation that happens to resolve.
    """
    metrics: list[EvalMetric] = []
    violations: list[Violation] = []

    cited = 0
    total = 0
    for question_id, answer in sorted(run.answers.items()):
        available = run.sources_for(question_id)
        for source_id in answer.source_ids:
            total += 1
            if not available or source_id in available:
                cited += 1
            else:
                violations.append(
                    Violation(
                        rule="citation_not_retrieved",
                        question_number=run.reference(question_id),
                        detail=(
                            f"cited {source_id}, which the retriever did not select; "
                            f"selected: {sorted(available) or 'nothing'}"
                        ),
                    )
                )

    traceability = 1.0 if total == 0 else cited / total
    metrics.append(
        EvalMetric(
            key="traceability",
            label="Citations resolving to a retrieved answer",
            value=round(traceability, 4),
            direction=MetricDirection.HIGHER_IS_BETTER,
            threshold=TRACEABILITY_THRESHOLD,
            passed=traceability >= TRACEABILITY_THRESHOLD,
            detail=f"{cited}/{total} citation(s) across {run.accepted} accepted answer(s)",
        )
    )

    # An accepted answer with an unresolved entity should be impossible: the
    # guardrail hard-fails it into an escalation. Measured anyway, because "the
    # guardrail is wired" and "no unresolved entity reached the response" are
    # different claims and only the second is what a reader needs.
    entity_failures = [
        Violation(
            rule="entity_unresolved_in_accepted_answer",
            question_number=run.reference(question_id),
            detail=f"accepted answer carries unsupported claim(s): {answer.unsupported_claims}",
        )
        for question_id, answer in sorted(run.answers.items())
        if answer.unsupported_claims
    ]
    violations.extend(entity_failures)
    metrics.append(
        EvalMetric(
            key="entity_failures",
            label="Unresolved entities in accepted answers",
            value=float(len(entity_failures)),
            direction=MetricDirection.MUST_BE_ZERO,
            threshold=0.0,
            passed=not entity_failures,
        )
    )

    numeric_failures: list[Violation] = []
    for question_id, answer in sorted(run.answers.items()):
        for claim in answer.unsupported_claims:
            offenders = [n for n in _NUMERIC.findall(claim) if n not in _TRIVIAL]
            if offenders:
                numeric_failures.append(
                    Violation(
                        rule="numeric_not_in_source",
                        question_number=run.reference(question_id),
                        detail=f"unsupported claim states {', '.join(offenders)}",
                    )
                )
    violations.extend(numeric_failures)
    metrics.append(
        EvalMetric(
            key="numeric_failures",
            label="Numbers no cited source states",
            value=float(len(numeric_failures)),
            direction=MetricDirection.MUST_BE_ZERO,
            threshold=0.0,
            passed=not numeric_failures,
        )
    )

    return CategoryResult(
        key=Category.GROUNDING,
        label="Grounding — every claim traceable to a retrieved answer",
        status=_status(metrics, violations),
        metrics=metrics,
        violations=violations,
        notes=[
            "The denominator is ACCEPTED answers. An escalated question is not an "
            "ungrounded answer; it is a question the system declined to answer.",
            "Hallucination rate is judged separately — see the quality category, "
            "which needs judge-model and therefore does not run in CI.",
        ],
    )


def compliance(run: RunUnderTest) -> CategoryResult:
    """Mandatory coverage 100%, limit violations 0, forbidden content 0."""
    counts = summarise(run.compliance)
    mandatory_total = sum(1 for question in run.questions if question.mandatory)
    answered = mandatory_total - counts["mandatory_missing"]
    coverage = 1.0 if mandatory_total == 0 else answered / mandatory_total

    violations = [
        Violation(
            rule="mandatory_unanswered",
            question_number=run.reference(result.question_id),
            detail="a mandatory question has no answer in the response",
        )
        for result in run.compliance
        if not result.mandatory_answered
    ]
    violations.extend(
        Violation(
            rule="word_limit_exceeded",
            question_number=run.reference(result.question_id),
            detail="the answer exceeds the limit the document stated",
        )
        for result in run.compliance
        if not result.within_word_limit
    )
    violations.extend(
        Violation(
            rule=f"forbidden_content:{hit.rule}",
            question_number=run.reference(result.question_id),
            detail=f"matched {hit.matched_text!r}",
        )
        for result in run.compliance
        for hit in result.forbidden_content_hits
    )

    metrics = [
        EvalMetric(
            key="mandatory_coverage",
            label="Mandatory questions answered",
            value=round(coverage, 4),
            direction=MetricDirection.HIGHER_IS_BETTER,
            threshold=1.0,
            passed=coverage >= 1.0,
            detail=f"{answered}/{mandatory_total} mandatory question(s)",
        ),
        EvalMetric(
            key="limit_violations",
            label="Word-limit violations",
            value=float(counts["limit_violations"]),
            direction=MetricDirection.MUST_BE_ZERO,
            threshold=0.0,
            passed=counts["limit_violations"] == 0,
        ),
        EvalMetric(
            key="forbidden_hits",
            label="Forbidden content in the response",
            value=float(counts["forbidden_hits"]),
            direction=MetricDirection.MUST_BE_ZERO,
            threshold=0.0,
            passed=counts["forbidden_hits"] == 0,
        ),
    ]

    return CategoryResult(
        key=Category.COMPLIANCE,
        label="Compliance — mandatory questions answered, word limits respected",
        status=_status(metrics, violations),
        metrics=metrics,
        violations=violations,
        notes=[
            "A mandatory question that ESCALATED counts as unanswered. The response "
            "is not complete until a human writes it, and a coverage number that "
            "ignored escalations would describe only the questions that went well.",
        ],
    )


def operational(run: RunUnderTest, *, budget_usd: float) -> CategoryResult:
    """Cost per run, per question and per ACCEPTED answer; latency per stage.

    COST PER ACCEPTED ANSWER IS THE ONE THAT MATTERS, and it is deliberately the
    harshest of the three: a run that escalates everything is cheap per question
    and infinitely expensive per answer. Reporting only cost-per-question would
    make a system that answers nothing look efficient.
    """
    cost = run.totals.cost_usd
    questions = max(run.totals.questions, 1)
    per_question = cost / questions
    per_accepted = cost / run.accepted if run.accepted else float("inf")

    metrics = [
        EvalMetric(
            key="cost_per_run",
            label="Cost per run (USD)",
            value=round(cost, 6),
            direction=MetricDirection.LOWER_IS_BETTER,
            threshold=budget_usd,
            passed=cost <= budget_usd,
            detail=(
                "complete"
                if run.cost_is_complete
                else "FLOOR — some calls were unpriced by the gateway"
            ),
        ),
        EvalMetric(
            key="cost_per_question",
            label="Cost per question (USD)",
            value=round(per_question, 6),
            direction=MetricDirection.LOWER_IS_BETTER,
        ),
        EvalMetric(
            key="cost_per_accepted_answer",
            label="Cost per ACCEPTED answer (USD)",
            value=round(per_accepted, 6) if run.accepted else 0.0,
            direction=MetricDirection.LOWER_IS_BETTER,
            detail=(
                f"{run.accepted} accepted of {run.totals.questions}"
                if run.accepted
                else "no answer was accepted; cost per accepted answer is undefined"
            ),
        ),
        EvalMetric(
            key="tokens_per_run",
            label="Tokens per run",
            value=float(run.totals.tokens_used),
            direction=MetricDirection.LOWER_IS_BETTER,
        ),
    ]
    metrics.extend(
        EvalMetric(
            key=f"latency_{timing.stage}",
            label=f"Stage latency — {timing.stage} (s)",
            value=round(timing.seconds, 3),
            direction=MetricDirection.LOWER_IS_BETTER,
        )
        for timing in run.timings
    )

    notes = [
        "Cost per ACCEPTED answer is the number to read. A run that escalates "
        "everything is cheap per question and produces nothing.",
    ]
    if not run.cost_is_complete:
        notes.append(
            "The gateway did not price every call, so cost is a FLOOR rather than a "
            "total. The USD gate is applied to the floor: an under-count that "
            "exceeds the ceiling has exceeded it."
        )

    return CategoryResult(
        key=Category.OPERATIONAL,
        label="Operational — cost and latency against budget",
        status=_status(metrics, []),
        metrics=metrics,
        notes=notes,
    )
