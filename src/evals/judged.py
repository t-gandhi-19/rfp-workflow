"""The quality category, and the hallucination half of grounding (D16).

THE ONLY CATEGORY THAT SPENDS MONEY TO MEASURE. One judge call per accepted
answer, on `judge-model`, which resolves to a different model family than the
drafter — asserted structurally in `tests/unit/test_model_pins.py`, not left to
the config comment that used to be the only thing holding it.

EVAL RUNS ONLY. The ledger REFUSES a judge call in a production run
(`JudgeInProductionError`) rather than counting one, because the correct number
of judge calls outside an eval is not "few", it is none.

WHAT IS DELIBERATELY NOT MEASURED. Judge-vs-human agreement. D16 accepts the
risk that these scores are uncalibrated against human opinion, and the honest
consequence is that the report says "judged" rather than "good". A spot-check
sample and an agreement number would be the fix; inventing one from the judge's
own confidence would not.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from src.evals.contracts import (
    CategoryResult,
    CategoryStatus,
    EvalMetric,
    MetricDirection,
    Violation,
)
from src.evals.registry import Category
from src.evals.run_under_test import RunUnderTest

logger = logging.getLogger("rfp.evals.judged")

#: The rubric's dimensions, in the order the file states them.
DIMENSIONS = ("relevance", "specificity", "directness", "tone")

#: §3 gates. A 1-5 rubric, so 3.5 is "better than the midpoint" rather than an
#: arbitrary bar — the anchors put generic marketing prose at 1 and a checkable,
#: directly-responsive answer at 5.
QUALITY_THRESHOLD = 3.5

#: Zero-tolerance. A hallucination is not a low score, it is a false statement.
HALLUCINATION_THRESHOLD = 0.0


@dataclass(frozen=True)
class JudgeVerdict:
    """One judged answer."""

    question_id: str
    relevance: int
    specificity: int
    directness: int
    tone: int
    grounded: bool
    ungrounded_spans: tuple[str, ...] = ()
    notes: str = ""

    @property
    def mean(self) -> float:
        return (self.relevance + self.specificity + self.directness + self.tone) / 4


def parse_verdict(question_id: str, raw: str) -> JudgeVerdict | None:
    """Read one judge reply, or None if it is unusable.

    None rather than a raise: one unparseable verdict degrades ONE answer's
    quality score, and taking the whole eval down over it would lose every other
    measurement in the run. The count of unusable verdicts is reported, so a
    judge that has started returning prose is visible rather than silently
    shrinking the denominator.
    """
    try:
        payload = json.loads(raw)
        return JudgeVerdict(
            question_id=question_id,
            relevance=int(payload["relevance"]),
            specificity=int(payload["specificity"]),
            directness=int(payload["directness"]),
            tone=int(payload["tone"]),
            grounded=bool(payload["grounded"]),
            ungrounded_spans=tuple(str(s) for s in payload.get("ungrounded_spans", ())),
            notes=str(payload.get("notes", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("judge verdict for %s is unusable (%s)", question_id, exc)
        return None


def quality(
    run: RunUnderTest, verdicts: list[JudgeVerdict], *, unusable: int = 0
) -> CategoryResult:
    """Per-dimension means, and the hallucination rate.

    ESCALATED QUESTIONS ARE NOT SCORED, per the rubric's own scoring rules: an
    escalation is scored on whether it was CORRECT, which is the adversarial
    category's job. Including them here would average a placeholder into the
    prose scores.
    """
    if not verdicts:
        return CategoryResult(
            key=Category.QUALITY,
            label="Quality — LLM-judged, per D16",
            status=CategoryStatus.NOT_IMPLEMENTED,
            not_implemented_reason=(
                "No judged answers in this run. Either nothing was accepted, or the "
                "run was invoked without judge-model. This is a fact about the "
                "invocation, not about the build."
            ),
        )

    metrics: list[EvalMetric] = []
    for dimension in DIMENSIONS:
        scores = [getattr(v, dimension) for v in verdicts]
        mean = sum(scores) / len(scores)
        metrics.append(
            EvalMetric(
                key=dimension,
                label=f"{dimension.title()} (1-5, judged)",
                value=round(mean, 3),
                direction=MetricDirection.HIGHER_IS_BETTER,
                threshold=QUALITY_THRESHOLD,
                passed=mean >= QUALITY_THRESHOLD,
                detail=f"{len(scores)} answer(s) judged",
            )
        )

    overall = sum(v.mean for v in verdicts) / len(verdicts)
    metrics.append(
        EvalMetric(
            key="overall",
            label="Overall quality (mean of the four dimensions)",
            value=round(overall, 3),
            direction=MetricDirection.HIGHER_IS_BETTER,
            threshold=QUALITY_THRESHOLD,
            passed=overall >= QUALITY_THRESHOLD,
        )
    )

    hallucinated = [v for v in verdicts if not v.grounded]
    rate = len(hallucinated) / len(verdicts)
    metrics.append(
        EvalMetric(
            key="hallucination_rate",
            label="Answers the judge found ungrounded",
            value=round(rate, 4),
            direction=MetricDirection.MUST_BE_ZERO,
            threshold=HALLUCINATION_THRESHOLD,
            passed=rate <= HALLUCINATION_THRESHOLD,
            detail=f"{len(hallucinated)}/{len(verdicts)}",
        )
    )

    if unusable:
        metrics.append(
            EvalMetric(
                key="unusable_verdicts",
                label="Judge replies that could not be parsed",
                value=float(unusable),
                direction=MetricDirection.MUST_BE_ZERO,
                detail=(
                    "Reported but NOT gated: an unusable verdict is a fact about the "
                    "judge, not about the answer it failed to score."
                ),
            )
        )

    violations = [
        Violation(
            rule="hallucination",
            question_number=run.reference(verdict.question_id),
            detail=(
                f"judge found {len(verdict.ungrounded_spans)} ungrounded span(s): "
                f"{verdict.ungrounded_spans[0] if verdict.ungrounded_spans else verdict.notes}"
            ),
        )
        for verdict in hallucinated
    ]

    return CategoryResult(
        key=Category.QUALITY,
        label="Quality — LLM-judged, per D16",
        status=(
            CategoryStatus.FAIL
            if violations or any(m.passed is False for m in metrics)
            else CategoryStatus.PASS
        ),
        metrics=metrics,
        violations=violations,
        notes=[
            "Scored by judge-model, a DIFFERENT model family from the drafter, so a "
            "model never grades its own prose. Asserted in tests/unit/test_model_pins.py.",
            "Escalated questions are not scored here — the rubric scores an escalation "
            "on whether it was correct, which is the adversarial category.",
            "JUDGE-VS-HUMAN AGREEMENT IS NOT MEASURED. D16 accepts that these scores "
            "are uncalibrated against human opinion; the report therefore says "
            "'judged', not 'good'.",
        ],
    )
