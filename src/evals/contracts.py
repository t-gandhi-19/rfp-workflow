"""Typed results for the eval harness (build prompt §3).

Every category produces the same shape, so the report renderer, the Postgres
writer and the delta comparison are written once rather than per category. A
category that is not implemented yet still produces a `CategoryResult` — with
`status=NOT_IMPLEMENTED` and no metrics — because a category missing from the
report reads as "nothing to say", while one greyed out and named reads as "not
measured yet". Those are different claims and only the second is true.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CategoryStatus(StrEnum):
    # ruff's hardcoded-password heuristic fires on a member literally named
    # PASS. It is a verdict, not a credential.
    PASS = "PASS"  # noqa: S105
    FAIL = "FAIL"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class MetricDirection(StrEnum):
    """Which way is better. Needed to colour a delta against a previous SHA."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    #: A count that should be exactly zero — staleness, leaks, paraphrase hits.
    MUST_BE_ZERO = "must_be_zero"


class EvalMetric(BaseModel):
    """One number, with everything needed to judge it later.

    `threshold` and `passed` are stored rather than recomputed at render time:
    a report is a record of what was decided, and a threshold that moved after
    the fact would silently rewrite history.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: float
    direction: MetricDirection
    #: None where a metric is reported but not gated (MRR, the tail statistic).
    threshold: float | None = None
    passed: bool | None = None
    detail: str | None = None


class Violation(BaseModel):
    """A zero-tolerance breach, named so it can be acted on without re-running.

    Deliberately not a bare string: the eval that finds a stale answer in a
    candidate list has to say which question, which answer and which rule, or
    the report obliges someone to reproduce it before they can start.
    """

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(min_length=1)
    question_number: str | None = None
    detail: str = Field(min_length=1)


class CategoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    status: CategoryStatus
    metrics: list[EvalMetric] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list)
    #: Free-form, rendered under the table. Where a category explains itself —
    #: what the numbers mean, what was not measured.
    notes: list[str] = Field(default_factory=list)
    #: Why a category is NOT_IMPLEMENTED, so the report says what is missing.
    not_implemented_reason: str | None = None

    @property
    def gated_metrics(self) -> list[EvalMetric]:
        return [metric for metric in self.metrics if metric.passed is not None]
