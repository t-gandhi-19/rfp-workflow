"""Run-level contracts: live state, final result, and eval scores.

`RunState` is the checkpoint. It is upserted through write-api at every stage
transition and every per-question status change, keyed on (run_id, question_id)
so a resumed run skips completed questions and re-executes pending ones
(build prompt §13).
"""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from src.contracts.checks import ComplianceResult
from src.contracts.drafting import DraftedAnswer
from src.contracts.enums import HaltReason, QuestionStatus, RunStage


class RunState(BaseModel):
    """Checkpointed state of one run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    rfp_id: str = Field(min_length=1)
    stage: RunStage
    per_question_status: dict[str, QuestionStatus] = Field(default_factory=dict)
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    started_at: AwareDatetime
    updated_at: AwareDatetime
    halted_reason: HaltReason | None = None

    @model_validator(mode="after")
    def _halt_state_is_consistent(self) -> RunState:
        """HALTED and halted_reason travel together, in both directions.

        A halted run whose reason is missing is unreadable in the dashboard, and
        a reason attached to a live run would misreport it as dead.
        """
        if self.stage is RunStage.HALTED and self.halted_reason is None:
            raise ValueError("stage=HALTED requires halted_reason")
        if self.stage is not RunStage.HALTED and self.halted_reason is not None:
            raise ValueError(f"halted_reason set but stage is {self.stage}, not HALTED")
        return self

    @model_validator(mode="after")
    def _clock_moves_forward(self) -> RunState:
        if self.updated_at < self.started_at:
            raise ValueError("updated_at precedes started_at")
        return self


class RunTotals(BaseModel):
    """Roll-up counters for one run.

    `answered` counts answers that passed guardrails and were not escalated —
    the same denominator the harness uses for cost per accepted answer.
    """

    model_config = ConfigDict(extra="forbid")

    questions: int = Field(ge=0)
    answered: int = Field(ge=0)
    escalated: int = Field(ge=0)
    failed: int = Field(ge=0)
    tokens_used: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _parts_do_not_exceed_whole(self) -> RunTotals:
        accounted = self.answered + self.escalated + self.failed
        if accounted > self.questions:
            raise ValueError(
                f"answered+escalated+failed ({accounted}) exceeds questions ({self.questions})"
            )
        return self


class ArtifactPaths(BaseModel):
    """Where a run's outputs landed on disk, under `out/<run_id>/`."""

    model_config = ConfigDict(extra="forbid")

    response_docx: str | None = None
    escalations_json: str | None = None
    eval_report_html: str | None = None


class EvalScore(BaseModel):
    """One metric from one eval run, keyed by git SHA for regression deltas.

    `passed` is supplied rather than derived: some metrics pass by exceeding the
    threshold (recall) and others by staying at or below it (hallucination rate,
    forbidden-content hits), so direction lives with the metric definition in
    the harness.
    """

    model_config = ConfigDict(extra="forbid")

    git_sha: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: float
    threshold: float
    passed: bool


class RunResult(BaseModel):
    """Terminal result of a run — what the human reviewer receives."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    answers: list[DraftedAnswer] = Field(default_factory=list)
    compliance: list[ComplianceResult] = Field(default_factory=list)
    eval_scores: list[EvalScore] | None = None
    totals: RunTotals
    trace_url: str | None = None
    artifact_paths: ArtifactPaths = Field(default_factory=ArtifactPaths)
