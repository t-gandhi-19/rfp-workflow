"""Typed access to `config/limits.yaml` — the budgets the controller enforces.

CLAUDE.md rule 14: the deterministic controller enforces these, and the agent
framework is never trusted to respect them. That is what makes this a config
file read by the controller rather than a set of parameters handed to crewAI:
the ceiling has to be checked by code that runs whether or not the framework
does what it was asked.

LiteLLM mirrors the same ceilings independently, so a controller bug cannot run
up a bill on its own. Two layers, one source of truth, and neither of them a
number written into Python.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field

from src.contracts.thresholds import load_config_yaml


class PerRunBudget(BaseModel):
    """Breaching either ceiling halts the run with partials already saved."""

    model_config = ConfigDict(extra="forbid")

    max_usd: float = Field(gt=0.0)
    max_tokens: int = Field(gt=0)


class PerQuestionBudget(BaseModel):
    """The hard call budget. No loops, no redraft cycles (rule 14)."""

    model_config = ConfigDict(extra="forbid")

    rerank_calls: int = Field(ge=0)
    draft_calls: int = Field(ge=0)
    critique_calls: int = Field(ge=0)
    validation_retries: int = Field(ge=0)


class PerRunCalls(BaseModel):
    model_config = ConfigDict(extra="forbid")

    triage: int = Field(ge=0)
    extract_assist: int = Field(ge=0)


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_run: PerRunBudget
    per_question: PerQuestionBudget
    per_run_calls: PerRunCalls


class ConcurrencyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Ceiling on the per-question fan-out semaphore. Raising it moves latency,
    #: not cost — the per-question call budget is what bounds spend.
    question_fanout: int = Field(gt=0)


class GatewayLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(ge=0)
    backoff_initial_seconds: float = Field(gt=0.0)
    backoff_multiplier: float = Field(ge=1.0)
    timeouts_seconds: dict[str, float]


class StageLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Wall-clock ceiling per stage before the controller gives up and escalates
    #: the affected questions.
    timeouts_seconds: dict[str, float]


class LimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    budget: BudgetConfig
    concurrency: ConcurrencyConfig
    gateway: GatewayLimits
    stages: StageLimits


@lru_cache(maxsize=1)
def limits_config() -> LimitsConfig:
    """Parsed, validated `limits.yaml`. Cached after first read."""
    return LimitsConfig.model_validate(load_config_yaml("limits.yaml"))


def reload_limits() -> None:
    """Drop the cache so the next read picks up a new ``RFP_CONFIG_DIR``."""
    limits_config.cache_clear()
