"""Typed access to `config/scoring.yaml`.

Contracts enforce a threshold (a draft below the confidence floor *must* be
flagged for SME review), and that threshold is a config value, never a literal
in Python. This module is the single place the YAML is read, so the number
appears exactly once in the codebase.

The config is cached after first read. Tests that need a different threshold
point ``RFP_CONFIG_DIR`` at a fixture directory and call :func:`reload_config`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FinalScoreWeights(BaseModel):
    """Blend weights for `final_score` (build prompt §10 D)."""

    model_config = ConfigDict(extra="forbid")

    vector_graph: float = Field(ge=0.0, le=1.0)
    rerank: float = Field(ge=0.0, le=1.0)


class FinalScoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: FinalScoreWeights


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(gt=0)
    rerank_top_n: int = Field(gt=0)
    match_floor: float = Field(ge=0.0, le=1.0)


class OutcomeMultipliers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    won: float = Field(gt=0.0)
    unknown: float = Field(gt=0.0)
    lost: float = Field(gt=0.0)


class RecencyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decay_days: float = Field(gt=0.0)


class GraphMultiplierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: OutcomeMultipliers
    recency: RecencyConfig
    evidence_bonus: float = Field(gt=0.0)


class ConfidenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    escalation_threshold: float = Field(ge=0.0, le=1.0)
    clamp_min: float = Field(ge=0.0, le=1.0)
    clamp_max: float = Field(ge=0.0, le=1.0)


class ScoringConfig(BaseModel):
    """Whole of `config/scoring.yaml`, validated on load."""

    model_config = ConfigDict(extra="forbid")

    version: int
    retrieval: RetrievalConfig
    final_score: FinalScoreConfig
    graph_multiplier: GraphMultiplierConfig
    confidence: ConfidenceConfig


def config_dir() -> Path:
    """Directory holding the YAML config tree.

    ``RFP_CONFIG_DIR`` overrides the repo-relative default, which is what lets
    tests supply their own thresholds and what lets containers mount config at
    a different path.
    """
    override = os.environ.get("RFP_CONFIG_DIR")
    if override:
        return Path(override)
    # src/contracts/thresholds.py -> src/contracts -> src -> repo root
    return Path(__file__).resolve().parents[2] / "config"


def _load_yaml(filename: str) -> dict[str, Any]:
    path = config_dir() / filename
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data: object = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return dict(data)


@lru_cache(maxsize=1)
def scoring_config() -> ScoringConfig:
    """Parsed, validated `scoring.yaml`. Cached after first read."""
    return ScoringConfig.model_validate(_load_yaml("scoring.yaml"))


def reload_config() -> None:
    """Drop the cache so the next read picks up a new ``RFP_CONFIG_DIR``."""
    scoring_config.cache_clear()


def confidence_escalation_threshold() -> float:
    """Confidence below this obliges `needs_sme_review` (build prompt §6)."""
    return scoring_config().confidence.escalation_threshold
