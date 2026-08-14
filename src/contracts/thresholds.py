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
from typing import Any, Literal

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


class RerankConfig(BaseModel):
    """Stage C — one batched call per question (build prompt §10 C)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    #: Gateway alias. Application code never names a model.
    alias: str = Field(min_length=1)
    #: Concrete provider tag the alias must resolve to; asserted by preflight.
    tag: str = Field(min_length=1)
    #: Zero, so identical runs rerank identically. Anything else makes the
    #: retrieval evals measure noise as well as quality.
    temperature: float = Field(ge=0.0, le=2.0)
    timeout_seconds: float = Field(gt=0.0)


class RetrievalConfig(BaseModel):
    """No match floor here, deliberately (amendment O).

    The floor is derived from corpus statistics and lives in the calibration
    artifact. `extra="forbid"` is what makes that stick: a config still carrying
    `match_floor` or `match_floor_calibrated` is rejected at load rather than
    quietly resolving to a stale value in unknown units.
    """

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(gt=0)
    rerank_top_n: int = Field(gt=0)


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


class PreferenceConfig(BaseModel):
    """D17: preference reorders qualifying candidates; it never gates them."""

    model_config = ConfigDict(extra="forbid")

    #: Recency becomes a nudge in [floor, 1.0] rather than a gate in [0, 1].
    recency_floor: float = Field(ge=0.0, le=1.0)
    clamp_min: float = Field(gt=0.0)
    clamp_max: float = Field(gt=0.0)


class FloorDerivation(BaseModel):
    """HOW the floor is computed. The value it computes to lives in the artifact.

    The anchor names are constrained to real artifact fields rather than left as
    free strings: they are used to read attributes off the artifact, and a typo
    should fail at config load rather than at the first retrieval.
    """

    model_config = ConfigDict(extra="forbid")

    background_anchor: Literal["bg_p50", "bg_p95", "bg_p99"]
    same_topic_anchor: Literal["same_topic_p05", "same_topic_p50"]
    #: 0.0 sits on the background anchor, 1.0 on the same-topic anchor.
    midpoint_weight: float = Field(ge=0.0, le=1.0)


class CommissionedProbes(BaseModel):
    """Margins measured at commissioning, in calibrated space (D19).

    None until commissioned; calibration refuses rather than assuming a value it
    was never given.
    """

    model_config = ConfigDict(extra="forbid")

    #: How far each unanswerable sat BELOW the floor.
    unanswerable_2_7: float | None = Field(default=None, ge=0.0)
    unanswerable_2_8: float | None = Field(default=None, ge=0.0)
    unanswerable_3_5: float | None = Field(default=None, ge=0.0)
    #: How far the WEAKEST answerable sat above it.
    min_answerable: float | None = Field(default=None, ge=0.0)

    def commissioned(self) -> bool:
        return all(
            value is not None
            for value in (
                self.unanswerable_2_7,
                self.unanswerable_2_8,
                self.unanswerable_3_5,
                self.min_answerable,
            )
        )


class FloorDiscrimination(BaseModel):
    """Tier 2 (D19): does the derived floor still tell answerable from not?

    Replaces the `same_p05 - bg_p99` tail statistic, which was demoted to a
    reported diagnostic. That statistic gated a PROXY: the same-subject lower
    tail is set by the hardest legitimate paraphrases, which is an open-ended
    authorship boundary with no crisp edge, so adding data to firm the estimate
    also adds worst-match candidates. It failed while every operational margin
    held or improved — the definition of a proxy diverging from its target.
    """

    model_config = ConfigDict(extra="forbid")

    #: Fraction of each commissioned margin that must survive.
    retention: float = Field(gt=0.0, le=1.0)
    commissioned: CommissionedProbes


class SeparationGuards(BaseModel):
    """Two tiers, watching two different failures (D18, revised by D19).

    These protect the FUTURE — they detect change against a commissioned
    baseline. The absolute quality standard is the zero-tolerance retrieval
    evals on real embeddings, which bind regardless of these parameters.
    """

    model_config = ConfigDict(extra="forbid")

    #: Tier 1. Absolute floor under the MEDIAN gap: catches wholesale collapse.
    body_min: float = Field(ge=0.0)
    #: Tier 2. Probe-based, on the derived floor's actual discrimination.
    floor_discrimination: FloorDiscrimination


class CalibrationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_background_pairs: int = Field(gt=0)
    min_same_topic_pairs: int = Field(gt=0)
    separation_guards: SeparationGuards
    floor_derivation: FloorDerivation


class ScoringConfig(BaseModel):
    """Whole of `config/scoring.yaml`, validated on load."""

    model_config = ConfigDict(extra="forbid")

    version: int
    retrieval: RetrievalConfig
    rerank: RerankConfig
    preference: PreferenceConfig
    calibration: CalibrationSettings
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


def load_config_yaml(filename: str) -> dict[str, Any]:
    """Read one YAML file from the config tree as a mapping.

    Public because `limits.yaml` is loaded the same way and by the same rules —
    same directory, same `RFP_CONFIG_DIR` override, same "a config that is not a
    mapping is a broken config" failure. A second copy of this in the controller
    would be a second place for the override to stop working.
    """
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
    return ScoringConfig.model_validate(load_config_yaml("scoring.yaml"))


def reload_config() -> None:
    """Drop the cache so the next read picks up a new ``RFP_CONFIG_DIR``."""
    scoring_config.cache_clear()


def confidence_escalation_threshold() -> float:
    """Confidence below this obliges `needs_sme_review` (build prompt §6)."""
    return scoring_config().confidence.escalation_threshold
