"""The config loader — thresholds live in YAML, never as literals in Python.

The point of these tests is that changing `config/scoring.yaml` actually changes
behaviour. If the threshold were hardcoded somewhere, the override test below
would still pass on the default and quietly prove nothing, so it asserts against
a value the shipped config does not use.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.contracts import thresholds
from src.contracts.thresholds import ScoringConfig, confidence_escalation_threshold, reload_config


@pytest.fixture(autouse=True)
def _clean_config_cache() -> object:
    """Every test starts and ends with a cold cache and no env override."""
    reload_config()
    yield
    reload_config()


class TestShippedConfig:
    def test_loads_and_validates(self) -> None:
        config = thresholds.scoring_config()
        assert isinstance(config, ScoringConfig)
        assert config.version == 1

    def test_matches_the_documented_defaults(self) -> None:
        """These are the numbers in the build prompt §10; drift should be loud."""
        config = thresholds.scoring_config()
        assert config.retrieval.top_k == 20
        assert config.retrieval.rerank_top_n == 8
        # Amendment O: config holds no floor VALUE at all — only the rule that
        # derives one. The value lives in the calibration artifact.
        assert config.calibration.floor_derivation.midpoint_weight == pytest.approx(0.5)
        assert config.calibration.floor_derivation.background_anchor == "bg_p99"
        assert config.final_score.weights.vector_graph == pytest.approx(0.5)
        assert config.final_score.weights.rerank == pytest.approx(0.5)
        assert config.graph_multiplier.outcome.won == pytest.approx(1.15)
        assert config.graph_multiplier.outcome.unknown == pytest.approx(1.0)
        assert config.graph_multiplier.outcome.lost == pytest.approx(0.85)
        assert config.graph_multiplier.recency.decay_days == pytest.approx(540)
        assert config.graph_multiplier.evidence_bonus == pytest.approx(1.1)
        assert config.confidence.escalation_threshold == pytest.approx(0.6)

    def test_blend_weights_sum_to_one(self) -> None:
        weights = thresholds.scoring_config().final_score.weights
        assert weights.vector_graph + weights.rerank == pytest.approx(1.0)

    def test_won_outranks_unknown_outranks_lost(self) -> None:
        """The ordering property the scoring math depends on (build prompt §10 B)."""
        outcome = thresholds.scoring_config().graph_multiplier.outcome
        assert outcome.won > outcome.unknown > outcome.lost


class TestOverride:
    def test_config_dir_can_be_redirected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shipped = thresholds.scoring_config().model_dump()
        shipped["confidence"]["escalation_threshold"] = 0.42
        (tmp_path / "scoring.yaml").write_text(yaml.safe_dump(shipped), encoding="utf-8")

        monkeypatch.setenv("RFP_CONFIG_DIR", str(tmp_path))
        reload_config()

        assert confidence_escalation_threshold() == pytest.approx(0.42)

    def test_missing_config_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RFP_CONFIG_DIR", str(tmp_path / "nope"))
        reload_config()
        with pytest.raises(FileNotFoundError):
            thresholds.scoring_config()

    def test_rejects_an_unknown_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """extra='forbid' catches a typo in config rather than silently ignoring it."""
        shipped = thresholds.scoring_config().model_dump()
        shipped["confidence"]["escalaton_threshold"] = 0.5
        (tmp_path / "scoring.yaml").write_text(yaml.safe_dump(shipped), encoding="utf-8")

        monkeypatch.setenv("RFP_CONFIG_DIR", str(tmp_path))
        reload_config()
        with pytest.raises(ValueError, match="Extra inputs"):
            thresholds.scoring_config()
