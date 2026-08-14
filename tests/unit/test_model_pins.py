"""Every Ollama reference in the gateway config is an actual pin.

`ollama/llama3.2` resolves to whatever `:latest` points at today and changes on
the next pull, so two machines can run different weights behind the same alias
and nothing says so. Making that a preflight failure turns the pinning rule from
documentation into something the build enforces.

The shipped config is asserted here too, so a future alias added without a tag
fails immediately rather than at whatever point someone re-reads the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.gateway.model_pins import (
    DEFAULT_LITELLM_CONFIG,
    GatewayModel,
    drifting_models,
    model_family,
    parse_alias_references,
    parse_gateway_models,
)
from src.gateway.preflight import check_tags_are_explicit


def _write_config(path: Path, models: list[tuple[str, str]]) -> Path:
    config = {
        "model_list": [
            {"model_name": alias, "litellm_params": {"model": reference}}
            for alias, reference in models
        ]
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class TestTagClassification:
    @pytest.mark.parametrize(
        "tag", ["llama3.2:3b", "llama3.1:8b", "nomic-embed-text:v1.5", "qwen2.5:7b-instruct"]
    )
    def test_versioned_tags_are_pins(self, tag: str) -> None:
        model = GatewayModel(alias="a", reference=f"ollama/{tag}", tag=tag)
        assert model.is_drifting is False
        assert model.violation() is None

    def test_a_bare_name_is_not_a_pin(self) -> None:
        model = GatewayModel(alias="triage-model", reference="ollama/llama3.2", tag="llama3.2")
        assert model.is_drifting is True
        violation = model.violation()
        assert violation is not None and "no tag" in violation

    def test_explicit_latest_is_not_a_pin_either(self) -> None:
        """Being visible does not make it stable."""
        model = GatewayModel(
            alias="embed-model", reference="ollama/nomic:latest", tag="nomic:latest"
        )
        assert model.is_drifting is True
        violation = model.violation()
        assert violation is not None and ":latest is not a pin" in violation

    def test_the_fix_command_names_the_exact_tag(self) -> None:
        model = GatewayModel(alias="a", reference="ollama/llama3.1:8b", tag="llama3.1:8b")
        assert model.pull_command == "ollama pull llama3.1:8b"


class TestParsing:
    def test_only_ollama_aliases_are_collected(self, tmp_path: Path) -> None:
        """Groq versioning is the provider's problem and has no local host."""
        path = _write_config(
            tmp_path / "c.yaml",
            [
                ("triage-model", "ollama/llama3.2:3b"),
                ("drafter-model", "groq/llama-3.3-70b-versatile"),
            ],
        )
        models = parse_gateway_models(path)
        assert [m.alias for m in models] == ["triage-model"]

    def test_several_aliases_may_share_a_tag(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path / "c.yaml",
            [("triage-model", "ollama/llama3.2:3b"), ("loginterp-model", "ollama/llama3.2:3b")],
        )
        assert {m.tag for m in parse_gateway_models(path)} == {"llama3.2:3b"}

    def test_an_empty_config_yields_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("model_list: []\n", encoding="utf-8")
        assert parse_gateway_models(path) == []


class TestPreflightRejectsDrift:
    def test_a_bare_reference_fails_preflight(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path / "c.yaml", [("triage-model", "ollama/llama3.2")])
        result = check_tags_are_explicit(parse_gateway_models(path))
        assert result.ok is False
        assert "triage-model" in result.detail
        assert result.fix is not None and ":latest is not a pin" in result.fix

    def test_an_explicit_latest_fails_preflight(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path / "c.yaml", [("embed-model", "ollama/nomic:latest")])
        assert check_tags_are_explicit(parse_gateway_models(path)).ok is False

    def test_every_offender_is_named_not_just_the_first(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path / "c.yaml",
            [("triage-model", "ollama/llama3.2"), ("loginterp-model", "ollama/mistral:latest")],
        )
        result = check_tags_are_explicit(parse_gateway_models(path))
        assert "triage-model" in result.detail
        assert "loginterp-model" in result.detail

    def test_a_fully_pinned_config_passes(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path / "c.yaml",
            [("triage-model", "ollama/llama3.2:3b"), ("rerank-model", "ollama/llama3.1:8b")],
        )
        assert check_tags_are_explicit(parse_gateway_models(path)).ok is True


class TestTheShippedConfig:
    def test_every_ollama_reference_is_pinned(self) -> None:
        """The rule applied to the config we actually ship."""
        models = parse_gateway_models()
        assert models, "the gateway config should declare Ollama aliases"
        offenders = [m.violation() for m in drifting_models(models)]
        assert offenders == []

    def test_preflight_accepts_it(self) -> None:
        assert check_tags_are_explicit(parse_gateway_models()).ok is True

    def test_the_embedding_pin_agrees_with_the_gateway(self) -> None:
        """A drifted alias would embed with a different model than we pinned."""
        from src.contracts.embedding import embedding_config

        config = embedding_config()
        model = next(m for m in parse_gateway_models() if m.alias == config.model.alias)
        assert model.tag == config.model.tag

    def test_the_rerank_pin_agrees_with_the_gateway(self) -> None:
        from src.contracts.thresholds import scoring_config

        rerank = scoring_config().rerank
        model = next(m for m in parse_gateway_models() if m.alias == rerank.alias)
        assert model.tag == rerank.tag

    def test_the_config_path_is_the_real_one(self) -> None:
        assert DEFAULT_LITELLM_CONFIG.is_file()


class TestFamilyExtraction:
    @pytest.mark.parametrize(
        ("reference", "expected"),
        [
            ("groq/llama-3.3-70b-versatile", "llama"),
            ("groq/qwen/qwen3-32b", "qwen"),
            ("ollama/llama3.2:3b", "llama"),
            ("ollama/llama3.1:8b", "llama"),
            ("ollama/nomic-embed-text:v1.5", "nomic"),
        ],
    )
    def test_version_size_and_vendor_are_stripped(self, reference: str, expected: str) -> None:
        assert model_family(reference) == expected

    def test_the_same_family_on_two_providers_is_still_one_family(self) -> None:
        """The comparison that matters is 'same weights lineage', not 'same
        provider'. A judge on `ollama/llama3.2` grading a drafter on
        `groq/llama-3.3-70b` is the self-grading D16 forbids, even though the
        two references share no prefix at all."""
        assert model_family("ollama/llama3.2:3b") == model_family("groq/llama-3.3-70b-versatile")


class TestTheJudgeIsNotSelfGrading:
    """D16, made structural rather than left in a config comment.

    The gateway config says "A different model family from the drafter on
    purpose" above `judge-model`. That sentence is the only thing that was
    holding the requirement: pointing judge-model at
    `groq/llama-3.3-70b-versatile` would have kept every test in the repository
    green — including the prompt-contract test, since the PROMPT would still
    say different-family — while making every quality number in the eval report
    a model's opinion of its own prose.
    """

    def test_the_judge_alias_exists_in_the_gateway(self) -> None:
        assert "judge-model" in parse_alias_references()

    def test_the_judge_is_a_different_family_from_the_drafter(self) -> None:
        references = parse_alias_references()
        judge = model_family(references["judge-model"])
        drafter = model_family(references["drafter-model"])
        assert judge != drafter, (
            f"judge-model and drafter-model are both the '{judge}' family. "
            f"D16 requires a different family so a model never grades its own prose."
        )

    def test_the_judge_is_a_different_family_from_the_critic(self) -> None:
        """The critic already scored this answer and lowered its confidence. A
        judge of the same family re-applying the same priors is not the
        independent measurement the report presents it as."""
        references = parse_alias_references()
        assert model_family(references["judge-model"]) != model_family(references["critic-model"])
