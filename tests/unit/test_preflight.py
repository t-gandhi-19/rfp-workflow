"""Preflight checks, driven against a stubbed Ollama and a stubbed gateway.

The model-absent and wrong-width cases are the whole point: both are awkward to
reproduce for real (one means uninstalling someone's model, the other means
installing a differently-shaped one), and both are exactly what preflight is for.

Both stubs are injected as httpx clients rather than patched onto the httpx
module. A global patch also catches every other client in the process — which is
how the first version of these tests silently disabled the gateway's own mock
and produced eleven confusing failures.
"""

from __future__ import annotations

import httpx
import pytest

from src.contracts.embedding import EmbeddingConfig
from src.contracts.thresholds import RerankConfig
from src.gateway.client import GatewayClient
from src.gateway.ollama_admin import is_installed, normalise_tag
from src.gateway.preflight import (
    check_embedding_width,
    check_model_installed,
    check_ollama_reachable,
    check_rerank_model_installed,
    format_report,
    ollama_base_url_for_host,
    run_preflight,
)

OLLAMA = "http://localhost:11434"
GATEWAY = GatewayClient(base_url="http://gateway.test", api_key="k")

CONFIG = EmbeddingConfig.model_validate(
    {
        "version": 1,
        "model": {"alias": "embed-model", "tag": "nomic-embed-text", "dimensions": 768},
        "index": {
            "name": "question_embedding",
            "label": "Question",
            "property": "embedding",
            "similarity": "cosine",
        },
        "install_command": "ollama pull nomic-embed-text",
    }
)

RERANK = RerankConfig.model_validate(
    {
        "enabled": True,
        "alias": "rerank-model",
        "tag": "llama3.1:8b",
        "temperature": 0,
        "timeout_seconds": 180,
    }
)

INSTALLED = ["nomic-embed-text:latest", "llama3.1:8b", "llama3.2:latest"]


def ollama(models: list[str] | None, *, unreachable: bool = False) -> httpx.AsyncClient:
    """An Ollama stub reporting `models`, or refusing connections."""

    def handler(request: httpx.Request) -> httpx.Response:
        if unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"models": [{"name": n} for n in (models or [])]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def gateway(dimensions: int, *, status: int = 200) -> httpx.AsyncClient:
    """A gateway stub returning an embedding of the given width."""

    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, text="upstream unavailable")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.01] * dimensions}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestTagNormalisation:
    """`ollama pull nomic-embed-text` installs `nomic-embed-text:latest`."""

    def test_latest_suffix_is_collapsed(self) -> None:
        assert normalise_tag("nomic-embed-text:latest") == "nomic-embed-text"

    def test_explicit_versions_are_preserved(self) -> None:
        assert normalise_tag("llama3.2:3b") == "llama3.2:3b"

    def test_pinned_tag_matches_its_latest_form(self) -> None:
        """Without this, a model sitting right there reports as missing."""
        assert is_installed("nomic-embed-text", ["nomic-embed-text:latest"]) is True

    def test_a_different_model_does_not_match(self) -> None:
        assert is_installed("nomic-embed-text", ["llama3.2:3b"]) is False


class TestReachability:
    async def test_reports_ok_when_ollama_answers(self) -> None:
        async with ollama(INSTALLED) as client:
            result = await check_ollama_reachable(OLLAMA, client=client)
        assert result.ok is True
        assert "3 model(s) installed" in result.detail

    async def test_reports_failure_with_a_fix_when_unreachable(self) -> None:
        async with ollama(None, unreachable=True) as client:
            result = await check_ollama_reachable(OLLAMA, client=client)
        assert result.ok is False
        assert "cannot reach Ollama" in result.detail
        assert result.fix is not None and "ollama serve" in result.fix


class TestModelPresence:
    async def test_passes_when_the_pinned_model_is_installed(self) -> None:
        async with ollama(INSTALLED) as client:
            result = await check_model_installed(OLLAMA, CONFIG, client=client)
        assert result.ok is True

    async def test_fails_with_the_literal_pull_command_when_absent(self) -> None:
        """The required behaviour: name the exact command that fixes it."""
        async with ollama(["llama3.2:latest"]) as client:
            result = await check_model_installed(OLLAMA, CONFIG, client=client)
        assert result.ok is False
        assert result.fix == "ollama pull nomic-embed-text"
        assert "llama3.2" in result.detail

    async def test_fails_cleanly_when_nothing_is_installed(self) -> None:
        async with ollama([]) as client:
            result = await check_model_installed(OLLAMA, CONFIG, client=client)
        assert result.ok is False
        assert "(none)" in result.detail

    async def test_fails_with_the_pull_command_when_ollama_is_down(self) -> None:
        async with ollama(None, unreachable=True) as client:
            result = await check_model_installed(OLLAMA, CONFIG, client=client)
        assert result.ok is False
        assert result.fix == "ollama pull nomic-embed-text"


class TestRerankModelPresence:
    """Presence only — invoking an 8B model on CPU would make preflight take
    minutes, which would get it skipped, which defeats the point of having it."""

    async def test_passes_when_installed(self) -> None:
        async with ollama(INSTALLED) as client:
            result = await check_rerank_model_installed(OLLAMA, RERANK, client=client)
        assert result.ok is True

    async def test_fails_with_the_literal_pull_command_when_absent(self) -> None:
        async with ollama(["nomic-embed-text:latest"]) as client:
            result = await check_rerank_model_installed(OLLAMA, RERANK, client=client)
        assert result.ok is False
        assert result.fix == "ollama pull llama3.1:8b"

    async def test_fails_with_the_pull_command_when_ollama_is_down(self) -> None:
        async with ollama(None, unreachable=True) as client:
            result = await check_rerank_model_installed(OLLAMA, RERANK, client=client)
        assert result.ok is False
        assert result.fix == "ollama pull llama3.1:8b"

    async def test_skipped_when_rerank_is_disabled(self) -> None:
        """Scoring runs without it, so a missing model is not a failure."""
        disabled = RERANK.model_copy(update={"enabled": False})
        async with ollama([]) as client:
            result = await check_rerank_model_installed(OLLAMA, disabled, client=client)
        assert result.ok is True
        assert "disabled" in result.detail


class TestEmbeddingWidth:
    """The check that matters most — a width mismatch is otherwise silent."""

    async def test_passes_at_the_pinned_dimension(self) -> None:
        async with gateway(768) as client:
            result = await check_embedding_width(GATEWAY, CONFIG, client=client)
        assert result.ok is True
        assert "768 dimensions" in result.detail

    @pytest.mark.parametrize("wrong", [384, 512, 1024, 1536])
    async def test_fails_on_any_other_width(self, wrong: int) -> None:
        async with gateway(wrong) as client:
            result = await check_embedding_width(GATEWAY, CONFIG, client=client)
        assert result.ok is False
        assert f"returned {wrong} dimensions" in result.detail
        assert "768" in result.detail
        assert result.fix is not None and "make reembed" in result.fix

    async def test_fails_readably_when_the_gateway_is_down(self) -> None:
        async with gateway(768, status=503) as client:
            result = await check_embedding_width(GATEWAY, CONFIG, client=client)
        assert result.ok is False
        assert result.fix is not None and "make up" in result.fix


class TestFullRun:
    async def test_all_green_when_everything_is_in_place(self) -> None:
        async with ollama(INSTALLED) as oc, gateway(768) as gc:
            results = await run_preflight(
                {}, config=CONFIG, gateway=GATEWAY, ollama_client=oc, http_client=gc
            )
        assert [r.ok for r in results] == [True, True, True, True]
        assert "All checks passed" in format_report(results)

    async def test_reports_every_failure_not_just_the_first(self) -> None:
        """One run should tell you everything that is wrong."""
        async with ollama(["llama3.2:latest"]) as oc, gateway(384) as gc:
            results = await run_preflight(
                {}, config=CONFIG, gateway=GATEWAY, ollama_client=oc, http_client=gc
            )
        # Embedding model missing, rerank model missing, and the wrong width —
        # all three reported from one run rather than one at a time.
        assert [r.ok for r in results] == [True, False, False, False]
        report = format_report(results)
        assert "3 check(s) failed" in report
        assert "ollama pull nomic-embed-text" in report
        assert "ollama pull llama3.1:8b" in report

    async def test_report_prints_the_fix_verbatim(self) -> None:
        async with ollama([]) as oc, gateway(768) as gc:
            results = await run_preflight(
                {}, config=CONFIG, gateway=GATEWAY, ollama_client=oc, http_client=gc
            )
        assert "fix:  ollama pull nomic-embed-text" in format_report(results)

    async def test_a_down_ollama_fails_two_checks_not_one(self) -> None:
        async with ollama(None, unreachable=True) as oc, gateway(768) as gc:
            results = await run_preflight(
                {}, config=CONFIG, gateway=GATEWAY, ollama_client=oc, http_client=gc
            )
        assert [r.ok for r in results] == [False, False, False, True]


class TestHostUrl:
    def test_defaults_to_localhost_not_the_container_alias(self) -> None:
        """host.docker.internal does not resolve from the host shell."""
        assert ollama_base_url_for_host({}) == "http://localhost:11434"

    def test_respects_an_explicit_override(self) -> None:
        assert (
            ollama_base_url_for_host({"OLLAMA_BASE_URL_HOST": "http://ollama.box:11434"})
            == "http://ollama.box:11434"
        )


class TestShippedConfigMatchesTheGateway:
    """The pin is only meaningful if config and the gateway agree."""

    def test_embedding_config_loads(self) -> None:
        from src.contracts.embedding import embedding_config

        config = embedding_config()
        assert config.model.tag == "nomic-embed-text"
        assert config.model.dimensions == 768
        assert config.index.similarity == "cosine"

    def test_litellm_resolves_the_alias_to_the_pinned_tag(self) -> None:
        """A drifted alias would embed with a different model than we pinned."""
        from pathlib import Path

        import yaml

        from src.contracts.embedding import embedding_config

        config = embedding_config()
        litellm = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "docker/litellm/config.yaml").read_text(
                encoding="utf-8"
            )
        )
        entry = next(m for m in litellm["model_list"] if m["model_name"] == config.model.alias)
        assert entry["litellm_params"]["model"] == f"ollama/{config.model.tag}"
