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

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from neo4j.exceptions import AuthError, ServiceUnavailable

from src.contracts.embedding import EmbeddingConfig
from src.gateway import preflight
from src.gateway.client import GatewayClient
from src.gateway.model_pins import GatewayModel
from src.gateway.ollama_admin import is_installed, normalise_tag
from src.gateway.preflight import (
    check_embedding_width,
    check_gateway_models_installed,
    check_ollama_reachable,
    check_tags_are_explicit,
    format_report,
    ollama_base_url_for_host,
    run_preflight,
)
from src.retrieval.calibration import (
    GEOMETRY,
    POPULATION,
    CalibrationArtifact,
    CalibrationError,
)

OLLAMA = "http://localhost:11434"
GATEWAY = GatewayClient(base_url="http://gateway.test", api_key="k")

CONFIG = EmbeddingConfig.model_validate(
    {
        "version": 1,
        "model": {
            "alias": "embed-model",
            "tag": "nomic-embed-text:v1.5",
            "dimensions": 768,
            "task_prefixes": {"document": "search_document: ", "query": "search_query: "},
        },
        "index": {
            "name": "question_embedding",
            "label": "Question",
            "property": "embedding",
            "similarity": "cosine",
        },
        "install_command": "ollama pull nomic-embed-text:v1.5",
    }
)


def model(alias: str, tag: str) -> GatewayModel:
    return GatewayModel(alias=alias, reference=f"ollama/{tag}", tag=tag)


MODELS = [
    model("triage-model", "llama3.2:3b"),
    model("rerank-model", "llama3.1:8b"),
    model("embed-model", "nomic-embed-text:v1.5"),
]

INSTALLED = ["llama3.1:8b", "llama3.2:3b", "nomic-embed-text:v1.5"]


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
    def test_latest_suffix_is_collapsed(self) -> None:
        assert normalise_tag("nomic-embed-text:latest") == "nomic-embed-text"

    def test_explicit_versions_are_preserved(self) -> None:
        assert normalise_tag("llama3.2:3b") == "llama3.2:3b"

    def test_an_exact_tag_matches(self) -> None:
        assert is_installed("llama3.2:3b", ["llama3.2:3b"]) is True

    def test_a_different_version_does_not_match(self) -> None:
        """3b and 8b are different models, not different names for one."""
        assert is_installed("llama3.2:3b", ["llama3.2:1b"]) is False


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


class TestGatewayModelsInstalled:
    """One loop over the parsed config, so a future alias is covered on arrival."""

    async def test_all_present(self) -> None:
        async with ollama(INSTALLED) as client:
            results = await check_gateway_models_installed(OLLAMA, MODELS, client=client)
        assert [r.ok for r in results] == [True, True, True]

    async def test_a_missing_model_names_its_own_pull_command(self) -> None:
        async with ollama(["llama3.2:3b", "nomic-embed-text:v1.5"]) as client:
            results = await check_gateway_models_installed(OLLAMA, MODELS, client=client)
        failed = [r for r in results if not r.ok]
        assert len(failed) == 1
        assert failed[0].fix == "ollama pull llama3.1:8b"

    async def test_aliases_sharing_a_tag_report_once(self) -> None:
        """Four aliases on one tag should not bury the other checks."""
        shared = [
            model("triage-model", "llama3.2:3b"),
            model("extract-assist-model", "llama3.2:3b"),
            model("loginterp-model", "llama3.2:3b"),
        ]
        async with ollama([]) as client:
            results = await check_gateway_models_installed(OLLAMA, shared, client=client)
        assert len(results) == 1
        assert "triage-model" in results[0].name
        assert "loginterp-model" in results[0].name

    async def test_unreachable_ollama_fails_every_model(self) -> None:
        async with ollama(None, unreachable=True) as client:
            results = await check_gateway_models_installed(OLLAMA, MODELS, client=client)
        assert [r.ok for r in results] == [False, False, False]

    async def test_an_empty_config_is_itself_a_failure(self) -> None:
        """A config that parsed to nothing means the check is silently vacuous."""
        async with ollama(INSTALLED) as client:
            results = await check_gateway_models_installed(OLLAMA, [], client=client)
        assert [r.ok for r in results] == [False]


class TestExplicitTags:
    def test_a_pinned_config_passes(self) -> None:
        assert check_tags_are_explicit(MODELS).ok is True

    def test_a_bare_reference_fails(self) -> None:
        result = check_tags_are_explicit([model("triage-model", "llama3.2")])
        assert result.ok is False
        assert "no tag" in result.detail

    def test_an_explicit_latest_fails(self) -> None:
        result = check_tags_are_explicit([model("embed-model", "nomic-embed-text:latest")])
        assert result.ok is False
        assert ":latest is not a pin" in result.detail


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
        assert result.fix is not None and "make reembed" in result.fix

    async def test_fails_readably_when_the_gateway_is_down(self) -> None:
        async with gateway(768, status=503) as client:
            result = await check_embedding_width(GATEWAY, CONFIG, client=client)
        assert result.ok is False
        assert result.fix is not None and "make up" in result.fix


class TestNeo4jAuth:
    """Amendment Q. A healthcheck cannot see authentication.

    Neo4j reports healthy as soon as it is listening, and `NEO4J_AUTH` applies
    only when the data volume is FIRST initialised — so a volume created before
    a credential change keeps the old password while compose, `.env` and the
    container environment all agree on the new one. Everything reads as correct
    and every connection is rejected.
    """

    async def test_a_rejected_credential_names_both_causes_in_order(self) -> None:
        """The URI first, the stale volume second — that is the order to check.

        A rejected credential looks identical whether the volume is stale or the
        URI points at another project's Neo4j on the conventional port. Naming
        only the volume sent one session to `make nuke` for a problem nuking
        could not fix; the wrong port was the actual cause.
        """
        driver = MagicMock()
        driver.verify_connectivity = AsyncMock(side_effect=AuthError("nope"))
        driver.close = AsyncMock()
        with patch.object(preflight, "AsyncGraphDatabase") as factory:
            factory.driver.return_value = driver
            result = await preflight.check_neo4j_auth(
                {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_PASSWORD": "x"}
            )
        assert result.ok is False
        assert result.fix is not None
        assert "NEO4J_BOLT_PORT_HOST" in result.fix
        assert "stale" in result.fix
        assert "make nuke" in result.fix
        # The URI it actually dialled, so the wrong-server case is visible.
        assert "bolt://localhost:7687" in result.detail

    async def test_an_unreachable_server_is_distinguished_from_a_bad_password(self) -> None:
        """Different causes, different fixes: start the stack vs reset the volume."""
        driver = MagicMock()
        driver.verify_connectivity = AsyncMock(side_effect=ServiceUnavailable("down"))
        driver.close = AsyncMock()
        with patch.object(preflight, "AsyncGraphDatabase") as factory:
            factory.driver.return_value = driver
            result = await preflight.check_neo4j_auth(
                {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_PASSWORD": "x"}
            )
        assert result.ok is False
        assert result.fix is not None and "make up" in result.fix
        assert "stale data volume" not in (result.fix or "")

    async def test_a_missing_password_is_caught_before_dialling(self) -> None:
        result = await preflight.check_neo4j_auth({"NEO4J_URI": "bolt://localhost:7687"})
        assert result.ok is False
        assert "NEO4J_PASSWORD is not set" in result.detail

    async def test_accepted_credentials_pass(self) -> None:
        driver = MagicMock()
        driver.verify_connectivity = AsyncMock(return_value=None)
        driver.close = AsyncMock()
        with patch.object(preflight, "AsyncGraphDatabase") as factory:
            factory.driver.return_value = driver
            result = await preflight.check_neo4j_auth(
                {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_PASSWORD": "x"}
            )
        assert result.ok is True
        driver.close.assert_awaited()

    async def test_the_driver_is_closed_even_when_auth_fails(self) -> None:
        """A leaked driver would keep the process alive after preflight exits."""
        driver = MagicMock()
        driver.verify_connectivity = AsyncMock(side_effect=AuthError("nope"))
        driver.close = AsyncMock()
        with patch.object(preflight, "AsyncGraphDatabase") as factory:
            factory.driver.return_value = driver
            await preflight.check_neo4j_auth(
                {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_PASSWORD": "x"}
            )
        driver.close.assert_awaited()


class TestCalibrationFreshness:
    """Amendment J. Retrieval is fail-closed on the calibration artifact, so a
    missing or stale one should surface here rather than as a run refusing
    partway through — correct behaviour, discovered at the worst moment.
    """

    def test_a_missing_artifact_fails_and_names_the_fix(self, tmp_path: Path) -> None:
        with patch.object(preflight, "load_for_current_corpus") as load:
            load.side_effect = CalibrationError("calibration artifact missing at /x")
            result = preflight.check_calibration_is_fresh()
        assert result.ok is False
        assert result.fix is not None and "make calibrate" in result.fix

    def test_a_stale_artifact_fails(self) -> None:
        """Stale is the case worth naming: it loads fine and describes nothing."""
        with patch.object(preflight, "load_for_current_corpus") as load:
            load.side_effect = CalibrationError(
                "calibration was measured over corpus abc but the corpus is now def"
            )
            result = preflight.check_calibration_is_fresh()
        assert result.ok is False
        assert "corpus" in result.detail

    def test_a_current_artifact_passes_and_reports_the_floor(self) -> None:
        """The derived floor is shown, because it is the number that matters."""
        artifact = CalibrationArtifact(
            geometry=GEOMETRY,
            population=POPULATION,
            embed_model_tag="nomic-embed-text:v1.5",
            corpus_hash="abcdef0123456789",
            computed_at="2026-08-13T00:00:00+00:00",
            background_pair_count=12192,
            same_topic_pair_count=240,
            bg_p50=0.60,
            bg_p95=0.72,
            bg_p99=0.75,
            same_topic_p05=0.85,
            same_topic_p50=0.90,
            derived_floor=0.666667,
        )
        with patch.object(preflight, "load_for_current_corpus", return_value=artifact):
            result = preflight.check_calibration_is_fresh()
        assert result.ok is True
        assert "abcdef0123456789" in result.detail
        assert "0.6667" in result.detail


class TestFullRun:
    async def test_all_green_when_everything_is_in_place(self) -> None:
        """Ingest's variant: no calibration check, because nothing is calibrated
        before the corpus it describes has been loaded."""
        async with ollama(INSTALLED) as oc, gateway(768) as gc:
            results = await run_preflight(
                {},
                config=CONFIG,
                models=MODELS,
                gateway=GATEWAY,
                ollama_client=oc,
                http_client=gc,
                include_calibration=False,
                include_neo4j=False,
            )
        # reachable + explicit-tags + one per distinct tag + width
        assert len(results) == 6
        assert all(r.ok for r in results)
        assert "All checks passed" in format_report(results)

    async def test_the_calibration_check_is_included_by_default(self) -> None:
        """Standalone `make preflight` checks it; `make ingest` does not."""
        async with ollama(INSTALLED) as oc, gateway(768) as gc:
            with patch.object(preflight, "load_for_current_corpus") as load:
                load.side_effect = CalibrationError("missing")
                results = await run_preflight(
                    {},
                    config=CONFIG,
                    models=MODELS,
                    gateway=GATEWAY,
                    ollama_client=oc,
                    http_client=gc,
                    include_neo4j=False,
                )
        assert len(results) == 7
        assert results[-1].ok is False

    async def test_reports_every_failure_not_just_the_first(self) -> None:
        """One run should tell you everything that is wrong."""
        async with ollama(["llama3.2:3b"]) as oc, gateway(384) as gc:
            results = await run_preflight(
                {},
                config=CONFIG,
                models=MODELS,
                gateway=GATEWAY,
                ollama_client=oc,
                http_client=gc,
                include_calibration=False,
                include_neo4j=False,
            )
        report = format_report(results)
        assert "3 check(s) failed" in report
        assert "ollama pull llama3.1:8b" in report
        assert "ollama pull nomic-embed-text:v1.5" in report

    async def test_a_drifting_config_fails_the_run(self) -> None:
        """Implicit tags are a preflight failure, not a style note."""
        drifting = [model("triage-model", "llama3.2")]
        async with ollama(["llama3.2:3b"]) as oc, gateway(768) as gc:
            results = await run_preflight(
                {},
                config=CONFIG,
                models=drifting,
                gateway=GATEWAY,
                ollama_client=oc,
                http_client=gc,
            )
        assert any(not r.ok and "explicitly tagged" in r.name for r in results)

    async def test_report_prints_the_fix_verbatim(self) -> None:
        async with ollama([]) as oc, gateway(768) as gc:
            results = await run_preflight(
                {}, config=CONFIG, models=MODELS, gateway=GATEWAY, ollama_client=oc, http_client=gc
            )
        assert "fix:  ollama pull llama3.1:8b" in format_report(results)


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
    def test_embedding_config_loads(self) -> None:
        from src.contracts.embedding import embedding_config

        config = embedding_config()
        assert config.model.tag == "nomic-embed-text:v1.5"
        assert config.model.dimensions == 768
        assert config.index.similarity == "cosine"
