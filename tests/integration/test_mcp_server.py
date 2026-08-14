"""The six tools, through the running container, over real HTTP.

WHAT THIS ADDS OVER THE UNIT SUITE. `tests/unit/test_mcp_tools.py` proves the
tool registry is well-formed and that no contract could carry Cypher; it proves
nothing about the deployed service, because it never starts one. Everything here
crosses a real network boundary into a container that was built from the
Dockerfile, authenticated by the real Keycloak, reading a real graph.

The gap that shape closes is specific. Auth, the schema boundary and
confidentiality are each enforced by code the unit suite already covers — and
each of them can be *wired* wrong without any of that code changing: a container
missing an environment variable, an issuer that differs from write-api's by a
hostname, a role the realm never actually grants. Those failures live in the
wiring, and only a live call can see them.

**Confidentiality is proved through the tool boundary, not only under it.**
`tests/integration/test_graph_queries.py` already proves `find_similar_questions`
excludes Bluepine's ANS-0014 from a Meridian retrieval. That proof is about the
query. This one is about the *path an agent actually takes* — HTTP, JSON,
contract validation, session, query, envelope, JSON again — because the tool
layer is where `requesting_customer` could be dropped, defaulted or ignored
without touching a line of the query it wraps.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

import httpx
import pytest

from src.contracts.embedding import EmbedRole, embedding_config
from src.gateway.client import GatewayClient
from src.gateway.fake_embedder import fake_embedding, fake_embeddings_enabled
from src.mcp_server.tools import TOOLS_BY_NAME
from tests.live import mcp_base, require_mcp_server

pytestmark = pytest.mark.integration

MCP_BASE = mcp_base()
KEYCLOAK_BASE = os.environ.get("KEYCLOAK_BASE", "http://localhost:8080")
WRITE_API_BASE = os.environ.get("WRITE_API_BASE", "http://localhost:8001")
REALM = os.environ.get("KEYCLOAK_REALM", "rfp")
TOKEN_URL = f"{KEYCLOAK_BASE}/realms/{REALM}/protocol/openid-connect/token"

#: The compose container, for the audit-log assertion. Named by variable so a
#: differently-named project can still run this rather than silently skipping.
MCP_CONTAINER = os.environ.get("MCP_CONTAINER", "rfp-workflow-mcp-server-1")

MERIDIAN = "Meridian Insurance Group"
BLUEPINE = "Bluepine Health Systems"
#: The corpus's only confidential answer, and the question it answers.
SECRET_QUESTION = "HQ-0014"
SECRET_ANSWER = "ANS-0014"


@pytest.fixture(scope="module", autouse=True)
def _live_mcp() -> None:
    require_mcp_server()


@pytest.fixture
async def http() -> Any:
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


async def _token(client: httpx.AsyncClient, client_id: str, secret_env: str) -> str:
    response = await client.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": os.environ[secret_env],
        },
    )
    response.raise_for_status()
    token: str = response.json()["access_token"]
    return token


async def retriever_token(client: httpx.AsyncClient) -> str:
    """The caller these tools exist for: `rfp-reader` + `kg-reader`."""
    return await _token(client, "retriever-sa", "RETRIEVER_SA_SECRET")


async def no_kg_reader_token(client: httpx.AsyncClient) -> str:
    """`extractor-sa` holds `rfp-reader` only.

    A real, valid, correctly-audienced token that the realm genuinely issues —
    which is what separates a 403 from a 401 and makes the distinction testable
    rather than asserted.
    """
    return await _token(client, "extractor-sa", "EXTRACTOR_SA_SECRET")


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def call_tool(
    client: httpx.AsyncClient, name: str, payload: dict[str, Any], token: str
) -> httpx.Response:
    return await client.post(f"{MCP_BASE}/tools/{name}", json=payload, headers=bearer(token))


async def embed_query(text: str) -> list[float]:
    """Query-side embedding, by whichever path this environment declares.

    The stand-in when `RFP_FAKE_EMBEDDINGS` is set, because CI's graph was
    ingested with the stand-in and its vectors ARE stand-in vectors; the gateway
    otherwise. Asking the environment rather than hardcoding either is the point
    — a test that hardcoded `fake_embedding` could not notice the environment
    failing to declare itself.
    """
    config = embedding_config()
    if fake_embeddings_enabled():
        return fake_embedding(text, config.model.dimensions, role=EmbedRole.QUERY)
    client = GatewayClient.from_env()
    vectors = await client.embed([text], alias=config.model.alias, role=EmbedRole.QUERY)
    return vectors[0]


def valid_payloads(embedding: list[float]) -> dict[str, dict[str, Any]]:
    """One valid input per tool. Keyed by tool name so the parametrisation below
    is derived from `TOOLS_BY_NAME` rather than from a second hand-written list
    that could fall out of step with it."""
    return {
        "retrieve_candidates": {
            "embedding": embedding,
            "requesting_customer": MERIDIAN,
            "k": 5,
        },
        "get_full_answers": {"question_id": "HQ-0001", "requesting_customer": MERIDIAN},
        "get_evidence": {"answer_id": "ANS-0001"},
        "entity_exists": {"name": "CloudNova Partners", "entity_type": "vendor"},
        "get_sme_for_capability": {"capability_id": "CAP-0001"},
        "coverage_gaps": {"rfp_id": "HRFP-CUS-0001"},
    }


# ---------------------------------------------------------------------------


class TestTheServiceIsUp:
    async def test_health_reports_the_build_it_is_running(self, http: httpx.AsyncClient) -> None:
        """Unauthenticated on purpose: a healthcheck a container cannot call is
        not a healthcheck, and the response carries no graph data."""
        response = await http.get(f"{MCP_BASE}/health")
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "ok"
        assert response.json()["git_sha"]


class TestEveryToolAnswersAnAuthorisedCaller:
    """All six, individually. A suite that exercised one tool and asserted the
    others 'work the same way' would miss exactly the tool whose handler was
    wired to the wrong query function."""

    @pytest.mark.parametrize("tool_name", sorted(TOOLS_BY_NAME))
    async def test_retriever_sa_can_invoke_it(
        self, http: httpx.AsyncClient, tool_name: str
    ) -> None:
        token = await retriever_token(http)
        payload = valid_payloads(await embed_query("cloud migration approach"))[tool_name]
        response = await call_tool(http, tool_name, payload, token)
        assert response.status_code == 200, response.text

    @pytest.mark.parametrize("tool_name", sorted(TOOLS_BY_NAME))
    async def test_its_response_satisfies_its_published_output_contract(
        self, http: httpx.AsyncClient, tool_name: str
    ) -> None:
        """A 200 is not evidence the envelope is the one the manifest promises.

        Validating the body against the tool's own output model closes that: a
        handler returning a bare list, or the wrong envelope, is a 200 with a
        body no caller could parse against the published schema.
        """
        token = await retriever_token(http)
        payload = valid_payloads(await embed_query("cloud migration approach"))[tool_name]
        response = await call_tool(http, tool_name, payload, token)
        assert response.status_code == 200, response.text
        TOOLS_BY_NAME[tool_name].output_model.model_validate(response.json())

    async def test_the_manifest_lists_exactly_the_six(self, http: httpx.AsyncClient) -> None:
        token = await retriever_token(http)
        response = await http.get(f"{MCP_BASE}/tools", headers=bearer(token))
        assert response.status_code == 200, response.text
        assert {tool["name"] for tool in response.json()["tools"]} == set(TOOLS_BY_NAME)


class TestTheRoleBoundary:
    """401 is 'I do not know who you are'; 403 is 'I know, and you may not'."""

    async def test_no_token_is_401(self, http: httpx.AsyncClient) -> None:
        response = await http.post(f"{MCP_BASE}/tools/get_evidence", json={"answer_id": "ANS-0001"})
        assert response.status_code == 401

    async def test_a_token_without_kg_reader_is_403(self, http: httpx.AsyncClient) -> None:
        token = await no_kg_reader_token(http)
        response = await call_tool(http, "get_evidence", {"answer_id": "ANS-0001"}, token)
        assert response.status_code == 403

    async def test_the_manifest_is_behind_the_same_role(self, http: httpx.AsyncClient) -> None:
        """Listing what a caller may not invoke is an information leak in
        miniature — it hands an unauthorised caller the attack surface."""
        token = await no_kg_reader_token(http)
        response = await http.get(f"{MCP_BASE}/tools", headers=bearer(token))
        assert response.status_code == 403

    async def test_the_refusal_names_the_role_and_the_caller(self, http: httpx.AsyncClient) -> None:
        token = await no_kg_reader_token(http)
        response = await call_tool(http, "get_evidence", {"answer_id": "ANS-0001"}, token)
        detail = response.json()["detail"]
        assert "kg-reader" in detail
        assert "extractor-sa" in detail

    async def test_the_semantics_are_write_apis_because_the_code_is(
        self, http: httpx.AsyncClient
    ) -> None:
        """The same caller, refused by both services, refused identically.

        mcp-server imports write-api's `require_role` rather than reimplementing
        it, and this is the assertion that keeps that true: the two refusals are
        compared field for field, with only the role name differing. Two
        implementations kept in step by hand would drift here first — a different
        status, a `message` where the other says `detail`, a body an agent's
        error handling would have to special-case per service.
        """
        token = await no_kg_reader_token(http)

        from_mcp = await call_tool(http, "get_evidence", {"answer_id": "ANS-0001"}, token)
        from_write_api = await http.put(
            f"{WRITE_API_BASE}/v1/drafts/run-mcp-integration/q-mcp-integration",
            json={
                "run_id": "run-mcp-integration",
                "question_id": "q-mcp-integration",
                "answer_markdown": "unreachable",
                "citations": [],
                "confidence": 0.5,
                "status": "drafted",
            },
            headers=bearer(token),
        )

        assert from_mcp.status_code == from_write_api.status_code == 403
        assert set(from_mcp.json()) == set(from_write_api.json()) == {"detail"}
        # Same sentence, same shape, differing only in the role each guards.
        assert from_mcp.json()["detail"] == from_write_api.json()["detail"].replace(
            "draft-writer", "kg-reader"
        )


class TestTheSchemaBoundary:
    """Malformed input fails HERE, with a message that names what was wrong.

    An agent is not a debugger. A 422 that says only 'validation error' costs a
    retry that cannot succeed, because nothing in the response tells the caller
    what to change.
    """

    async def test_an_unknown_field_is_refused_and_named(self, http: httpx.AsyncClient) -> None:
        """`extra="forbid"`, end to end. The misspelling matters: served
        silently, this would be a retrieval computed without a customer."""
        token = await retriever_token(http)
        response = await call_tool(
            http,
            "get_full_answers",
            {"question_id": "HQ-0001", "requesting_custmer": MERIDIAN},
            token,
        )
        assert response.status_code == 422
        body = response.json()["detail"]
        assert body["tool"] == "get_full_answers"
        assert any("requesting_custmer" in error["field"] for error in body["errors"])

    async def test_a_missing_required_field_is_refused_and_named(
        self, http: httpx.AsyncClient
    ) -> None:
        token = await retriever_token(http)
        response = await call_tool(http, "get_full_answers", {"question_id": "HQ-0001"}, token)
        assert response.status_code == 422
        fields = {error["field"] for error in response.json()["detail"]["errors"]}
        assert "requesting_customer" in fields

    async def test_a_wrongly_typed_field_is_refused_and_named(
        self, http: httpx.AsyncClient
    ) -> None:
        token = await retriever_token(http)
        response = await call_tool(
            http,
            "retrieve_candidates",
            {"embedding": "not-a-vector", "requesting_customer": MERIDIAN},
            token,
        )
        assert response.status_code == 422
        fields = {error["field"] for error in response.json()["detail"]["errors"]}
        assert "embedding" in fields

    async def test_an_out_of_range_value_is_refused(self, http: httpx.AsyncClient) -> None:
        """`k` is bounded at 50. Unbounded, one call could drain the graph."""
        token = await retriever_token(http)
        response = await call_tool(
            http,
            "retrieve_candidates",
            {
                "embedding": await embed_query("anything"),
                "requesting_customer": MERIDIAN,
                "k": 5000,
            },
            token,
        )
        assert response.status_code == 422
        assert "k" in {error["field"] for error in response.json()["detail"]["errors"]}

    async def test_a_wrong_width_embedding_says_both_widths(self, http: httpx.AsyncClient) -> None:
        """The handler's own complaint, not the schema's.

        Width is a fact about the deployed index, so the contract cannot state
        it; the message carries both numbers because 'wrong dimensions' without
        them is not actionable.
        """
        token = await retriever_token(http)
        expected = embedding_config().model.dimensions
        response = await call_tool(
            http,
            "retrieve_candidates",
            {"embedding": [0.1] * 7, "requesting_customer": MERIDIAN},
            token,
        )
        assert response.status_code == 422
        message = response.json()["detail"]["message"]
        assert "7" in message and str(expected) in message

    async def test_an_unknown_tool_is_404_and_says_there_is_no_raw_query(
        self, http: httpx.AsyncClient
    ) -> None:
        """The refusal doubles as documentation, at the moment an agent is most
        likely to be reaching for an escape hatch that does not exist."""
        token = await retriever_token(http)
        response = await call_tool(http, "run_cypher", {"cypher": "MATCH (n) RETURN n"}, token)
        assert response.status_code == 404
        assert "no raw-query tool" in response.json()["detail"]

    async def test_the_role_is_checked_before_the_schema(self, http: httpx.AsyncClient) -> None:
        """Order matters: a 422 to an unauthorised caller confirms a tool's
        contract to someone who may not use it."""
        token = await no_kg_reader_token(http)
        response = await call_tool(http, "get_full_answers", {"garbage": True}, token)
        assert response.status_code == 403


class TestConfidentialityThroughTheToolBoundary:
    """The Bluepine/Meridian proof, repeated where an agent actually stands.

    Under the tool layer this is already proved against the query. Repeated here
    because the tool layer is a place `requesting_customer` could be dropped,
    defaulted, or read from the wrong field without the query changing at all.
    """

    async def test_bluepines_answer_is_in_no_candidate_list_for_meridian(
        self, http: httpx.AsyncClient
    ) -> None:
        """The strongest query available: the confidential question's own text,
        which makes its answer the nearest possible neighbour."""
        token = await retriever_token(http)
        response = await call_tool(
            http,
            "retrieve_candidates",
            {
                "embedding": await embed_query(_secret_question_text()),
                "requesting_customer": MERIDIAN,
                "k": 50,
            },
            token,
        )
        assert response.status_code == 200, response.text
        candidates = response.json()["candidates"]
        assert candidates, "a k=50 retrieval returning nothing would pass vacuously"

        reachable = {answer_id for c in candidates for answer_id in c["answer_ids"]}
        assert SECRET_ANSWER not in reachable
        assert SECRET_QUESTION not in {c["question_id"] for c in candidates}

    async def test_the_same_call_for_bluepine_does_return_it(self, http: httpx.AsyncClient) -> None:
        """Confidentiality is not deletion.

        Without this, the test above passes just as well against a tool that
        returns nothing to anyone — and a retrieval layer that had quietly
        stopped working would read as a security success.
        """
        token = await retriever_token(http)
        response = await call_tool(
            http,
            "retrieve_candidates",
            {
                "embedding": await embed_query(_secret_question_text()),
                "requesting_customer": BLUEPINE,
                "k": 50,
            },
            token,
        )
        assert response.status_code == 200, response.text
        candidates = response.json()["candidates"]
        reachable = {answer_id for c in candidates for answer_id in c["answer_ids"]}
        assert SECRET_ANSWER in reachable

    async def test_get_full_answers_hides_it_from_meridian(self, http: httpx.AsyncClient) -> None:
        """The second customer-scoped tool, asked directly for the question that
        owns the secret. Retrieval is not the only way to name an id."""
        token = await retriever_token(http)
        response = await call_tool(
            http,
            "get_full_answers",
            {"question_id": SECRET_QUESTION, "requesting_customer": MERIDIAN},
            token,
        )
        assert response.status_code == 200, response.text
        assert SECRET_ANSWER not in {a["answer_id"] for a in response.json()["answers"]}

    async def test_get_full_answers_returns_it_to_bluepine(self, http: httpx.AsyncClient) -> None:
        token = await retriever_token(http)
        response = await call_tool(
            http,
            "get_full_answers",
            {"question_id": SECRET_QUESTION, "requesting_customer": BLUEPINE},
            token,
        )
        assert response.status_code == 200, response.text
        assert SECRET_ANSWER in {a["answer_id"] for a in response.json()["answers"]}

    async def test_the_customer_cannot_be_omitted(self, http: httpx.AsyncClient) -> None:
        """No default means no default, over HTTP.

        An optional visibility parameter is one forgotten argument away from a
        leak, and a tool boundary is exactly where arguments get forgotten.
        """
        token = await retriever_token(http)
        response = await call_tool(
            http,
            "retrieve_candidates",
            {"embedding": await embed_query("anything"), "k": 5},
            token,
        )
        assert response.status_code == 422
        fields = {error["field"] for error in response.json()["detail"]["errors"]}
        assert "requesting_customer" in fields


class TestTheCallerIsInTheLog:
    """A read that surfaced confidential material must be as auditable as a write.

    Read from the CONTAINER'S OWN log rather than a caplog handler: the claim is
    about what the deployed service records, and an in-process capture would
    prove only that the logging call exists in the source.
    """

    @pytest.fixture(autouse=True)
    def _docker(self) -> None:
        if _docker_cli() is None:
            pytest.skip("docker CLI not available; cannot read the container's log")

    def container_log(self) -> str:
        resolved = _docker_cli()
        assert resolved is not None  # the fixture skipped otherwise
        result = subprocess.run(  # noqa: S603
            [resolved, "logs", "--tail", "400", MCP_CONTAINER],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            # Both streams. The WSL shim reports its refusal on stdout, so a
            # stderr-only message left this skip blank and unexplained — which is
            # how the docker.exe ordering above went unnoticed for a round.
            reason = (result.stderr.strip() or result.stdout.strip() or "no output").splitlines()
            pytest.skip(f"cannot read logs for {MCP_CONTAINER}: {reason[0]}")
        return result.stdout + result.stderr

    async def test_a_successful_call_logs_the_subject_and_the_tool(
        self, http: httpx.AsyncClient
    ) -> None:
        token = await retriever_token(http)
        subject = _subject_of(token)
        response = await call_tool(http, "get_evidence", {"answer_id": "ANS-0001"}, token)
        assert response.status_code == 200, response.text

        log = self.container_log()
        assert f"subject={subject}" in log
        assert "tool=get_evidence" in log

    async def test_a_refused_call_is_logged_too(self, http: httpx.AsyncClient) -> None:
        """An audit trail that records only successes cannot answer the question
        it exists for: what did this caller try?"""
        token = await retriever_token(http)
        subject = _subject_of(token)
        response = await call_tool(http, "get_full_answers", {"question_id": "HQ-0001"}, token)
        assert response.status_code == 422

        log = self.container_log()
        assert "mcp.tool.invalid" in log
        assert f"subject={subject}" in log


class TestNoRawQueryPathExistsOnTheLiveService:
    """The structural claim, restated against what the container actually serves.

    `tests/unit/test_mcp_tools.py` asserts this over the in-process registry.
    Both are needed: the unit form catches a bad tool the moment it is written,
    and this one catches a container serving a manifest that is not the one this
    repo builds — an older image, a stale layer, a second registry.
    """

    #: Same list the unit assertions use, imported by value rather than by
    #: reference so a weakening there does not silently weaken this too.
    QUERY_SMELLS = (
        "cypher",
        "query_string",
        "raw",
        "statement",
        "sql",
        "script",
        "eval",
        "exec",
    )

    async def test_no_served_tool_is_named_like_a_raw_query(self, http: httpx.AsyncClient) -> None:
        token = await retriever_token(http)
        response = await http.get(f"{MCP_BASE}/tools", headers=bearer(token))
        for tool in response.json()["tools"]:
            assert not any(smell in tool["name"].lower() for smell in self.QUERY_SMELLS)

    async def test_no_served_input_schema_has_a_field_that_could_carry_one(
        self, http: httpx.AsyncClient
    ) -> None:
        token = await retriever_token(http)
        response = await http.get(f"{MCP_BASE}/tools", headers=bearer(token))
        for tool in response.json()["tools"]:
            for field in tool["input_schema"].get("properties", {}):
                assert not any(smell in field.lower() for smell in self.QUERY_SMELLS), (
                    f"{tool['name']}.{field}"
                )

    async def test_every_served_input_schema_forbids_extra_fields(
        self, http: httpx.AsyncClient
    ) -> None:
        """`additionalProperties: false` is how `extra="forbid"` reaches a caller
        reading the manifest — without it an agent is told unknown fields are
        acceptable, and only finds out otherwise at call time."""
        token = await retriever_token(http)
        response = await http.get(f"{MCP_BASE}/tools", headers=bearer(token))
        for tool in response.json()["tools"]:
            assert tool["input_schema"].get("additionalProperties") is False, tool["name"]


# ---------------------------------------------------------------------------
# Helpers that read fixtures or tokens
# ---------------------------------------------------------------------------


def _docker_cli() -> str | None:
    """The docker CLI, under either name it goes by here.

    `docker` on a Linux runner; `docker.exe` from WSL on the Windows build host,
    where Docker Desktop runs on the Windows side and WSL interop is how a Linux
    shell reaches it. Naming both is the difference between this assertion
    RUNNING on the developing machine and being skipped there — and it is the
    only assertion in the suite that the audit trail exists at all, so a skip
    would mean nobody had checked.

    That is not hypothetical: the per-call log was silently discarded by uvicorn
    until this test ran against the container and found nothing there.

    ORDER MATTERS on the Windows host. Docker Desktop puts BOTH names on the
    PATH a WSL shell inherits, and the Linux-side `docker` is a shim that exits
    non-zero with "could not be found in this WSL 2 distro" unless WSL
    integration is enabled for the distro. Preferring `docker.exe` picks the one
    that answers; on a Linux runner `docker.exe` does not exist, so the order
    costs nothing there.
    """
    for name in ("docker.exe", "docker"):
        resolved = shutil.which(name)
        if resolved is not None:
            return resolved
    return None


def _secret_question_text() -> str:
    from pathlib import Path

    fixtures = Path(__file__).resolve().parents[2] / "fixtures"
    with (fixtures / "qa_pairs.json").open(encoding="utf-8") as handle:
        pairs = json.load(handle)
    secret = next(pair for pair in pairs if pair["confidential"])
    assert secret["answer_id"] == SECRET_ANSWER, (
        "the corpus's confidential answer moved; this module's constants name "
        f"{SECRET_ANSWER} but the fixtures now say {secret['answer_id']}"
    )
    text: str = secret["question"]
    return text


def _subject_of(token: str) -> str:
    """The `sub` claim, read WITHOUT verifying.

    Legitimate here and nowhere else: this is a test asserting what the server
    logged about a token the server itself verified, so the claim is being used
    as a lookup key, not as an authorisation decision.
    """
    import base64

    payload = token.split(".")[1]
    padded = payload + "=" * (-len(payload) % 4)
    claims: dict[str, Any] = json.loads(base64.urlsafe_b64decode(padded))
    subject: str = claims["sub"]
    return subject
