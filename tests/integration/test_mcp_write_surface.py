"""The two Phase 4 MCP tools, and the roles that separate them.

WHY BOTH DIRECTIONS. Before Phase 4 every tool sat behind `kg-reader` on the
route, so "which role does this tool need" had exactly one possible answer.
Adding a WRITE tool to that route would have made every existing retriever able
to persist an answer, and nothing would have failed. The negative tests here go
both ways on purpose:

* `retriever-sa` holds `kg-reader` and NOT `draft-writer` — it may retrieve and
  may not save.
* `assembler-sa` holds `draft-writer` and NOT `kg-reader` — it may save and may
  not retrieve.

Either one alone would be consistent with a single shared role.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import pytest

from src.contracts import DraftedAnswer, RunState
from tests.live import mcp_base, require_mcp_server, require_write_api_and_keycloak

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

MCP_BASE = mcp_base()
KEYCLOAK = os.environ.get("KEYCLOAK_BASE", "http://localhost:8080")
REALM = os.environ.get("KEYCLOAK_REALM", "rfp")
TOKEN_URL = f"{KEYCLOAK}/realms/{REALM}/protocol/openid-connect/token"
WRITE_API = os.environ.get("WRITE_API_BASE", "http://localhost:8001")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module", autouse=True)
def _live() -> None:
    require_mcp_server()
    require_write_api_and_keycloak()


@pytest.fixture
async def http() -> Any:
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


async def token_for(client: httpx.AsyncClient, client_id: str, secret_env: str) -> str:
    secret = os.environ.get(secret_env)
    if not secret:
        pytest.skip(f"{secret_env} is not set")
    response = await client.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": secret,
        },
    )
    response.raise_for_status()
    token: str = response.json()["access_token"]
    return token


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def call(
    client: httpx.AsyncClient, name: str, payload: dict[str, Any], token: str
) -> httpx.Response:
    return await client.post(f"{MCP_BASE}/tools/{name}", json=payload, headers=bearer(token))


def answer(question_id: str = "GQ-001") -> dict[str, Any]:
    return DraftedAnswer(
        question_id=question_id,
        answer_text="A grounded synthetic answer.",
        source_ids=["ANS-0001"],
        confidence=0.82,
        needs_sme_review=False,
    ).model_dump(mode="json")


async def seed_run(client: httpx.AsyncClient, run_id: str) -> None:
    """A run row must exist before a draft can reference it (FK)."""
    from datetime import UTC, datetime

    from src.contracts import RunStage

    now = datetime.now(UTC)
    state = RunState(
        run_id=run_id,
        rfp_id=f"rfp-{run_id}",
        stage=RunStage.DRAFTING,
        started_at=now,
        updated_at=now,
    )
    token = await token_for(client, "drafter-sa", "DRAFTER_SA_SECRET")
    response = await client.put(
        f"{WRITE_API}/v1/runs/{run_id}",
        json=state.model_dump(mode="json"),
        headers=bearer(token),
    )
    response.raise_for_status()


class TestSaveDraft:
    async def test_the_drafter_can_save(self, http: httpx.AsyncClient) -> None:
        run_id = f"mcpw-{uuid.uuid4().hex[:10]}"
        await seed_run(http, run_id)
        token = await token_for(http, "drafter-sa", "DRAFTER_SA_SECRET")
        response = await call(http, "save_draft", {"run_id": run_id, "answer": answer()}, token)
        assert response.status_code == 200, response.text
        assert response.json()["written"] == "draft"

    async def test_the_row_records_the_original_caller(self, http: httpx.AsyncClient) -> None:
        """The token is FORWARDED, not replaced.

        If mcp-server minted its own service account for this, every row would
        say mcp-server wrote it — and "which principal produced this artifact"
        is the one question an audit trail is asked.
        """
        run_id = f"mcpw-{uuid.uuid4().hex[:10]}"
        await seed_run(http, run_id)
        token = await token_for(http, "drafter-sa", "DRAFTER_SA_SECRET")
        response = await call(http, "save_draft", {"run_id": run_id, "answer": answer()}, token)
        assert "mcp" not in response.json()["written_by"].lower()

    async def test_it_is_idempotent(self, http: httpx.AsyncClient) -> None:
        """Every write in this system is an upsert on a natural key, which is
        what makes resume safe."""
        run_id = f"mcpw-{uuid.uuid4().hex[:10]}"
        await seed_run(http, run_id)
        token = await token_for(http, "drafter-sa", "DRAFTER_SA_SECRET")
        first = await call(http, "save_draft", {"run_id": run_id, "answer": answer()}, token)
        second = await call(http, "save_draft", {"run_id": run_id, "answer": answer()}, token)
        assert first.json() == second.json()

    async def test_an_ungrounded_answer_is_refused_at_the_boundary(
        self, http: httpx.AsyncClient
    ) -> None:
        """`DraftedAnswer` refuses an uncited, unescalated answer, so nothing
        ungrounded reaches the table through this tool — and the refusal happens
        before a connection is opened."""
        run_id = f"mcpw-{uuid.uuid4().hex[:10]}"
        await seed_run(http, run_id)
        token = await token_for(http, "drafter-sa", "DRAFTER_SA_SECRET")
        bad = answer() | {"source_ids": []}
        response = await call(http, "save_draft", {"run_id": run_id, "answer": bad}, token)
        assert response.status_code == 422

    async def test_a_retriever_may_not_save(self, http: httpx.AsyncClient) -> None:
        """Direction one: kg-reader is not draft-writer."""
        token = await token_for(http, "retriever-sa", "RETRIEVER_SA_SECRET")
        response = await call(http, "save_draft", {"run_id": "any", "answer": answer()}, token)
        assert response.status_code == 403
        assert "draft-writer" in response.text

    async def test_the_role_is_checked_before_the_schema(self, http: httpx.AsyncClient) -> None:
        """A 422 naming the fields of a tool you may not invoke tells an
        unauthorised caller the shape of what it cannot call."""
        token = await token_for(http, "retriever-sa", "RETRIEVER_SA_SECRET")
        response = await call(http, "save_draft", {"nonsense": True}, token)
        assert response.status_code == 403

    async def test_no_token_is_401_not_403(self, http: httpx.AsyncClient) -> None:
        response = await http.post(
            f"{MCP_BASE}/tools/save_draft", json={"run_id": "x", "answer": answer()}
        )
        assert response.status_code == 401


class TestGetRunState:
    async def test_a_reader_can_read_a_run(self, http: httpx.AsyncClient) -> None:
        run_id = f"mcpr-{uuid.uuid4().hex[:10]}"
        await seed_run(http, run_id)
        token = await token_for(http, "retriever-sa", "RETRIEVER_SA_SECRET")
        response = await call(http, "get_run_state", {"run_id": run_id}, token)
        assert response.status_code == 200, response.text
        assert response.json()["state"]["run_id"] == run_id

    async def test_an_unknown_run_is_null_not_an_error(self, http: httpx.AsyncClient) -> None:
        """A legitimate answer. A 500 would tell an agent to retry something
        that will never succeed."""
        token = await token_for(http, "retriever-sa", "RETRIEVER_SA_SECRET")
        response = await call(http, "get_run_state", {"run_id": "never-existed"}, token)
        assert response.status_code == 200
        assert response.json()["state"] is None

    async def test_the_response_satisfies_the_published_contract(
        self, http: httpx.AsyncClient
    ) -> None:
        run_id = f"mcpr-{uuid.uuid4().hex[:10]}"
        await seed_run(http, run_id)
        token = await token_for(http, "retriever-sa", "RETRIEVER_SA_SECRET")
        response = await call(http, "get_run_state", {"run_id": run_id}, token)
        assert RunState.model_validate(response.json()["state"]).run_id == run_id

    async def test_a_caller_without_rfp_reader_is_refused(self, http: httpx.AsyncClient) -> None:
        """`ingest-sa` holds kg-reader and kg-writer, and no rfp-reader."""
        token = await token_for(http, "ingest-sa", "INGEST_SA_SECRET")
        response = await call(http, "get_run_state", {"run_id": "any"}, token)
        assert response.status_code == 403
        assert "rfp-reader" in response.text


class TestTheOtherDirection:
    """`assembler-sa` holds draft-writer and NOT kg-reader.

    Without this half, every assertion above would also pass on a server that
    still used one shared role for everything.
    """

    async def test_it_may_save_a_draft(self, http: httpx.AsyncClient) -> None:
        run_id = f"mcpa-{uuid.uuid4().hex[:10]}"
        await seed_run(http, run_id)
        token = await token_for(http, "assembler-sa", "ASSEMBLER_SA_SECRET")
        response = await call(http, "save_draft", {"run_id": run_id, "answer": answer()}, token)
        assert response.status_code == 200, response.text

    async def test_it_may_not_read_the_graph(self, http: httpx.AsyncClient) -> None:
        token = await token_for(http, "assembler-sa", "ASSEMBLER_SA_SECRET")
        response = await call(http, "get_evidence", {"answer_id": "ANS-0001"}, token)
        assert response.status_code == 403
        assert "kg-reader" in response.text

    async def test_the_manifest_stays_behind_the_read_role(self, http: httpx.AsyncClient) -> None:
        """Listing tools a caller may not invoke is an information leak in
        miniature, and that has not changed by adding two of them."""
        token = await token_for(http, "assembler-sa", "ASSEMBLER_SA_SECRET")
        response = await http.get(f"{MCP_BASE}/tools", headers=bearer(token))
        assert response.status_code == 403


class TestTheManifest:
    async def test_it_lists_all_eight(self, http: httpx.AsyncClient) -> None:
        token = await token_for(http, "retriever-sa", "RETRIEVER_SA_SECRET")
        response = await http.get(f"{MCP_BASE}/tools", headers=bearer(token))
        names = {tool["name"] for tool in response.json()["tools"]}
        assert {"save_draft", "get_run_state"} <= names
        assert len(names) == 8
