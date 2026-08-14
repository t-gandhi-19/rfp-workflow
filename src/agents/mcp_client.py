"""The agents' only door to the graph (CLAUDE.md rule 4).

Agents never write Cypher. They call named tools on the MCP server, which
executes the tested, parameterised functions in `src/graph/queries.py` and
nothing else — there is no raw-query endpoint to reach for.

IDENTITY IS PER AGENT, not per process. `retriever-sa` and `drafter-sa` hold
different roles, and the tool call carries whichever token belongs to the agent
making it, so the server's audit log records who asked rather than recording
that "the pipeline" asked. Passing one shared token would make every refusal
test in `tests/integration/test_mcp_server.py` a statement about a service
account no agent actually is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

from src.contracts import EntityCheckResult, EntityType


class McpError(RuntimeError):
    """A tool call failed. The caller escalates its question; it does not retry.

    The graph is not flaky in a way a second identical call fixes — a 403 is a
    role that was never granted, a 422 is a payload that will not become valid,
    and a 404 is a tool that does not exist. Retrying any of them spends the
    budget to receive the same answer.
    """


def mcp_base() -> str:
    return os.environ.get("MCP_BASE", f"http://localhost:{os.environ.get('MCP_PORT_HOST', '8002')}")


def keycloak_base() -> str:
    return os.environ.get(
        "KEYCLOAK_BASE", f"http://localhost:{os.environ.get('KEYCLOAK_PORT_HOST', '8080')}"
    )


@dataclass
class McpClient:
    """One agent's authenticated tool caller."""

    client_id: str
    secret_env: str
    base_url: str = field(default_factory=mcp_base)
    keycloak_url: str = field(default_factory=keycloak_base)
    realm: str = field(default_factory=lambda: os.environ.get("KEYCLOAK_REALM", "rfp"))
    timeout: float = 30.0
    _token: str | None = field(default=None, repr=False)

    async def _authenticate(self, client: httpx.AsyncClient) -> str:
        if self._token is not None:
            return self._token
        secret = os.environ.get(self.secret_env)
        if not secret:
            raise McpError(
                f"{self.secret_env} is not set; {self.client_id} cannot obtain a token. "
                "Load it with: set -a && . ./.env && set +a"
            )
        try:
            response = await client.post(
                f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": secret,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise McpError(f"{self.client_id} could not obtain a token: {exc}") from exc
        token: str = response.json()["access_token"]
        self._token = token
        return token

    async def call(
        self, tool: str, payload: dict[str, object], *, client: httpx.AsyncClient
    ) -> dict[str, object]:
        """Invoke one named tool and return its typed envelope as JSON."""
        token = await self._authenticate(client)
        try:
            response = await client.post(
                f"{self.base_url}/tools/{tool}",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            raise McpError(f"mcp-server unreachable for '{tool}': {exc}") from exc
        if response.status_code != 200:
            raise McpError(
                f"mcp-server refused '{tool}' for {self.client_id}: "
                f"{response.status_code} {response.text[:300]}"
            )
        body: dict[str, object] = response.json()
        return body

    # -- the synchronous door, for crewAI tools ---------------------------

    def _authenticate_sync(self, client: httpx.Client) -> str:
        if self._token is not None:
            return self._token
        secret = os.environ.get(self.secret_env)
        if not secret:
            raise McpError(f"{self.secret_env} is not set; {self.client_id} cannot obtain a token")
        try:
            response = client.post(
                f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": secret,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise McpError(f"{self.client_id} could not obtain a token: {exc}") from exc
        token: str = response.json()["access_token"]
        self._token = token
        return token

    def call_sync(self, tool: str, payload: dict[str, object]) -> dict[str, object]:
        """The same call, from a synchronous caller.

        crewAI invokes tools synchronously from inside an agent's own loop. The
        alternative — driving the async client from that thread — means nesting
        an event loop inside the one already running the controller's fan-out,
        which deadlocks rather than failing cleanly. A separate sync path is
        less elegant and considerably easier to reason about.
        """
        with httpx.Client(timeout=self.timeout) as client:
            token = self._authenticate_sync(client)
            try:
                response = client.post(
                    f"{self.base_url}/tools/{tool}",
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )
            except httpx.HTTPError as exc:
                raise McpError(f"mcp-server unreachable for '{tool}': {exc}") from exc
        if response.status_code != 200:
            raise McpError(
                f"mcp-server refused '{tool}' for {self.client_id}: "
                f"{response.status_code} {response.text[:300]}"
            )
        body: dict[str, object] = response.json()
        return body

    # -- the three tiers, and the two grounding tools ---------------------

    async def retrieve_candidates(
        self,
        *,
        embedding: list[float],
        domain: str,
        requesting_customer: str,
        k: int,
        client: httpx.AsyncClient,
    ) -> list[dict[str, object]]:
        """Tier 1. Summaries only — full text is a separate, explicit fetch."""
        body = await self.call(
            "retrieve_candidates",
            {
                "embedding": embedding,
                "domain": domain,
                "requesting_customer": requesting_customer,
                "k": k,
            },
            client=client,
        )
        candidates: list[dict[str, object]] = body["candidates"]  # type: ignore[assignment]
        return candidates

    async def get_full_answers(
        self, *, question_id: str, requesting_customer: str, client: httpx.AsyncClient
    ) -> list[dict[str, object]]:
        """Tier 2. Full answer text, still filtered to what this customer may see."""
        body = await self.call(
            "get_full_answers",
            {"question_id": question_id, "requesting_customer": requesting_customer},
            client=client,
        )
        answers: list[dict[str, object]] = body["answers"]  # type: ignore[assignment]
        return answers

    async def get_evidence(
        self, *, answer_id: str, client: httpx.AsyncClient
    ) -> dict[str, object] | None:
        """Tier 3. Provenance: author, customer, outcome, evidence codes."""
        body = await self.call("get_evidence", {"answer_id": answer_id}, client=client)
        lineage = body.get("lineage")
        return lineage if isinstance(lineage, dict) else None

    async def entity_exists(
        self, *, name: str, entity_type: EntityType, client: httpx.AsyncClient
    ) -> EntityCheckResult:
        """The closed-world grounding check (build prompt §15)."""
        body = await self.call(
            "entity_exists",
            {"name": name, "entity_type": entity_type.value},
            client=client,
        )
        return EntityCheckResult.model_validate(body["resolution"])

    async def get_sme_for_capability(
        self, *, capability_id: str, client: httpx.AsyncClient
    ) -> list[dict[str, object]]:
        body = await self.call(
            "get_sme_for_capability", {"capability_id": capability_id}, client=client
        )
        smes: list[dict[str, object]] = body["smes"]  # type: ignore[assignment]
        return smes


#: The identities the agents hold. `retriever-sa` reads the graph; `drafter-sa`
#: also writes drafts. Neither holds `submitter`, which is granted to nothing.
RETRIEVER = ("retriever-sa", "RETRIEVER_SA_SECRET")
DRAFTER = ("drafter-sa", "DRAFTER_SA_SECRET")


def retriever_client() -> McpClient:
    return McpClient(client_id=RETRIEVER[0], secret_env=RETRIEVER[1])


def drafter_client() -> McpClient:
    return McpClient(client_id=DRAFTER[0], secret_env=DRAFTER[1])
