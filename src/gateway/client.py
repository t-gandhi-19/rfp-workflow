"""The only egress to models (CLAUDE.md rule 5).

Everything above this module names a gateway *alias* — ``embed-model``,
``drafter-model`` — never a provider or a model name. That is what keeps the
alias table, the retry policy, the budget ceiling, and cost accounting in one
place instead of scattered across call sites.

No provider SDK is imported here either: the LiteLLM proxy speaks an
OpenAI-compatible HTTP API, so plain httpx is enough.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


class GatewayError(RuntimeError):
    """The gateway could not satisfy a request.

    Callers escalate the affected question rather than crashing the run
    (build prompt §18); they never fall back to a provider directly.
    """


def gateway_base_url() -> str:
    """Where the proxy lives.

    Containers reach it by service name; a host-side caller such as
    ``make preflight`` reaches it on the published port, so the host form is
    overridable without changing the container configuration.
    """
    explicit = os.environ.get("LITELLM_BASE_URL_HOST") or os.environ.get("LITELLM_BASE_URL")
    if explicit:
        return explicit
    port = os.environ.get("LITELLM_PORT_HOST", "4000")
    return f"http://localhost:{port}"


@dataclass(frozen=True)
class GatewayClient:
    """Thin client over the LiteLLM proxy."""

    base_url: str
    api_key: str
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> GatewayClient:
        return cls(
            base_url=gateway_base_url(),
            api_key=os.environ.get("LITELLM_MASTER_KEY", ""),
            timeout=float(os.environ.get("GATEWAY_TIMEOUT_SECONDS", "120")),
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def embed(
        self,
        texts: list[str],
        *,
        alias: str,
        client: httpx.AsyncClient | None = None,
    ) -> list[list[float]]:
        """Embed texts through ``alias``, preserving input order.

        `client` is injectable so tests can drive this against a mock transport
        without a live proxy.
        """
        if not texts:
            return []

        payload = {"model": alias, "input": texts}
        owned = client is None
        http = client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await http.post(
                f"{self.base_url.rstrip('/')}/v1/embeddings",
                json=payload,
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            raise GatewayError(f"gateway unreachable at {self.base_url}: {exc}") from exc
        finally:
            if owned:
                await http.aclose()

        if response.status_code >= 300:
            raise GatewayError(
                f"gateway returned {response.status_code} for alias '{alias}': {response.text[:400]}"
            )

        data = response.json().get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise GatewayError(
                f"gateway returned {len(data) if isinstance(data, list) else 'no'} embeddings "
                f"for {len(texts)} input(s)"
            )
        # The API is not required to return results in request order; index says
        # where each belongs, and silently mispairing text to vector would poison
        # retrieval in a way nothing downstream could detect.
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        return [[float(value) for value in item["embedding"]] for item in ordered]
