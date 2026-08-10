"""Ollama inventory queries.

This asks Ollama *what is installed*; it never asks it to run anything. Model
calls go through the LiteLLM proxy (CLAUDE.md rule 5) — see
:mod:`src.gateway.client`. Inventory has no equivalent through the proxy, and
this file lives in ``src/gateway/`` because that is where provider-specific
knowledge belongs.
"""

from __future__ import annotations

import httpx


def normalise_tag(tag: str) -> str:
    """Collapse Ollama's implicit ``:latest`` so tags compare sensibly.

    ``ollama pull nomic-embed-text`` installs a model that reports itself as
    ``nomic-embed-text:latest``. Comparing raw strings would say the pinned
    model is missing when it is sitting right there.
    """
    return tag[: -len(":latest")] if tag.endswith(":latest") else tag


async def installed_tags(
    base_url: str,
    *,
    timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """Every model tag installed in the Ollama instance at ``base_url``.

    `client` is injectable so tests can drive this against a mock transport.
    Injection rather than patching `httpx` globally, because a global patch also
    catches every *other* client in the same process — which is exactly how the
    first version of these tests broke the gateway's own mock.

    Raises httpx errors on an unreachable or unhealthy endpoint; the caller
    turns those into a readable check failure.
    """
    owned = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await http.get(f"{base_url.rstrip('/')}/api/tags")
        response.raise_for_status()
        payload = response.json()
    finally:
        if owned:
            await http.aclose()
    models = payload.get("models") or []
    return sorted(normalise_tag(str(model.get("name", ""))) for model in models)


def is_installed(wanted: str, installed: list[str]) -> bool:
    """Whether the pinned tag is present, ignoring an implicit ``:latest``."""
    return normalise_tag(wanted) in {normalise_tag(tag) for tag in installed}
