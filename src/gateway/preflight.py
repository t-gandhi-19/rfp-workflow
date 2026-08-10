"""Preflight: prove the embedding path works before anything writes to the graph.

Ingest computes embeddings for the whole corpus and stores them in a Neo4j
vector index built for a fixed width. Three things can be wrong, and only one of
them announces itself:

* Ollama is not running — obvious, fails immediately.
* The pinned model is not installed — a confusing 404 from deep inside ingest.
* The model returns a different width than the index expects — **silent**. The
  index rejects or mismatches vectors and retrieval quietly degrades.

So the third check is a real probe embedding whose length is measured, not
assumed from documentation. `make ingest` refuses to run until all three pass.

The checks are plain functions returning results so they can be tested against a
stubbed endpoint — including the model-absent case, which is otherwise awkward
to reproduce without uninstalling someone's model.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.contracts.embedding import EmbeddingConfig, embedding_config
from src.gateway.client import GatewayClient, GatewayError
from src.gateway.ollama_admin import installed_tags, is_installed


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    #: Literal command that fixes it, printed verbatim. None when there is no
    #: single command to give.
    fix: str | None = None


def ollama_base_url_for_host(env: dict[str, str]) -> str:
    """Where a host-side caller reaches Ollama.

    Containers use ``host.docker.internal``, which does not resolve from the
    host shell that runs ``make preflight`` — so the host form is separate.
    """
    return env.get("OLLAMA_BASE_URL_HOST") or "http://localhost:11434"


async def check_ollama_reachable(
    base_url: str, *, timeout: float = 10.0, client: httpx.AsyncClient | None = None
) -> CheckResult:
    try:
        tags = await installed_tags(base_url, timeout=timeout, client=client)
    except httpx.HTTPError as exc:
        return CheckResult(
            name="ollama reachable",
            ok=False,
            detail=f"cannot reach Ollama at {base_url} ({type(exc).__name__})",
            fix="ollama serve        # and ensure OLLAMA_HOST=0.0.0.0 so containers can reach it",
        )
    return CheckResult(
        name="ollama reachable",
        ok=True,
        detail=f"{base_url} — {len(tags)} model(s) installed",
    )


async def check_model_installed(
    base_url: str,
    config: EmbeddingConfig,
    *,
    timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
) -> CheckResult:
    wanted = config.model.tag
    try:
        tags = await installed_tags(base_url, timeout=timeout, client=client)
    except httpx.HTTPError as exc:
        return CheckResult(
            name=f"pinned model '{wanted}' installed",
            ok=False,
            detail=f"cannot reach Ollama at {base_url} ({type(exc).__name__})",
            fix=config.install_command,
        )
    if not is_installed(wanted, tags):
        return CheckResult(
            name=f"pinned model '{wanted}' installed",
            ok=False,
            detail=f"not installed. Present: {', '.join(tags) or '(none)'}",
            fix=config.install_command,
        )
    return CheckResult(
        name=f"pinned model '{wanted}' installed",
        ok=True,
        detail="present",
    )


async def check_embedding_width(
    gateway: GatewayClient,
    config: EmbeddingConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> CheckResult:
    """Embed a probe string and measure the result.

    This is the check that matters most, because a width mismatch is the only
    one of the three failure modes that produces no error at ingest time.
    """
    expected = config.model.dimensions
    alias = config.model.alias
    try:
        vectors = await gateway.embed(
            ["preflight probe: dimension check"], alias=alias, client=client
        )
    except GatewayError as exc:
        return CheckResult(
            name=f"probe embedding is {expected}-dim",
            ok=False,
            detail=str(exc),
            fix="make up            # the LiteLLM gateway must be running",
        )

    actual = len(vectors[0]) if vectors else 0
    if actual != expected:
        return CheckResult(
            name=f"probe embedding is {expected}-dim",
            ok=False,
            detail=(
                f"alias '{alias}' returned {actual} dimensions, but the Neo4j vector index "
                f"is built for {expected}. Every stored vector would be unusable."
            ),
            fix=(
                "Reconcile config/embedding.yaml with docker/litellm/config.yaml, "
                "then: make reembed"
            ),
        )
    return CheckResult(
        name=f"probe embedding is {expected}-dim",
        ok=True,
        detail=f"alias '{alias}' returned {actual} dimensions",
    )


async def run_preflight(
    env: dict[str, str],
    *,
    config: EmbeddingConfig | None = None,
    gateway: GatewayClient | None = None,
    ollama_client_timeout: float = 10.0,
    ollama_client: httpx.AsyncClient | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> list[CheckResult]:
    """Run every check and return all results — never short-circuit.

    Reporting all three at once means one run tells you everything that is
    wrong, rather than revealing the next problem only after you fix this one.
    """
    resolved = config or embedding_config()
    base_url = ollama_base_url_for_host(env)
    gw = gateway or GatewayClient.from_env()

    reachable = await check_ollama_reachable(
        base_url, timeout=ollama_client_timeout, client=ollama_client
    )
    installed = await check_model_installed(
        base_url, resolved, timeout=ollama_client_timeout, client=ollama_client
    )
    width = await check_embedding_width(gw, resolved, client=http_client)
    return [reachable, installed, width]


def format_report(results: list[CheckResult]) -> str:
    """Human-readable report; fixes are printed as literal commands."""
    lines = ["", "preflight — embedding path", ""]
    for result in results:
        lines.append(f"  [{'ok' if result.ok else 'FAIL'}] {result.name}")
        lines.append(f"         {result.detail}")
        if not result.ok and result.fix:
            lines.append(f"         fix:  {result.fix}")
    lines.append("")
    if all(result.ok for result in results):
        lines.append("All checks passed. Safe to ingest.")
    else:
        failed = sum(1 for result in results if not result.ok)
        lines.append(f"{failed} check(s) failed. Ingest will refuse to run until they pass.")
    lines.append("")
    return "\n".join(lines)
