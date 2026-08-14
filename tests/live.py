"""Reachability probes, so stack-dependent tests skip rather than fail.

A fresh clone with nothing running should produce a green run with skips. A red
run is a signal that something is broken; "you have not started Docker yet" is
not that signal, and letting it look like one trains people to ignore red.

Each probe returns ``None`` when the service is usable, or a short reason when
it is not. :func:`require` turns any reason into a skip that names the exact
command to fix it.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable

import httpx
import pytest

#: How to start what these tests need. Named in every skip message.
COMPOSE_UP = "make up  (or: docker compose --profile infra --profile app up -d --wait)"

DEFAULT_TIMEOUT = 3.0


def keycloak_base() -> str:
    return os.environ.get("KEYCLOAK_BASE", "http://localhost:8080")


def write_api_base() -> str:
    return os.environ.get("WRITE_API_BASE", "http://localhost:8001")


def mcp_base() -> str:
    return os.environ.get("MCP_BASE", "http://localhost:8002")


def neo4j_bolt() -> tuple[str, int]:
    host = os.environ.get("NEO4J_HOST", "localhost")
    port = int(os.environ.get("NEO4J_BOLT_PORT_HOST", "7687"))
    return host, port


def http_ok(label: str, url: str, timeout: float = DEFAULT_TIMEOUT) -> Callable[[], str | None]:
    """Probe an HTTP endpoint, requiring a 2xx."""

    def _probe() -> str | None:
        try:
            response = httpx.get(url, timeout=timeout)
        except Exception as exc:
            return f"{label} unreachable at {url} ({type(exc).__name__})"
        if response.status_code >= 300:
            return f"{label} at {url} returned {response.status_code}"
        return None

    return _probe


def tcp_open(
    label: str, host: str, port: int, timeout: float = DEFAULT_TIMEOUT
) -> Callable[[], str | None]:
    """Probe a raw TCP port, for services that speak no HTTP (Bolt)."""

    def _probe() -> str | None:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return None
        except OSError as exc:
            return f"{label} unreachable at {host}:{port} ({type(exc).__name__})"

    return _probe


def require(*probes: Callable[[], str | None]) -> None:
    """Skip the calling test unless every probe reports the service usable."""
    reasons = [reason for probe in probes if (reason := probe()) is not None]
    if reasons:
        pytest.skip(
            "live stack not available:\n  - "
            + "\n  - ".join(reasons)
            + f"\nStart it with:  {COMPOSE_UP}"
        )


def require_env(*names: str) -> None:
    """Skip unless the named variables are in the environment.

    A reachable stack is not enough: these tests authenticate with the service
    account secrets from .env. Without them the tests die on a KeyError deep
    inside a helper, which reads like a broken test rather than an unloaded
    environment.
    """
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.skip(
            f"environment not loaded — missing: {', '.join(missing)}\n"
            "Load it with:  set -a && . ./.env && set +a"
        )


#: Secrets the write-api integration tests authenticate with.
WRITE_API_SECRETS = (
    "KEYCLOAK_ADMIN_PASSWORD",
    "DRAFTER_SA_SECRET",
    "TRIAGE_SA_SECRET",
    "EVALS_SA_SECRET",
)


def require_write_api_and_keycloak() -> None:
    """The pair write-api tests need, plus the credentials they authenticate with.

    write-api's /health performs a real database round-trip, so a 2xx from it is
    also proof that Postgres is up and migrated — no separate probe needed.
    """
    require(
        http_ok("Keycloak", f"{keycloak_base()}/realms/rfp/.well-known/openid-configuration"),
        http_ok("write-api (and Postgres behind it)", f"{write_api_base()}/health"),
    )
    require_env(*WRITE_API_SECRETS)


#: What the graph tests authenticate with. Checked alongside the port probe
#: because a port is not an identity — see `require_neo4j`.
NEO4J_SECRETS = ("NEO4J_PASSWORD",)


def require_neo4j() -> None:
    """Bolt reachable AND the credentials to use it.

    A RAW TCP PROBE CANNOT TELL WHOSE DATABASE THAT IS. On a machine running a
    second project's stack — which is the normal case on the build host, hence
    the 5xxxx port overrides in `.env` — port 7687 is open and belongs to
    somebody else. A fresh clone with no `.env` then defaults to 7687, passes
    the probe, and every graph test fails with

        Neo.ClientError.Security.Unauthorized — missing key `credentials`

    which is 38 red tests saying nothing about this repository. Amendment B
    requires a fresh clone to be green-with-skips; requiring the password makes
    the skip say "environment not loaded", which is both true and actionable.

    This does not make the probe prove identity — nothing cheap does. It makes
    the *unloaded environment* case, which is the reachable one, report itself
    honestly instead of as a wall of authentication failures.
    """
    require(tcp_open("Neo4j Bolt", *neo4j_bolt()))
    require_env(*NEO4J_SECRETS)


#: Secrets the mcp-server integration tests authenticate with. `retriever-sa` is
#: the caller under test; `extractor-sa` is the one deliberately without
#: `kg-reader`, which is what makes the 403 assertion mean anything.
MCP_SECRETS = ("RETRIEVER_SA_SECRET", "EXTRACTOR_SA_SECRET")


def require_mcp_server() -> None:
    """mcp-server, plus the Keycloak that issues its tokens and the Neo4j behind it.

    All three, because a tool call exercises all three and a skip that named only
    one would send the reader to the wrong service. mcp-server's /health does not
    touch the graph — it deliberately reports the process, not its dependencies —
    so Neo4j is probed separately rather than inferred from a 200.
    """
    require(
        http_ok("Keycloak", f"{keycloak_base()}/realms/rfp/.well-known/openid-configuration"),
        http_ok("mcp-server", f"{mcp_base()}/health"),
        tcp_open("Neo4j Bolt", *neo4j_bolt()),
    )
    require_env(*MCP_SECRETS)
