"""Preflight: prove the local model path works before anything writes to the graph.

Four classes of problem, in increasing order of how long they take to notice:

* **Ollama is not running** — obvious, fails immediately.
* **A pinned model is not installed** — a confusing 404 from deep inside a run.
* **A reference is not really a pin** (`ollama/llama3.2`, or an explicit
  `:latest`) — invisible until two machines quietly disagree about what an alias
  means, by which point the divergence is weeks old.
* **The embedding model returns a different width than the index expects** —
  **silent**. Vectors go in unusable and retrieval simply gets worse.

The model checks walk the gateway config rather than a hardcoded list, so an
alias added in a later phase is covered the day it appears rather than the day
someone remembers to add a check for it. The embedding model gets one deeper
check on top: a real probe embedding whose length is measured, not read from
documentation.

`make ingest` refuses to run until all of them pass.

Every check is a plain function returning a result, so they can be driven
against a stubbed endpoint — including the model-absent case, which is otherwise
awkward to reproduce without uninstalling someone's model.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from src.contracts.embedding import EmbeddingConfig, EmbedRole, embedding_config
from src.gateway.client import GatewayClient, GatewayError
from src.gateway.model_pins import GatewayModel, drifting_models, parse_gateway_models
from src.gateway.ollama_admin import installed_tags, is_installed
from src.retrieval.calibration import CalibrationError, load_for_current_corpus


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


def check_tags_are_explicit(models: list[GatewayModel]) -> CheckResult:
    """Every Ollama reference in the gateway config names a version.

    A static check — it reads the config, not the host. It exists because a
    drifting reference is invisible until two machines quietly disagree about
    what `triage-model` means, and by then the divergence is weeks old.
    """
    name = "every gateway Ollama reference is explicitly tagged"
    violations = [model.violation() for model in drifting_models(models)]
    if violations:
        return CheckResult(
            name=name,
            ok=False,
            detail="; ".join(v for v in violations if v),
            fix=(
                "Write the installed, versioned tag in docker/litellm/config.yaml "
                "(e.g. ollama/llama3.2:3b, not ollama/llama3.2). :latest is not a pin — "
                "it moves on the next pull, so two machines can run different weights "
                "behind the same alias."
            ),
        )
    return CheckResult(
        name=name,
        ok=True,
        detail=f"{len(models)} Ollama alias(es), all versioned",
    )


async def check_gateway_models_installed(
    base_url: str,
    models: list[GatewayModel],
    *,
    timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
) -> list[CheckResult]:
    """Every Ollama-backed alias in the gateway config is installed on the host.

    One loop over the parsed config rather than a hardcoded list, so an alias
    added in a later phase is covered the day it appears.

    Presence only. Reranking has no fixed-shape output to probe the way an
    embedding has a width, and invoking an 8B model on a CPU host would make
    preflight take minutes — which would get it skipped, which defeats it. The
    embedding model gets the deeper check separately.
    """
    if not models:
        return [
            CheckResult(
                name="gateway Ollama models installed",
                ok=False,
                detail="no Ollama aliases found in the gateway config",
                fix="Check docker/litellm/config.yaml — model_list looks empty or unparsed.",
            )
        ]

    try:
        tags = await installed_tags(base_url, timeout=timeout, client=client)
    except httpx.HTTPError as exc:
        return [
            CheckResult(
                name=f"'{model.tag}' installed (alias {model.alias})",
                ok=False,
                detail=f"cannot reach Ollama at {base_url} ({type(exc).__name__})",
                fix=model.pull_command,
            )
            for model in models
        ]

    # One result per DISTINCT tag: several aliases share llama3.2:3b, and
    # repeating an identical failure four times buries the other checks.
    results: list[CheckResult] = []
    for tag in sorted({model.tag for model in models}):
        sharing = sorted(model.alias for model in models if model.tag == tag)
        label = f"'{tag}' installed (alias{'es' if len(sharing) > 1 else ''}: {', '.join(sharing)})"
        if is_installed(tag, tags):
            results.append(CheckResult(name=label, ok=True, detail="present"))
        else:
            results.append(
                CheckResult(
                    name=label,
                    ok=False,
                    detail=f"not installed. Present: {', '.join(tags) or '(none)'}",
                    fix=f"ollama pull {tag}",
                )
            )
    return results


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
            ["preflight probe: dimension check"],
            alias=alias,
            role=EmbedRole.QUERY,
            client=client,
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


async def check_neo4j_auth(env: dict[str, str]) -> CheckResult:
    """The graph accepts the credentials in `.env` (amendment Q).

    A healthcheck cannot see this. Neo4j reports healthy as soon as it is
    listening, and `NEO4J_AUTH` is applied only when the data volume is FIRST
    initialised — so a volume created before a credential change keeps the old
    password forever while compose, `.env` and the container environment all
    agree on the new one. Everything looks correct and every connection is
    rejected.

    The other cause is more mundane and, here, more likely: the URI points at a
    DIFFERENT Neo4j. Another stack on this machine holds the conventional 7687,
    and it rejects these credentials in exactly the same way. Both surface as an
    AuthError from deep inside ingest, so the check reports the URI it actually
    dialled and the fix names both causes in the order worth checking them.

    A third mode does not reach authentication at all: the URI may name a host
    that does not resolve from wherever this is running. Every failure mode
    returns a CheckResult — none raises — because this probe runs alongside
    seven others whose results are worth having even when it fails.
    """
    name = "neo4j accepts the configured credentials"
    uri = env.get("NEO4J_URI") or f"bolt://localhost:{env.get('NEO4J_BOLT_PORT_HOST', '7687')}"
    user = env.get("NEO4J_USER", "neo4j")
    password = env.get("NEO4J_PASSWORD", "")
    if not password:
        return CheckResult(
            name=name,
            ok=False,
            detail="NEO4J_PASSWORD is not set",
            fix="cp .env.example .env    # then set NEO4J_PASSWORD",
        )

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        await driver.verify_connectivity()
    except AuthError:
        return CheckResult(
            name=name,
            ok=False,
            # The URI is in the detail because the likeliest cause is that it is
            # the WRONG ONE. Another project's Neo4j on the conventional port
            # will reject these credentials in exactly the same way a stale
            # volume does, and the message that named only the volume sent one
            # session to `make nuke` for a problem nuking could not fix.
            detail=f"credentials rejected by {uri}",
            fix=(
                "check the URI above is THIS project's Neo4j (NEO4J_BOLT_PORT_HOST in .env; "
                "another stack may hold the default port). If it is, the data volume is "
                "stale from a prior password — NEO4J_AUTH applies only on first "
                "initialisation — and make nuke resets it."
            ),
        )
    except ValueError:
        # The driver raises a bare ValueError("Cannot resolve address ...") when
        # the host does not resolve. It is neither a ServiceUnavailable nor an
        # OSError, so it used to escape this function AND run_preflight, killing
        # the harness with a resolver traceback and silencing every other
        # check's result — worse than a failing probe, because a failing probe
        # still lets the other seven report.
        #
        # Causes in check order. The first is by far the likeliest: `.env` sets
        # NEO4J_URI to the compose SERVICE NAME, which resolves only inside the
        # compose network, so any host-side caller that did not override it is
        # dialling a name that cannot exist there.
        return CheckResult(
            name=name,
            ok=False,
            detail=f"cannot resolve the host in {uri}",
            fix=(
                "1. if that is a compose service name and this is running on the HOST, "
                "override it: NEO4J_URI=bolt://localhost:$NEO4J_BOLT_PORT_HOST "
                "(the make targets apply this as HOST_NEO4J). "
                "2. otherwise the named host is not running: make up"
            ),
        )
    except (ServiceUnavailable, OSError) as exc:
        return CheckResult(
            name=name,
            ok=False,
            detail=f"cannot reach Neo4j at {uri} ({type(exc).__name__})",
            fix="make up            # the Neo4j container must be running",
        )
    finally:
        await driver.close()
    return CheckResult(name=name, ok=True, detail=f"{uri} accepted {user}")


def check_calibration_is_fresh() -> CheckResult:
    """The calibration artifact exists and still describes this model and corpus.

    Amendment J. Retrieval is fail-closed on this, so without the check the
    first sign of a missing or stale artifact is a run refusing partway through
    — which is correct behaviour discovered at the worst moment. Preflight is
    where the other silent-until-late failures are caught, so it belongs here.

    Staleness is the case worth naming: an artifact measured against a different
    model, or against the corpus as it was before a regeneration, loads fine and
    describes a distribution that no longer exists.
    """
    name = "calibration artifact is present and current"
    try:
        artifact = load_for_current_corpus()
    except CalibrationError as exc:
        return CheckResult(
            name=name,
            ok=False,
            detail=str(exc).splitlines()[0],
            fix="make calibrate     # after 'make up' and 'make ingest'",
        )
    return CheckResult(
        name=name,
        ok=True,
        detail=(
            f"measured {artifact.computed_at} over corpus {artifact.corpus_hash}; "
            f"separation {artifact.separation:+.4f}, "
            f"derived floor {artifact.derived_floor:.4f}"
        ),
    )


async def run_preflight(
    env: dict[str, str],
    *,
    config: EmbeddingConfig | None = None,
    models: list[GatewayModel] | None = None,
    gateway: GatewayClient | None = None,
    ollama_client_timeout: float = 10.0,
    ollama_client: httpx.AsyncClient | None = None,
    http_client: httpx.AsyncClient | None = None,
    include_calibration: bool = True,
    include_neo4j: bool = True,
) -> list[CheckResult]:
    """Run every check and return all results — never short-circuit.

    Reporting them all at once means one run tells you everything that is
    wrong, rather than revealing the next problem only after you fix this one.

    `include_calibration` is False when preflight runs AS PART OF ingest. On a
    clean machine there is no corpus to have calibrated yet, so the check could
    not pass by construction and would block the very command that produces the
    thing it looks for. Run standalone — which is what `make preflight` does —
    it is included, because by then an artifact is expected to exist.
    """
    resolved = config or embedding_config()
    resolved_models = models if models is not None else parse_gateway_models()
    base_url = ollama_base_url_for_host(env)
    gw = gateway or GatewayClient.from_env()

    results = [
        await check_ollama_reachable(base_url, timeout=ollama_client_timeout, client=ollama_client),
        check_tags_are_explicit(resolved_models),
    ]
    results.extend(
        await check_gateway_models_installed(
            base_url, resolved_models, timeout=ollama_client_timeout, client=ollama_client
        )
    )
    # The embedding model is the one alias with a deeper check: its output has a
    # fixed width that the Neo4j index depends on, and a mismatch is silent.
    results.append(await check_embedding_width(gw, resolved, client=http_client))
    # Amendment Q. Runs in BOTH modes, including before ingest — a stale volume
    # is exactly what blocks ingest, so the check that names it has to run first.
    if include_neo4j:
        results.append(await check_neo4j_auth(env))
    if include_calibration:
        results.append(check_calibration_is_fresh())
    return results


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
