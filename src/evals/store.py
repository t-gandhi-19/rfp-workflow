"""Persisting eval metrics, and reading the previous SHA back for deltas.

WRITES GO THROUGH THE WRITE-API, never directly to Postgres (CLAUDE.md rule 4).
The harness authenticates as `evals-sa`, which holds `eval-writer` and — proved
by `tests/integration/test_live_stack.py` — cannot write a draft. So a harness
bug can corrupt its own scoreboard and nothing else.

READS GO DIRECTLY, and the asymmetry is deliberate rather than an oversight: the
delta comparison needs the previous SHA's rows, there is no read endpoint, and
adding one would put a query surface on the write path. `app_reader` is a
read-only role, the query is parameterised, and it lives here rather than
anywhere an agent can reach it.

KEYED BY GIT SHA, which is what makes "did this commit make retrieval worse?"
answerable at all. The table's primary key is (git_sha, run_id, metric), so a
re-run at the same SHA updates rather than accumulating — an eval you can run
twice is one you will run twice, and two rows for one measurement is a scoreboard
that cannot be read.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

import httpx

from src.contracts.run import EvalScore
from src.evals.contracts import CategoryResult

logger = logging.getLogger("rfp.evals.store")

#: The role the harness holds, and the client it authenticates as.
EVALS_CLIENT_ID = "evals-sa"
# The NAME of the variable holding the secret, not a secret. ruff's heuristic
# fires on the word; the value it names is in .env, which is gitignored.
EVALS_SECRET_ENV = "EVALS_SA_SECRET"  # noqa: S105


#: What a real commit key looks like: a full 40-hex SHA, optionally `-dirty`.
#:
#: The baseline query filters on this because `eval_results` is not the
#: harness's private table — the integration suite writes rows keyed `integration`
#: and `itest-<random>` to prove the write path, and those would otherwise become
#: the "previous SHA" a real run differences against. Running the test suite must
#: not change what the scoreboard says about the code.
COMMIT_SHAPE = r"^[0-9a-f]{40}(-dirty)?$"


class EvalStoreError(RuntimeError):
    """Raised when persistence fails in a way the caller must not ignore."""


def git_sha() -> str:
    """The commit the numbers were produced at.

    Amendment D: a report carries the SHA it was produced at. `--dirty` is
    appended when the worktree is not clean, because a number measured against
    uncommitted code is not a number about that commit, and a scoreboard that
    silently attributes it to one is worse than a scoreboard with a gap.

    GIT IS ASKED FIRST, and `GIT_SHA` is only a fallback. The first version had
    that precedence the other way and every local run keyed its rows to
    `local-dev` — because `.env` sets `GIT_SHA=local-dev` for the CONTAINERS to
    report as their build, and the harness silently inherited it. One shared
    name, two different questions: "what build is this container?" and "what
    commit produced these numbers?". In a checkout only git can answer the
    second, so in a checkout git is the answer.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        # Outside a checkout — a container, a source tarball — GIT_SHA is the
        # only answer available, and here it is the right one.
        if override := os.environ.get("GIT_SHA"):
            return override
        raise EvalStoreError(
            "cannot determine the git SHA, and eval rows are keyed by it. "
            "Set GIT_SHA explicitly if running outside a checkout."
        ) from exc
    sha = result.stdout.strip()

    dirty = subprocess.run(
        ["git", "status", "--porcelain"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return f"{sha}-dirty" if dirty.stdout.strip() else sha


def to_scores(results: list[CategoryResult], *, sha: str, run_id: str) -> list[EvalScore]:
    """Flatten categories into rows.

    ONLY GATED METRICS ARE PERSISTED. `EvalScore` requires a threshold and a
    `passed`, and an ungated metric has neither — MRR is reported because it
    describes the shape of a result, not because anything should fail on it.
    Writing it with an invented threshold of 0.0 would make it look gated
    forever after, to everyone reading the table rather than this comment.

    NOT_IMPLEMENTED categories contribute nothing, for the same reason they carry
    no metrics: a placeholder row would be differenced against the next SHA and
    turn "not measured" into an apparent regression on the day it is measured.
    """
    return [
        EvalScore(
            git_sha=sha,
            run_id=run_id,
            # Namespaced by category: `extraction.recall` and a future
            # `grounding.recall` are different metrics, and an unqualified
            # `recall` would silently overwrite one with the other under the
            # (git_sha, run_id, metric) primary key.
            metric=f"{result.key}.{metric.key}",
            value=metric.value,
            threshold=metric.threshold,
            passed=metric.passed,
        )
        for result in results
        for metric in result.gated_metrics
        if metric.threshold is not None and metric.passed is not None
    ]


async def _token(client: httpx.AsyncClient, *, keycloak_base: str, realm: str) -> str:
    secret = os.environ.get(EVALS_SECRET_ENV)
    if not secret:
        raise EvalStoreError(
            f"{EVALS_SECRET_ENV} is not set; the harness authenticates as "
            f"{EVALS_CLIENT_ID} to write eval results. Load it with: "
            "set -a && . ./.env && set +a"
        )
    response = await client.post(
        f"{keycloak_base}/realms/{realm}/protocol/openid-connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": EVALS_CLIENT_ID,
            "client_secret": secret,
        },
    )
    response.raise_for_status()
    token: str = response.json()["access_token"]
    return token


async def persist(scores: list[EvalScore]) -> int:
    """POST the rows through the write-api. Returns how many were written."""
    if not scores:
        return 0

    keycloak_base = os.environ.get(
        "KEYCLOAK_BASE", f"http://localhost:{os.environ.get('KEYCLOAK_PORT_HOST', '8080')}"
    )
    realm = os.environ.get("KEYCLOAK_REALM", "rfp")
    write_api = os.environ.get(
        "WRITE_API_BASE", f"http://localhost:{os.environ.get('WRITE_API_PORT', '8001')}"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        token = await _token(client, keycloak_base=keycloak_base, realm=realm)
        response = await client.post(
            f"{write_api}/v1/eval-results",
            json={"scores": [score.model_dump(mode="json") for score in scores]},
            headers={"Authorization": f"Bearer {token}"},
        )
    if response.status_code != 200:
        raise EvalStoreError(
            f"write-api refused the eval results: {response.status_code} {response.text}"
        )
    return len(scores)


@dataclass(frozen=True)
class Delta:
    """One metric's movement between two SHAs."""

    metric: str
    previous: float
    current: float

    @property
    def change(self) -> float:
        return self.current - self.previous


async def previous_scores(*, current_sha: str) -> tuple[str | None, dict[str, float]]:
    """The most recent PRIOR SHA's metrics, and which SHA they came from.

    Returns `(None, {})` when there is no history or the database is
    unreachable. A missing baseline is a normal state — the first run at a new
    clone has none — and the report says "no previous SHA" rather than failing,
    because a harness that refuses to produce a report without history is one
    nobody can bootstrap.

    Ordered by `created_at` rather than by any notion of git ancestry: the table
    records when a measurement was taken, and it has no way to know whether one
    SHA is an ancestor of another. The report says which SHA it compared against
    so the reader can judge whether the comparison is meaningful.

    Uses `reader_url()` rather than a DSN assembled here. The first version built
    its own and named `postgresql+asyncpg`, a driver this project does not ship.
    That cost a 35-minute rerank-ON run: the baseline lookup happens AFTER every
    expensive measurement, so it took the whole run down at the last step with
    all the work already done and nothing written.

    ENGINE CONSTRUCTION IS INSIDE THE `try` for exactly that reason. It used to
    sit above it, so a bad URL raised past the handler instead of degrading —
    the opposite of what the handler was written for. Anything that can go wrong
    reaching the database belongs to the "report says no baseline" path, not to
    the "lose the run" path.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

    from src.state.db import reader_url

    engine: AsyncEngine | None = None
    try:
        engine = create_async_engine(reader_url())
        async with engine.connect() as conn:
            found = await conn.execute(
                text(
                    "SELECT git_sha FROM eval_results "
                    "WHERE git_sha <> :sha AND git_sha ~ :shape "
                    "GROUP BY git_sha ORDER BY max(created_at) DESC LIMIT 1"
                ),
                {"sha": current_sha, "shape": COMMIT_SHAPE},
            )
            row = found.first()
            if row is None:
                return None, {}
            previous_sha = str(row[0])

            rows = await conn.execute(
                text("SELECT metric, value FROM eval_results WHERE git_sha = :sha"),
                {"sha": previous_sha},
            )
            return previous_sha, {str(m): float(v) for m, v in rows.all()}
    except Exception as exc:
        # Deliberately broad: the caller renders "no previous SHA" and the run
        # continues. A delta is a convenience; the measurements are the product,
        # and failing the harness because a comparison was unavailable would be
        # the tail wagging the dog.
        #
        # NOT SILENT, though. Degrading quietly and degrading invisibly are
        # different decisions, and only the first one was ever intended: a
        # baseline that vanishes without a word is indistinguishable from a
        # fresh clone, and the difference matters to whoever reads the report.
        logger.warning(
            "eval baseline unavailable (%s: %s) — the report will say 'no previous SHA'. "
            "This is expected on a fresh clone and a defect anywhere else.",
            type(exc).__name__,
            exc,
        )
        return None, {}
    finally:
        # None when construction itself failed, which is now inside the try.
        if engine is not None:
            await engine.dispose()


def deltas(current: list[EvalScore], previous: dict[str, float]) -> list[Delta]:
    """Movements for metrics present in BOTH runs.

    A metric measured for the first time has no delta — reporting one against an
    implied zero would show every new metric as a large improvement.
    """
    return [
        Delta(metric=score.metric, previous=previous[score.metric], current=score.value)
        for score in current
        if score.metric in previous
    ]
