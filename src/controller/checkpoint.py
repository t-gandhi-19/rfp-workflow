"""Checkpointing a run, and reading one back to resume it.

WRITES GO THROUGH THE WRITE-API (rule 4). The controller authenticates as
`drafter-sa`, which holds `draft-writer`. It cannot write eval results and it
cannot write the graph, so a controller bug corrupts a run and nothing else.

READS GO DIRECTLY, through the SELECT-only `app_reader` identity — the same
asymmetry, for the same reason, as `src/evals/store.py`: resume needs whole rows
back, there is no read endpoint on the write path, and adding one would put a
query surface on the only thing allowed to mutate state. `get_run_state` arrives
on the MCP server in 1e for AGENT-facing reads; this is the controller's own,
and the controller is not an agent.

EVERY WRITE IS AN IDEMPOTENT UPSERT on a natural key, which is the whole basis
of resume: re-running a question that already completed overwrites its row with
the same values instead of producing a second one.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import httpx

from src.contracts import DraftedAnswer, QuestionStatus, RunStage, RunState

logger = logging.getLogger("rfp.controller.checkpoint")

#: The controller's identity. Holds rfp-reader, kg-reader, draft-writer — and
#: deliberately not `submitter`, which is granted to nothing (rule 2).
CONTROLLER_CLIENT_ID = "drafter-sa"
# The NAME of the variable, not a secret; the value lives in gitignored .env.
CONTROLLER_SECRET_ENV = "DRAFTER_SA_SECRET"  # noqa: S105


class CheckpointError(RuntimeError):
    """A checkpoint could not be written or read.

    Raised rather than swallowed. A run whose state is not durable cannot be
    resumed, and continuing to spend model calls on one is spending money to
    produce something unrecoverable.
    """


def keycloak_base() -> str:
    return os.environ.get(
        "KEYCLOAK_BASE", f"http://localhost:{os.environ.get('KEYCLOAK_PORT_HOST', '8080')}"
    )


def write_api_base() -> str:
    return os.environ.get(
        "WRITE_API_BASE", f"http://localhost:{os.environ.get('WRITE_API_PORT', '8001')}"
    )


@dataclass
class Checkpointer:
    """Writes run state through the write-api, holding one token per run."""

    base_url: str = field(default_factory=write_api_base)
    keycloak_url: str = field(default_factory=keycloak_base)
    realm: str = field(default_factory=lambda: os.environ.get("KEYCLOAK_REALM", "rfp"))
    timeout: float = 30.0
    _token: str | None = field(default=None, repr=False)

    async def _authenticate(self, client: httpx.AsyncClient) -> str:
        if self._token is not None:
            return self._token
        secret = os.environ.get(CONTROLLER_SECRET_ENV)
        if not secret:
            raise CheckpointError(
                f"{CONTROLLER_SECRET_ENV} is not set; the controller authenticates as "
                f"{CONTROLLER_CLIENT_ID} to checkpoint a run. Load it with: "
                "set -a && . ./.env && set +a"
            )
        try:
            response = await client.post(
                f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": CONTROLLER_CLIENT_ID,
                    "client_secret": secret,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CheckpointError(f"could not obtain a token from Keycloak: {exc}") from exc
        token: str = response.json()["access_token"]
        self._token = token
        return token

    async def _put(self, client: httpx.AsyncClient, path: str, payload: dict[str, object]) -> None:
        """PUT once, and once more if the CONNECTION died rather than the request.

        A run holds one client across a fan-out that can take many minutes, and
        an idle keep-alive connection gets closed by the server or an
        intermediary in that time. httpx surfaces that as
        `Server disconnected without sending a response` on the next use — a
        transport failure, not a refusal, and one that succeeds immediately on a
        fresh connection.

        This is not a general retry policy. Only `TransportError` is retried,
        and only once: a 4xx is a refusal that will be refused again, and every
        write here is an idempotent upsert on a natural key, so re-sending one
        that may already have landed cannot double-write.

        Found by a real run, where a dropped connection during an escalation
        took down a run that had already survived the error being escalated.
        """
        token = await self._authenticate(client)
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {token}"}

        for attempt in (1, 2):
            try:
                response = await client.put(url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                if attempt == 1:
                    logger.info("write-api connection dropped on %s; retrying once", path)
                    continue
                raise CheckpointError(f"write-api unreachable for {path}: {exc}") from exc
            except httpx.HTTPError as exc:
                raise CheckpointError(f"write-api unreachable for {path}: {exc}") from exc
            if response.status_code != 200:
                raise CheckpointError(
                    f"write-api refused {path}: {response.status_code} {response.text[:300]}"
                )
            return

    async def save_run(self, state: RunState, *, client: httpx.AsyncClient) -> None:
        """Checkpoint the run. Called at every stage transition (§13)."""
        await self._put(client, f"/v1/runs/{state.run_id}", state.model_dump(mode="json"))

    async def save_question_status(
        self,
        *,
        run_id: str,
        question_id: str,
        status: QuestionStatus,
        client: httpx.AsyncClient,
    ) -> None:
        """Checkpoint one question's position. Called on every change (§13)."""
        await self._put(
            client,
            f"/v1/runs/{run_id}/questions/{question_id}/status",
            {"status": status.value},
        )

    async def save_draft(
        self, *, run_id: str, answer: DraftedAnswer, client: httpx.AsyncClient
    ) -> None:
        await self._put(
            client,
            f"/v1/drafts/{run_id}/{answer.question_id}",
            answer.model_dump(mode="json"),
        )


@dataclass(frozen=True)
class ResumableRun:
    """What a resume needs: where the run got to, and what it already produced.

    `answers` holds only questions that reached a terminal state. A question
    checkpointed as PENDING or DRAFTED is re-executed from the start on resume,
    because a half-finished question is not a result — and re-running it is free
    of side effects, since every write is an upsert on its natural key.
    """

    state: RunState
    answers: dict[str, DraftedAnswer]


#: Statuses whose work is finished and must NOT be re-executed on resume.
#: DRAFTED and CRITIQUED are deliberately absent: both mean the question was
#: mid-pipeline when the run died, and finishing it needs the steps that follow.
TERMINAL_STATUSES = frozenset(
    {QuestionStatus.COMPLETE, QuestionStatus.ESCALATED, QuestionStatus.FAILED}
)


async def load_run(run_id: str) -> ResumableRun:
    """Read a run back for `make resume RUN=<id>`.

    Raises rather than returning None for an unknown run: `make resume` was
    given an id by a human, and the useful answer to a typo is an error naming
    the id, not an empty run that appears to succeed instantly.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from src.state.db import reader_url

    engine = create_async_engine(reader_url())
    try:
        async with engine.connect() as conn:
            run_row = (
                await conn.execute(
                    text(
                        "SELECT run_id, rfp_id, stage, tokens_used, cost_usd, "
                        "started_at, updated_at, halted_reason "
                        "FROM runs WHERE run_id = :run_id"
                    ),
                    {"run_id": run_id},
                )
            ).first()
            if run_row is None:
                raise CheckpointError(
                    f"no run '{run_id}' in the database. `make resume RUN=<id>` takes a "
                    f"run id from a previous run."
                )

            status_rows = (
                await conn.execute(
                    text("SELECT question_id, status FROM question_status WHERE run_id = :run_id"),
                    {"run_id": run_id},
                )
            ).all()

            draft_rows = (
                await conn.execute(
                    text(
                        "SELECT question_id, answer_text, source_ids, confidence, "
                        "needs_sme_review, unsupported_claims, escalation_reason "
                        "FROM drafts WHERE run_id = :run_id"
                    ),
                    {"run_id": run_id},
                )
            ).all()
    finally:
        await engine.dispose()

    per_question = {str(qid): QuestionStatus(status) for qid, status in status_rows}
    state = RunState(
        run_id=str(run_row[0]),
        rfp_id=str(run_row[1]),
        stage=RunStage(run_row[2]),
        per_question_status=per_question,
        tokens_used=int(run_row[3]),
        cost_usd=float(run_row[4]),
        started_at=run_row[5],
        updated_at=run_row[6],
        halted_reason=run_row[7],
    )

    answers = {
        str(row[0]): DraftedAnswer(
            question_id=str(row[0]),
            answer_text=str(row[1]),
            source_ids=list(row[2] or []),
            confidence=float(row[3]),
            needs_sme_review=bool(row[4]),
            unsupported_claims=list(row[5] or []),
            escalation_reason=row[6],
        )
        for row in draft_rows
        if per_question.get(str(row[0])) in TERMINAL_STATUSES
    }
    return ResumableRun(state=state, answers=answers)
