"""write-api — the only path to Postgres (CLAUDE.md rule 4, build prompt §17).

Everything here is a PUT or a batch POST onto a natural key, because every write
in this system is an idempotent upsert rather than an append. PUT is the honest
verb for that: sending the same body twice leaves the same state, which is
exactly what `make resume` relies on.

Request bodies are the Pydantic contracts themselves, so a malformed payload
fails at this boundary with a readable error instead of halfway through a stage.

Phase 1 exposes the endpoints needed to checkpoint a run and to record drafts
and eval results. The remaining artifact endpoints (critiques, compliance,
entity checks, retrieval results) arrive with the phases that produce them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from src.contracts import DraftedAnswer, EvalScore, QuestionStatus, RunState
from src.state.db import dispose_engine, get_engine
from src.write_api import repository
from src.write_api.auth import Principal, build_verifier, require_role
from src.write_api.settings import get_settings

# Roles are mapped per endpoint rather than per service. The eval harness needs
# to record metrics; it has no business writing an answer. Giving it its own
# role keeps that boundary enforced by the token rather than by convention.
DRAFT_WRITER = "draft-writer"
EVAL_WRITER = "eval-writer"


class Ack(BaseModel):
    """What a successful write returns: what was written, and by whom."""

    model_config = ConfigDict(extra="forbid")

    written: str
    key: str
    written_by: str


class QuestionStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: QuestionStatus


class EvalScoreBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[EvalScore] = Field(min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.verifier = build_verifier()
    yield
    await dispose_engine()


app = FastAPI(
    title="rfp-workflow write-api",
    version="0.1.0",
    summary="The only write path to Postgres. Every write is an idempotent upsert.",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness plus a real database round-trip.

    A health check that does not touch the database would report healthy while
    every write fails, which is the opposite of useful for `depends_on`.
    """
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"database unavailable: {exc}",
        ) from exc
    return {"status": "ok", "git_sha": get_settings().git_sha}


@app.put("/v1/runs/{run_id}", response_model=Ack, tags=["runs"])
async def put_run(
    run_id: str,
    state: RunState,
    principal: Annotated[Principal, Depends(require_role(DRAFT_WRITER))],
) -> Ack:
    """Checkpoint run state. Idempotent on run_id."""
    if state.run_id != run_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"path run_id '{run_id}' does not match body run_id '{state.run_id}'",
        )
    async with get_engine().begin() as conn:
        await repository.upsert_run(conn, state, written_by=principal.subject)
    return Ack(written="run", key=run_id, written_by=principal.subject)


@app.put(
    "/v1/runs/{run_id}/questions/{question_id}/status",
    response_model=Ack,
    tags=["runs"],
)
async def put_question_status(
    run_id: str,
    question_id: str,
    update: QuestionStatusUpdate,
    principal: Annotated[Principal, Depends(require_role(DRAFT_WRITER))],
) -> Ack:
    """Record one question's status. Idempotent on (run_id, question_id)."""
    async with get_engine().begin() as conn:
        await repository.upsert_question_status(
            conn,
            run_id=run_id,
            question_id=question_id,
            status=update.status,
            written_by=principal.subject,
        )
    return Ack(
        written="question_status",
        key=f"{run_id}/{question_id}",
        written_by=principal.subject,
    )


@app.put("/v1/drafts/{run_id}/{question_id}", response_model=Ack, tags=["drafts"])
async def put_draft(
    run_id: str,
    question_id: str,
    answer: DraftedAnswer,
    principal: Annotated[Principal, Depends(require_role(DRAFT_WRITER))],
) -> Ack:
    """Persist a drafted answer. Idempotent on (run_id, question_id).

    The DraftedAnswer contract has already refused anything uncited and
    unescalated, so that check happens before a connection is even opened.
    """
    if answer.question_id != question_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"path question_id '{question_id}' does not match "
                f"body question_id '{answer.question_id}'"
            ),
        )
    async with get_engine().begin() as conn:
        await repository.upsert_draft(
            conn, run_id=run_id, answer=answer, written_by=principal.subject
        )
    return Ack(written="draft", key=f"{run_id}/{question_id}", written_by=principal.subject)


@app.post("/v1/eval-results", response_model=Ack, tags=["evals"])
async def post_eval_results(
    batch: EvalScoreBatch,
    principal: Annotated[Principal, Depends(require_role(EVAL_WRITER))],
) -> Ack:
    """Persist eval metrics, keyed by git SHA. Idempotent per (sha, run, metric)."""
    async with get_engine().begin() as conn:
        await repository.upsert_eval_scores(conn, batch.scores, written_by=principal.subject)
    return Ack(
        written="eval_results",
        key=f"{len(batch.scores)} metric(s)",
        written_by=principal.subject,
    )
