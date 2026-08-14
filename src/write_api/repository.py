"""Idempotent upserts — the only code that writes to Postgres.

Every function here is `INSERT ... ON CONFLICT DO UPDATE` on the table's natural
key. That is what makes `make resume` safe: re-running a question that already
completed overwrites its row with the same values rather than creating a second
one, so a resumed run and an uninterrupted run produce identical state.

All values are bound parameters via SQLAlchemy Core. No SQL string is ever built
by interpolation, and `tests/security/test_no_fstring_sql.py` walks the AST of
this module to keep it that way.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection

from src.contracts import DraftedAnswer, EvalScore, QuestionStatus, RunState
from src.retrieval.calibration import CalibrationArtifact
from src.state.tables import (
    calibration_artifacts,
    drafts,
    eval_results,
    question_status,
    runs,
)


async def _upsert(
    conn: AsyncConnection,
    table: Any,
    values: dict[str, Any],
    *,
    key: list[str],
) -> None:
    """INSERT, or UPDATE every non-key column when the natural key collides."""
    statement = pg_insert(table).values(**values)
    updatable = {name: statement.excluded[name] for name in values if name not in key}
    await conn.execute(statement.on_conflict_do_update(index_elements=key, set_=updatable))


async def upsert_run(conn: AsyncConnection, state: RunState, *, written_by: str) -> None:
    """Checkpoint a run's state. Called at every stage transition (§13)."""
    await _upsert(
        conn,
        runs,
        {
            "run_id": state.run_id,
            "rfp_id": state.rfp_id,
            "stage": state.stage.value,
            "tokens_used": state.tokens_used,
            "cost_usd": state.cost_usd,
            "started_at": state.started_at,
            "updated_at": state.updated_at,
            "halted_reason": state.halted_reason.value if state.halted_reason else None,
            "written_by": written_by,
        },
        key=["run_id"],
    )


async def upsert_question_status(
    conn: AsyncConnection,
    *,
    run_id: str,
    question_id: str,
    status: QuestionStatus,
    written_by: str,
) -> None:
    """Record one question's lifecycle position within a run."""
    await _upsert(
        conn,
        question_status,
        {
            "run_id": run_id,
            "question_id": question_id,
            "status": status.value,
            "updated_at": func.now(),
            "written_by": written_by,
        },
        key=["run_id", "question_id"],
    )


async def upsert_draft(
    conn: AsyncConnection,
    *,
    run_id: str,
    answer: DraftedAnswer,
    written_by: str,
) -> None:
    """Persist a drafted answer.

    The contract has already enforced that an unescalated answer carries at
    least one source id and no unsupported claims, so nothing ungrounded can
    reach this table through this path.
    """
    await _upsert(
        conn,
        drafts,
        {
            "run_id": run_id,
            "question_id": answer.question_id,
            "answer_text": answer.answer_text,
            "source_ids": list(answer.source_ids),
            "confidence": answer.confidence,
            "needs_sme_review": answer.needs_sme_review,
            "unsupported_claims": list(answer.unsupported_claims),
            "escalation_reason": answer.escalation_reason,
            "updated_at": func.now(),
            "written_by": written_by,
        },
        key=["run_id", "question_id"],
    )


async def upsert_calibration(
    conn: AsyncConnection,
    artifact: CalibrationArtifact,
    *,
    written_by: str,
) -> None:
    """Record the measured retrieval calibration.

    Keyed on (model, corpus, geometry): re-measuring an unchanged model and
    corpus overwrites, so the table states the current calibration rather than
    logging every attempt to compute it.

    `derived_floor` is stored rather than recomputed on read. The rule that
    derives it may change; the number a given run actually judged against must
    stay recoverable either way.
    """
    await _upsert(
        conn,
        calibration_artifacts,
        {
            "embed_model_tag": artifact.embed_model_tag,
            "corpus_hash": artifact.corpus_hash,
            "geometry": artifact.geometry,
            "computed_at": datetime.fromisoformat(artifact.computed_at),
            "background_pair_count": artifact.background_pair_count,
            "same_topic_pair_count": artifact.same_topic_pair_count,
            "bg_p50": artifact.bg_p50,
            "bg_p95": artifact.bg_p95,
            "bg_p99": artifact.bg_p99,
            "same_topic_p05": artifact.same_topic_p05,
            "same_topic_p50": artifact.same_topic_p50,
            "derived_floor": artifact.derived_floor,
            "written_by": written_by,
        },
        key=["embed_model_tag", "corpus_hash", "geometry"],
    )


async def upsert_eval_scores(
    conn: AsyncConnection,
    scores: list[EvalScore],
    *,
    written_by: str,
) -> None:
    """Persist eval metrics, keyed by git SHA so regressions are diffable."""
    for score in scores:
        await _upsert(
            conn,
            eval_results,
            {
                "git_sha": score.git_sha,
                "run_id": score.run_id,
                "metric": score.metric,
                "value": score.value,
                "threshold": score.threshold,
                "passed": score.passed,
                "written_by": written_by,
            },
            key=["git_sha", "run_id", "metric"],
        )
