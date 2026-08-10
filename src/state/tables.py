"""SQLAlchemy Core table definitions — the shape of the run record.

Two properties are deliberate throughout:

**Natural keys, not surrogate ids.** Every artifact table is keyed on
``(run_id, question_id)`` (or ``(git_sha, run_id, metric)``). That makes every
write an idempotent upsert, which is what lets a killed run resume and produce a
byte-identical artifact rather than a duplicated one (build prompt §13).

**Every row records who wrote it.** ``written_by`` carries the JWT subject from
the token that authorized the write. It is what joins the Postgres record to the
Keycloak audit trail and the OTel spans, and it is how the Phase 6 dashboard can
show which service account produced which artifact.

Definitions live here rather than in ``write_api`` because state persistence is
shared: write-api writes through them and the MCP server, dashboard, and log
interpreter read through them with a SELECT-only role.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

# Explicit naming convention so Alembic generates stable constraint names and a
# future migration can always refer to one by name.
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

ID = String(64)
NOW = text("now()")


rfp_documents = Table(
    "rfp_documents",
    metadata,
    Column("id", ID, primary_key=True),
    Column("source_filename", Text, nullable=False),
    Column("format", String(8), nullable=False),
    Column("customer_name", Text, nullable=False),
    Column("domain", String(64), nullable=False),
    Column("detected_deadline", Date, nullable=True),
    Column("intake_timestamp", DateTime(timezone=True), nullable=False),
    Column("raw_text", Text, nullable=False),
    Column("sections", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("written_by", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=NOW),
)


questions = Table(
    "questions",
    metadata,
    Column("id", ID, primary_key=True),
    Column("rfp_id", ID, ForeignKey("rfp_documents.id", ondelete="CASCADE"), nullable=False),
    Column("text", Text, nullable=False),
    Column("normalized_text", Text, nullable=False),
    Column("section", Text, nullable=False),
    Column("question_type", String(32), nullable=False),
    Column("word_limit", Integer, nullable=True),
    Column("mandatory", Boolean, nullable=False, server_default=text("false")),
    # `order` is reserved in SQL; the contract field keeps its name, the column
    # does not.
    Column("question_order", Integer, nullable=False),
    Column("written_by", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    Index("ix_questions_rfp_id_order", "rfp_id", "question_order"),
)


runs = Table(
    "runs",
    metadata,
    Column("run_id", ID, primary_key=True),
    Column("rfp_id", ID, nullable=False),
    Column("stage", String(32), nullable=False),
    Column("tokens_used", Integer, nullable=False, server_default=text("0")),
    Column("cost_usd", Numeric(12, 6), nullable=False, server_default=text("0")),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("halted_reason", String(32), nullable=True),
    Column("trace_url", Text, nullable=True),
    Column("written_by", Text, nullable=False),
    Index("ix_runs_rfp_id", "rfp_id"),
    Index("ix_runs_stage", "stage"),
)


question_status = Table(
    "question_status",
    metadata,
    Column("run_id", ID, ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("question_id", ID, nullable=False),
    Column("status", String(32), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    Column("written_by", Text, nullable=False),
    PrimaryKeyConstraint("run_id", "question_id"),
)


retrieval_results = Table(
    "retrieval_results",
    metadata,
    Column("run_id", ID, ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("question_id", ID, nullable=False),
    Column("candidates", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("status", String(16), nullable=False),
    Column("floor_used", Numeric(6, 4), nullable=False),
    Column("written_by", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    PrimaryKeyConstraint("run_id", "question_id"),
)


drafts = Table(
    "drafts",
    metadata,
    Column("run_id", ID, ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("question_id", ID, nullable=False),
    Column("answer_text", Text, nullable=False),
    Column("source_ids", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("confidence", Numeric(6, 4), nullable=False),
    Column("needs_sme_review", Boolean, nullable=False),
    Column("unsupported_claims", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("escalation_reason", Text, nullable=True),
    Column("written_by", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    PrimaryKeyConstraint("run_id", "question_id"),
    Index("ix_drafts_needs_sme_review", "run_id", "needs_sme_review"),
)


critiques = Table(
    "critiques",
    metadata,
    Column("run_id", ID, ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("question_id", ID, nullable=False),
    Column("issues", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("confidence_delta", Numeric(6, 4), nullable=False),
    Column("added_unsupported_claims", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("written_by", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    PrimaryKeyConstraint("run_id", "question_id"),
)


compliance_results = Table(
    "compliance_results",
    metadata,
    Column("run_id", ID, ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("question_id", ID, nullable=False),
    Column("mandatory_answered", Boolean, nullable=False),
    Column("within_word_limit", Boolean, nullable=False),
    Column("forbidden_content_hits", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("template_slots_filled", Boolean, nullable=False),
    Column("written_by", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    PrimaryKeyConstraint("run_id", "question_id"),
)


entity_checks = Table(
    "entity_checks",
    metadata,
    Column("run_id", ID, ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("question_id", ID, nullable=False),
    Column("entity_text", Text, nullable=False),
    Column("entity_type", String(32), nullable=False),
    Column("resolved_node_id", ID, nullable=True),
    Column("passed", Boolean, nullable=False),
    Column("written_by", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    PrimaryKeyConstraint("run_id", "question_id", "entity_text"),
    Index("ix_entity_checks_failed", "run_id", "passed"),
)


escalations = Table(
    "escalations",
    metadata,
    Column("run_id", ID, ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("question_id", ID, nullable=False),
    Column("reason", Text, nullable=False),
    Column("sme_id", ID, nullable=True),
    Column("written_by", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    PrimaryKeyConstraint("run_id", "question_id"),
)


guardrail_hits = Table(
    "guardrail_hits",
    metadata,
    Column("run_id", ID, ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("question_id", ID, nullable=False),
    Column("guardrail", String(64), nullable=False),
    Column("details", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("written_by", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    # One row per guardrail per question: a re-run overwrites rather than
    # accumulating duplicates.
    PrimaryKeyConstraint("run_id", "question_id", "guardrail"),
)


eval_results = Table(
    "eval_results",
    metadata,
    Column("git_sha", String(64), nullable=False),
    Column("run_id", ID, nullable=False),
    Column("metric", String(128), nullable=False),
    Column("value", Numeric(12, 6), nullable=False),
    Column("threshold", Numeric(12, 6), nullable=False),
    Column("passed", Boolean, nullable=False),
    Column("written_by", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    PrimaryKeyConstraint("git_sha", "run_id", "metric"),
    # The dashboard's regression view reads metrics by SHA in commit order.
    Index("ix_eval_results_metric_sha", "metric", "git_sha"),
)


__all__ = [
    "compliance_results",
    "critiques",
    "drafts",
    "entity_checks",
    "escalations",
    "eval_results",
    "guardrail_hits",
    "metadata",
    "question_status",
    "questions",
    "retrieval_results",
    "rfp_documents",
    "runs",
]
