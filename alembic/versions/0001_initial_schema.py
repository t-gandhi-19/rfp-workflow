"""Initial schema: run record, artifacts, and eval results.

Revision ID: 0001
Revises:
Create Date: Phase 1

Every artifact table is keyed on its natural key so writes are idempotent
upserts and a resumed run cannot duplicate work. `written_by` carries the JWT
subject that authorized each write, joining these rows to the Keycloak audit
trail and to OTel spans.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ID = sa.String(64)
NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table(
        "rfp_documents",
        sa.Column("id", ID, nullable=False),
        sa.Column("source_filename", sa.Text(), nullable=False),
        sa.Column("format", sa.String(8), nullable=False),
        sa.Column("customer_name", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(64), nullable=False),
        sa.Column("detected_deadline", sa.Date(), nullable=True),
        sa.Column("intake_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("sections", JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_rfp_documents"),
    )

    op.create_table(
        "questions",
        sa.Column("id", ID, nullable=False),
        sa.Column("rfp_id", ID, nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("section", sa.Text(), nullable=False),
        sa.Column("question_type", sa.String(32), nullable=False),
        sa.Column("word_limit", sa.Integer(), nullable=True),
        sa.Column("mandatory", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("question_order", sa.Integer(), nullable=False),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["rfp_id"],
            ["rfp_documents.id"],
            name="fk_questions_rfp_id_rfp_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_questions"),
    )
    op.create_index("ix_questions_rfp_id_order", "questions", ["rfp_id", "question_order"])

    op.create_table(
        "runs",
        sa.Column("run_id", ID, nullable=False),
        sa.Column("rfp_id", ID, nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("tokens_used", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("cost_usd", sa.Numeric(12, 6), server_default=sa.text("0"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("halted_reason", sa.String(32), nullable=True),
        sa.Column("trace_url", sa.Text(), nullable=True),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("run_id", name="pk_runs"),
    )
    op.create_index("ix_runs_rfp_id", "runs", ["rfp_id"])
    op.create_index("ix_runs_stage", "runs", ["stage"])

    op.create_table(
        "question_status",
        sa.Column("run_id", ID, nullable=False),
        sa.Column("question_id", ID, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.run_id"], name="fk_question_status_run_id_runs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("run_id", "question_id", name="pk_question_status"),
    )

    op.create_table(
        "retrieval_results",
        sa.Column("run_id", ID, nullable=False),
        sa.Column("question_id", ID, nullable=False),
        sa.Column("candidates", JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("floor_used", sa.Numeric(6, 4), nullable=False),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.run_id"], name="fk_retrieval_results_run_id_runs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("run_id", "question_id", name="pk_retrieval_results"),
    )

    op.create_table(
        "drafts",
        sa.Column("run_id", ID, nullable=False),
        sa.Column("question_id", ID, nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=False),
        sa.Column("source_ids", JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("confidence", sa.Numeric(6, 4), nullable=False),
        sa.Column("needs_sme_review", sa.Boolean(), nullable=False),
        sa.Column(
            "unsupported_claims", JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False
        ),
        sa.Column("escalation_reason", sa.Text(), nullable=True),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.run_id"], name="fk_drafts_run_id_runs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("run_id", "question_id", name="pk_drafts"),
    )
    op.create_index("ix_drafts_needs_sme_review", "drafts", ["run_id", "needs_sme_review"])

    op.create_table(
        "critiques",
        sa.Column("run_id", ID, nullable=False),
        sa.Column("question_id", ID, nullable=False),
        sa.Column("issues", JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("confidence_delta", sa.Numeric(6, 4), nullable=False),
        sa.Column(
            "added_unsupported_claims",
            JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.run_id"], name="fk_critiques_run_id_runs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("run_id", "question_id", name="pk_critiques"),
    )

    op.create_table(
        "compliance_results",
        sa.Column("run_id", ID, nullable=False),
        sa.Column("question_id", ID, nullable=False),
        sa.Column("mandatory_answered", sa.Boolean(), nullable=False),
        sa.Column("within_word_limit", sa.Boolean(), nullable=False),
        sa.Column(
            "forbidden_content_hits",
            JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("template_slots_filled", sa.Boolean(), nullable=False),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.run_id"],
            name="fk_compliance_results_run_id_runs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "question_id", name="pk_compliance_results"),
    )

    op.create_table(
        "entity_checks",
        sa.Column("run_id", ID, nullable=False),
        sa.Column("question_id", ID, nullable=False),
        sa.Column("entity_text", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.String(32), nullable=False),
        sa.Column("resolved_node_id", ID, nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.run_id"], name="fk_entity_checks_run_id_runs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("run_id", "question_id", "entity_text", name="pk_entity_checks"),
    )
    op.create_index("ix_entity_checks_failed", "entity_checks", ["run_id", "passed"])

    op.create_table(
        "escalations",
        sa.Column("run_id", ID, nullable=False),
        sa.Column("question_id", ID, nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("sme_id", ID, nullable=True),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.run_id"], name="fk_escalations_run_id_runs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("run_id", "question_id", name="pk_escalations"),
    )

    op.create_table(
        "guardrail_hits",
        sa.Column("run_id", ID, nullable=False),
        sa.Column("question_id", ID, nullable=False),
        sa.Column("guardrail", sa.String(64), nullable=False),
        sa.Column("details", JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.run_id"], name="fk_guardrail_hits_run_id_runs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("run_id", "question_id", "guardrail", name="pk_guardrail_hits"),
    )

    op.create_table(
        "eval_results",
        sa.Column("git_sha", sa.String(64), nullable=False),
        sa.Column("run_id", ID, nullable=False),
        sa.Column("metric", sa.String(128), nullable=False),
        sa.Column("value", sa.Numeric(12, 6), nullable=False),
        sa.Column("threshold", sa.Numeric(12, 6), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.PrimaryKeyConstraint("git_sha", "run_id", "metric", name="pk_eval_results"),
    )
    op.create_index("ix_eval_results_metric_sha", "eval_results", ["metric", "git_sha"])


def downgrade() -> None:
    op.drop_table("eval_results")
    op.drop_table("guardrail_hits")
    op.drop_table("escalations")
    op.drop_table("entity_checks")
    op.drop_table("compliance_results")
    op.drop_table("critiques")
    op.drop_table("drafts")
    op.drop_table("retrieval_results")
    op.drop_table("question_status")
    op.drop_table("runs")
    op.drop_table("questions")
    op.drop_table("rfp_documents")
