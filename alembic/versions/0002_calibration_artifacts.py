"""Calibration artifacts: the measured anchors the match floor is derived from.

Revision ID: 0002
Revises: 0001
Create Date: Phase 3, amendment J

Retrieval refuses to run without a valid artifact, so this is not a reporting
table — it is the record of what the floor a run judged against actually meant.

Keyed on (embed_model_tag, corpus_hash, geometry) because those three are what
make the statistics valid: change the model, the corpus, or which side of the
prefix asymmetry was measured, and the numbers describe something else. Writing
the same combination twice overwrites, so the table states the current
calibration rather than logging attempts to compute it.

`derived_floor` is stored rather than recomputed on read. The rule that derives
it may change; the number a given run judged against must stay recoverable.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table(
        "calibration_artifacts",
        sa.Column("embed_model_tag", sa.String(128), nullable=False),
        sa.Column("corpus_hash", sa.String(64), nullable=False),
        sa.Column("geometry", sa.String(64), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("background_pair_count", sa.Integer(), nullable=False),
        sa.Column("same_topic_pair_count", sa.Integer(), nullable=False),
        sa.Column("bg_p50", sa.Numeric(8, 6), nullable=False),
        sa.Column("bg_p95", sa.Numeric(8, 6), nullable=False),
        sa.Column("bg_p99", sa.Numeric(8, 6), nullable=False),
        sa.Column("same_topic_p05", sa.Numeric(8, 6), nullable=False),
        sa.Column("same_topic_p50", sa.Numeric(8, 6), nullable=False),
        sa.Column("derived_floor", sa.Numeric(8, 6), nullable=False),
        sa.Column("written_by", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.PrimaryKeyConstraint(
            "embed_model_tag", "corpus_hash", "geometry", name="pk_calibration_artifacts"
        ),
    )


def downgrade() -> None:
    op.drop_table("calibration_artifacts")
