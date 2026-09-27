"""Background tasks: what was asked, where it ran, how it ended.

Revision ID: 0031
Revises: 0030
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("parent_session_id", sa.String(64), nullable=True),
        sa.Column("profile", sa.String(64), nullable=True),
        sa.Column("subject", sa.String(128), nullable=True),
        sa.Column("task", sa.Text(), nullable=False),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("usage", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_by", sa.String(128), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'done', 'failed', 'cancelled', 'interrupted')",
            name="ck_tasks_status_valid",
        ),
    )
    op.create_index("ix_tasks_parent_session_id", "tasks", ["parent_session_id"])
    op.create_index("ix_tasks_status", "tasks", ["status"])


def downgrade() -> None:
    op.drop_index("ix_tasks_status", table_name="tasks")
    op.drop_index("ix_tasks_parent_session_id", table_name="tasks")
    op.drop_table("tasks")
