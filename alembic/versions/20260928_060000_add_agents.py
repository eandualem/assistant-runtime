"""Persistent agents: one continuing session each, and the messages sent to them.

Revision ID: 0035
Revises: 0034
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0035"
down_revision = "0034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("profile", sa.String(64), nullable=True),
        sa.Column("subject", sa.String(128), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_by", sa.String(128), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('active', 'stopped')", name="ck_agents_status_valid"),
    )
    op.create_index(
        "uq_agents_active_identity",
        "agents",
        [sa.text("coalesce(profile, '')"), sa.text("coalesce(subject, '')")],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_index(
        "uq_agents_active_session",
        "agents",
        ["session_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_table(
        "agent_messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id"), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=True),
        sa.Column("parent_session_id", sa.String(64), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
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
            name="ck_agent_messages_status_valid",
        ),
    )
    op.create_index(
        "ix_agent_messages_agent_id_created_at", "agent_messages", ["agent_id", "created_at"]
    )
    op.create_index(
        "ix_agent_messages_parent_session_id", "agent_messages", ["parent_session_id"]
    )
    op.create_index("ix_agent_messages_status", "agent_messages", ["status"])


def downgrade() -> None:
    op.drop_table("agent_messages")
    op.drop_table("agents")
