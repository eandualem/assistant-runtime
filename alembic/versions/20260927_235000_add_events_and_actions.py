"""Event records, and host-written action records with owner confirmations.

Revision ID: 0032
Revises: 0031
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None

_JSON = postgresql.JSONB(astext_type=sa.Text())


def _now() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("event_id", sa.String(200), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("agent", sa.String(128), nullable=True),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("payload", _JSON, nullable=True),
        sa.Column("target_session_id", sa.String(64), nullable=True),
        sa.Column("history", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("delivery", _JSON, nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        _now(),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heard_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("source", "event_id", name="uq_events_source_event_id"),
        sa.CheckConstraint("direction IN ('inbound', 'outbound')", name="ck_events_direction"),
        sa.CheckConstraint(
            "severity IN ('info', 'warning', 'critical')", name="ck_events_severity"
        ),
        sa.CheckConstraint(
            "status IN ('received', 'delivered', 'pending', 'heard')", name="ck_events_status"
        ),
    )
    op.create_index("ix_events_agent", "events", ["agent"])
    op.create_table(
        "actions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("arguments", _JSON, nullable=True),
        sa.Column("profile", sa.String(64), nullable=True),
        sa.Column("subject", sa.String(128), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("history", _JSON, nullable=False),
        sa.Column("confirmed_by", sa.String(128), nullable=True),
        sa.Column("results", _JSON, nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("proposed_text", sa.Text(), nullable=True),
        sa.Column("revised_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(128), nullable=False),
        _now(),
        sa.CheckConstraint(
            "status IN ('proposed', 'scheduled', 'sending', 'sent', 'failed', 'discarded', "
            "'undone')",
            name="ck_actions_status",
        ),
    )
    op.create_index("ix_actions_status", "actions", ["status"])
    op.create_table(
        "action_confirmations",
        sa.Column("seq", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "action_id",
            sa.BigInteger(),
            sa.ForeignKey("actions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("recipient", sa.String(200), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("text_sha256", sa.String(64), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("key_epoch", sa.Integer(), nullable=True),
        sa.Column("sender", sa.String(200), nullable=True),
        sa.Column("audience", sa.String(200), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("result", _JSON, nullable=True),
        sa.Column("reconciled", sa.String(16), nullable=True),
        sa.CheckConstraint("kind IN ('message', 'steer')", name="ck_confirmations_kind"),
        sa.CheckConstraint(
            "source IN ('button', 'typed', 'voice')", name="ck_confirmations_source"
        ),
        sa.CheckConstraint(
            "status IN ('confirmed', 'sent', 'failed')", name="ck_confirmations_status"
        ),
        sa.CheckConstraint(
            "reconciled IS NULL OR reconciled IN ('matched', 'altered', 'missing', 'undelivered')",
            name="ck_confirmations_reconciled",
        ),
    )
    op.create_index("ix_action_confirmations_action_id", "action_confirmations", ["action_id"])


def downgrade() -> None:
    op.drop_index("ix_action_confirmations_action_id", table_name="action_confirmations")
    op.drop_table("action_confirmations")
    op.drop_index("ix_actions_status", table_name="actions")
    op.drop_table("actions")
    op.drop_index("ix_events_agent", table_name="events")
    op.drop_table("events")
