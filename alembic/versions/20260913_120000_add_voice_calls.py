"""Checkpoint optional voice calls without changing the backend message tree.

Revision ID: 0023
Revises: 0022
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "voice_calls",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(64),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("snapshot", postgresql.JSONB(), nullable=False),
    )
    op.create_index("ix_voice_calls_session_id", "voice_calls", ["session_id"])


def downgrade() -> None:
    op.drop_table("voice_calls")
