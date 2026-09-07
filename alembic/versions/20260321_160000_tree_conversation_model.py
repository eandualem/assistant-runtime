"""tree conversation model

Revision ID: 0017
Revises: 0016
Create Date: 2026-03-21 16:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0017"
down_revision: Union[str, None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("parent_id", sa.String(length=64), nullable=True),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column(
            "message_type",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'standard'"),
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("segments", JSONB(), nullable=True),
        sa.Column("usage", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role_valid"),
        sa.CheckConstraint(
            "message_type IN ('standard', 'guidance')",
            name="ck_messages_message_type_valid",
        ),
        sa.ForeignKeyConstraint(["parent_id"], ["messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_parent_id", "messages", ["parent_id"], unique=False)
    op.create_index(
        "ix_messages_session_id_created_at",
        "messages",
        ["session_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "uq_messages_single_root_per_session",
        "messages",
        ["session_id"],
        unique=True,
        postgresql_where=sa.text("parent_id IS NULL"),
    )

    op.drop_column("sessions", "message_history")


def downgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("message_history", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
    )

    op.drop_index("uq_messages_single_root_per_session", table_name="messages")
    op.drop_index("ix_messages_session_id_created_at", table_name="messages")
    op.drop_index("ix_messages_parent_id", table_name="messages")
    op.drop_table("messages")
