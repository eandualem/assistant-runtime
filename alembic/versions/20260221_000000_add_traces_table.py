"""add traces table

Revision ID: 0004
Revises: 0003
Create Date: 2026-02-21 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "traces",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("events", sa.dialects.postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("user_message", sa.Text(), nullable=True),
        sa.Column("is_continuation", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_traces")),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name=op.f("fk_traces_session_id_sessions"),
            ondelete="CASCADE",
        ),
    )
    op.create_index(op.f("ix_traces_session_id"), "traces", ["session_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_traces_session_id"), table_name="traces")
    op.drop_table("traces")
