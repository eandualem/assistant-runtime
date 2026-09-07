"""add telegram binding columns to sessions

Revision ID: 0016
Revises: 0015
Create Date: 2026-03-21 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0016"
down_revision: Union[str, None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("telegram_chat_id", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column("telegram_bound_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_sessions_telegram_chat_id_bound_at",
        "sessions",
        ["telegram_chat_id", "telegram_bound_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_sessions_telegram_chat_id_bound_at", table_name="sessions")
    op.drop_column("sessions", "telegram_bound_at")
    op.drop_column("sessions", "telegram_chat_id")
