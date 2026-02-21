"""add model settings columns to user_settings

Revision ID: 0008
Revises: 0007
Create Date: 2026-02-21 40:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("user_settings", sa.Column("summarization_model", sa.Text(), nullable=True))
    op.add_column("user_settings", sa.Column("working_memory_model", sa.Text(), nullable=True))
    op.add_column("user_settings", sa.Column("default_image_model", sa.Text(), nullable=True))
    op.add_column("user_settings", sa.Column("default_video_model", sa.Text(), nullable=True))
    op.add_column("user_settings", sa.Column("subagent_model", sa.Text(), nullable=True))
    op.add_column(
        "user_settings", sa.Column("subagent_thinking_budget", sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("user_settings", "subagent_thinking_budget")
    op.drop_column("user_settings", "subagent_model")
    op.drop_column("user_settings", "default_video_model")
    op.drop_column("user_settings", "default_image_model")
    op.drop_column("user_settings", "working_memory_model")
    op.drop_column("user_settings", "summarization_model")
