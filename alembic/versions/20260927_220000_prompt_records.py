"""Record the system prompt each assistant message was produced with.

``messages.prompt`` holds the record (profile, subject, artifact versions,
the dynamic fragments); the stable part of the prompt is stored once in
``prompt_snapshots`` under its SHA-256, since it repeats across turns.

Revision ID: 0030
Revises: 0029
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prompt_snapshots",
        sa.Column("hash", sa.String(64), primary_key=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.add_column(
        "messages", sa.Column("prompt", postgresql.JSONB(astext_type=sa.Text()), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("messages", "prompt")
    op.drop_table("prompt_snapshots")
