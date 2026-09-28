"""Store what the model was given, so later turns replay it unchanged.

``messages.model_messages`` holds a row's part of the conversation as
Pydantic AI messages (``ModelMessagesTypeAdapter``): a user row's request, an
assistant row's responses, tool returns and the requests its turn added.
``steering.attachments`` keeps a steering message's reference attachments.

Revision ID: 0036
Revises: 0035
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0036"
down_revision = "0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("model_messages", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "steering",
        sa.Column("attachments", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("steering", "attachments")
    op.drop_column("messages", "model_messages")
