"""Index trace rows by creation time for age-based retention.

Revision ID: 0024
Revises: 0023
"""

from alembic import op

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_traces_created_at", "traces", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_traces_created_at", table_name="traces")
