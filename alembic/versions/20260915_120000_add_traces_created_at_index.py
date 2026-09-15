"""Index trace rows by creation time and drop stored screenshot payloads.

Screenshots are no longer written to traces; the rows written before this
release still carry them, so the payloads are cleared. The column stays
(nullable, unused) so older installations downgrade cleanly.

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
    op.execute("UPDATE traces SET screenshot = NULL WHERE screenshot IS NOT NULL")


def downgrade() -> None:
    op.drop_index("ix_traces_created_at", table_name="traces")
