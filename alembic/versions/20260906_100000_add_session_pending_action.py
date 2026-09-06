"""add pending_action to sessions

Revision ID: 0022
Revises: 0021
Create Date: 2026-09-06 10:00:00.000000

The one host-tool call a session is waiting on used to live in process
memory only, so a restart turned every unanswered call stale. It is now
stored on the session row and restored on load; rows created before this
revision have no pending action, and their unanswered calls are recorded
as ``unknown`` when the session is next loaded.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0022"
down_revision: Union[str, None] = "0021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("pending_action", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sessions", "pending_action")
