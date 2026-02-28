"""drop pending_tool_call column from sessions table

Frontend tool infrastructure removed — the backend is backend-only.
No more DeferredToolRequests, no continuation flow, no pending tool call tracking.

Revision ID: 0012
Revises: 0011
Create Date: 2026-02-28 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("sessions", "pending_tool_call")


def downgrade() -> None:
    op.add_column("sessions", sa.Column("pending_tool_call", JSONB, nullable=True))
