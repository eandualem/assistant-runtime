"""add owner_id to sessions

Revision ID: 0021
Revises: 0020
Create Date: 2026-09-05 13:00:00.000000

Rows created before ownership existed keep a NULL owner: only administrators
can reach them until one assigns an owner (PATCH /api/sessions/{id}/owner).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0021"
down_revision: Union[str, None] = "0020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("sessions", sa.Column("owner_id", sa.String(length=128), nullable=True))
    op.create_index("ix_sessions_owner_id", "sessions", ["owner_id"])


def downgrade() -> None:
    op.drop_index("ix_sessions_owner_id", table_name="sessions")
    op.drop_column("sessions", "owner_id")
