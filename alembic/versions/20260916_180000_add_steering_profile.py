"""Preserve the requested profile on queued steering.

Revision ID: 0026
Revises: 0025
"""

import sqlalchemy as sa
from alembic import op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("steering", sa.Column("profile", sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column("steering", "profile")
