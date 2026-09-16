"""Add the codex_service_tier tunable to the runtime settings row.

Revision ID: 0025
Revises: 0024
"""

import sqlalchemy as sa
from alembic import op

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("user_settings", sa.Column("codex_service_tier", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("user_settings", "codex_service_tier")
