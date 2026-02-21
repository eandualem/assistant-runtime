"""add screenshot column to traces

Revision ID: 0005
Revises: 0004
Create Date: 2026-02-21 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("traces", sa.Column("screenshot", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("traces", "screenshot")
