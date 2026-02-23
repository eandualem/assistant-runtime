"""widen session id columns from varchar(36) to varchar(64)

Revision ID: 0010
Revises: 3293fd0bfee2
Create Date: 2026-02-22 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: Union[str, None] = "3293fd0bfee2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "sessions",
        "id",
        existing_type=sa.String(36),
        type_=sa.String(64),
        existing_nullable=False,
    )
    op.alter_column(
        "traces",
        "session_id",
        existing_type=sa.String(36),
        type_=sa.String(64),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "traces",
        "session_id",
        existing_type=sa.String(64),
        type_=sa.String(36),
        existing_nullable=False,
    )
    op.alter_column(
        "sessions",
        "id",
        existing_type=sa.String(64),
        type_=sa.String(36),
        existing_nullable=False,
    )
