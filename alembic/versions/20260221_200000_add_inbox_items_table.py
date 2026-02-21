"""add inbox_items table

Revision ID: 0006
Revises: 0005
Create Date: 2026-02-21 20:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "inbox_items",
        sa.Column(
            "id",
            sa.String(36),
            primary_key=True,
            server_default=sa.func.gen_random_uuid().cast(sa.String),
        ),
        sa.Column("from_agent", sa.String(50), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False, server_default=sa.text("'info'")),
        sa.Column("context", JSONB(), nullable=True),
        sa.Column("surfaced", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_inbox_items_surfaced", "inbox_items", ["surfaced"])
    op.create_index("ix_inbox_items_created_at", "inbox_items", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_inbox_items_created_at", table_name="inbox_items")
    op.drop_index("ix_inbox_items_surfaced", table_name="inbox_items")
    op.drop_table("inbox_items")
