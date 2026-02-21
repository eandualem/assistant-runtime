"""add db constraints and inbox composite index

Revision ID: 0009
Revises: 0008
Create Date: 2026-02-21 50:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_check_constraint(
        "ck_inbox_items_severity_valid",
        "inbox_items",
        "severity IN ('info', 'action_needed', 'urgent')",
    )
    op.create_check_constraint(
        "ck_user_settings_singleton_id",
        "user_settings",
        "id = 'default'",
    )
    op.create_index(
        "ix_inbox_items_surfaced_severity_created_at",
        "inbox_items",
        ["surfaced", "severity", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_inbox_items_surfaced_severity_created_at", table_name="inbox_items")
    op.drop_constraint("ck_user_settings_singleton_id", "user_settings", type_="check")
    op.drop_constraint("ck_inbox_items_severity_valid", "inbox_items", type_="check")
