"""Messages a host appends to a conversation without a model turn (role ``host``).

Revision ID: 0034
Revises: 0033
"""

import sqlalchemy as sa
from alembic import op

revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_messages_role_valid", "messages", type_="check")
    op.create_check_constraint(
        "ck_messages_role_valid", "messages", "role IN ('user', 'assistant', 'host')"
    )


def downgrade() -> None:
    # Deleting host messages would cascade to every message that follows them.
    count = op.get_bind().execute(sa.text("SELECT count(*) FROM messages WHERE role = 'host'"))
    if count.scalar():
        raise RuntimeError(
            "Host messages exist; downgrading below 0034 would delete the conversations "
            "that continue after them"
        )
    op.drop_constraint("ck_messages_role_valid", "messages", type_="check")
    op.create_check_constraint("ck_messages_role_valid", "messages", "role IN ('user', 'assistant')")
