"""Messages a host appends to a conversation without a model turn (role ``host``).

Revision ID: 0034
Revises: 0033
"""

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
    op.execute("DELETE FROM messages WHERE role = 'host'")
    op.drop_constraint("ck_messages_role_valid", "messages", type_="check")
    op.create_check_constraint("ck_messages_role_valid", "messages", "role IN ('user', 'assistant')")
