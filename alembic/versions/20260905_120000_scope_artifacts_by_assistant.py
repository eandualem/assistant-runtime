"""scope artifacts by assistant profile

Revision ID: 0020
Revises: 0019
Create Date: 2026-09-05 12:00:00.000000

Existing rows were written by the original technical-operator assistant, so they
are assigned to the ``technical_operator`` profile.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0020"
down_revision: Union[str, None] = "0019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "artifacts",
        sa.Column(
            "assistant",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("'technical_operator'"),
        ),
    )
    op.drop_constraint("uq_artifacts_name_version", "artifacts", type_="unique")
    op.create_unique_constraint(
        "uq_artifacts_assistant_name_version", "artifacts", ["assistant", "name", "version"]
    )
    op.drop_index("ix_artifacts_name_is_active", table_name="artifacts")
    op.create_index(
        "ix_artifacts_assistant_name_is_active",
        "artifacts",
        ["assistant", "name", "is_active"],
    )


def downgrade() -> None:
    op.drop_index("ix_artifacts_assistant_name_is_active", table_name="artifacts")
    op.create_index("ix_artifacts_name_is_active", "artifacts", ["name", "is_active"])
    op.drop_constraint("uq_artifacts_assistant_name_version", "artifacts", type_="unique")
    op.create_unique_constraint("uq_artifacts_name_version", "artifacts", ["name", "version"])
    op.drop_column("artifacts", "assistant")
