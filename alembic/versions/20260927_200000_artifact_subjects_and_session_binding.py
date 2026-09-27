"""Scope artifact versions by subject; bind a session to a profile and subject.

A subject-scoped artifact keeps one version history per subject (for
example, per agent an assistant keeps track of); ``subject`` is empty for
every profile-scoped version, which is all existing rows. A session created
for a subject records its profile and subject so that later turns, host
continuations and delivered messages keep them.

Revision ID: 0029
Revises: 0028
"""

import sqlalchemy as sa
from alembic import op

revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "artifacts",
        sa.Column("subject", sa.String(128), nullable=False, server_default=""),
    )
    op.drop_constraint("uq_artifacts_assistant_name_version", "artifacts", type_="unique")
    op.create_unique_constraint(
        "uq_artifacts_assistant_subject_name_version",
        "artifacts",
        ["assistant", "subject", "name", "version"],
    )
    op.drop_index("uq_artifacts_one_active", table_name="artifacts")
    op.create_index(
        "uq_artifacts_one_active",
        "artifacts",
        ["assistant", "subject", "name"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )
    op.drop_index("ix_artifacts_assistant_name_is_active", table_name="artifacts")
    op.create_index(
        "ix_artifacts_assistant_subject_name_is_active",
        "artifacts",
        ["assistant", "subject", "name", "is_active"],
    )
    op.add_column("sessions", sa.Column("profile", sa.String(64), nullable=True))
    op.add_column("sessions", sa.Column("subject", sa.String(128), nullable=True))
    op.create_index("ix_sessions_profile_subject", "sessions", ["profile", "subject"])


def downgrade() -> None:
    op.drop_index("ix_sessions_profile_subject", table_name="sessions")
    op.drop_column("sessions", "subject")
    op.drop_column("sessions", "profile")
    # Subject histories have no place in the older schema.
    op.execute("DELETE FROM artifacts WHERE subject <> ''")
    op.drop_index("ix_artifacts_assistant_subject_name_is_active", table_name="artifacts")
    op.create_index(
        "ix_artifacts_assistant_name_is_active", "artifacts", ["assistant", "name", "is_active"]
    )
    op.drop_index("uq_artifacts_one_active", table_name="artifacts")
    op.create_index(
        "uq_artifacts_one_active",
        "artifacts",
        ["assistant", "name"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )
    op.drop_constraint("uq_artifacts_assistant_subject_name_version", "artifacts", type_="unique")
    op.create_unique_constraint(
        "uq_artifacts_assistant_name_version", "artifacts", ["assistant", "name", "version"]
    )
    op.drop_column("artifacts", "subject")
