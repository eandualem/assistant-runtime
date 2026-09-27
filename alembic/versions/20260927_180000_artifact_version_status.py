"""Give artifact versions a lifecycle: pending, active, superseded, rejected.

A proposal used to be an inactive row, indistinguishable from a version
that had been active before. Existing rows are classified once: the active
row is ``active``; an inactive row newer than the active one (or any
inactive row where none is active) is ``pending``; every other row is
``superseded``. A version left behind by a rollback is newer than the
active one, so it is listed as pending again; approving or rejecting it
settles it.

Revision ID: 0028
Revises: 0027
"""

import sqlalchemy as sa
from alembic import op

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "artifacts",
        "proposed_by",
        type_=sa.String(128),
        existing_type=sa.String(32),
        existing_nullable=False,
    )
    op.add_column(
        "artifacts",
        sa.Column("status", sa.String(16), nullable=False, server_default="superseded"),
    )
    op.add_column(
        "artifacts",
        sa.Column("actor_kind", sa.String(16), nullable=False, server_default="host"),
    )
    op.add_column("artifacts", sa.Column("rationale", sa.Text(), nullable=True))
    op.add_column("artifacts", sa.Column("decided_by", sa.String(128), nullable=True))
    op.add_column(
        "artifacts", sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("artifacts", sa.Column("decision_reason", sa.Text(), nullable=True))

    # At most one active row per artifact: where several are, the newest wins.
    op.execute(
        "UPDATE artifacts AS a SET is_active = false WHERE a.is_active AND EXISTS ("
        "SELECT 1 FROM artifacts AS b WHERE b.assistant = a.assistant AND b.name = a.name "
        "AND b.is_active AND b.version > a.version)"
    )
    op.execute("UPDATE artifacts SET status = 'active' WHERE is_active")
    op.execute(
        "UPDATE artifacts AS a SET status = 'pending' WHERE NOT a.is_active AND a.version > "
        "coalesce((SELECT b.version FROM artifacts AS b WHERE b.assistant = a.assistant "
        "AND b.name = a.name AND b.is_active), 0)"
    )
    op.execute("UPDATE artifacts SET actor_kind = 'assistant' WHERE proposed_by = 'assistant'")
    op.execute("UPDATE artifacts SET actor_kind = 'seed' WHERE proposed_by = 'system'")
    op.alter_column("artifacts", "status", server_default=None)
    op.alter_column("artifacts", "actor_kind", server_default=None)

    op.create_check_constraint(
        "ck_artifacts_status_valid",
        "artifacts",
        "status IN ('active', 'pending', 'superseded', 'rejected')",
    )
    op.create_check_constraint(
        "ck_artifacts_status_is_active", "artifacts", "(status = 'active') = is_active"
    )
    op.create_check_constraint(
        "ck_artifacts_actor_kind_valid", "artifacts", "actor_kind IN ('assistant', 'host', 'seed')"
    )
    op.create_index(
        "uq_artifacts_one_active",
        "artifacts",
        ["assistant", "name"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )


def downgrade() -> None:
    op.drop_index("uq_artifacts_one_active", table_name="artifacts")
    op.drop_constraint("ck_artifacts_actor_kind_valid", "artifacts", type_="check")
    op.drop_constraint("ck_artifacts_status_is_active", "artifacts", type_="check")
    op.drop_constraint("ck_artifacts_status_valid", "artifacts", type_="check")
    for column in ("decision_reason", "decided_at", "decided_by", "rationale", "actor_kind"):
        op.drop_column("artifacts", column)
    op.drop_column("artifacts", "status")
    op.execute("UPDATE artifacts SET proposed_by = left(proposed_by, 32)")
    op.alter_column(
        "artifacts",
        "proposed_by",
        type_=sa.String(32),
        existing_type=sa.String(128),
        existing_nullable=False,
    )
