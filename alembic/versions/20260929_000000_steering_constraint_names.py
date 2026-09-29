"""Name the steering table's constraints as the models do.

The table was created as ``guidance`` under the naming convention, and 0019
renamed it to ``steering``. Its constraints kept their ``guidance`` names,
because 0019 looked for Postgres's default names instead. Each one is renamed
only while it still has its old name; renaming the primary key renames its
index too.

Revision ID: 0038
Revises: 0037
"""

import sqlalchemy as sa
from alembic import op

revision = "0038"
down_revision = "0037"
branch_labels = None
depends_on = None

_RENAMES = (
    ("pk_guidance", "pk_steering"),
    ("ck_guidance_ck_guidance_status_valid", "ck_steering_ck_steering_status_valid"),
    ("fk_guidance_session_id_sessions", "fk_steering_session_id_sessions"),
)


def _exists(conn, name: str) -> bool:
    return conn.execute(
        sa.text(
            """
            select exists (
                select 1
                from pg_constraint c
                join pg_class t on t.oid = c.conrelid
                join pg_namespace n on n.oid = t.relnamespace
                where n.nspname = current_schema()
                  and t.relname = 'steering'
                  and c.conname = :name
            )
            """
        ),
        {"name": name},
    ).scalar_one()


def _rename(pairs) -> None:
    conn = op.get_bind()
    for old, new in pairs:
        if _exists(conn, old) and not _exists(conn, new):
            conn.execute(sa.text(f'ALTER TABLE steering RENAME CONSTRAINT "{old}" TO "{new}"'))


def upgrade() -> None:
    _rename(_RENAMES)


def downgrade() -> None:
    _rename((new, old) for old, new in _RENAMES)
