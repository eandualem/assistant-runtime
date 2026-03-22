"""rename guidance storage to steering

Revision ID: 0019
Revises: 0018
Create Date: 2026-03-22 09:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table_exists(table_name: str) -> bool:
    conn = op.get_bind()
    return conn.execute(
        sa.text(
            """
            select exists (
                select 1
                from information_schema.tables
                where table_schema = current_schema()
                  and table_name = :table_name
            )
            """
        ),
        {"table_name": table_name},
    ).scalar_one()


def _constraint_exists(table_name: str, constraint_name: str) -> bool:
    conn = op.get_bind()
    return conn.execute(
        sa.text(
            """
            select exists (
                select 1
                from pg_constraint c
                join pg_class t on t.oid = c.conrelid
                join pg_namespace n on n.oid = t.relnamespace
                where n.nspname = current_schema()
                  and t.relname = :table_name
                  and c.conname = :constraint_name
            )
            """
        ),
        {"table_name": table_name, "constraint_name": constraint_name},
    ).scalar_one()


def _index_exists(index_name: str) -> bool:
    conn = op.get_bind()
    return conn.execute(
        sa.text(
            """
            select exists (
                select 1
                from pg_indexes
                where schemaname = current_schema()
                  and indexname = :index_name
            )
            """
        ),
        {"index_name": index_name},
    ).scalar_one()


def _rename_table_and_related_objects(
    *,
    old_table: str,
    new_table: str,
    old_check: str,
    new_check: str,
    old_created_index: str,
    new_created_index: str,
    old_status_index: str,
    new_status_index: str,
) -> None:
    conn = op.get_bind()

    if _table_exists(old_table) and not _table_exists(new_table):
        conn.execute(sa.text(f'ALTER TABLE "{old_table}" RENAME TO "{new_table}"'))

    if _constraint_exists(new_table, old_check) and not _constraint_exists(new_table, new_check):
        conn.execute(
            sa.text(f'ALTER TABLE "{new_table}" RENAME CONSTRAINT "{old_check}" TO "{new_check}"')
        )

    if _index_exists(old_created_index) and not _index_exists(new_created_index):
        conn.execute(sa.text(f'ALTER INDEX "{old_created_index}" RENAME TO "{new_created_index}"'))

    if _index_exists(old_status_index) and not _index_exists(new_status_index):
        conn.execute(sa.text(f'ALTER INDEX "{old_status_index}" RENAME TO "{new_status_index}"'))

    old_pkey = f"{old_table}_pkey"
    new_pkey = f"{new_table}_pkey"
    if _constraint_exists(new_table, old_pkey) and not _constraint_exists(new_table, new_pkey):
        conn.execute(
            sa.text(f'ALTER TABLE "{new_table}" RENAME CONSTRAINT "{old_pkey}" TO "{new_pkey}"')
        )


def upgrade() -> None:
    _rename_table_and_related_objects(
        old_table="guidance",
        new_table="steering",
        old_check="ck_guidance_status_valid",
        new_check="ck_steering_status_valid",
        old_created_index="ix_guidance_session_id_created_at",
        new_created_index="ix_steering_session_id_created_at",
        old_status_index="ix_guidance_session_id_status_created_at",
        new_status_index="ix_steering_session_id_status_created_at",
    )


def downgrade() -> None:
    _rename_table_and_related_objects(
        old_table="steering",
        new_table="guidance",
        old_check="ck_steering_status_valid",
        new_check="ck_guidance_status_valid",
        old_created_index="ix_steering_session_id_created_at",
        new_created_index="ix_guidance_session_id_created_at",
        old_status_index="ix_steering_session_id_status_created_at",
        new_status_index="ix_guidance_session_id_status_created_at",
    )
