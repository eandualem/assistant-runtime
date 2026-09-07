"""add guidance table

Revision ID: 0018
Revises: 0017
Create Date: 2026-03-21 18:00:00.000000

"""

from __future__ import annotations

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0018"
down_revision: Union[str, None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_messages = sa.table(
    "messages",
    sa.column("id", sa.String(length=64)),
    sa.column("session_id", sa.String(length=64)),
    sa.column("parent_id", sa.String(length=64)),
    sa.column("message_type", sa.String(length=16)),
    sa.column("content", sa.Text()),
    sa.column("created_at", sa.DateTime(timezone=True)),
)

_guidance = sa.table(
    "guidance",
    sa.column("id", sa.String(length=64)),
    sa.column("session_id", sa.String(length=64)),
    sa.column("content", sa.Text()),
    sa.column("status", sa.String(length=16)),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("delivered_at", sa.DateTime(timezone=True)),
)

_MESSAGE_TYPE_CONSTRAINT_NAME = "ck_messages_ck_messages_message_type_valid"
_LEGACY_GUIDANCE_TYPES = {"guidance", "steering"}


def _find_messages_message_type_constraint() -> str | None:
    """Return the existing message_type check constraint name, if present."""
    conn = op.get_bind()
    return conn.execute(
        sa.text(
            """
            select c.conname
            from pg_constraint c
            join pg_class t on t.oid = c.conrelid
            where t.relname = 'messages'
              and c.contype = 'c'
              and c.conname like '%message_type_valid'
            order by c.conname
            limit 1
            """
        )
    ).scalar_one_or_none()


def _replace_messages_message_type_constraint(expression: str) -> None:
    """Replace the messages.message_type check constraint with a new expression."""
    conn = op.get_bind()
    existing_name = _find_messages_message_type_constraint()
    if existing_name is not None:
        conn.execute(sa.text(f'ALTER TABLE messages DROP CONSTRAINT "{existing_name}"'))
    conn.execute(
        sa.text(
            f'ALTER TABLE messages ADD CONSTRAINT "{_MESSAGE_TYPE_CONSTRAINT_NAME}" '
            f"CHECK ({expression})"
        )
    )


def _guidance_parent_targets(rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, str | None]]:
    """Map each legacy guidance row to the nearest non-guidance ancestor."""
    guidance_ids = {row["id"] for row in rows if row["message_type"] in _LEGACY_GUIDANCE_TYPES}
    parent_by_id = {row["id"]: row["parent_id"] for row in rows}
    targets: dict[str, str | None] = {}

    for guidance_id in guidance_ids:
        target = parent_by_id.get(guidance_id)
        seen = {guidance_id}
        while target in guidance_ids and target not in seen:
            seen.add(target)
            target = parent_by_id.get(target)
        targets[guidance_id] = target

    return guidance_ids, targets


def _migrate_legacy_guidance_rows() -> None:
    """Move old tree-style guidance/steering rows into the new guidance table."""
    conn = op.get_bind()
    rows = [
        dict(row)
        for row in conn.execute(
            sa.select(
                _messages.c.id,
                _messages.c.session_id,
                _messages.c.parent_id,
                _messages.c.message_type,
                _messages.c.content,
                _messages.c.created_at,
            )
        ).mappings()
    ]

    legacy_guidance = [row for row in rows if row["message_type"] in _LEGACY_GUIDANCE_TYPES]
    if not legacy_guidance:
        return

    conn.execute(
        sa.insert(_guidance),
        [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "content": row["content"],
                "status": "delivered",
                "created_at": row["created_at"],
                "delivered_at": row["created_at"],
            }
            for row in legacy_guidance
        ],
    )

    guidance_ids, parent_targets = _guidance_parent_targets(rows)
    reparent_rows = [
        {
            "message_id": row["id"],
            "parent_id": parent_targets[row["parent_id"]],
        }
        for row in rows
        if row["message_type"] not in _LEGACY_GUIDANCE_TYPES and row["parent_id"] in guidance_ids
    ]

    if reparent_rows:
        conn.execute(
            sa.text("UPDATE messages SET parent_id = :parent_id WHERE id = :message_id"),
            reparent_rows,
        )

    conn.execute(sa.delete(_messages).where(_messages.c.id.in_(sorted(guidance_ids))))


def upgrade() -> None:
    op.create_table(
        "guidance",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'delivered', 'promoted')",
            name="ck_guidance_status_valid",
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_guidance_session_id_created_at",
        "guidance",
        ["session_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_guidance_session_id_status_created_at",
        "guidance",
        ["session_id", "status", "created_at"],
        unique=False,
    )

    _migrate_legacy_guidance_rows()

    _replace_messages_message_type_constraint("message_type = 'standard'")


def downgrade() -> None:
    _replace_messages_message_type_constraint("message_type IN ('standard', 'guidance')")
    op.drop_index("ix_guidance_session_id_status_created_at", table_name="guidance")
    op.drop_index("ix_guidance_session_id_created_at", table_name="guidance")
    op.drop_table("guidance")
