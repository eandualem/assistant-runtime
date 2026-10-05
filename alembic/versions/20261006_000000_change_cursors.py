"""Record when rows last changed, for change cursors.

Messages, tasks, agent messages, actions, events and voice calls change after
they are created, and a host reconciling them needs to read what changed
since its last pass. Each gets ``updated_at``, set by every write, and an
``(updated_at, id)`` index (messages also per session). Existing rows take
the latest time they already record. Voice calls had no time columns: they
also get ``created_at``, both taken from the snapshot's own times (Unix
seconds) where it has them.

Revision ID: 0039
Revises: 0038
"""

import sqlalchemy as sa
from alembic import op

revision = "0039"
down_revision = "0038"
branch_labels = None
depends_on = None

# The latest time each row already records; GREATEST ignores NULLs.
_BACKFILL = {
    "messages": "created_at",
    "tasks": "GREATEST(created_at, started_at, finished_at)",
    "agent_messages": "GREATEST(created_at, started_at, finished_at)",
    "actions": "GREATEST(created_at, decided_at, revised_at)",
    "events": "GREATEST(created_at, delivered_at, heard_at)",
}


def _snapshot_time(field: str) -> str:
    return (
        f"CASE WHEN jsonb_typeof(snapshot -> '{field}') = 'number' "
        f"THEN to_timestamp((snapshot ->> '{field}')::double precision) END"
    )


def _timestamp(name: str) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


def upgrade() -> None:
    for table, latest in _BACKFILL.items():
        op.add_column(table, _timestamp("updated_at"))
        op.execute(f"UPDATE {table} SET updated_at = {latest}")
        op.create_index(f"ix_{table}_updated_at_id", table, ["updated_at", "id"])
    op.create_index(
        "ix_messages_session_id_updated_at_id", "messages", ["session_id", "updated_at", "id"]
    )

    op.add_column("voice_calls", _timestamp("created_at"))
    op.add_column("voice_calls", _timestamp("updated_at"))
    op.execute(
        f"UPDATE voice_calls SET created_at = COALESCE({_snapshot_time('created_at')}, now())"
    )
    op.execute(
        "UPDATE voice_calls SET updated_at = "
        f"GREATEST(created_at, {_snapshot_time('last_activity_at')})"
    )
    op.create_index("ix_voice_calls_updated_at_id", "voice_calls", ["updated_at", "id"])


def downgrade() -> None:
    op.drop_index("ix_voice_calls_updated_at_id", table_name="voice_calls")
    op.drop_column("voice_calls", "updated_at")
    op.drop_column("voice_calls", "created_at")
    op.drop_index("ix_messages_session_id_updated_at_id", table_name="messages")
    for table in reversed(_BACKFILL):
        op.drop_index(f"ix_{table}_updated_at_id", table_name=table)
        op.drop_column(table, "updated_at")
