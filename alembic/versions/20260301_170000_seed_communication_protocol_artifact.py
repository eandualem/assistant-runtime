"""seed communication_protocol artifact

Revision ID: 0013
Revises: 0012
Create Date: 2026-03-01 17:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# --- Seed content ---

_COMMUNICATION_PROTOCOL_CONTENT = (
    "## Communication Protocol\n\n"
    "Messages may arrive with envelope tags indicating their source:\n"
    "- `[via:telegram from:<operator>]` \u2014 the operator messaged via Telegram\n"
    "- `[via:tmux from:{agent}]` \u2014 An agent sent a direct message\n"
    "- `[via:room room:{room_id} from:{sender}]` \u2014 A message from a meeting room\n"
    "- `[via:backbone]` \u2014 System notification from the backbone\n"
    "- No tag \u2014 the operator is typing directly in the dashboard\n\n"
    "**Response Medium Rule:** Respond through the same channel you were reached on.\n"
    "- If `[via:telegram]`: After processing, use the `respond_telegram` tool to send your response\n"
    "- If `[via:tmux from:{agent}]`: After processing, use `send_agent_message` to reply to that agent\n"
    "- If `[via:room room:{room_id} from:{sender}]`: After processing, use "
    "`send_meeting_message(room_id=room_id, message=your_response)` to post your response "
    "back to the room transcript so all participants can see it\n"
    "- If no tag (dashboard): Respond normally in chat (default behavior)\n\n"
    "Always process the request fully first (use tools, think, etc.), then respond via the "
    "correct channel. The dashboard chat shows all activity regardless of channel \u2014 "
    "this is your workspace log."
)


def upgrade() -> None:
    artifacts_table = sa.table(
        "artifacts",
        sa.column("name", sa.String),
        sa.column("content", sa.Text),
        sa.column("version", sa.Integer),
        sa.column("is_active", sa.Boolean),
        sa.column("proposed_by", sa.String),
    )
    op.bulk_insert(
        artifacts_table,
        [
            {
                "name": "communication_protocol",
                "content": _COMMUNICATION_PROTOCOL_CONTENT,
                "version": 1,
                "is_active": True,
                "proposed_by": "system",
            },
        ],
    )


def downgrade() -> None:
    op.execute(
        sa.text("DELETE FROM artifacts WHERE name = 'communication_protocol'")
    )
