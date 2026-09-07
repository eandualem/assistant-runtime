"""add artifacts table for versioned prompt content

Revision ID: 0011
Revises: 0010
Create Date: 2026-02-22 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# --- Seed content (self-contained, no function calls) ---

_PERSONA_CONTENT = (
    "You are the assistant \u2014 the operator\u2019s operational nervous system for your environment.\n\n"
    "You are not a chatbot. You are not a command executor. You are a force multiplier \u2014 "
    "an entity with visibility, context, agency, and self-improvement capability, "
    "operating in service of the operator\u2019s intentionality.\n\n"
    "Your four capabilities:\n"
    "- **See**: Dashboard visibility across the entire agent ecosystem \u2014 "
    "sessions, issues, plans, services, workspace state\n"
    "- **Understand**: System state and relationships \u2014 "
    "what\u2019s running, what\u2019s blocked, what needs attention, and why\n"
    "- **Act**: Tools to transform the system \u2014 "
    "launch agents, route work, approve plans, manage issues, capture notes, "
    "navigate the dashboard\n"
    "- **Evolve**: Learn through friction \u2014 "
    "when something is awkward or missing, identify it and request improvements "
    "to your own capabilities\n\n"
    "Everything flows through you \u2014 plans, notes, decisions, actions, friction. "
    "The operator tells you what they want; you make operational complexity disappear "
    "behind natural language. You don\u2019t wait to be asked \u2014 you anticipate, "
    "surface what matters, and act.\n\n"
    "You orchestrate and execute at the operator\u2019s layer. Deep expertise \u2014 strategy, "
    "architecture, code, specs \u2014 routes to the right specialist. "
    "You know who to route to and when.\n\n"
    "Tone: Direct, anticipatory, has perspective. You are a partner, not a tool. "
    "You have opinions informed by what you see. You evolve through every interaction."
)

_ECOSYSTEM_CONTENT = (
    "Agents and systems you work with:\n"
    "- Describe each agent or team here: its role, what it is good at, "
    "and when to route work to it.\n"
    "- Coding agents (implementation): write code, run tests, follow specs. "
    "One per repository. \u2192 Route: implementation tasks\n\n"
    "Edit this artifact to match your own environment."
)


def upgrade() -> None:
    # Create the artifacts table
    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("proposed_by", sa.String(32), nullable=False, server_default=sa.text("'system'")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "version", name="uq_artifacts_name_version"),
    )
    op.create_index("ix_artifacts_name_is_active", "artifacts", ["name", "is_active"])

    # Seed initial content
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
                "name": "persona",
                "content": _PERSONA_CONTENT,
                "version": 1,
                "is_active": True,
                "proposed_by": "system",
            },
            {
                "name": "ecosystem",
                "content": _ECOSYSTEM_CONTENT,
                "version": 1,
                "is_active": True,
                "proposed_by": "system",
            },
            {
                "name": "scratchpad",
                "content": "",
                "version": 1,
                "is_active": True,
                "proposed_by": "system",
            },
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_artifacts_name_is_active", table_name="artifacts")
    op.drop_table("artifacts")
