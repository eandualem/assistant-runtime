"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from assistant_runtime.services.database.base import Base


class SessionORM(Base):
    """Persistent session storage — maps to the 'sessions' table."""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_telegram_chat_id_bound_at", "telegram_chat_id", "telegram_bound_at"),
        Index("ix_sessions_owner_id", "owner_id"),
        Index("ix_sessions_profile_subject", "profile", "subject"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    """The principal that created the session; NULL for rows that predate ownership."""
    profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    """Set together when the session was created for a subject; NULL otherwise."""
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    turn_number: Mapped[int] = mapped_column(Integer, default=0)
    working_memory: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    pending_action: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    """The one host-tool call waiting for its continuation; NULL when none is pending."""
    telegram_chat_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    telegram_bound_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now() + interval '24 hours'"),
        nullable=False,
    )


class VoiceCallORM(Base):
    """Checkpointed voice transcript/usage; delegated turns remain normal messages."""

    __tablename__ = "voice_calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)


class MessageORM(Base):
    """Tree-structured conversation messages."""

    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'host')", name="ck_messages_role_valid"),
        CheckConstraint(
            "message_type = 'standard'",
            name="ck_messages_message_type_valid",
        ),
        Index("ix_messages_session_id_created_at", "session_id", "created_at"),
        Index("ix_messages_parent_id", "parent_id"),
        Index(
            "uq_messages_single_root_per_session",
            "session_id",
            unique=True,
            postgresql_where=text("parent_id IS NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("messages.id", ondelete="CASCADE"), nullable=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    message_type: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'standard'")
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    segments: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    usage: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    prompt: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    """What the assistant message was produced with (see ``app/assistant/prompt_record``)."""
    model_messages: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    """The row's part of the conversation as Pydantic AI messages, replayed to the model."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SteeringORM(Base):
    """Out-of-band steering records — separate from the message tree."""

    __tablename__ = "steering"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'delivered', 'promoted')",
            name="ck_steering_status_valid",
        ),
        Index("ix_steering_session_id_created_at", "session_id", "created_at"),
        Index("ix_steering_session_id_status_created_at", "session_id", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    attachments: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    """Reference attachments sent with the steering message (``host_context.Attachment``)."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TraceORM(Base):
    """Debug trace storage — one row per assistant request (stream)."""

    __tablename__ = "traces"
    # Retention deletes by age; without this the startup cleanup scans the table.
    __table_args__ = (Index("ix_traces_created_at", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    events: Mapped[list] = mapped_column(JSONB, default=list)
    user_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_continuation: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    # The table still has a nullable ``screenshot`` column; migration 0024 cleared
    # it and nothing maps it, so a screenshot can no longer end up in a trace.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class InboxItemORM(Base):
    """Inbox items — agents push messages for the assistant to surface contextually."""

    __tablename__ = "inbox_items"
    __table_args__ = (
        CheckConstraint(
            "severity IN ('info', 'action_needed', 'urgent')",
            name="ck_inbox_items_severity_valid",
        ),
        Index(
            "ix_inbox_items_surfaced_severity_created_at",
            "surfaced",
            "severity",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, server_default=func.gen_random_uuid().cast(String)
    )
    from_agent: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'info'"))
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    surfaced: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class UserSettingsORM(Base):
    """Persisted user settings — single-row table with id='default'."""

    __tablename__ = "user_settings"
    __table_args__ = (CheckConstraint("id = 'default'", name="ck_user_settings_singleton_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    default_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    thinking_budget: Mapped[int | None] = mapped_column(Integer, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_turns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enable_working_memory: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    summarization_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    working_memory_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_image_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_video_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    subagent_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    subagent_thinking_budget: Mapped[int | None] = mapped_column(Integer, nullable=True)
    codex_service_tier: Mapped[str | None] = mapped_column(String(16), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ArtifactORM(Base):
    """Versioned prompt artifacts — mutable data backing system prompt fragments."""

    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint(
            "assistant",
            "subject",
            "name",
            "version",
            name="uq_artifacts_assistant_subject_name_version",
        ),
        Index(
            "ix_artifacts_assistant_subject_name_is_active",
            "assistant",
            "subject",
            "name",
            "is_active",
        ),
        Index(
            "uq_artifacts_one_active",
            "assistant",
            "subject",
            "name",
            unique=True,
            postgresql_where=text("is_active"),
        ),
        CheckConstraint(
            "status IN ('active', 'pending', 'superseded', 'rejected')",
            name="ck_artifacts_status_valid",
        ),
        CheckConstraint("(status = 'active') = is_active", name="ck_artifacts_status_is_active"),
        CheckConstraint(
            "actor_kind IN ('assistant', 'host', 'seed')", name="ck_artifacts_actor_kind_valid"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    assistant: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default=text("'technical_operator'")
    )
    subject: Mapped[str] = mapped_column(String(128), nullable=False, server_default=text("''"))
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    proposed_by: Mapped[str] = mapped_column(
        String(128), nullable=False, server_default=text("'system'")
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    actor_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PromptSnapshotORM(Base):
    """The stable part of a system prompt, stored once under its SHA-256."""

    __tablename__ = "prompt_snapshots"

    hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TaskORM(Base):
    """A background task: a turn run in its own session outside any conversation turn."""

    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'done', 'failed', 'cancelled', 'interrupted')",
            name="ck_tasks_status_valid",
        ),
        Index("ix_tasks_parent_session_id", "parent_session_id"),
        Index("ix_tasks_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    usage: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AgentORM(Base):
    """A persistent agent: one continuing session its messages run in, in order."""

    __tablename__ = "agents"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'stopped')", name="ck_agents_status_valid"),
        # One active agent per session; per profile and subject below the class.
        Index(
            "uq_agents_active_session",
            "session_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # The tunables its turns send as a request's config, validated by TunableOverrides.
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


Index(
    "uq_agents_active_identity",
    func.coalesce(AgentORM.profile, ""),
    func.coalesce(AgentORM.subject, ""),
    unique=True,
    postgresql_where=text("status = 'active'"),
)


class AgentMessageORM(Base):
    """A message to a persistent agent, and how the turn that answered it ended."""

    __tablename__ = "agent_messages"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'done', 'failed', 'cancelled', 'interrupted')",
            name="ck_agent_messages_status_valid",
        ),
        Index("ix_agent_messages_agent_id_created_at", "agent_id", "created_at"),
        Index("ix_agent_messages_parent_session_id", "parent_session_id"),
        Index("ix_agent_messages_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(36), ForeignKey("agents.id"), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parent_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    usage: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EventORM(Base):
    """An event from another system (inbound) or a notice for the owner (outbound)."""

    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("source", "event_id", name="uq_events_source_event_id"),
        CheckConstraint("direction IN ('inbound', 'outbound')", name="ck_events_direction"),
        CheckConstraint("severity IN ('info', 'warning', 'critical')", name="ck_events_severity"),
        CheckConstraint(
            "status IN ('received', 'delivered', 'pending', 'heard')", name="ck_events_status"
        ),
        Index("ix_events_agent", "agent"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(200), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    agent: Mapped[str | None] = mapped_column(String(128), nullable=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    target_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    history: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    delivery: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heard_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ActionORM(Base):
    """An action a host proposed and carried out, with its status history and results."""

    __tablename__ = "actions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('proposed', 'scheduled', 'sending', 'sent', 'failed', 'discarded', "
            "'undone')",
            name="ck_actions_status",
        ),
        Index("ix_actions_status", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    history: Mapped[list] = mapped_column(JSONB, nullable=False)
    confirmed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    results: Mapped[dict] = mapped_column(JSONB, nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    proposed_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    revised_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ActionConfirmationORM(Base):
    """The owner's confirmation of one action for one recipient, written before it is sent."""

    __tablename__ = "action_confirmations"
    __table_args__ = (
        CheckConstraint("kind IN ('message', 'steer')", name="ck_confirmations_kind"),
        CheckConstraint("source IN ('button', 'typed', 'voice')", name="ck_confirmations_source"),
        CheckConstraint(
            "status IN ('confirmed', 'sent', 'failed')", name="ck_confirmations_status"
        ),
        CheckConstraint(
            "reconciled IS NULL OR reconciled IN ('matched', 'altered', 'missing', 'undelivered')",
            name="ck_confirmations_reconciled",
        ),
        Index("ix_action_confirmations_action_id", "action_id"),
    )

    seq: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    action_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("actions.id", ondelete="CASCADE"), nullable=False
    )
    recipient: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    key_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sender: Mapped[str | None] = mapped_column(String(200), nullable=True)
    audience: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[dict | list | str | None] = mapped_column(JSONB, nullable=True)
    reconciled: Mapped[str | None] = mapped_column(String(16), nullable=True)


class HostStateORM(Base):
    """A small versioned value a host keeps under a namespace and key."""

    __tablename__ = "host_state"

    namespace: Mapped[str] = mapped_column(String(64), primary_key=True)
    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    """A deleted key stays as a tombstone, so its version never repeats."""
    updated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class OAuthTokenORM(Base):
    """OAuth token storage — encrypted tokens for provider subscriptions."""

    __tablename__ = "oauth_tokens"

    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    # "api_key" (the provider-key store) or "login" (a ChatGPT/Codex login): each
    # owner has its own row, so neither overwrites or deletes the other's data.
    kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    # A login's access token for kind "login".
    encrypted_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_id_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
