"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
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

from lovely_assistant.services.database.base import Base


class SessionORM(Base):
    """Persistent session storage — maps to the 'sessions' table."""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_telegram_chat_id_bound_at", "telegram_chat_id", "telegram_bound_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    turn_number: Mapped[int] = mapped_column(Integer, default=0)
    working_memory: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    telegram_chat_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    telegram_bound_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now() + interval '24 hours'"),
        nullable=False,
    )


class MessageORM(Base):
    """Tree-structured conversation messages."""

    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role_valid"),
        CheckConstraint(
            "message_type IN ('standard', 'guidance')",
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TraceORM(Base):
    """Debug trace storage — one row per assistant request (stream)."""

    __tablename__ = "traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    events: Mapped[list] = mapped_column(JSONB, default=list)
    user_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_continuation: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    screenshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class InboxItemORM(Base):
    """Inbox items — agents push messages for Jarvis to surface contextually."""

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
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ArtifactORM(Base):
    """Versioned prompt artifacts — mutable data backing system prompt fragments."""

    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_artifacts_name_version"),
        Index("ix_artifacts_name_is_active", "name", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    proposed_by: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'system'")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OAuthTokenORM(Base):
    """OAuth token storage — encrypted tokens for provider subscriptions."""

    __tablename__ = "oauth_tokens"

    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    encrypted_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_id_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
