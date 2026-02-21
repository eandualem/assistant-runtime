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
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from lovely_assistant.services.database.base import Base


class SessionORM(Base):
    """Persistent session storage — maps to the 'sessions' table."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    turn_number: Mapped[int] = mapped_column(Integer, default=0)
    message_history: Mapped[list] = mapped_column(JSONB, default=list)
    working_memory: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    pending_tool_call: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now() + interval '24 hours'"),
        nullable=False,
    )


class TraceORM(Base):
    """Debug trace storage — one row per assistant request (stream)."""

    __tablename__ = "traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), index=True
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
