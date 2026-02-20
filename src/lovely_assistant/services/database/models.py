"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
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


class UserSettingsORM(Base):
    """Persisted user settings — single-row table with id='default'."""

    __tablename__ = "user_settings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    default_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    thinking_budget: Mapped[int | None] = mapped_column(Integer, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_turns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enable_working_memory: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
