"""Database repositories — query layer over ORM models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import case, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from lovely_assistant.services.database.models import (
    ArtifactORM,
    InboxItemORM,
    MessageORM,
    OAuthTokenORM,
    SessionORM,
    TraceORM,
    UserSettingsORM,
)

# Severity ordering for unsurfaced inbox queries (highest priority first).
_SEVERITY_ORDER = {"urgent": 0, "action_needed": 1, "info": 2}


class SessionRepository:
    """CRUD operations for sessions. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        session_id: str,
        title: str | None = None,
        expires_at: datetime | None = None,
    ) -> SessionORM:
        """Create a new session row."""
        values: dict[str, object] = {
            "id": session_id,
            "title": title,
            "turn_number": 0,
        }
        if expires_at is not None:
            values["expires_at"] = expires_at

        result = await self._session.execute(
            insert(SessionORM).values(**values).returning(SessionORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def get(self, session_id: str) -> SessionORM | None:
        """Get a non-expired session by ID, or None."""
        result = await self._session.execute(
            select(SessionORM).where(
                SessionORM.id == session_id,
                SessionORM.expires_at > func.now(),
            )
        )
        return result.scalar_one_or_none()

    async def list_all(self, limit: int = 50, offset: int = 0) -> list[SessionORM]:
        """List non-expired sessions ordered by most recently updated."""
        result = await self._session.execute(
            select(SessionORM)
            .where(SessionORM.expires_at > func.now())
            .order_by(SessionORM.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_latest_by_telegram_chat_id(self, chat_id: str) -> SessionORM | None:
        """Return the newest Telegram-bound session for a chat, or None."""
        result = await self._session.execute(
            select(SessionORM)
            .where(
                SessionORM.telegram_chat_id == chat_id,
                SessionORM.expires_at > func.now(),
            )
            .order_by(SessionORM.telegram_bound_at.desc(), SessionORM.updated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def update(self, session_id: str, **fields) -> None:
        """Update specific fields on a session. No-op if session doesn't exist."""
        if not fields:
            return
        stmt = (
            update(SessionORM)
            .where(
                SessionORM.id == session_id,
                SessionORM.expires_at > func.now(),
            )
            .values(**fields, updated_at=func.now())
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if a row was deleted."""
        result = await self._session.execute(delete(SessionORM).where(SessionORM.id == session_id))
        await self._session.flush()
        return (result.rowcount or 0) > 0

    async def exists(self, session_id: str) -> bool:
        """Check if a non-expired session exists."""
        result = await self._session.execute(
            select(SessionORM.id).where(
                SessionORM.id == session_id,
                SessionORM.expires_at > func.now(),
            )
        )
        return result.scalar_one_or_none() is not None

    async def cleanup_expired(self) -> int:
        """Delete all expired sessions. Returns count of deleted rows."""
        result = await self._session.execute(
            delete(SessionORM).where(SessionORM.expires_at <= func.now())
        )
        await self._session.flush()
        return result.rowcount or 0

    async def upsert(
        self,
        session_id: str,
        *,
        title: str | None = None,
        turn_number: int = 0,
        working_memory: dict | None = None,
        telegram_chat_id: str | None = None,
        telegram_bound_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        """Atomic INSERT ... ON CONFLICT DO UPDATE.

        Eliminates the race condition in check-then-insert patterns.
        """
        values: dict[str, object] = {
            "id": session_id,
            "title": title,
            "turn_number": turn_number,
            "working_memory": working_memory,
            "telegram_chat_id": telegram_chat_id,
            "telegram_bound_at": telegram_bound_at,
        }
        if expires_at is not None:
            values["expires_at"] = expires_at

        stmt = pg_insert(SessionORM).values(**values)

        update_fields: dict[str, object] = {
            "title": stmt.excluded.title,
            "turn_number": stmt.excluded.turn_number,
            "working_memory": stmt.excluded.working_memory,
            "telegram_chat_id": stmt.excluded.telegram_chat_id,
            "telegram_bound_at": stmt.excluded.telegram_bound_at,
            "updated_at": func.now(),
        }
        if expires_at is not None:
            update_fields["expires_at"] = stmt.excluded.expires_at

        stmt = stmt.on_conflict_do_update(index_elements=["id"], set_=update_fields)
        await self._session.execute(stmt)
        await self._session.flush()


class MessageRepository:
    """CRUD operations for conversation messages."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        message_id: str,
        session_id: str,
        parent_id: str | None,
        role: str,
        message_type: str,
        content: str,
        segments: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> MessageORM:
        result = await self._session.execute(
            insert(MessageORM)
            .values(
                id=message_id,
                session_id=session_id,
                parent_id=parent_id,
                role=role,
                message_type=message_type,
                content=content,
                segments=segments,
                usage=usage,
            )
            .returning(MessageORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def get(self, message_id: str) -> MessageORM | None:
        result = await self._session.execute(select(MessageORM).where(MessageORM.id == message_id))
        return result.scalar_one_or_none()

    async def list_by_session(self, session_id: str) -> list[MessageORM]:
        result = await self._session.execute(
            select(MessageORM)
            .where(MessageORM.session_id == session_id)
            .order_by(MessageORM.created_at.asc())
        )
        return list(result.scalars().all())

    async def count_by_session(self, session_id: str) -> int:
        result = await self._session.execute(
            select(func.count(MessageORM.id)).where(MessageORM.session_id == session_id)
        )
        return int(result.scalar_one() or 0)

    async def count_by_sessions(self, session_ids: list[str]) -> dict[str, int]:
        if not session_ids:
            return {}
        result = await self._session.execute(
            select(MessageORM.session_id, func.count(MessageORM.id))
            .where(MessageORM.session_id.in_(session_ids))
            .group_by(MessageORM.session_id)
        )
        return {session_id: int(count) for session_id, count in result.all()}

    async def update(
        self,
        message_id: str,
        *,
        content: str | None = None,
        segments: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        fields: dict[str, Any] = {}
        if content is not None:
            fields["content"] = content
        if segments is not None:
            fields["segments"] = segments
        if usage is not None:
            fields["usage"] = usage
        if not fields:
            return
        await self._session.execute(update(MessageORM).where(MessageORM.id == message_id).values(**fields))
        await self._session.flush()


class TraceRepository:
    """CRUD operations for debug traces. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        trace_id: str,
        session_id: str,
        events: list[dict[str, Any]],
        *,
        user_message: str | None = None,
        is_continuation: bool = False,
        duration_ms: float | None = None,
        screenshot: str | None = None,
    ) -> TraceORM:
        """Create a new trace row."""
        result = await self._session.execute(
            insert(TraceORM)
            .values(
                id=trace_id,
                session_id=session_id,
                events=events,
                user_message=user_message,
                is_continuation=is_continuation,
                duration_ms=duration_ms,
                screenshot=screenshot,
            )
            .returning(TraceORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def list_by_session(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> list[TraceORM]:
        """List traces for a session, ordered by creation time."""
        result = await self._session.execute(
            select(TraceORM)
            .where(TraceORM.session_id == session_id)
            .order_by(TraceORM.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())


class SettingsRepository:
    """CRUD operations for user settings. Uses flush() — caller owns commit."""

    SETTINGS_ID: str = "default"

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self) -> UserSettingsORM | None:
        """Get the settings row, or None."""
        result = await self._session.execute(
            select(UserSettingsORM).where(UserSettingsORM.id == self.SETTINGS_ID)
        )
        return result.scalar_one_or_none()

    async def save(self, state: dict[str, Any]) -> UserSettingsORM:
        """Upsert settings — create or update the single settings row."""
        if not state:
            row = await self.get()
            if row is not None:
                return row
            result = await self._session.execute(
                insert(UserSettingsORM).values(id=self.SETTINGS_ID).returning(UserSettingsORM)
            )
            await self._session.flush()
            return result.scalar_one()

        stmt = pg_insert(UserSettingsORM).values(id=self.SETTINGS_ID, **state)
        result = await self._session.execute(
            stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    **{key: getattr(stmt.excluded, key) for key in state},
                    "updated_at": func.now(),
                },
            ).returning(UserSettingsORM)
        )
        await self._session.flush()
        return result.scalar_one()


class InboxRepository:
    """CRUD operations for inbox items. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        from_agent: str,
        message: str,
        severity: str = "info",
        context: dict[str, Any] | None = None,
    ) -> InboxItemORM:
        """Create a new inbox item."""
        result = await self._session.execute(
            insert(InboxItemORM)
            .values(
                from_agent=from_agent,
                message=message,
                severity=severity,
                context=context,
            )
            .returning(InboxItemORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def list_unsurfaced(self, limit: int = 20) -> list[InboxItemORM]:
        """List unsurfaced items ordered by severity (urgent first) then created_at desc."""
        severity_sort = case(
            _SEVERITY_ORDER,
            value=InboxItemORM.severity,
            else_=99,
        )
        result = await self._session.execute(
            select(InboxItemORM)
            .where(InboxItemORM.surfaced.is_(False))
            .order_by(severity_sort, InboxItemORM.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_surfaced(self, item_id: str) -> InboxItemORM | None:
        """Mark an item as surfaced. Returns the item or None if not found."""
        result = await self._session.execute(
            update(InboxItemORM)
            .where(InboxItemORM.id == item_id)
            .values(surfaced=True)
            .returning(InboxItemORM)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            await self._session.flush()
        return row

    async def list_all(self, limit: int = 50, offset: int = 0) -> list[InboxItemORM]:
        """List all items ordered by created_at desc."""
        result = await self._session.execute(
            select(InboxItemORM)
            .order_by(InboxItemORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())


class ArtifactRepository:
    """CRUD operations for versioned prompt artifacts. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_active(self, name: str) -> ArtifactORM | None:
        """Get the active version of an artifact by name."""
        result = await self._session.execute(
            select(ArtifactORM).where(
                ArtifactORM.name == name,
                ArtifactORM.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def get_all_active(self) -> list[ArtifactORM]:
        """Get all active artifacts, ordered by name.

        Excludes artifacts with ``debug_`` prefix (test probes).
        """
        result = await self._session.execute(
            select(ArtifactORM)
            .where(
                ArtifactORM.is_active.is_(True),
                ~ArtifactORM.name.startswith("debug_"),
            )
            .order_by(ArtifactORM.name)
        )
        return list(result.scalars().all())

    async def delete_by_name(self, name: str) -> int:
        """Delete all versions of an artifact by name. Returns count deleted."""
        result = await self._session.execute(delete(ArtifactORM).where(ArtifactORM.name == name))
        await self._session.flush()
        return result.rowcount or 0

    async def get_history(self, name: str, limit: int = 20) -> list[ArtifactORM]:
        """Get version history for an artifact, newest first."""
        result = await self._session.execute(
            select(ArtifactORM)
            .where(ArtifactORM.name == name)
            .order_by(ArtifactORM.version.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def propose(self, name: str, content: str, proposed_by: str) -> ArtifactORM:
        """Create a new version of an artifact (inactive until approved).

        Auto-increments version based on MAX(version) for this name.
        """
        max_result = await self._session.execute(
            select(func.max(ArtifactORM.version)).where(ArtifactORM.name == name)
        )
        current_max = max_result.scalar_one_or_none() or 0
        next_version = current_max + 1

        result = await self._session.execute(
            insert(ArtifactORM)
            .values(
                name=name,
                content=content,
                version=next_version,
                is_active=False,
                proposed_by=proposed_by,
            )
            .returning(ArtifactORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def approve(self, name: str, version: int) -> ArtifactORM | None:
        """Approve a version: deactivate current active, activate target.

        Returns None if the target version doesn't exist.
        """
        result = await self._session.execute(
            select(ArtifactORM.id).where(
                ArtifactORM.name == name,
                ArtifactORM.version == version,
            )
        )
        target_id = result.scalar_one_or_none()
        if target_id is None:
            return None

        await self._session.execute(
            update(ArtifactORM)
            .where(ArtifactORM.name == name, ArtifactORM.is_active.is_(True))
            .values(is_active=False)
        )

        result = await self._session.execute(
            update(ArtifactORM)
            .where(ArtifactORM.id == target_id)
            .values(is_active=True)
            .returning(ArtifactORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def rollback(self, name: str, version: int) -> ArtifactORM | None:
        """Rollback to a previous version — same mechanics as approve."""
        return await self.approve(name, version)

    async def update_scratchpad(self, content: str, proposed_by: str = "jarvis") -> ArtifactORM:
        """Propose and auto-approve a scratchpad update in one step."""
        row = await self.propose("scratchpad", content, proposed_by)
        await self._session.execute(
            update(ArtifactORM)
            .where(
                ArtifactORM.name == "scratchpad",
                ArtifactORM.is_active.is_(True),
                ArtifactORM.id != row.id,
            )
            .values(is_active=False)
        )

        result = await self._session.execute(
            update(ArtifactORM)
            .where(ArtifactORM.id == row.id)
            .values(is_active=True)
            .returning(ArtifactORM)
        )
        await self._session.flush()
        return result.scalar_one()


class OAuthTokenRepository:
    """CRUD operations for OAuth tokens. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, provider: str) -> OAuthTokenORM | None:
        """Get a token by provider name."""
        result = await self._session.execute(
            select(OAuthTokenORM).where(OAuthTokenORM.provider == provider)
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        provider: str,
        *,
        encrypted_api_key: str | None = None,
        encrypted_refresh_token: str | None = None,
        encrypted_id_token: str | None = None,
        expires_at: float | None = None,
        email: str | None = None,
    ) -> None:
        """Insert or update an OAuth token for a provider."""
        values: dict[str, object] = {
            "provider": provider,
            "encrypted_api_key": encrypted_api_key,
            "encrypted_refresh_token": encrypted_refresh_token,
            "encrypted_id_token": encrypted_id_token,
            "expires_at": expires_at,
            "email": email,
        }

        stmt = pg_insert(OAuthTokenORM).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["provider"],
            set_={
                "encrypted_api_key": stmt.excluded.encrypted_api_key,
                "encrypted_refresh_token": stmt.excluded.encrypted_refresh_token,
                "encrypted_id_token": stmt.excluded.encrypted_id_token,
                "expires_at": stmt.excluded.expires_at,
                "email": stmt.excluded.email,
                "updated_at": func.now(),
            },
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def delete(self, provider: str) -> bool:
        """Delete a token by provider. Returns True if deleted."""
        result = await self._session.execute(
            delete(OAuthTokenORM).where(OAuthTokenORM.provider == provider)
        )
        await self._session.flush()
        return (result.rowcount or 0) > 0
