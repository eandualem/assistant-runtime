"""Database repositories — query layer over ORM models."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import case, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.services.database.models import (
    ArtifactORM,
    InboxItemORM,
    MessageORM,
    OAuthTokenORM,
    SessionORM,
    SteeringORM,
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
        owner_id: str | None = None,
        profile: str | None = None,
        subject: str | None = None,
    ) -> SessionORM:
        """Create a new session row."""
        values: dict[str, object] = {
            "id": session_id,
            "title": title,
            "turn_number": 0,
            "owner_id": owner_id,
            "profile": profile,
            "subject": subject,
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

    async def list_all(
        self, limit: int = 50, offset: int = 0, *, owner_id: str | None = None
    ) -> list[SessionORM]:
        """List non-expired sessions, most recently updated first.

        With ``owner_id``, only that principal's sessions are listed.
        """
        stmt = select(SessionORM).where(SessionORM.expires_at > func.now())
        if owner_id is not None:
            stmt = stmt.where(SessionORM.owner_id == owner_id)
        result = await self._session.execute(
            stmt.order_by(SessionORM.updated_at.desc()).limit(limit).offset(offset)
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
        owner_id: str | None = None,
        pending_action: dict | None = None,
        profile: str | None = None,
        subject: str | None = None,
    ) -> None:
        """Atomic INSERT ... ON CONFLICT DO UPDATE.

        Eliminates the race condition in check-then-insert patterns. The
        owner is written on insert only: reassignment goes through
        ``update()``, so a stale state save can never restore an owner an
        administrator cleared or changed. The profile and subject binding
        is written on insert only too: it never changes.
        """
        values: dict[str, object] = {
            "id": session_id,
            "title": title,
            "turn_number": turn_number,
            "working_memory": working_memory,
            "telegram_chat_id": telegram_chat_id,
            "telegram_bound_at": telegram_bound_at,
            "owner_id": owner_id,
            "pending_action": pending_action,
            "profile": profile,
            "subject": subject,
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
            "pending_action": stmt.excluded.pending_action,
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
        await self._session.execute(
            update(MessageORM).where(MessageORM.id == message_id).values(**fields)
        )
        await self._session.flush()


class SteeringRepository:
    """CRUD operations for steering records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        steering_id: str,
        session_id: str,
        content: str,
        status: str,
        profile: str | None = None,
        delivered_at: datetime | None = None,
    ) -> SteeringORM:
        result = await self._session.execute(
            insert(SteeringORM)
            .values(
                id=steering_id,
                session_id=session_id,
                profile=profile,
                content=content,
                status=status,
                delivered_at=delivered_at,
            )
            .returning(SteeringORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def list_by_session(self, session_id: str) -> list[SteeringORM]:
        result = await self._session.execute(
            select(SteeringORM)
            .where(SteeringORM.session_id == session_id)
            .order_by(SteeringORM.created_at.asc(), SteeringORM.id.asc())
        )
        return list(result.scalars().all())

    async def mark_status(
        self,
        steering_ids: list[str],
        *,
        status: str,
        delivered_at: datetime | None,
    ) -> None:
        if not steering_ids:
            return

        await self._session.execute(
            update(SteeringORM)
            .where(SteeringORM.id.in_(steering_ids))
            .values(status=status, delivered_at=delivered_at)
        )
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
            )
            .returning(TraceORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def delete_older_than(self, cutoff: datetime) -> int:
        """Delete traces created before ``cutoff``. Returns the number of rows removed."""
        result = await self._session.execute(delete(TraceORM).where(TraceORM.created_at < cutoff))
        await self._session.flush()
        return result.rowcount or 0

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
    """CRUD operations for versioned prompt artifacts. Uses flush() — caller owns commit.

    Every method takes the profile ``scope`` first and a ``subject`` (empty
    for a profile-scoped artifact): rows of different assistants or subjects
    never mix, even when artifact names coincide.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _key(scope: str, subject: str, name: str) -> Any:
        return (
            (ArtifactORM.assistant == scope)
            & (ArtifactORM.subject == subject)
            & (ArtifactORM.name == name)
        )

    async def lock(self, scope: str, name: str, *, subject: str = "") -> None:
        """Serialize mutations until commit, including an artifact's first version."""
        key = json.dumps([scope, name] if not subject else [scope, subject, name])
        await self._session.execute(
            select(func.pg_advisory_xact_lock(func.hashtextextended(key, 0)))
        )

    async def get_active(self, scope: str, name: str, *, subject: str = "") -> ArtifactORM | None:
        """Get the active version of an artifact by name."""
        result = await self._session.execute(
            select(ArtifactORM).where(
                self._key(scope, subject, name), ArtifactORM.is_active.is_(True)
            )
        )
        return result.scalar_one_or_none()

    async def get_all_active(self, scope: str, *, subject: str = "") -> list[ArtifactORM]:
        """Get all active artifacts of the scope and subject, ordered by name."""
        result = await self._session.execute(
            select(ArtifactORM)
            .where(
                ArtifactORM.assistant == scope,
                ArtifactORM.subject == subject,
                ArtifactORM.is_active.is_(True),
            )
            .order_by(ArtifactORM.name)
        )
        return list(result.scalars().all())

    async def get_subjects(self, scope: str) -> list[str]:
        """The subjects with stored versions in the scope, sorted."""
        result = await self._session.execute(
            select(ArtifactORM.subject)
            .where(ArtifactORM.assistant == scope, ArtifactORM.subject != "")
            .distinct()
            .order_by(ArtifactORM.subject)
        )
        return list(result.scalars().all())

    async def delete_by_name(self, scope: str, name: str, *, subject: str = "") -> int:
        """Delete all versions of an artifact by name. Returns count deleted."""
        result = await self._session.execute(
            delete(ArtifactORM).where(self._key(scope, subject, name))
        )
        await self._session.flush()
        return result.rowcount or 0

    async def get_history(
        self, scope: str, name: str, limit: int = 20, *, subject: str = ""
    ) -> list[ArtifactORM]:
        """Get version history for an artifact, newest first."""
        result = await self._session.execute(
            select(ArtifactORM)
            .where(self._key(scope, subject, name))
            .order_by(ArtifactORM.version.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_version(
        self, scope: str, name: str, version: int, *, subject: str = ""
    ) -> ArtifactORM | None:
        """One stored version, or None."""
        result = await self._session.execute(
            select(ArtifactORM).where(
                self._key(scope, subject, name), ArtifactORM.version == version
            )
        )
        return result.scalar_one_or_none()

    async def get_by_status(
        self, scope: str, status: str, limit: int | None = None, before_id: int | None = None
    ) -> list[ArtifactORM]:
        """Versions of the scope (every subject) in ``status``, newest first; ``before_id`` pages back."""
        query = (
            select(ArtifactORM)
            .where(ArtifactORM.assistant == scope, ArtifactORM.status == status)
            .order_by(ArtifactORM.id.desc())
        )
        if before_id is not None:
            query = query.where(ArtifactORM.id < before_id)
        if limit is not None:
            query = query.limit(limit)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def propose(
        self,
        scope: str,
        name: str,
        content: str,
        proposed_by: str,
        *,
        subject: str = "",
        actor_kind: str = "host",
        rationale: str | None = None,
    ) -> ArtifactORM:
        """Create a pending version inside the caller's locked artifact transaction."""
        max_result = await self._session.execute(
            select(func.max(ArtifactORM.version)).where(self._key(scope, subject, name))
        )
        result = await self._session.execute(
            insert(ArtifactORM)
            .values(
                assistant=scope,
                subject=subject,
                name=name,
                content=content,
                version=(max_result.scalar_one_or_none() or 0) + 1,
                is_active=False,
                status="pending",
                proposed_by=proposed_by,
                actor_kind=actor_kind,
                rationale=rationale,
            )
            .returning(ArtifactORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def approve(
        self,
        scope: str,
        name: str,
        version: int,
        *,
        subject: str = "",
        decided_by: str | None = None,
    ) -> ArtifactORM | None:
        """Activate a version; the previously active one becomes superseded.

        ``decided_by`` is recorded when the target was a pending proposal.
        Returns None if the target version doesn't exist.
        """
        target = await self.get_version(scope, name, version, subject=subject)
        if target is None:
            return None
        target_id, was_pending = target.id, target.status == "pending"

        # Deactivate first: at most one active row per artifact (a unique index).
        await self._session.execute(
            update(ArtifactORM)
            .where(
                self._key(scope, subject, name),
                ArtifactORM.is_active.is_(True),
                ArtifactORM.id != target_id,
            )
            .values(is_active=False, status="superseded")
        )
        values: dict[str, Any] = {"is_active": True, "status": "active"}
        if decided_by is not None and was_pending:
            values.update(decided_by=decided_by, decided_at=func.now())
        result = await self._session.execute(
            update(ArtifactORM)
            .where(ArtifactORM.id == target_id)
            .values(**values)
            .returning(ArtifactORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def reject(
        self,
        scope: str,
        name: str,
        version: int,
        *,
        subject: str = "",
        decided_by: str,
        reason: str | None,
    ) -> ArtifactORM | None:
        """Mark a pending version rejected; None when it is not pending (or absent)."""
        result = await self._session.execute(
            update(ArtifactORM)
            .where(
                self._key(scope, subject, name),
                ArtifactORM.version == version,
                ArtifactORM.status == "pending",
            )
            .values(
                status="rejected",
                decided_by=decided_by,
                decided_at=func.now(),
                decision_reason=reason,
            )
            .returning(ArtifactORM)
        )
        await self._session.flush()
        return result.scalar_one_or_none()

    async def prune_superseded(self, scope: str, name: str, keep: int, *, subject: str = "") -> int:
        """Delete superseded versions beyond the newest ``keep``. Returns count deleted."""
        kept = (
            select(ArtifactORM.id)
            .where(self._key(scope, subject, name), ArtifactORM.status == "superseded")
            .order_by(ArtifactORM.version.desc())
            .limit(keep)
        )
        result = await self._session.execute(
            delete(ArtifactORM).where(
                self._key(scope, subject, name),
                ArtifactORM.status == "superseded",
                ArtifactORM.id.not_in(kept.scalar_subquery()),
            )
        )
        await self._session.flush()
        return result.rowcount or 0


class OAuthTokenRepository:
    """CRUD operations for OAuth tokens. Uses flush() — caller owns commit.

    ``kind`` is "api_key" for the provider-key store and "login" for a
    ChatGPT/Codex login; every operation touches only the row of that kind.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, provider: str, kind: str) -> OAuthTokenORM | None:
        """Get the provider's token of one kind."""
        result = await self._session.execute(
            select(OAuthTokenORM).where(
                OAuthTokenORM.provider == provider, OAuthTokenORM.kind == kind
            )
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        provider: str,
        kind: str,
        *,
        encrypted_api_key: str | None = None,
        encrypted_refresh_token: str | None = None,
        encrypted_id_token: str | None = None,
        expires_at: float | None = None,
        email: str | None = None,
    ) -> None:
        """Insert or update the provider's token of one kind."""
        values: dict[str, object] = {
            "provider": provider,
            "kind": kind,
            "encrypted_api_key": encrypted_api_key,
            "encrypted_refresh_token": encrypted_refresh_token,
            "encrypted_id_token": encrypted_id_token,
            "expires_at": expires_at,
            "email": email,
        }

        stmt = pg_insert(OAuthTokenORM).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["provider", "kind"],
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

    async def delete(self, provider: str, kind: str) -> bool:
        """Delete the provider's token of one kind. Returns True if deleted."""
        result = await self._session.execute(
            delete(OAuthTokenORM).where(
                OAuthTokenORM.provider == provider, OAuthTokenORM.kind == kind
            )
        )
        await self._session.flush()
        return (result.rowcount or 0) > 0
