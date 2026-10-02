"""Artifacts queries; the caller owns the transaction."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.services.database.models import (
    ArtifactORM,
)


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
