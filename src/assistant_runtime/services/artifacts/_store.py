"""Version stores: in memory for the lifetime of the process, or Postgres.

Internal module — only accessed through ArtifactService (interface.py).
Both stores key versions by ``(scope, subject, name)``: the scope is the
profile name, so different assistants never share artifacts, and the
subject is empty for a profile-scoped artifact and names the subject (for
example, an agent the assistant keeps track of) for a subject-scoped one.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Protocol

from assistant_runtime.services.artifacts.models import ArtifactVersion


class ArtifactStore(Protocol):
    """What the service needs from a version store."""

    def transaction(
        self, scope: str, name: str, *, subject: str = ""
    ) -> AbstractAsyncContextManager[ArtifactStore]: ...

    @property
    def durable(self) -> bool: ...

    async def get_active(
        self, scope: str, name: str, *, subject: str = ""
    ) -> ArtifactVersion | None: ...

    async def get_all_active(self, scope: str, *, subject: str = "") -> list[ArtifactVersion]: ...

    async def get_history(
        self, scope: str, name: str, limit: int, *, subject: str = ""
    ) -> list[ArtifactVersion]: ...

    async def get_version(
        self, scope: str, name: str, version: int, *, subject: str = ""
    ) -> ArtifactVersion | None: ...

    async def get_by_status(
        self, scope: str, status: str, limit: int | None, before_id: int | None = None
    ) -> list[ArtifactVersion]: ...

    async def get_subjects(self, scope: str) -> list[str]: ...

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
    ) -> ArtifactVersion: ...

    async def activate(
        self,
        scope: str,
        name: str,
        version: int,
        *,
        subject: str = "",
        decided_by: str | None = None,
    ) -> ArtifactVersion | None: ...

    async def reject(
        self,
        scope: str,
        name: str,
        version: int,
        *,
        subject: str = "",
        decided_by: str,
        reason: str | None,
    ) -> ArtifactVersion | None: ...

    async def prune_superseded(
        self, scope: str, name: str, keep: int, *, subject: str = ""
    ) -> int: ...

    async def delete(self, scope: str, name: str, *, subject: str = "") -> int: ...


class InMemoryArtifactStore:
    """Versions kept in the process; lost on restart."""

    durable = False

    def __init__(self) -> None:
        self._versions: dict[tuple[str, str, str], list[ArtifactVersion]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._next_id = 1

    @asynccontextmanager
    async def transaction(
        self, scope: str, name: str, *, subject: str = ""
    ) -> AsyncIterator[ArtifactStore]:
        async with self._lock:
            yield self

    async def get_active(
        self, scope: str, name: str, *, subject: str = ""
    ) -> ArtifactVersion | None:
        return next(
            (v for v in self._versions.get((scope, subject, name), []) if v.is_active), None
        )

    async def get_all_active(self, scope: str, *, subject: str = "") -> list[ArtifactVersion]:
        active = [
            v
            for (item_scope, item_subject, _), versions in self._versions.items()
            if item_scope == scope and item_subject == subject
            for v in versions
            if v.is_active
        ]
        return sorted(active, key=lambda v: v.name)

    async def get_history(
        self, scope: str, name: str, limit: int, *, subject: str = ""
    ) -> list[ArtifactVersion]:
        versions = self._versions.get((scope, subject, name), [])
        return sorted(versions, key=lambda v: v.version, reverse=True)[:limit]

    async def get_version(
        self, scope: str, name: str, version: int, *, subject: str = ""
    ) -> ArtifactVersion | None:
        return next(
            (v for v in self._versions.get((scope, subject, name), []) if v.version == version),
            None,
        )

    async def get_by_status(
        self, scope: str, status: str, limit: int | None, before_id: int | None = None
    ) -> list[ArtifactVersion]:
        rows = [
            v
            for (item_scope, _, _), versions in self._versions.items()
            if item_scope == scope
            for v in versions
            if v.status == status and (before_id is None or (v.id or 0) < before_id)
        ]
        return sorted(rows, key=lambda v: v.id or 0, reverse=True)[:limit]  # None: all

    async def get_subjects(self, scope: str) -> list[str]:
        return sorted(
            {
                subject
                for (item_scope, subject, _), versions in self._versions.items()
                if item_scope == scope and subject and versions
            }
        )

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
    ) -> ArtifactVersion:
        versions = self._versions[(scope, subject, name)]
        row = ArtifactVersion(
            name=name,
            content=content,
            version=max((v.version for v in versions), default=0) + 1,
            is_active=False,
            proposed_by=proposed_by,
            created_at=datetime.now(UTC),
            id=self._next_id,
            status="pending",
            actor_kind=actor_kind,
            rationale=rationale,
            subject=subject,
        )
        self._next_id += 1
        versions.append(row)
        return row

    async def activate(
        self,
        scope: str,
        name: str,
        version: int,
        *,
        subject: str = "",
        decided_by: str | None = None,
    ) -> ArtifactVersion | None:
        versions = self._versions.get((scope, subject, name), [])
        target = next((v for v in versions if v.version == version), None)
        if target is None:
            return None
        decision = (
            {"decided_by": decided_by, "decided_at": datetime.now(UTC)}
            if decided_by is not None and target.status == "pending"
            else {}
        )
        updated = [
            replace(v, is_active=True, status="active", **decision)
            if v.version == version
            else replace(v, is_active=False, status="superseded")
            if v.is_active
            else v
            for v in versions
        ]
        self._versions[(scope, subject, name)] = updated
        return next(v for v in updated if v.version == version)

    async def reject(
        self,
        scope: str,
        name: str,
        version: int,
        *,
        subject: str = "",
        decided_by: str,
        reason: str | None,
    ) -> ArtifactVersion | None:
        versions = self._versions.get((scope, subject, name), [])
        for index, v in enumerate(versions):
            if v.version == version and v.status == "pending":
                versions[index] = replace(
                    v,
                    status="rejected",
                    decided_by=decided_by,
                    decided_at=datetime.now(UTC),
                    decision_reason=reason,
                )
                return versions[index]
        return None

    async def prune_superseded(self, scope: str, name: str, keep: int, *, subject: str = "") -> int:
        versions = self._versions.get((scope, subject, name), [])
        superseded = sorted(
            (v for v in versions if v.status == "superseded"),
            key=lambda v: v.version,
            reverse=True,
        )
        dropped = {v.version for v in superseded[keep:]}
        self._versions[(scope, subject, name)] = [v for v in versions if v.version not in dropped]
        return len(dropped)

    async def delete(self, scope: str, name: str, *, subject: str = "") -> int:
        return len(self._versions.pop((scope, subject, name), []))


class DatabaseArtifactStore:
    """Versions in the ``artifacts`` table, through ``ArtifactRepository``."""

    durable = True

    def __init__(self, database_service: Any, *, session: Any = None) -> None:
        self._database = database_service
        self._transaction_session = session

    @asynccontextmanager
    async def _session_context(self) -> AsyncIterator[Any]:
        if self._transaction_session is not None:
            yield self._transaction_session
        else:
            async with self._database.session_context() as session:
                yield session

    @asynccontextmanager
    async def transaction(
        self, scope: str, name: str, *, subject: str = ""
    ) -> AsyncIterator[ArtifactStore]:
        async with self._database.session_context() as session:
            await self._repository(session).lock(scope, name, subject=subject)
            yield DatabaseArtifactStore(self._database, session=session)

    @staticmethod
    def _to_version(row: Any) -> ArtifactVersion:
        return ArtifactVersion(
            name=row.name,
            content=row.content,
            version=row.version,
            is_active=bool(row.is_active),
            proposed_by=row.proposed_by,
            created_at=row.created_at,
            id=row.id,
            status=row.status,
            actor_kind=row.actor_kind,
            rationale=row.rationale,
            decided_by=row.decided_by,
            decided_at=row.decided_at,
            decision_reason=row.decision_reason,
            subject=row.subject,
        )

    def _repository(self, session: Any) -> Any:
        from assistant_runtime.services.database.repositories import ArtifactRepository

        return ArtifactRepository(session)

    async def get_active(
        self, scope: str, name: str, *, subject: str = ""
    ) -> ArtifactVersion | None:
        async with self._session_context() as session:
            row = await self._repository(session).get_active(scope, name, subject=subject)
        return self._to_version(row) if row is not None else None

    async def get_all_active(self, scope: str, *, subject: str = "") -> list[ArtifactVersion]:
        async with self._session_context() as session:
            rows = await self._repository(session).get_all_active(scope, subject=subject)
        return [self._to_version(row) for row in rows]

    async def get_history(
        self, scope: str, name: str, limit: int, *, subject: str = ""
    ) -> list[ArtifactVersion]:
        async with self._session_context() as session:
            rows = await self._repository(session).get_history(
                scope, name, limit=limit, subject=subject
            )
        return [self._to_version(row) for row in rows]

    async def get_version(
        self, scope: str, name: str, version: int, *, subject: str = ""
    ) -> ArtifactVersion | None:
        async with self._session_context() as session:
            row = await self._repository(session).get_version(scope, name, version, subject=subject)
        return self._to_version(row) if row is not None else None

    async def get_by_status(
        self, scope: str, status: str, limit: int | None, before_id: int | None = None
    ) -> list[ArtifactVersion]:
        async with self._session_context() as session:
            rows = await self._repository(session).get_by_status(
                scope, status, limit=limit, before_id=before_id
            )
        return [self._to_version(row) for row in rows]

    async def get_subjects(self, scope: str) -> list[str]:
        async with self._session_context() as session:
            return await self._repository(session).get_subjects(scope)

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
    ) -> ArtifactVersion:
        async with self._session_context() as session:
            row = await self._repository(session).propose(
                scope,
                name,
                content,
                proposed_by,
                subject=subject,
                actor_kind=actor_kind,
                rationale=rationale,
            )
            return self._to_version(row)

    async def activate(
        self,
        scope: str,
        name: str,
        version: int,
        *,
        subject: str = "",
        decided_by: str | None = None,
    ) -> ArtifactVersion | None:
        async with self._session_context() as session:
            row = await self._repository(session).approve(
                scope, name, version, subject=subject, decided_by=decided_by
            )
            return self._to_version(row) if row is not None else None

    async def reject(
        self,
        scope: str,
        name: str,
        version: int,
        *,
        subject: str = "",
        decided_by: str,
        reason: str | None,
    ) -> ArtifactVersion | None:
        async with self._session_context() as session:
            row = await self._repository(session).reject(
                scope, name, version, subject=subject, decided_by=decided_by, reason=reason
            )
            return self._to_version(row) if row is not None else None

    async def prune_superseded(self, scope: str, name: str, keep: int, *, subject: str = "") -> int:
        async with self._session_context() as session:
            return await self._repository(session).prune_superseded(
                scope, name, keep, subject=subject
            )

    async def delete(self, scope: str, name: str, *, subject: str = "") -> int:
        async with self._session_context() as session:
            return await self._repository(session).delete_by_name(scope, name, subject=subject)
