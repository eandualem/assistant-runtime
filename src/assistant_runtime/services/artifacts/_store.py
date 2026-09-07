"""Version stores: in memory for the lifetime of the process, or Postgres.

Internal module — only accessed through ArtifactService (interface.py).
Both stores key versions by ``(scope, name)``; the scope is the profile
name, so different assistants never share artifacts.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Protocol

from assistant_runtime.services.artifacts.models import ArtifactVersion


class ArtifactStore(Protocol):
    """What the service needs from a version store."""

    @property
    def durable(self) -> bool: ...

    async def get_active(self, scope: str, name: str) -> ArtifactVersion | None: ...

    async def get_all_active(self, scope: str) -> list[ArtifactVersion]: ...

    async def get_history(self, scope: str, name: str, limit: int) -> list[ArtifactVersion]: ...

    async def propose(
        self, scope: str, name: str, content: str, proposed_by: str
    ) -> ArtifactVersion: ...

    async def activate(self, scope: str, name: str, version: int) -> ArtifactVersion | None: ...

    async def delete(self, scope: str, name: str) -> int: ...


class InMemoryArtifactStore:
    """Versions kept in the process; lost on restart."""

    durable = False

    def __init__(self) -> None:
        self._versions: dict[tuple[str, str], list[ArtifactVersion]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._next_id = 1

    async def get_active(self, scope: str, name: str) -> ArtifactVersion | None:
        return next((v for v in self._versions.get((scope, name), []) if v.is_active), None)

    async def get_all_active(self, scope: str) -> list[ArtifactVersion]:
        active = [
            v
            for (item_scope, _), versions in self._versions.items()
            if item_scope == scope
            for v in versions
            if v.is_active
        ]
        return sorted(active, key=lambda v: v.name)

    async def get_history(self, scope: str, name: str, limit: int) -> list[ArtifactVersion]:
        versions = self._versions.get((scope, name), [])
        return sorted(versions, key=lambda v: v.version, reverse=True)[:limit]

    async def propose(
        self, scope: str, name: str, content: str, proposed_by: str
    ) -> ArtifactVersion:
        async with self._lock:
            versions = self._versions[(scope, name)]
            row = ArtifactVersion(
                name=name,
                content=content,
                version=max((v.version for v in versions), default=0) + 1,
                is_active=False,
                proposed_by=proposed_by,
                created_at=datetime.now(UTC),
                id=self._next_id,
            )
            self._next_id += 1
            versions.append(row)
            return row

    async def activate(self, scope: str, name: str, version: int) -> ArtifactVersion | None:
        async with self._lock:
            versions = self._versions.get((scope, name), [])
            if not any(v.version == version for v in versions):
                return None
            updated = [
                ArtifactVersion(**{**vars(v), "is_active": v.version == version}) for v in versions
            ]
            self._versions[(scope, name)] = updated
            return next(v for v in updated if v.version == version)

    async def delete(self, scope: str, name: str) -> int:
        async with self._lock:
            return len(self._versions.pop((scope, name), []))


class DatabaseArtifactStore:
    """Versions in the ``artifacts`` table, through ``ArtifactRepository``."""

    durable = True

    def __init__(self, database_service: Any) -> None:
        self._database = database_service

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
        )

    def _repository(self, session: Any) -> Any:
        from assistant_runtime.services.database.repositories import ArtifactRepository

        return ArtifactRepository(session)

    async def get_active(self, scope: str, name: str) -> ArtifactVersion | None:
        async with self._database.session_context() as session:
            row = await self._repository(session).get_active(scope, name)
        return self._to_version(row) if row is not None else None

    async def get_all_active(self, scope: str) -> list[ArtifactVersion]:
        async with self._database.session_context() as session:
            rows = await self._repository(session).get_all_active(scope)
        return [self._to_version(row) for row in rows]

    async def get_history(self, scope: str, name: str, limit: int) -> list[ArtifactVersion]:
        async with self._database.session_context() as session:
            rows = await self._repository(session).get_history(scope, name, limit=limit)
        return [self._to_version(row) for row in rows]

    async def propose(
        self, scope: str, name: str, content: str, proposed_by: str
    ) -> ArtifactVersion:
        async with self._database.session_context() as session:
            row = await self._repository(session).propose(scope, name, content, proposed_by)
            return self._to_version(row)

    async def activate(self, scope: str, name: str, version: int) -> ArtifactVersion | None:
        async with self._database.session_context() as session:
            row = await self._repository(session).approve(scope, name, version)
            return self._to_version(row) if row is not None else None

    async def delete(self, scope: str, name: str) -> int:
        async with self._database.session_context() as session:
            return await self._repository(session).delete_by_name(scope, name)
