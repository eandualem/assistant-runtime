"""Host state: in process memory, or the ``host_state`` table.

Internal module — only accessed through HostStateService (interface.py).
Each write is one conditional step, so a stale write never overwrites a
newer one.
"""

from __future__ import annotations

import asyncio
from dataclasses import fields
from datetime import UTC, datetime
from typing import Any, Protocol

from assistant_runtime.services.host_state.models import HostStateEntry

_FIELDS = tuple(f.name for f in fields(HostStateEntry))


class HostStateStore(Protocol):
    durable: bool

    async def get(self, namespace: str, key: str) -> HostStateEntry | None: ...

    async def list(self, namespace: str, limit: int) -> list[HostStateEntry]: ...

    async def put(
        self, namespace: str, key: str, value: Any, by: str, expected_version: int | None
    ) -> HostStateEntry | None: ...

    async def delete(self, namespace: str, key: str, expected_version: int | None) -> bool: ...


class InMemoryHostStateStore:
    """Values kept in the process; lost on restart."""

    durable = False

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], HostStateEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, namespace: str, key: str) -> HostStateEntry | None:
        return self._entries.get((namespace, key))

    async def list(self, namespace: str, limit: int) -> list[HostStateEntry]:
        entries = [e for (ns, _), e in self._entries.items() if ns == namespace]
        return sorted(entries, key=lambda e: e.key)[:limit]

    async def put(
        self, namespace: str, key: str, value: Any, by: str, expected_version: int | None
    ) -> HostStateEntry | None:
        async with self._lock:
            current = self._entries.get((namespace, key))
            version = current.version if current is not None else 0
            if expected_version is not None and expected_version != version:
                return None
            entry = HostStateEntry(
                namespace, key, value, version + 1, by, updated_at=datetime.now(UTC)
            )
            self._entries[(namespace, key)] = entry
            return entry

    async def delete(self, namespace: str, key: str, expected_version: int | None) -> bool:
        async with self._lock:
            current = self._entries.get((namespace, key))
            if current is None or expected_version not in (None, current.version):
                return False
            del self._entries[(namespace, key)]
            return True


class DatabaseHostStateStore:
    """Values in the ``host_state`` table."""

    durable = True

    def __init__(self, database_service: Any) -> None:
        self._database = database_service

    @staticmethod
    def _repository(session: Any) -> Any:
        from assistant_runtime.services.database.repositories import HostStateRepository

        return HostStateRepository(session)

    @staticmethod
    def _entry(row: Any) -> HostStateEntry:
        return HostStateEntry(**{name: getattr(row, name) for name in _FIELDS})

    async def get(self, namespace: str, key: str) -> HostStateEntry | None:
        async with self._database.session_context() as session:
            row = await self._repository(session).get(namespace, key)
            return self._entry(row) if row is not None else None

    async def list(self, namespace: str, limit: int) -> list[HostStateEntry]:
        async with self._database.session_context() as session:
            return [self._entry(r) for r in await self._repository(session).list(namespace, limit)]

    async def put(
        self, namespace: str, key: str, value: Any, by: str, expected_version: int | None
    ) -> HostStateEntry | None:
        async with self._database.session_context() as session:
            row = await self._repository(session).put(namespace, key, value, by, expected_version)
            return self._entry(row) if row is not None else None

    async def delete(self, namespace: str, key: str, expected_version: int | None) -> bool:
        async with self._database.session_context() as session:
            return await self._repository(session).delete(namespace, key, expected_version)
