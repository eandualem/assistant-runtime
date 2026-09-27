"""Host state: in process memory, or the ``host_state`` table.

Internal module — only accessed through HostStateService (interface.py).
Each write is one conditional step, so a stale write never overwrites a
newer one.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import fields, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from assistant_runtime.services.host_state.models import HostStateEntry

_FIELDS = tuple(f.name for f in fields(HostStateEntry))


class HostStateStore(Protocol):
    durable: bool

    async def get(self, namespace: str, key: str) -> HostStateEntry | None: ...

    async def list(self, namespace: str, after: str | None, limit: int) -> list[HostStateEntry]: ...

    async def put(
        self, namespace: str, key: str, value: Any, by: str, expected_version: int | None
    ) -> HostStateEntry | None: ...

    async def delete(
        self, namespace: str, key: str, by: str, expected_version: int | None
    ) -> bool: ...


class InMemoryHostStateStore:
    """Values kept in the process; lost on restart.

    Values are copied in and out, so a caller's object never changes what
    is stored behind the version check. A deleted key keeps its version.
    """

    durable = False

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], HostStateEntry] = {}
        self._deleted: dict[tuple[str, str], int] = {}  # tombstones: the version at deletion
        self._lock = asyncio.Lock()

    @staticmethod
    def _copy(entry: HostStateEntry) -> HostStateEntry:
        return replace(entry, value=copy.deepcopy(entry.value))

    async def get(self, namespace: str, key: str) -> HostStateEntry | None:
        entry = self._entries.get((namespace, key))
        return self._copy(entry) if entry is not None else None

    async def list(self, namespace: str, after: str | None, limit: int) -> list[HostStateEntry]:
        entries = [
            e
            for (ns, key), e in self._entries.items()
            if ns == namespace and (after is None or key > after)
        ]
        return [self._copy(e) for e in sorted(entries, key=lambda e: e.key)[:limit]]

    async def put(
        self, namespace: str, key: str, value: Any, by: str, expected_version: int | None
    ) -> HostStateEntry | None:
        async with self._lock:
            current = self._entries.get((namespace, key))
            version = current.version if current is not None else 0
            if expected_version is not None and expected_version != version:
                return None
            if current is None:
                version = self._deleted.pop((namespace, key), 0)
            entry = HostStateEntry(
                namespace,
                key,
                copy.deepcopy(value),
                version + 1,
                by,
                updated_at=datetime.now(UTC),
            )
            self._entries[(namespace, key)] = entry
            return self._copy(entry)

    async def delete(self, namespace: str, key: str, by: str, expected_version: int | None) -> bool:
        async with self._lock:
            current = self._entries.get((namespace, key))
            if current is None or expected_version not in (None, current.version):
                return False
            del self._entries[(namespace, key)]
            self._deleted[(namespace, key)] = current.version + 1
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

    async def list(self, namespace: str, after: str | None, limit: int) -> list[HostStateEntry]:
        async with self._database.session_context() as session:
            rows = await self._repository(session).list(namespace, after, limit)
            return [self._entry(r) for r in rows]

    async def put(
        self, namespace: str, key: str, value: Any, by: str, expected_version: int | None
    ) -> HostStateEntry | None:
        async with self._database.session_context() as session:
            row = await self._repository(session).put(namespace, key, value, by, expected_version)
            return self._entry(row) if row is not None else None

    async def delete(self, namespace: str, key: str, by: str, expected_version: int | None) -> bool:
        async with self._database.session_context() as session:
            return await self._repository(session).delete(namespace, key, by, expected_version)
