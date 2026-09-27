"""Event records: in process memory, or the ``events`` table.

Internal module — only accessed through EventLogService (interface.py).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import fields, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from assistant_runtime.app.event_log.models import EventRecord

_FIELDS = tuple(f.name for f in fields(EventRecord))


class EventTransaction(Protocol):
    async def create_if_new(self, **values: Any) -> tuple[EventRecord, bool]: ...

    async def get(self, event_id: int, *, lock: bool = False) -> EventRecord | None: ...

    async def update(self, event_id: int, **values: Any) -> EventRecord: ...

    async def list(self, *, after: int, limit: int, **filters: Any) -> list[EventRecord]: ...


class EventStore(Protocol):
    durable: bool

    def transaction(self) -> contextlib.AbstractAsyncContextManager[EventTransaction]: ...


class InMemoryEventStore:
    """Events kept in the process; lost on restart."""

    durable = False

    def __init__(self) -> None:
        self._events: dict[int, EventRecord] = {}
        self._keys: dict[tuple[str, str], int] = {}
        self._next = 1
        self._lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[InMemoryEventStore]:
        async with self._lock:
            yield self

    async def create_if_new(self, **values: Any) -> tuple[EventRecord, bool]:
        key = (values["source"], values["event_id"])
        if key in self._keys:
            return self._events[self._keys[key]], False
        record = EventRecord(id=self._next, created_at=datetime.now(UTC), **values)
        self._events[record.id] = record
        self._keys[key] = record.id
        self._next += 1
        return record, True

    async def get(self, event_id: int, *, lock: bool = False) -> EventRecord | None:
        return self._events.get(event_id)

    async def update(self, event_id: int, **values: Any) -> EventRecord:
        record = self._events[event_id] = replace(self._events[event_id], **values)
        return record

    async def list(self, *, after: int, limit: int, **filters: Any) -> list[EventRecord]:
        records = [
            r
            for r in self._events.values()
            if r.id > after
            and all(value is None or getattr(r, name) == value for name, value in filters.items())
        ]
        return sorted(records, key=lambda r: r.id)[:limit]


class DatabaseEventStore:
    """Events in the ``events`` table."""

    durable = True

    def __init__(self, database_service: Any) -> None:
        self._database = database_service

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[_DatabaseTransaction]:
        from assistant_runtime.services.database.repositories import EventRepository

        async with self._database.session_context() as session:
            yield _DatabaseTransaction(EventRepository(session))


class _DatabaseTransaction:
    def __init__(self, repository: Any) -> None:
        self._repository = repository

    @staticmethod
    def _record(row: Any) -> EventRecord:
        return EventRecord(**{name: getattr(row, name) for name in _FIELDS})

    async def create_if_new(self, **values: Any) -> tuple[EventRecord, bool]:
        row, created = await self._repository.create_if_new(**values)
        return self._record(row), created

    async def get(self, event_id: int, *, lock: bool = False) -> EventRecord | None:
        row = await self._repository.get(event_id, lock=lock)
        return self._record(row) if row is not None else None

    async def update(self, event_id: int, **values: Any) -> EventRecord:
        return self._record(await self._repository.update(event_id, **values))

    async def list(self, *, after: int, limit: int, **filters: Any) -> list[EventRecord]:
        rows = await self._repository.list(after=after, limit=limit, **filters)
        return [self._record(row) for row in rows]
