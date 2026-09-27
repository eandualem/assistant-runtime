"""Task records: in process memory, or the ``tasks`` table.

Internal module — only accessed through TaskService (interface.py).
"""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from assistant_runtime.app.tasks.models import TaskRecord

_FIELDS = tuple(f.name for f in fields(TaskRecord))


class TaskStore(Protocol):
    durable: bool

    async def create(self, record: TaskRecord) -> TaskRecord: ...

    async def update(
        self, task_id: str, *, only_from: tuple[str, ...] | None = None, **values: Any
    ) -> TaskRecord | None: ...

    async def get(self, task_id: str) -> TaskRecord | None: ...

    async def list(
        self,
        *,
        created_by: str | None,
        parent_session_id: str | None,
        status: str | None,
        limit: int,
    ) -> list[TaskRecord]: ...

    async def mark_unfinished(self, status: str, error: str) -> list[TaskRecord]: ...


class InMemoryTaskStore:
    """Tasks kept in the process; lost on restart."""

    durable = False

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}

    async def create(self, record: TaskRecord) -> TaskRecord:
        record = replace(record, created_at=record.created_at or datetime.now(UTC))
        self._tasks[record.id] = record
        return record

    async def update(
        self, task_id: str, *, only_from: tuple[str, ...] | None = None, **values: Any
    ) -> TaskRecord | None:
        record = self._tasks.get(task_id)
        if record is None or (only_from is not None and record.status not in only_from):
            return None
        record = self._tasks[task_id] = replace(record, **values)
        return record

    async def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    async def list(
        self,
        *,
        created_by: str | None,
        parent_session_id: str | None,
        status: str | None,
        limit: int,
    ) -> list[TaskRecord]:
        records = [
            r
            for r in self._tasks.values()
            if (created_by is None or r.created_by == created_by)
            and (parent_session_id is None or r.parent_session_id == parent_session_id)
            and (status is None or r.status == status)
        ]
        return list(reversed(records))[:limit]

    async def mark_unfinished(self, status: str, error: str) -> list[TaskRecord]:
        return []  # nothing survives a restart in memory


class DatabaseTaskStore:
    """Tasks in the ``tasks`` table."""

    durable = True

    def __init__(self, database_service: Any) -> None:
        self._database = database_service

    @staticmethod
    def _repository(session: Any) -> Any:
        from assistant_runtime.services.database.repositories import TaskRepository

        return TaskRepository(session)

    @staticmethod
    def _record(row: Any) -> TaskRecord:
        return TaskRecord(**{name: getattr(row, name) for name in _FIELDS})

    async def create(self, record: TaskRecord) -> TaskRecord:
        values = {name: getattr(record, name) for name in _FIELDS if name != "created_at"}
        async with self._database.session_context() as session:
            return self._record(await self._repository(session).create(**values))

    async def update(
        self, task_id: str, *, only_from: tuple[str, ...] | None = None, **values: Any
    ) -> TaskRecord | None:
        async with self._database.session_context() as session:
            row = await self._repository(session).update(task_id, only_from=only_from, **values)
            return self._record(row) if row is not None else None

    async def get(self, task_id: str) -> TaskRecord | None:
        async with self._database.session_context() as session:
            row = await self._repository(session).get(task_id)
            return self._record(row) if row is not None else None

    async def list(
        self,
        *,
        created_by: str | None,
        parent_session_id: str | None,
        status: str | None,
        limit: int,
    ) -> list[TaskRecord]:
        async with self._database.session_context() as session:
            rows = await self._repository(session).list(
                created_by=created_by,
                parent_session_id=parent_session_id,
                status=status,
                limit=limit,
            )
            return [self._record(row) for row in rows]

    async def mark_unfinished(self, status: str, error: str) -> list[TaskRecord]:
        async with self._database.session_context() as session:
            rows = await self._repository(session).mark_unfinished(status, error)
            return [self._record(row) for row in rows]
