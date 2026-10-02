"""Task and persistent-agent records: in process memory, or the database.

Internal module — only accessed through TaskService (interface.py).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from assistant_runtime.app.tasks.models import AgentMessageRecord, AgentRecord, TaskRecord

_FIELDS = tuple(f.name for f in fields(TaskRecord))
_AGENT_FIELDS = tuple(f.name for f in fields(AgentRecord))
_MESSAGE_FIELDS = tuple(f.name for f in fields(AgentMessageRecord))


class AgentTakenError(Exception):
    """Another active agent has this profile and subject, or this session."""


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

    async def create_agent(self, record: AgentRecord) -> AgentRecord: ...

    async def update_agent(self, agent_id: str, **values: Any) -> AgentRecord | None: ...

    async def get_agent(self, agent_id: str) -> AgentRecord | None: ...

    async def list_agents(
        self, *, created_by: str | None, status: str | None
    ) -> list[AgentRecord]: ...

    async def create_message(self, record: AgentMessageRecord) -> AgentMessageRecord: ...

    async def update_message(
        self, message_id: str, *, only_from: tuple[str, ...] | None = None, **values: Any
    ) -> AgentMessageRecord | None: ...

    async def get_message(self, message_id: str) -> AgentMessageRecord | None: ...

    async def list_messages(
        self,
        *,
        agent_id: str | None,
        created_by: str | None,
        parent_session_id: str | None,
        status: str | None,
        limit: int,
    ) -> list[AgentMessageRecord]: ...

    async def mark_unfinished_messages(
        self, status: str, error: str
    ) -> list[AgentMessageRecord]: ...


class InMemoryTaskStore:
    """Tasks kept in the process; lost on restart."""

    durable = False

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._agents: dict[str, AgentRecord] = {}
        self._messages: dict[str, AgentMessageRecord] = {}

    async def create(self, record: TaskRecord) -> TaskRecord:
        record = replace(deepcopy(record), created_at=record.created_at or datetime.now(UTC))
        self._tasks[record.id] = record
        return deepcopy(record)

    async def update(
        self, task_id: str, *, only_from: tuple[str, ...] | None = None, **values: Any
    ) -> TaskRecord | None:
        record = self._tasks.get(task_id)
        if record is None or (only_from is not None and record.status not in only_from):
            return None
        record = self._tasks[task_id] = replace(record, **deepcopy(values))
        return deepcopy(record)

    async def get(self, task_id: str) -> TaskRecord | None:
        return deepcopy(self._tasks.get(task_id))

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
        return deepcopy(list(reversed(records))[:limit])

    async def mark_unfinished(self, status: str, error: str) -> list[TaskRecord]:
        return []  # nothing survives a restart in memory

    async def create_agent(self, record: AgentRecord) -> AgentRecord:
        record = replace(deepcopy(record), created_at=record.created_at or datetime.now(UTC))
        self._agents[record.id] = record
        return deepcopy(record)

    async def update_agent(self, agent_id: str, **values: Any) -> AgentRecord | None:
        record = self._agents.get(agent_id)
        if record is None:
            return None
        record = self._agents[agent_id] = replace(record, **deepcopy(values))
        return deepcopy(record)

    async def get_agent(self, agent_id: str) -> AgentRecord | None:
        return deepcopy(self._agents.get(agent_id))

    async def list_agents(self, *, created_by: str | None, status: str | None) -> list[AgentRecord]:
        records = [
            r
            for r in self._agents.values()
            if (created_by is None or r.created_by == created_by)
            and (status is None or r.status == status)
        ]
        return deepcopy(list(reversed(records)))

    async def create_message(self, record: AgentMessageRecord) -> AgentMessageRecord:
        record = replace(deepcopy(record), created_at=record.created_at or datetime.now(UTC))
        self._messages[record.id] = record
        return deepcopy(record)

    async def update_message(
        self, message_id: str, *, only_from: tuple[str, ...] | None = None, **values: Any
    ) -> AgentMessageRecord | None:
        record = self._messages.get(message_id)
        if record is None or (only_from is not None and record.status not in only_from):
            return None
        record = self._messages[message_id] = replace(record, **deepcopy(values))
        return deepcopy(record)

    async def get_message(self, message_id: str) -> AgentMessageRecord | None:
        return deepcopy(self._messages.get(message_id))

    async def list_messages(
        self,
        *,
        agent_id: str | None,
        created_by: str | None,
        parent_session_id: str | None,
        status: str | None,
        limit: int,
    ) -> list[AgentMessageRecord]:
        records = [
            r
            for r in self._messages.values()
            if (agent_id is None or r.agent_id == agent_id)
            and (created_by is None or r.created_by == created_by)
            and (parent_session_id is None or r.parent_session_id == parent_session_id)
            and (status is None or r.status == status)
        ]
        return deepcopy(list(reversed(records))[:limit])

    async def mark_unfinished_messages(self, status: str, error: str) -> list[AgentMessageRecord]:
        return []  # nothing survives a restart in memory


class DatabaseTaskStore:
    """Tasks in the ``tasks`` table; agents in ``agents`` and ``agent_messages``."""

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

    @staticmethod
    def _agents(session: Any) -> Any:
        from assistant_runtime.services.database.repositories import AgentRepository

        return AgentRepository(session)

    @staticmethod
    def _agent(row: Any) -> AgentRecord:
        return AgentRecord(**{name: getattr(row, name) for name in _AGENT_FIELDS})

    @staticmethod
    def _message(row: Any) -> AgentMessageRecord:
        return AgentMessageRecord(**{name: getattr(row, name) for name in _MESSAGE_FIELDS})

    async def create_agent(self, record: AgentRecord) -> AgentRecord:
        from sqlalchemy.exc import IntegrityError

        values = {name: getattr(record, name) for name in _AGENT_FIELDS if name != "created_at"}
        try:
            async with self._database.session_context() as session:
                return self._agent(await self._agents(session).create(**values))
        except IntegrityError as e:  # another process started the same agent first
            raise AgentTakenError(str(e.orig)) from e

    async def update_agent(self, agent_id: str, **values: Any) -> AgentRecord | None:
        from sqlalchemy.exc import IntegrityError

        try:
            async with self._database.session_context() as session:
                row = await self._agents(session).update(agent_id, **values)
                return self._agent(row) if row is not None else None
        except IntegrityError as e:
            raise AgentTakenError(str(e.orig)) from e

    async def get_agent(self, agent_id: str) -> AgentRecord | None:
        async with self._database.session_context() as session:
            row = await self._agents(session).get(agent_id)
            return self._agent(row) if row is not None else None

    async def list_agents(self, *, created_by: str | None, status: str | None) -> list[AgentRecord]:
        async with self._database.session_context() as session:
            rows = await self._agents(session).list(created_by=created_by, status=status)
            return [self._agent(row) for row in rows]

    async def create_message(self, record: AgentMessageRecord) -> AgentMessageRecord:
        values = {name: getattr(record, name) for name in _MESSAGE_FIELDS if name != "created_at"}
        async with self._database.session_context() as session:
            return self._message(await self._agents(session).create_message(**values))

    async def update_message(
        self, message_id: str, *, only_from: tuple[str, ...] | None = None, **values: Any
    ) -> AgentMessageRecord | None:
        async with self._database.session_context() as session:
            row = await self._agents(session).update_message(
                message_id, only_from=only_from, **values
            )
            return self._message(row) if row is not None else None

    async def get_message(self, message_id: str) -> AgentMessageRecord | None:
        async with self._database.session_context() as session:
            row = await self._agents(session).get_message(message_id)
            return self._message(row) if row is not None else None

    async def list_messages(
        self,
        *,
        agent_id: str | None,
        created_by: str | None,
        parent_session_id: str | None,
        status: str | None,
        limit: int,
    ) -> list[AgentMessageRecord]:
        async with self._database.session_context() as session:
            rows = await self._agents(session).list_messages(
                agent_id=agent_id,
                created_by=created_by,
                parent_session_id=parent_session_id,
                status=status,
                limit=limit,
            )
            return [self._message(row) for row in rows]

    async def mark_unfinished_messages(self, status: str, error: str) -> list[AgentMessageRecord]:
        async with self._database.session_context() as session:
            rows = await self._agents(session).mark_unfinished_messages(status, error)
            return [self._message(row) for row in rows]
