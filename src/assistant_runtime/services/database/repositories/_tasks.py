"""Tasks queries; the caller owns the transaction."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.services.database.models import (
    AgentMessageORM,
    AgentORM,
    TaskORM,
)


class TaskRepository:
    """Background task records. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, **values: Any) -> TaskORM:
        result = await self._session.execute(insert(TaskORM).values(**values).returning(TaskORM))
        await self._session.flush()
        return result.scalar_one()

    async def update(
        self, task_id: str, *, only_from: tuple[str, ...] | None = None, **fields: Any
    ) -> TaskORM | None:
        """Change the task; with ``only_from``, only while its status is one of them."""
        statement = update(TaskORM).where(TaskORM.id == task_id)
        if only_from is not None:
            statement = statement.where(TaskORM.status.in_(only_from))
        result = await self._session.execute(statement.values(**fields).returning(TaskORM))
        await self._session.flush()
        return result.scalar_one_or_none()

    async def get(self, task_id: str) -> TaskORM | None:
        result = await self._session.execute(select(TaskORM).where(TaskORM.id == task_id))
        return result.scalar_one_or_none()

    async def list(
        self,
        *,
        created_by: str | None = None,
        parent_session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[TaskORM]:
        """Newest first, filtered by whatever is given."""
        query = select(TaskORM).order_by(TaskORM.created_at.desc(), TaskORM.id.desc())
        if created_by is not None:
            query = query.where(TaskORM.created_by == created_by)
        if parent_session_id is not None:
            query = query.where(TaskORM.parent_session_id == parent_session_id)
        if status is not None:
            query = query.where(TaskORM.status == status)
        result = await self._session.execute(query.limit(limit))
        return list(result.scalars().all())

    async def mark_unfinished(self, status: str, error: str) -> list[TaskORM]:
        """Finish every queued or running task with ``status`` (after a restart)."""
        result = await self._session.execute(
            update(TaskORM)
            .where(TaskORM.status.in_(("queued", "running")))
            .values(status=status, error=error, finished_at=func.now())
            .returning(TaskORM)
        )
        await self._session.flush()
        return list(result.scalars().all())


class AgentRepository:
    """Persistent agents and their messages. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, **values: Any) -> AgentORM:
        result = await self._session.execute(insert(AgentORM).values(**values).returning(AgentORM))
        await self._session.flush()
        return result.scalar_one()

    async def update(self, agent_id: str, **fields: Any) -> AgentORM | None:
        result = await self._session.execute(
            update(AgentORM).where(AgentORM.id == agent_id).values(**fields).returning(AgentORM)
        )
        await self._session.flush()
        return result.scalar_one_or_none()

    async def get(self, agent_id: str) -> AgentORM | None:
        result = await self._session.execute(select(AgentORM).where(AgentORM.id == agent_id))
        return result.scalar_one_or_none()

    async def list(
        self, *, created_by: str | None = None, status: str | None = None
    ) -> list[AgentORM]:
        """Newest first, filtered by whatever is given."""
        query = select(AgentORM).order_by(AgentORM.created_at.desc(), AgentORM.id.desc())
        if created_by is not None:
            query = query.where(AgentORM.created_by == created_by)
        if status is not None:
            query = query.where(AgentORM.status == status)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def create_message(self, **values: Any) -> AgentMessageORM:
        result = await self._session.execute(
            insert(AgentMessageORM).values(**values).returning(AgentMessageORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def update_message(
        self, message_id: str, *, only_from: tuple[str, ...] | None = None, **fields: Any
    ) -> AgentMessageORM | None:
        """Change the message; with ``only_from``, only while its status is one of them."""
        statement = update(AgentMessageORM).where(AgentMessageORM.id == message_id)
        if only_from is not None:
            statement = statement.where(AgentMessageORM.status.in_(only_from))
        result = await self._session.execute(statement.values(**fields).returning(AgentMessageORM))
        await self._session.flush()
        return result.scalar_one_or_none()

    async def get_message(self, message_id: str) -> AgentMessageORM | None:
        result = await self._session.execute(
            select(AgentMessageORM).where(AgentMessageORM.id == message_id)
        )
        return result.scalar_one_or_none()

    async def list_messages(
        self,
        *,
        agent_id: str | None = None,
        created_by: str | None = None,
        parent_session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[AgentMessageORM]:
        """Newest first, filtered by whatever is given."""
        query = select(AgentMessageORM).order_by(
            AgentMessageORM.created_at.desc(), AgentMessageORM.id.desc()
        )
        if agent_id is not None:
            query = query.where(AgentMessageORM.agent_id == agent_id)
        if created_by is not None:
            query = query.where(AgentMessageORM.created_by == created_by)
        if parent_session_id is not None:
            query = query.where(AgentMessageORM.parent_session_id == parent_session_id)
        if status is not None:
            query = query.where(AgentMessageORM.status == status)
        result = await self._session.execute(query.limit(limit))
        return list(result.scalars().all())

    async def mark_unfinished_messages(self, status: str, error: str) -> list[AgentMessageORM]:
        """Finish every queued or running message with ``status`` (after a restart)."""
        result = await self._session.execute(
            update(AgentMessageORM)
            .where(AgentMessageORM.status.in_(("queued", "running")))
            .values(status=status, error=error, finished_at=func.now())
            .returning(AgentMessageORM)
        )
        await self._session.flush()
        return list(result.scalars().all())
