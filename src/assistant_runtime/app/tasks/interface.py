"""TaskService — bounded pieces of work run as turns in the background.

A task is a turn in its own fresh session (``task-<id>``), started by the
model (the task tools), by a host (``POST /api/tasks``) or in process, and
run outside any conversation turn: the caller keeps serving while it runs.
Tasks about the same (profile, subject) run one at a time, in order; at
most ``max_concurrent`` run at once. When a task ends, its record says how
and ``task_finished`` is published on ``app.state.events``; nothing is
steered into the session that asked, so the host decides when and how a
result is reviewed. A restart marks unfinished tasks ``interrupted`` and
publishes that; nothing is replayed.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import ValidationError

from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.app.tasks._store import DatabaseTaskStore, InMemoryTaskStore, TaskStore
from assistant_runtime.app.tasks._tools import register_task_tools
from assistant_runtime.app.tasks.config import TasksConfig
from assistant_runtime.app.tasks.exceptions import (
    TaskError,
    TaskLimitError,
    TaskNotFoundError,
    TasksDisabledError,
)
from assistant_runtime.app.tasks.models import TaskRecord
from assistant_runtime.principal import LOCAL_PRINCIPAL, Principal

if TYPE_CHECKING:
    from assistant_runtime.app.streaming.interface import StreamingService
    from assistant_runtime.base.events import EventHub
    from assistant_runtime.services.database.interface import DatabaseService
    from assistant_runtime.services.tools.interface import ToolService

_INTERRUPTED = "The runtime stopped while the task was queued or running"
# A task's end is recorded once: later terminal updates find it finished and change nothing.
_UNFINISHED = ("queued", "running")


class TaskService:
    """Start, list, read and cancel background tasks. Implements LifecycleAware."""

    def __init__(
        self,
        config: TasksConfig,
        streaming_service: StreamingService,
        database_service: DatabaseService | None = None,
        events: EventHub | None = None,
        tool_service: ToolService | None = None,
        default_profile: str | None = None,
    ) -> None:
        self._config = config
        self._tool_service = tool_service
        self._default_profile = default_profile
        self._reserved = 0
        self._streaming = streaming_service
        self._database = database_service
        self._events = events
        self._store: TaskStore | None = None
        self._slots = asyncio.Semaphore(config.max_concurrent)
        self._subject_locks: dict[tuple[str | None, str], list[Any]] = {}
        self._running: dict[str, asyncio.Task[None]] = {}
        self._cancel_requested: set[str] = set()

    # --- lifecycle ---

    async def start(self) -> None:
        database = self._database
        if database is not None and getattr(database, "healthy", False):
            self._store = DatabaseTaskStore(database)
            for record in await self._store.mark_unfinished("interrupted", _INTERRUPTED):
                await self._publish(record)
        else:
            self._store = InMemoryTaskStore()
        if self._config.enabled and self._tool_service is not None:
            # The tool service starts first (registration order), so its registry is ready.
            register_task_tools(self._tool_service, self)
        logger.info("Task service started", enabled=self._config.enabled)

    async def stop(self) -> None:
        running = list(self._running.items())
        for _, task in running:
            task.cancel()
        for task_id, task in running:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            await self._finish_unstarted(task_id, "interrupted")
        self._store = None
        logger.info("Task service stopped")

    async def health_check(self) -> dict[str, Any]:
        return {
            "healthy": self._store is not None,
            "enabled": self._config.enabled,
            "running": len(self._running),
            "durable": self._store.durable if self._store is not None else None,
        }

    # --- public API ---

    async def start_task(
        self,
        task: str,
        *,
        profile: str | None = None,
        subject: str | None = None,
        context: str | None = None,
        parent_session_id: str | None = None,
        principal: Principal | None = None,
    ) -> TaskRecord:
        """Queue a task and return its record at once; it runs in the background."""
        if not self._config.enabled:
            raise TasksDisabledError("Background tasks are not enabled")
        store = self._require_store()
        task = (task or "").strip()
        if not task:
            raise TaskError("A task needs a description of the work")
        self._streaming.validate_profile(profile)
        try:  # the request's own rules for profile and subject names, checked up front
            AssistantRequest(
                id="check", session_id="task-check", content=task, profile=profile, subject=subject
            )
        except ValidationError as e:
            raise TaskError(f"Invalid profile or subject: {e.errors()[0]['msg']}") from e
        # Reserve a place before any await, so concurrent starts cannot all pass.
        if len(self._running) + self._reserved >= self._config.max_waiting:
            raise TaskLimitError(
                f"{len(self._running) + self._reserved} tasks are queued or running; "
                "wait for some to finish"
            )
        self._reserved += 1
        principal = principal or LOCAL_PRINCIPAL
        task_id = str(uuid.uuid4())
        try:
            record = await store.create(
                TaskRecord(
                    id=task_id,
                    session_id=f"task-{task_id}",
                    task=task,
                    status="queued",
                    created_by=principal.id,
                    # The default resolved now: omitting it and naming it are the same profile.
                    profile=profile or self._default_profile,
                    subject=subject,
                    context=(context or "").strip() or None,
                    parent_session_id=parent_session_id,
                )
            )
            self._running[task_id] = asyncio.create_task(
                self._run(record, principal), name=f"task-{task_id}"
            )
        finally:
            self._reserved -= 1
        logger.info("Task queued", task_id=task_id, profile=profile, subject=subject)
        return record

    async def get(self, task_id: str, principal: Principal | None = None) -> TaskRecord:
        record = await self._require_store().get(task_id)
        if record is None or not _visible(record, principal or LOCAL_PRINCIPAL):
            raise TaskNotFoundError(f"No task '{task_id}'")
        return record

    async def list(
        self,
        principal: Principal | None = None,
        *,
        parent_session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[TaskRecord]:
        """Newest first: the caller's own tasks, or every task for an administrator."""
        principal = principal or LOCAL_PRINCIPAL
        return await self._require_store().list(
            created_by=None if principal.is_admin else principal.id,
            parent_session_id=parent_session_id,
            status=status,
            limit=limit,
        )

    async def cancel(self, task_id: str, principal: Principal | None = None) -> TaskRecord:
        """Stop a queued or running task; a finished one is returned as it is."""
        record = await self.get(task_id, principal)
        running = self._running.get(task_id)
        if running is None or record.finished:
            return record
        self._cancel_requested.add(task_id)
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await running
        await self._finish_unstarted(task_id, "cancelled")
        return await self.get(task_id, principal)

    # --- running ---

    async def _run(self, record: TaskRecord, principal: Principal) -> None:
        store = self._require_store()
        final: TaskRecord | None = record
        try:
            async with self._subject_turn(record.profile, record.subject), self._slots:
                started = await store.update(
                    record.id, only_from=("queued",), status="running", started_at=_now()
                )
                if started is None:
                    return  # it ended before it could start; its end was recorded then
                request = AssistantRequest(
                    id=str(uuid.uuid4()),
                    session_id=record.session_id,
                    content=_content(record),
                    profile=record.profile,
                    subject=record.subject,
                )
                async with asyncio.timeout(self._config.timeout_seconds):
                    result = await self._streaming.run_message(request, principal=principal)
            if result.pending_tool_call:
                # A task has no host to perform an action and send its result back.
                raise TaskError(
                    "The task asked the host to perform "
                    f"'{result.pending_tool_call.get('tool_name')}', which a task cannot answer"
                )
            final = await store.update(
                record.id,
                only_from=_UNFINISHED,
                status="done",
                result=(result.content or "")[: self._config.result_max_chars],
                usage=result.usage,
                finished_at=_now(),
            )
        except asyncio.CancelledError:
            requested = record.id in self._cancel_requested
            final = await store.update(
                record.id,
                only_from=_UNFINISHED,
                status="cancelled" if requested else "interrupted",
                error=None if requested else _INTERRUPTED,
                finished_at=_now(),
            )
        except TimeoutError:
            final = await store.update(
                record.id,
                only_from=_UNFINISHED,
                status="failed",
                error=f"Timed out after {self._config.timeout_seconds} s",
                finished_at=_now(),
            )
        except Exception as e:
            logger.warning("Task failed", task_id=record.id, error=str(e))
            final = await store.update(
                record.id,
                only_from=_UNFINISHED,
                status="failed",
                error=str(e) or type(e).__name__,
                finished_at=_now(),
            )
        finally:
            self._running.pop(record.id, None)
            self._cancel_requested.discard(record.id)
        if final is not None:
            await self._publish(final)

    async def _finish_unstarted(self, task_id: str, status: str) -> None:
        """Record the end of a task whose worker was cancelled before it ever ran.

        asyncio does not enter a coroutine cancelled before its first step, so
        its own handler and ``finally`` never run; this does their work.
        """
        if self._running.pop(task_id, None) is None:
            return  # the worker ran and finished its record itself
        self._cancel_requested.discard(task_id)
        final = await self._require_store().update(
            task_id,
            only_from=_UNFINISHED,
            status=status,
            error=None if status == "cancelled" else _INTERRUPTED,
            finished_at=_now(),
        )
        if final is not None:
            await self._publish(final)

    @contextlib.asynccontextmanager
    async def _subject_turn(self, profile: str | None, subject: str | None):
        """One task at a time per (profile, subject), in arrival order."""
        if subject is None:
            yield
            return
        key = (profile, subject)
        entry = self._subject_locks.setdefault(key, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self._subject_locks.pop(key, None)

    async def _publish(self, record: TaskRecord) -> None:
        if self._events is None:
            return
        await self._events.publish(
            {
                "type": "task_finished",
                "task_id": record.id,
                "status": record.status,
                "session_id": record.session_id,
                "parent_session_id": record.parent_session_id,
                "profile": record.profile,
                "subject": record.subject,
                "result": record.result,
                "error": record.error,
            }
        )

    def _require_store(self) -> TaskStore:
        if self._store is None:
            raise TaskError("Task service not started")
        return self._store


def _visible(record: TaskRecord, principal: Principal) -> bool:
    return principal.is_admin or record.created_by == principal.id


def _content(record: TaskRecord) -> str:
    if not record.context:
        return record.task
    return f"{record.task}\n\nContext:\n{record.context}"


def _now() -> datetime:
    return datetime.now(UTC)
