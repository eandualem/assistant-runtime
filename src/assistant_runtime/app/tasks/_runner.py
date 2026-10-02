"""Scheduling and terminal bookkeeping for task and persistent-agent turns."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Coroutine, Hashable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NamedTuple

from loguru import logger

from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.app.tasks.config import TasksConfig
from assistant_runtime.app.tasks.exceptions import TaskError, TaskLimitError
from assistant_runtime.principal import Principal

if TYPE_CHECKING:
    from assistant_runtime.app.streaming.interface import StreamingService

INTERRUPTED = "The runtime stopped while it was queued or running"
_UNFINISHED = ("queued", "running")


class _Job(NamedTuple):
    """A queued or running background turn: a task, or a message to an agent."""

    worker: asyncio.Task[None]
    update: Callable[..., Awaitable[Any]]
    publish: Callable[[Any], Awaitable[None]]
    agent_id: str | None = None


@dataclass
class _Order:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class BackgroundTurnRunner:
    """Own queue capacity, per-key ordering, shared slots and worker lifetime."""

    def __init__(self, config: TasksConfig, streaming: StreamingService) -> None:
        self._config = config
        self._streaming = streaming
        self._reserved = 0
        self._slots = asyncio.Semaphore(config.max_concurrent)
        self._order_locks: dict[Hashable, _Order] = {}
        self._running: dict[str, _Job] = {}
        self._cancel_requested: set[str] = set()

    @property
    def count(self) -> int:
        return len(self._running)

    @contextlib.contextmanager
    def reserve(self) -> Iterator[None]:
        """Reserve queue capacity across asynchronous record creation."""
        count = self.count + self._reserved
        if count >= self._config.max_waiting:
            raise TaskLimitError(
                f"{count} background turns are queued or running; wait for some to finish"
            )
        self._reserved += 1
        try:
            yield
        finally:
            self._reserved -= 1

    def submit(
        self,
        job_id: str,
        work: Coroutine[Any, Any, None],
        update: Callable[..., Awaitable[Any]],
        publish: Callable[[Any], Awaitable[None]],
        *,
        name: str,
        agent_id: str | None = None,
    ) -> None:
        self._running[job_id] = _Job(
            asyncio.create_task(work, name=name), update, publish, agent_id
        )

    def mark_cancelled(self, job_id: str) -> None:
        self._cancel_requested.add(job_id)

    async def cancel_agent(self, agent_id: str) -> None:
        for job_id in [i for i, job in self._running.items() if job.agent_id == agent_id]:
            await self.cancel(job_id)

    async def stop(self) -> None:
        running = list(self._running.items())
        for _, job in running:
            job.worker.cancel()
        for job_id, job in running:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await job.worker
            await self._finish_unstarted(job_id, "interrupted")

    async def execute(
        self,
        job_id: str,
        order_key: Hashable | None,
        start: Callable[[], Awaitable[AssistantRequest | None]],
        update: Callable[..., Awaitable[Any]],
        publish: Callable[[Any], Awaitable[None]],
        principal: Principal,
    ) -> None:
        """Run one background turn in its order and a free slot; record and publish its end."""
        final: Any = None
        try:
            async with self._in_order(order_key), self._slots:
                request = await start()
                if request is None:
                    return
                async with asyncio.timeout(self._config.timeout_seconds):
                    result = await self._streaming.run_message(request, principal=principal)
            if result.pending_tool_call:
                # A background turn has no host to perform an action and send its result back.
                raise TaskError(
                    "The turn asked the host to perform "
                    f"'{result.pending_tool_call.get('tool_name')}', which a background "
                    "turn cannot answer"
                )
            final = await update(
                job_id,
                only_from=_UNFINISHED,
                status="done",
                result=(result.content or "")[: self._config.result_max_chars],
                usage=result.usage,
                finished_at=_now(),
            )
        except asyncio.CancelledError:
            requested = job_id in self._cancel_requested
            final = await update(
                job_id,
                only_from=_UNFINISHED,
                status="cancelled" if requested else "interrupted",
                error=None if requested else INTERRUPTED,
                finished_at=_now(),
            )
        except TimeoutError:
            final = await update(
                job_id,
                only_from=_UNFINISHED,
                status="failed",
                error=f"Timed out after {self._config.timeout_seconds} s",
                finished_at=_now(),
            )
        except Exception as e:
            logger.warning("Background turn failed", job_id=job_id, error=str(e))
            final = await update(
                job_id,
                only_from=_UNFINISHED,
                status="failed",
                error=str(e) or type(e).__name__,
                finished_at=_now(),
            )
        finally:
            self._running.pop(job_id, None)
            self._cancel_requested.discard(job_id)
        if final is not None:
            await publish(final)

    async def cancel(self, job_id: str) -> None:
        """Cancel a queued or running background turn and wait until its end is recorded."""
        job = self._running.get(job_id)
        if job is None:
            return
        self._cancel_requested.add(job_id)
        job.worker.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await job.worker
        await self._finish_unstarted(job_id, "cancelled")

    async def _finish_unstarted(self, job_id: str, status: str) -> None:
        """Record the end of a background turn whose worker was cancelled before it ever ran.

        asyncio does not enter a coroutine cancelled before its first step, so
        its own handler and ``finally`` never run; this does their work.
        """
        job = self._running.pop(job_id, None)
        if job is None:
            return  # the worker ran and finished its record itself
        self._cancel_requested.discard(job_id)
        final = await job.update(
            job_id,
            only_from=_UNFINISHED,
            status=status,
            error=None if status == "cancelled" else INTERRUPTED,
            finished_at=_now(),
        )
        if final is not None:
            await job.publish(final)

    @contextlib.asynccontextmanager
    async def _in_order(self, key: Hashable | None):
        """One background turn at a time per key, in arrival order; no key, no order."""
        if key is None:
            yield
            return
        entry = self._order_locks.setdefault(key, _Order())
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                self._order_locks.pop(key, None)


def _now() -> datetime:
    return datetime.now(UTC)
