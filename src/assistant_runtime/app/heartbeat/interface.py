"""HeartbeatService — the runtime's periodic check-in, delivered through the ingress."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from assistant_runtime.app.heartbeat.config import HeartbeatConfig
from assistant_runtime.app.heartbeat.exceptions import HeartbeatError

if TYPE_CHECKING:
    from assistant_runtime.app.ingress.interface import IngressService


class HeartbeatService:
    """Every ``interval_seconds``, hand the assistant its check-in message."""

    def __init__(self, config: HeartbeatConfig, ingress_service: IngressService) -> None:
        self._config = config
        self._ingress = ingress_service
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._last_heartbeat_at: datetime | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        """Start the background heartbeat loop when enabled."""
        self._started = True
        self._last_error = None
        if self._config.enabled:
            self._task = asyncio.create_task(self._run_loop(), name="assistant-heartbeat")
        logger.info(
            "Heartbeat service started",
            enabled=self._config.enabled,
            interval_seconds=self._config.interval_seconds,
        )

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._started = False
        logger.info("Heartbeat service stopped")

    async def health_check(self) -> dict[str, Any]:
        task_running = self._task is not None and not self._task.done()
        return {
            "healthy": self._started and (not self._config.enabled or task_running),
            "enabled": self._config.enabled,
            "task_running": task_running,
            "last_heartbeat_at": self._last_heartbeat_at.isoformat()
            if self._last_heartbeat_at is not None
            else None,
            "last_error": self._last_error,
        }

    async def enqueue_once(self) -> dict[str, Any]:
        """Deliver one heartbeat into the most recently active session."""
        if not self._started:
            raise HeartbeatError("Heartbeat service not started")
        # A heartbeat is only meaningful now: never queue one for later.
        result = await self._ingress.deliver(
            from_agent=self._config.from_agent,
            via="heartbeat",
            message=self._config.message,
            queue_when_unrouted=False,
        )
        if result.get("status") == "delivered":
            self._last_heartbeat_at = datetime.now(UTC)
            self._last_error = None
        return result

    async def _run_loop(self) -> None:
        try:
            if self._config.startup_delay_seconds:
                await asyncio.sleep(self._config.startup_delay_seconds)
            while True:
                try:
                    result = await self.enqueue_once()
                    logger.debug("Heartbeat tick completed", result=result)
                except Exception as exc:
                    self._last_error = str(exc)
                    logger.warning("Heartbeat tick failed", error=str(exc))
                await asyncio.sleep(self._config.interval_seconds)
        except asyncio.CancelledError:
            raise
