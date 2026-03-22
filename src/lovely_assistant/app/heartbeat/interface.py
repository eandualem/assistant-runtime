"""HeartbeatService — periodic Jarvis check-ins via the inbox injection path."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from lovely_assistant.app._injector import inject_inbox_message
from lovely_assistant.app.heartbeat.config import HeartbeatConfig
from lovely_assistant.app.heartbeat.exceptions import HeartbeatError

if TYPE_CHECKING:
    from lovely_assistant.app.assistant.interface import AssistantService
    from lovely_assistant.services.database.interface import DatabaseService


class HeartbeatService:
    """Periodic heartbeat injector for the most recently active Jarvis session."""

    def __init__(
        self,
        config: HeartbeatConfig,
        assistant_service: AssistantService,
        database_service: DatabaseService | None = None,
    ) -> None:
        self._config = config
        self._assistant = assistant_service
        self._db = database_service
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._last_heartbeat_at: datetime | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        """Start the background heartbeat loop when enabled."""
        self._started = True
        self._last_error = None
        if self._config.enabled:
            self._task = asyncio.create_task(self._run_loop(), name="jarvis-heartbeat")
        logger.info(
            "Heartbeat service started",
            enabled=self._config.enabled,
            interval_seconds=self._config.interval_seconds,
        )

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._started = False
        logger.info("Heartbeat service stopped")

    async def health_check(self) -> dict[str, Any]:
        """Report heartbeat loop state."""
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
        """Inject one heartbeat into the active Jarvis session."""
        if not self._started:
            raise HeartbeatError("Heartbeat service not started")
        if self._db is None:
            return {"status": "skipped", "reason": "database_unavailable"}

        session_id = await self._resolve_active_session_id()
        if session_id is None:
            return {"status": "skipped", "reason": "no_active_session"}

        result = await inject_inbox_message(
            db=self._db,
            assistant_service=self._assistant,
            from_agent=self._config.from_agent,
            via="heartbeat",
            message=self._config.message,
            session_id=session_id,
        )
        if result.get("status") == "delivered":
            self._last_heartbeat_at = datetime.now(UTC)
            self._last_error = None
        return result

    async def _resolve_active_session_id(self) -> str | None:
        """Pick the most recently active Jarvis session, if one exists."""
        sessions = self._assistant.get_session_store()
        if sessions is None:
            return None
        listed = await sessions.list_sessions(limit=1)
        if not listed:
            return None
        session_id = listed[0].get("session_id")
        return session_id if isinstance(session_id, str) and session_id else None

    async def _run_loop(self) -> None:
        """Background task that injects heartbeats on a fixed interval."""
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
