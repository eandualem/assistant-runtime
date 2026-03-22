"""Factory for heartbeat service lifecycle registration."""

from typing import Any

from loguru import logger

from lovely_assistant.app.heartbeat.interface import HeartbeatService
from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.config import AppSettings


async def register_heartbeat(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create HeartbeatService, store on app_state, and register with lifecycle."""
    settings = AppSettings()
    service = HeartbeatService(
        config=settings.heartbeat,
        assistant_service=app_state.assistant_service,
        database_service=getattr(app_state, "database_service", None),
    )
    app_state.heartbeat_service = service
    await lifecycle.register("heartbeat_service", service)
    logger.info("Heartbeat module registered")
