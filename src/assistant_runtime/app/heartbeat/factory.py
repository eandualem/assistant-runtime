"""Factory for heartbeat service lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.app.heartbeat.interface import HeartbeatService
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings


async def register_heartbeat(
    app_state: Any, lifecycle: LifecycleManager, *, settings: AppSettings | None = None
) -> None:
    """Create HeartbeatService, store on app_state, and register with lifecycle."""
    settings = settings if settings is not None else AppSettings()
    service = HeartbeatService(
        config=settings.heartbeat,
        ingress_service=app_state.ingress_service,
    )
    app_state.heartbeat_service = service
    await lifecycle.register("heartbeat_service", service)
    logger.info("Heartbeat module registered")
