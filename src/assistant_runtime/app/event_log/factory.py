"""Factory for event log lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.app.event_log.interface import EventLogService
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings


async def register_event_log(
    app_state: Any, lifecycle: LifecycleManager, *, settings: AppSettings | None = None
) -> None:
    """Create EventLogService (delivering through ingress), store it and register it."""
    settings = settings if settings is not None else AppSettings()
    service = EventLogService(
        settings.event_log,
        ingress_service=getattr(app_state, "ingress_service", None),
        database_service=getattr(app_state, "database_service", None),
    )
    app_state.event_log_service = service
    await lifecycle.register("event_log_service", service)
    logger.info("Event log module registered")
