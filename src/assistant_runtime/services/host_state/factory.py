"""Factory for host state lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.services.host_state.interface import HostStateService


async def register_host_state(
    app_state: Any, lifecycle: LifecycleManager, *, settings: AppSettings | None = None
) -> None:
    """Create HostStateService, store it on app_state and register it with the lifecycle."""
    settings = settings if settings is not None else AppSettings()
    service = HostStateService(settings.host_state, getattr(app_state, "database_service", None))
    app_state.host_state_service = service
    await lifecycle.register("host_state_service", service)
    logger.info("Host state module registered")
