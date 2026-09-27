"""Factory for action service lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.services.actions.interface import ActionService


async def register_actions(
    app_state: Any, lifecycle: LifecycleManager, *, settings: AppSettings | None = None
) -> None:
    """Create ActionService, store it on app_state and register it with the lifecycle."""
    settings = settings if settings is not None else AppSettings()
    service = ActionService(settings.actions, getattr(app_state, "database_service", None))
    app_state.action_service = service
    await lifecycle.register("action_service", service)
    logger.info("Actions module registered")
