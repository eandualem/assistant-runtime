"""Factory for access service lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.app.access.interface import AccessService
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings


async def register_access(
    app_state: Any, lifecycle: LifecycleManager, *, settings: AppSettings | None = None
) -> None:
    """Create AccessService from the settings and the definition's authenticator."""
    settings = settings if settings is not None else AppSettings()
    definition = getattr(app_state, "assistant_definition", None)
    service = AccessService(
        config=settings.access,
        authenticator=getattr(definition, "authenticate", None),
    )
    app_state.access_service = service
    await lifecycle.register("access_service", service)
    logger.info("Access module registered", mode=settings.access.mode)
