"""Factory for media service lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.services.media.interface import MediaService


async def register_media(
    app_state: Any, lifecycle: LifecycleManager, *, settings: AppSettings | None = None
) -> None:
    """Create MediaService, store on app_state, register with lifecycle."""
    settings = settings if settings is not None else AppSettings()
    service = MediaService(config=settings.media)
    app_state.media_service = service
    await lifecycle.register("media_service", service)
    logger.info("Media module registered")
