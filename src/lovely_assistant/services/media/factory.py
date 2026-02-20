"""Factory for media service lifecycle registration."""

from typing import Any

from loguru import logger

from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.config import AppSettings
from lovely_assistant.services.media.interface import MediaService


async def register_media(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create MediaService, store on app_state, register with lifecycle."""
    settings = AppSettings()
    service = MediaService(config=settings.media)
    app_state.media_service = service
    await lifecycle.register("media_service", service)
    logger.info("Media module registered")
