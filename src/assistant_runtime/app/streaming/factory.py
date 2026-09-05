"""Factory for streaming service lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.app.streaming.interface import StreamingService
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings


async def register_streaming(
    app_state: Any, lifecycle: LifecycleManager, *, settings: AppSettings | None = None
) -> None:
    """Create StreamingService, store on app_state, register with lifecycle.

    Depends on history_service, tool_service and assistant_service being
    already registered on app_state.
    """
    settings = settings if settings is not None else AppSettings()
    service = StreamingService(
        config=settings.streaming,
        history_service=app_state.history_service,
        tool_service=app_state.tool_service,
        assistant_service=app_state.assistant_service,
        database_service=getattr(app_state, "database_service", None),
    )
    app_state.streaming_service = service
    await lifecycle.register("streaming_service", service)
    logger.info("Streaming module registered")
