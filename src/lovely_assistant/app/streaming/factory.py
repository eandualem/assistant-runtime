"""Factory for streaming service lifecycle registration."""

from typing import Any

from loguru import logger

from lovely_assistant.app.streaming.interface import StreamingService
from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.config import AppSettings


async def register_streaming(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create StreamingService, store on app_state, register with lifecycle.

    Depends on llm_service, history_service, tool_service, and assistant_service
    being already registered on app_state.
    """
    settings = AppSettings()
    service = StreamingService(
        config=settings.streaming,
        llm_service=app_state.llm_service,
        history_service=app_state.history_service,
        tool_service=app_state.tool_service,
        assistant_service=app_state.assistant_service,
        assistant_config=settings.assistant,
    )
    app_state.streaming_service = service
    await lifecycle.register("streaming_service", service)
    logger.info("Streaming module registered")
