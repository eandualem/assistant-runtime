"""Factory for assistant service lifecycle registration."""

from typing import Any

from loguru import logger

from lovely_assistant.app.assistant.interface import AssistantService
from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.config import AppSettings


async def register_assistant(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create AssistantService, store on app_state, register with lifecycle.

    Depends on llm_service, history_service, and tool_service being already
    registered on app_state.
    """
    settings = AppSettings()
    service = AssistantService(
        config=settings.assistant,
        llm_service=app_state.llm_service,
        history_service=app_state.history_service,
        tool_service=app_state.tool_service,
    )
    app_state.assistant_service = service
    await lifecycle.register("assistant_service", service)
    logger.info("Assistant module registered")
