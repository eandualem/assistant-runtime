"""Factory for tool service lifecycle registration."""

from typing import Any

from loguru import logger

from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.config import AppSettings
from lovely_assistant.services.tools.interface import ToolService


async def register_tools(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create ToolService, store on app_state, register with lifecycle."""
    settings = AppSettings()
    service = ToolService(config=settings.tools)
    app_state.tool_service = service
    await lifecycle.register("tool_service", service)
    logger.info("Tool module registered")
