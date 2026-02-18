"""Factory for history service lifecycle registration."""

from typing import Any

from loguru import logger

from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.config import AppSettings
from lovely_assistant.services.history.interface import HistoryService


async def register_history(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create HistoryService, store on app_state, register with lifecycle."""
    settings = AppSettings()
    llm_service = app_state.llm_service  # Must be registered first
    service = HistoryService(config=settings.history, llm_service=llm_service)
    app_state.history_service = service
    await lifecycle.register("history_service", service)
    logger.info("History module registered")
