"""Factory for LLM service lifecycle registration."""

from typing import Any

from loguru import logger

from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.config import AppSettings
from lovely_assistant.services.llm.interface import LlmService


async def register_llm(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create LlmService, store on app_state, register with lifecycle."""
    settings = AppSettings()
    service = LlmService(config=settings.llm)
    oauth_service = getattr(app_state, "oauth_service", None)
    if oauth_service is not None:
        service.set_oauth_service(oauth_service)
    app_state.llm_service = service
    await lifecycle.register("llm_service", service)
    logger.info("LLM module registered")
