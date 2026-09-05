"""Factory for assistant service lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.app.assistant.interface import AssistantService
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings


async def register_assistant(
    app_state: Any, lifecycle: LifecycleManager, *, settings: AppSettings | None = None
) -> None:
    """Create AssistantService, store on app_state, register with lifecycle.

    Depends on llm_service, history_service, tool_service and artifact_service
    being already registered on app_state.
    """
    settings = settings if settings is not None else AppSettings()
    service = AssistantService(
        config=settings.assistant,
        llm_service=app_state.llm_service,
        history_service=app_state.history_service,
        tool_service=app_state.tool_service,
        artifact_service=app_state.artifact_service,
        runtime_settings=getattr(app_state, "runtime_settings", None),
        database_service=getattr(app_state, "database_service", None),
        definition=getattr(app_state, "assistant_definition", None),
    )
    app_state.assistant_service = service
    await lifecycle.register("assistant_service", service)
    logger.info("Assistant module registered")
