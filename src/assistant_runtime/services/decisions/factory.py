"""Factory for decision service lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.services.decisions.interface import DecisionService


async def register_decisions(
    app_state: Any, lifecycle: LifecycleManager, *, settings: AppSettings | None = None
) -> None:
    """Create DecisionService, store on app_state, register with lifecycle."""
    settings = settings if settings is not None else AppSettings()
    service = DecisionService(settings.decisions)
    app_state.decision_service = service
    await lifecycle.register("decision_service", service)
    logger.info("Decisions module registered")
