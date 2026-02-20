"""Factory for database service lifecycle registration."""

from typing import Any

from loguru import logger

from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.config import AppSettings
from lovely_assistant.services.database.interface import DatabaseService


async def register_database(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create DatabaseService, store on app_state, register with lifecycle."""
    settings = AppSettings()
    service = DatabaseService(config=settings.database)
    app_state.database_service = service
    await lifecycle.register("database_service", service)
    logger.info("Database module registered")
