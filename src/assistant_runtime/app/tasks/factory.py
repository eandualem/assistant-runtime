"""Factory for task service lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.app.tasks.interface import TaskService
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings


async def register_tasks(
    app_state: Any, lifecycle: LifecycleManager, *, settings: AppSettings | None = None
) -> None:
    """Create TaskService, store it on app_state and register it (it adds its tools on start)."""
    settings = settings if settings is not None else AppSettings()
    service = TaskService(
        config=settings.tasks,
        streaming_service=app_state.streaming_service,
        database_service=getattr(app_state, "database_service", None),
        events=getattr(app_state, "events", None),
        tool_service=getattr(app_state, "tool_service", None),
        default_profile=getattr(
            getattr(app_state, "assistant_service", None), "default_profile_name", None
        ),
    )
    app_state.task_service = service
    await lifecycle.register("task_service", service)
    logger.info("Tasks module registered", enabled=settings.tasks.enabled)
