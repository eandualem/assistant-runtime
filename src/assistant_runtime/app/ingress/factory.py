"""Factory for ingress service lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.app.ingress.interface import IngressService
from assistant_runtime.base.lifecycle import LifecycleManager


async def register_ingress(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create IngressService, store on app_state, register with lifecycle.

    Depends on assistant_service and streaming_service being registered.
    """
    service = IngressService(
        assistant_service=app_state.assistant_service,
        streaming_service=app_state.streaming_service,
        database_service=getattr(app_state, "database_service", None),
        socket_server=getattr(app_state, "sio", None),
    )
    app_state.ingress_service = service
    app_state.streaming_service.attach_ingress(service)
    await lifecycle.register("ingress_service", service)
    logger.info("Ingress module registered")
