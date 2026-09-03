"""Factory for OAuth service lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.services.oauth.interface import OAuthService


async def register_oauth(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create OAuthService, store on app_state, register with lifecycle."""
    settings = AppSettings()
    service = OAuthService(config=settings.oauth)

    # Inject database service dependency
    db_service = getattr(app_state, "database_service", None)
    if db_service is not None:
        service.set_database_service(db_service)

    app_state.oauth_service = service
    await lifecycle.register("oauth_service", service)
    logger.info("OAuth module registered")
