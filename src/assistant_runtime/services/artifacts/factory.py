"""Factory for artifact service lifecycle registration."""

from typing import Any

from loguru import logger

from assistant_runtime.artifacts import AssistantProfile, resolve_profile
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.services.artifacts.interface import ArtifactService


async def register_artifacts(
    app_state: Any,
    lifecycle: LifecycleManager,
    *,
    settings: AppSettings | None = None,
    profile: AssistantProfile | None = None,
) -> None:
    """Create ArtifactService for the resolved profile, store on app_state, register.

    The profile comes from the assistant definition when one is attached,
    else from ``ASSISTANT__PROFILE``, else it is the neutral built-in.
    """
    settings = settings if settings is not None else AppSettings()
    definition = getattr(app_state, "assistant_definition", None)
    if profile is None:
        profile = resolve_profile(settings.assistant.profile, getattr(definition, "profile", None))
    service = ArtifactService(
        config=settings.artifacts,
        profile=profile,
        database_service=getattr(app_state, "database_service", None),
    )
    app_state.artifact_service = service
    await lifecycle.register("artifact_service", service)
    logger.info("Artifact module registered", profile=profile.name)
