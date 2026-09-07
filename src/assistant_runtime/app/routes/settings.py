"""Settings endpoints — GET/PATCH /settings for the runtime overlay."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from assistant_runtime.app.access.deps import AdminDep, PrincipalDep
from assistant_runtime.app.assistant.config import TunableOverrides
from assistant_runtime.app.settings import RuntimeSettings

router = APIRouter()


def get_runtime_settings(request: Request) -> RuntimeSettings:
    """Access RuntimeSettings from app.state (created during lifespan)."""
    return request.app.state.runtime_settings


RuntimeSettingsDep = Annotated[RuntimeSettings, Depends(get_runtime_settings)]


@router.get("/settings")
async def get_settings(
    runtime_settings: RuntimeSettingsDep, principal: PrincipalDep
) -> dict[str, Any]:
    """Return current runtime settings with source annotations."""
    return runtime_settings.to_response_dict()


@router.patch("/settings")
async def patch_settings(
    body: TunableOverrides, runtime_settings: RuntimeSettingsDep, admin: AdminDep
) -> dict[str, Any]:
    """Update runtime settings. Omitted fields are unchanged; null clears the override."""
    updates = body.model_dump(exclude_unset=True)
    persisted = True
    if updates:
        persisted = await runtime_settings.update(**updates)
    response = runtime_settings.to_response_dict()
    response["persisted"] = persisted
    return response
