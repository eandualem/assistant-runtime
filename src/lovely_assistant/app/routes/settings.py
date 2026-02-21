"""Settings endpoints — GET/PATCH /settings for runtime configuration."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from lovely_assistant.app.settings import RuntimeSettings

router = APIRouter()


# --- Dependency ---


def get_runtime_settings(request: Request) -> RuntimeSettings:
    """Access RuntimeSettings from app.state (created during lifespan)."""
    return request.app.state.runtime_settings


RuntimeSettingsDep = Annotated[RuntimeSettings, Depends(get_runtime_settings)]


# --- Request model ---


class SettingsPatchRequest(BaseModel):
    """PATCH body for runtime settings. All fields optional — omitted = no change."""

    model_config = ConfigDict(extra="forbid")

    default_model: str | None = Field(default=None, description="Model override. None = clear.")
    thinking_budget: int | None = Field(
        default=None, ge=1, le=100_000, description="Thinking budget. None = clear."
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Temperature. None = clear."
    )
    max_turns: int | None = Field(
        default=None, ge=1, le=50, description="Max agent turns. None = clear."
    )
    enable_working_memory: bool | None = Field(
        default=None, description="Working memory toggle. None = clear."
    )
    summarization_model: str | None = Field(
        default=None, description="Summarization model. None = clear."
    )
    working_memory_model: str | None = Field(
        default=None, description="Working memory extraction model. None = clear."
    )
    default_image_model: str | None = Field(
        default=None, description="Default image generation model. None = clear."
    )
    default_video_model: str | None = Field(
        default=None, description="Default video generation model. None = clear."
    )
    subagent_model: str | None = Field(
        default=None, description="Subagent model override. None = clear."
    )
    subagent_thinking_budget: int | None = Field(
        default=None, ge=1, le=100_000, description="Subagent thinking budget. None = clear."
    )


# --- Endpoints ---


@router.get("/settings")
async def get_settings(runtime_settings: RuntimeSettingsDep) -> dict[str, Any]:
    """Return current runtime settings with source annotations."""
    return runtime_settings.to_response_dict()


@router.patch("/settings")
async def patch_settings(
    body: SettingsPatchRequest, runtime_settings: RuntimeSettingsDep
) -> dict[str, Any]:
    """Update runtime settings. Omitted fields are unchanged; null clears the override."""
    updates = body.model_dump(exclude_unset=True)
    persisted = True
    if updates:
        persisted = await runtime_settings.update(**updates)
    response = runtime_settings.to_response_dict()
    response["persisted"] = persisted
    return response
