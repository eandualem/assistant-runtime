"""Models endpoint — GET /models for the model registry."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from lovely_assistant.app.models_registry import get_defaults, get_models, get_provider_info

router = APIRouter()


@router.get("/models")
async def list_models(
    capability: str | None = Query(
        default=None, description="Filter by capability tag (e.g. text, vision, image-generation)"
    ),
    provider: str | None = Query(default=None, description="Filter by provider name"),
) -> dict[str, Any]:
    """Return available models, provider status, and current defaults."""
    models = get_models(capability=capability, provider=provider)
    return {
        "models": [m.model_dump() for m in models],
        "providers": {k: v.model_dump() for k, v in get_provider_info().items()},
        "defaults": get_defaults(),
    }
