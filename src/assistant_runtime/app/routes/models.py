"""Models endpoint — GET /models for the model registry."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from assistant_runtime.app.models_registry import get_defaults, get_models, get_provider_info

router = APIRouter()


@router.get("/models")
async def list_models(
    request: Request,
    capability: str | None = Query(
        default=None, description="Filter by capability tag (e.g. text, vision, image-generation)"
    ),
    provider: str | None = Query(default=None, description="Filter by provider name"),
) -> dict[str, Any]:
    """Return available models, provider status, and current defaults."""
    models = get_models(capability=capability, provider=provider)

    # Check OAuth connection status if service is available
    openai_oauth = None
    oauth_service = getattr(request.app.state, "oauth_service", None)
    if oauth_service is not None:
        status = oauth_service.get_device_code_status()
        openai_oauth = {
            "connected": status.connected,
            "email": status.email,
            "source": status.source,
        }

    result: dict[str, Any] = {
        "models": [m.model_dump() for m in models],
        "providers": {k: v.model_dump() for k, v in get_provider_info().items()},
        "defaults": get_defaults(),
    }
    if openai_oauth is not None:
        result["openai_oauth"] = openai_oauth

    return result
