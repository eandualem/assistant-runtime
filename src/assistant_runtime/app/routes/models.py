"""Models endpoint — ``GET /models``: the catalog, provider status and effective defaults."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from assistant_runtime.config import AppSettings
from assistant_runtime.model_catalog import get_models, get_provider_info

router = APIRouter()


@router.get("/models")
async def list_models(
    request: Request,
    capability: str | None = Query(
        default=None, description="Filter by capability tag (e.g. text, vision, image-generation)"
    ),
    provider: str | None = Query(default=None, description="Filter by provider name"),
) -> dict[str, Any]:
    """Return available models, provider status, and the defaults a request would get."""
    models = get_models(capability=capability, provider=provider)
    result: dict[str, Any] = {
        "models": [m.model_dump() for m in models],
        "providers": {k: v.model_dump() for k, v in get_provider_info().items()},
        "defaults": _effective_defaults(request),
    }

    oauth_service = getattr(request.app.state, "oauth_service", None)
    if oauth_service is not None:
        status = oauth_service.get_device_code_status()
        result["openai_oauth"] = {
            "connected": status.connected,
            "email": status.email,
            "source": status.source,
        }
    return result


def _effective_defaults(request: Request) -> dict[str, Any]:
    """The model each task uses right now: runtime overrides over the frozen settings.

    The primary and summarization models come from the LLM service when it is
    up, so the provider-aware fallback is reflected.
    """
    state = request.app.state
    settings = AppSettings()
    runtime = getattr(state, "runtime_settings", None)
    llm = getattr(state, "llm_service", None)

    def _tunable(name: str, frozen: Any) -> Any:
        return runtime.get(name, frozen) if runtime is not None else frozen

    primary = settings.llm.primary_model
    summarization = settings.llm.summarization_model
    if llm is not None:
        primary = llm.effective_primary_model()
        summarization = llm.effective_summarization_model()
    return {
        "primary_model": _tunable("default_model", None) or primary,
        "summarization_model": _tunable("summarization_model", None) or summarization,
        "working_memory_model": _tunable(
            "working_memory_model", settings.history.working_memory_model
        ),
        "default_image_model": _tunable("default_image_model", settings.media.default_image_model),
        "default_video_model": _tunable("default_video_model", settings.media.default_video_model),
        "subagent_model": _tunable("subagent_model", None),
        "subagent_thinking_budget": _tunable("subagent_thinking_budget", None),
    }
