"""Provider API key management — GET/PUT/DELETE for LLM provider credentials."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from assistant_runtime.app.access.deps import require_admin
from assistant_runtime.model_catalog import ALLOWED_PROVIDERS
from assistant_runtime.services.llm.exceptions import ProviderKeyStoreUnavailableError
from assistant_runtime.services.llm.interface import LlmService

router = APIRouter(prefix="/providers", tags=["providers"], dependencies=[Depends(require_admin)])

_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "openrouter": "OpenRouter",
}


# --- Dependencies ---


def get_llm_service(request: Request) -> LlmService:
    return request.app.state.llm_service


LlmServiceDep = Annotated[LlmService, Depends(get_llm_service)]


# --- Request / Response models ---


class SetApiKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str = Field(..., min_length=1, description="Provider API key")


# --- Endpoints ---


@router.get("")
async def list_providers(request: Request, llm: LlmServiceDep) -> list[dict[str, Any]]:
    """List all LLM providers with auth status and key source."""
    statuses = llm.get_provider_status()

    # Enrich with display names and OpenAI OAuth status
    oauth_service = getattr(request.app.state, "oauth_service", None)
    for entry in statuses:
        entry["name"] = _PROVIDER_DISPLAY_NAMES.get(entry["provider"], entry["provider"])
        entry["oauth"] = None
        if entry["provider"] == "openai" and oauth_service is not None:
            oauth_status = oauth_service.get_device_code_status()
            entry["oauth"] = {
                "connected": oauth_status.connected,
                "status": oauth_status.status,
                "email": oauth_status.email,
                "source": oauth_status.source,
                "expires_at": oauth_status.expires_at,
            }
            # If OAuth is connected but no direct API key, mark as configured via oauth
            if oauth_status.connected and not entry["configured"]:
                entry["configured"] = True
                entry["source"] = "oauth"

    return statuses


@router.put("/{provider}/api-key")
async def set_api_key(provider: str, body: SetApiKeyRequest, llm: LlmServiceDep) -> dict[str, Any]:
    """Store an API key for a provider (encrypted in Postgres) and activate it immediately."""
    if provider not in ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    try:
        await llm.store_provider_key(provider, body.api_key)
    except ProviderKeyStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    raw = body.api_key
    preview = f"...{raw[-4:]}" if len(raw) >= 4 else "***"
    return {"provider": provider, "status": "configured", "api_key_preview": preview}


@router.delete("/{provider}/api-key")
async def delete_api_key(provider: str, llm: LlmServiceDep) -> dict[str, Any]:
    """Remove a stored API key for a provider; a key from the environment stays in force."""
    if provider not in ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    try:
        deleted = await llm.delete_provider_key(provider)
    except ProviderKeyStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"provider": provider, "status": "removed", "was_stored": deleted}
