"""Provider API key management — GET/PUT/DELETE for LLM provider credentials."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from assistant_runtime.services.llm.config import ALLOWED_PROVIDERS
from assistant_runtime.services.llm.interface import LlmService

router = APIRouter(prefix="/providers", tags=["providers"])

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
async def set_api_key(
    provider: str,
    body: SetApiKeyRequest,
    request: Request,
    llm: LlmServiceDep,
) -> dict[str, Any]:
    """Store an API key for a provider (encrypted in DB) and activate it immediately."""
    if provider not in ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    oauth_service = getattr(request.app.state, "oauth_service", None)
    if oauth_service is None or not oauth_service.configured:
        raise HTTPException(
            status_code=503,
            detail="Encryption not configured — set OAUTH__ENCRYPTION_KEY to enable API key storage",
        )

    # Encrypt and store in DB
    fernet = oauth_service._fernet
    if fernet is None:
        raise HTTPException(status_code=503, detail="Encryption cipher not initialized")

    db_service = getattr(request.app.state, "database_service", None)
    if db_service is None:
        raise HTTPException(status_code=503, detail="Database service not available")

    encrypted = fernet.encrypt(body.api_key.encode()).decode()

    async with db_service.session_context() as session:
        from assistant_runtime.services.database.repositories import OAuthTokenRepository

        repo = OAuthTokenRepository(session)
        await repo.upsert(provider=provider, encrypted_api_key=encrypted)
        await session.commit()

    # Hot-reload into LLM service (takes effect immediately, no restart)
    await llm.reload_provider_key(provider, body.api_key)

    raw = body.api_key
    preview = f"...{raw[-4:]}" if len(raw) >= 4 else "***"
    return {"provider": provider, "status": "configured", "api_key_preview": preview}


@router.delete("/{provider}/api-key")
async def delete_api_key(
    provider: str,
    request: Request,
    llm: LlmServiceDep,
) -> dict[str, Any]:
    """Remove a stored API key for a provider."""
    if provider not in ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    db_service = getattr(request.app.state, "database_service", None)
    if db_service is None:
        raise HTTPException(status_code=503, detail="Database service not available")

    async with db_service.session_context() as session:
        from assistant_runtime.services.database.repositories import OAuthTokenRepository

        repo = OAuthTokenRepository(session)
        deleted = await repo.delete(provider)
        await session.commit()

    # Remove from LLM service
    await llm.remove_provider_key(provider)

    return {"provider": provider, "status": "removed", "was_stored": deleted}
