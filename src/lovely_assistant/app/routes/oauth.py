"""OAuth endpoints for OpenAI ChatGPT/Codex auth."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from lovely_assistant.services.oauth.deps import OAuthServiceDep

router = APIRouter(prefix="/oauth", tags=["oauth"])


@router.post("/openai/device-code")
async def initiate_device_code(service: OAuthServiceDep) -> dict[str, Any]:
    """Initiate OpenAI Device Code authorization flow.

    Returns user_code and verification_uri for the user to visit.
    """
    result = await service.initiate_device_code()
    return result.model_dump()


@router.post("/openai/codex-cli/sync")
async def sync_codex_cli(service: OAuthServiceDep) -> dict[str, Any]:
    """Sync OpenAI ChatGPT/Codex auth from the local Codex CLI."""
    status = await service.sync_from_codex_cli()
    return status.model_dump()


@router.get("/openai/status")
async def get_oauth_status(service: OAuthServiceDep) -> dict[str, Any]:
    """Get current OAuth authorization status and connection info."""
    status = service.get_device_code_status()
    return status.model_dump()


@router.delete("/openai")
async def disconnect_oauth(service: OAuthServiceDep) -> dict[str, str]:
    """Disconnect OpenAI OAuth — delete tokens and clear API key."""
    await service.disconnect()
    return {"status": "disconnected"}
