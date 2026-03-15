"""Chat endpoint — /chat (non-streaming)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from loguru import logger

from lovely_assistant.app.assistant.deps import AssistantServiceDep
from lovely_assistant.app.assistant.models import AssistantRequest, AssistantResult

router = APIRouter()


@router.post("/chat")
async def chat(
    request: Request,
    assistant_request: AssistantRequest,
    service: AssistantServiceDep,
) -> AssistantResult:
    """Process a chat message (non-streaming)."""
    logger.info(
        "Received chat request",
        session_id=assistant_request.session_id,
    )

    # Refresh OAuth token if needed (lightweight check)
    oauth_service = getattr(request.app.state, "oauth_service", None)
    if oauth_service is not None and oauth_service.configured and oauth_service.needs_refresh():
        try:
            await oauth_service.refresh()
        except Exception as exc:
            logger.warning("OAuth refresh failed before chat", error=str(exc))

    return await service.process_message(assistant_request)
