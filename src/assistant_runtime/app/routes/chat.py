"""Chat endpoint — ``POST /chat``, the non-streaming form of a turn."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from assistant_runtime.app.assistant.models import AssistantRequest, AssistantResult
from assistant_runtime.app.streaming.deps import StreamingServiceDep

router = APIRouter()


@router.post("/chat/{session_id}/cancel")
async def cancel_chat(session_id: str, service: StreamingServiceDep) -> dict[str, bool]:
    """Request cancellation; the active turn persists its snapshot before ending."""
    return {"cancel_requested": await service.cancel_session(session_id)}


@router.post("/chat")
async def chat(
    request: Request,
    assistant_request: AssistantRequest,
    service: StreamingServiceDep,
) -> AssistantResult:
    """Run one turn and return the final answer."""
    logger.info("Received chat request", session_id=assistant_request.session_id)

    if assistant_request.is_steering:
        raise HTTPException(
            status_code=422,
            detail="Steering messages are only supported over the streaming Socket.IO transport",
        )

    # Refresh OAuth token if needed (lightweight check)
    oauth_service = getattr(request.app.state, "oauth_service", None)
    if oauth_service is not None and oauth_service.configured and oauth_service.needs_refresh():
        try:
            await oauth_service.refresh()
        except Exception as exc:
            logger.warning("OAuth refresh failed before chat", error=str(exc))

    return await service.run_message(assistant_request)
