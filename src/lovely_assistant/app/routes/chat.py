"""Chat endpoint — /chat (non-streaming)."""

from __future__ import annotations

from fastapi import APIRouter
from loguru import logger

from lovely_assistant.app.assistant.deps import AssistantServiceDep
from lovely_assistant.app.assistant.models import AssistantRequest, AssistantResult

router = APIRouter()


@router.post("/chat")
async def chat(request: AssistantRequest, service: AssistantServiceDep) -> AssistantResult:
    """Process a chat message (non-streaming)."""
    logger.info(
        "Received chat request",
        session_id=request.session_id,
    )
    return await service.process_message(request)
