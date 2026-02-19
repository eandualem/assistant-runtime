"""Chat endpoints — /chat and /chat/stream."""

from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from lovely_assistant.app.assistant.deps import AssistantServiceDep
from lovely_assistant.app.assistant.models import AssistantRequest, AssistantResult
from lovely_assistant.app.streaming.deps import StreamingServiceDep

router = APIRouter()


@router.post("/chat")
async def chat(request: AssistantRequest, service: AssistantServiceDep) -> AssistantResult:
    """Process a chat message (non-streaming)."""
    return await service.process_message(request)


@router.post("/chat/stream")
async def chat_stream(request: AssistantRequest, service: StreamingServiceDep) -> StreamingResponse:
    """Process a chat message with SSE streaming."""

    async def event_generator():
        async for event in service.stream_message(request):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
