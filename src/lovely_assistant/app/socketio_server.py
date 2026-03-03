"""Socket.IO server — real-time transport layer for assistant streaming.

Replaces SSE with bidirectional socket.io communication. The AssistantNamespace
consumes StreamingService's async generator and emits events to session rooms.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import socketio
from loguru import logger

from lovely_assistant.app.assistant.models import AssistantRequest

if TYPE_CHECKING:
    from lovely_assistant.app.streaming.interface import StreamingService

# Maps event dict `type` to socket.io event name.
# debug_* events are mapped dynamically (all → assistant:debug).
_EVENT_TYPE_MAP: dict[str, str] = {
    "agent_status": "assistant:status",
    "thinking_delta": "assistant:thinking_delta",
    "text_delta": "assistant:text_delta",
    "tool_call": "assistant:tool_call",
    "tool_result": "assistant:tool_result",
    "tool_error": "assistant:tool_error",
    "final_response": "assistant:final_response",
    "error": "assistant:error",
}


def create_sio() -> socketio.AsyncServer:
    """Create and configure the Socket.IO async server."""
    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
    sio.register_namespace(AssistantNamespace("/assistant"))
    return sio


class AssistantNamespace(socketio.AsyncNamespace):
    """Socket.IO namespace for assistant streaming.

    Handles session room management, message streaming, and cancellation.
    """

    def __init__(self, namespace: str = "/assistant") -> None:
        super().__init__(namespace)
        self._active_streams: dict[str, asyncio.Task[None]] = {}

    @property
    def _streaming_service(self) -> StreamingService:
        return self.server.fastapi_app.state.streaming_service

    async def on_connect(self, sid: str, environ: dict, auth: Any = None) -> bool:
        logger.info("Socket.IO client connected", sid=sid, namespace=self.namespace)
        return True

    async def on_disconnect(self, sid: str) -> None:
        logger.info("Socket.IO client disconnected", sid=sid, namespace=self.namespace)

    async def on_assistant_join_session(self, sid: str, data: dict[str, Any]) -> None:
        """Client joins a session room for receiving events."""
        session_id = data.get("session_id") if isinstance(data, dict) else None
        if not session_id:
            await self.emit(
                "assistant:error", {"type": "validation", "message": "Missing session_id"}, to=sid
            )
            return
        room = f"session:{session_id}"
        self.enter_room(sid, room)
        logger.info("Client joined session room", sid=sid, session_id=session_id)

    async def on_assistant_message(self, sid: str, data: dict[str, Any]) -> None:
        """Handle incoming chat message — start streaming to session room."""
        if not isinstance(data, dict) or "session_id" not in data or "message" not in data:
            await self.emit(
                "assistant:error",
                {"type": "validation", "message": "Missing session_id or message"},
                to=sid,
            )
            return

        session_id = data["session_id"]
        room = f"session:{session_id}"

        # Guard: reject if stream already active for this session
        if session_id in self._active_streams:
            await self.emit(
                "assistant:error",
                {"type": "conflict", "message": "Stream already active for this session"},
                to=sid,
            )
            return

        # Auto-join room
        self.enter_room(sid, room)

        # Build request
        request = AssistantRequest(**data)

        # Start streaming task
        task = asyncio.create_task(self._run_stream(sid, session_id, request))
        self._active_streams[session_id] = task
        task.add_done_callback(lambda _t: self._active_streams.pop(session_id, None))

    async def on_assistant_cancel(self, sid: str, data: dict[str, Any]) -> None:
        """Cancel an active stream for a session."""
        session_id = data.get("session_id") if isinstance(data, dict) else None
        if not session_id:
            return

        task = self._active_streams.get(session_id)
        if task and not task.done():
            task.cancel()
            logger.info("Stream cancelled by client", sid=sid, session_id=session_id)

    async def _run_stream(self, sid: str, session_id: str, request: AssistantRequest) -> None:
        """Consume StreamingService generator and emit events to the session room."""
        room = f"session:{session_id}"
        try:
            async for event in self._streaming_service.stream_message(request):
                event_type = event.get("type", "")
                socket_event = _EVENT_TYPE_MAP.get(event_type)
                if socket_event is None:
                    # debug_* events → assistant:debug
                    if event_type.startswith("debug_"):
                        socket_event = "assistant:debug"
                    else:
                        socket_event = "assistant:unknown"
                await self.emit(socket_event, event, room=room)
        except asyncio.CancelledError:
            await self.emit(
                "assistant:error",
                {"type": "cancelled", "message": "Stream cancelled"},
                room=room,
            )
        except Exception as e:
            logger.error("Stream failed", session_id=session_id, error=str(e))
            await self.emit(
                "assistant:error",
                {"type": "internal", "message": f"Stream failed: {e}"},
                room=room,
            )
