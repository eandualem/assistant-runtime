"""Socket.IO server — real-time transport layer for assistant streaming.

The AssistantNamespace consumes StreamingService's async generator and emits
events directly to the requesting client (to=sid).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
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

    Handles message streaming and cancellation. Events are emitted directly
    to the requesting client via to=sid (1:1 model, matching the backbone's
    terminal namespace pattern).
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
        await self.enter_room(sid, room)
        machine_state = data.get("machine_state") if isinstance(data, dict) else None
        streaming_service = self._try_get_streaming_service()
        if streaming_service is not None:
            warm = getattr(streaming_service, "warm_session", None)
            if callable(warm):
                try:
                    maybe_awaitable = warm(session_id, machine_state)
                    if inspect.isawaitable(maybe_awaitable):
                        await maybe_awaitable
                except Exception as e:
                    logger.warning(
                        "Session warmup failed during join",
                        sid=sid,
                        session_id=session_id,
                        error=str(e),
                    )
        logger.info("Client joined session room", sid=sid, session_id=session_id)

    async def on_assistant_message(self, sid: str, data: dict[str, Any]) -> None:
        """Handle incoming chat message — start streaming to client."""
        if not isinstance(data, dict):
            await self.emit(
                "assistant:error",
                {"type": "validation", "message": "Invalid assistant request payload"},
                to=sid,
            )
            return

        # Build request — catch Pydantic validation errors
        try:
            request = AssistantRequest(**data)
        except Exception as e:
            logger.error("Request validation failed", sid=sid, error=str(e))
            await self.emit(
                "assistant:error",
                {"type": "validation", "message": f"Invalid request: {e}"},
                to=sid,
            )
            return

        session_id = request.session_id
        is_continuation = request.is_continuation

        if request.is_guidance:
            try:
                await self._streaming_service.queue_guidance(request)
            except Exception as e:
                logger.error(
                    "Guidance queueing failed",
                    sid=sid,
                    session_id=session_id,
                    error=str(e),
                )
                await self.emit(
                    "assistant:error",
                    {"type": "session", "message": str(e)},
                    to=sid,
                )
            return

        # If a stream is already active for this session, cancel it and replace.
        # Exception: continuations (tool_call_id set) should not cancel the
        # stream that dispatched the tool — the stream may still be draining
        # its finally block. Continuations go through the streaming service's
        # own validation (pending tool call matching).
        active_task = self._active_streams.get(session_id)
        if active_task is not None:
            if active_task.done():
                self._active_streams.pop(session_id, None)
            elif not is_continuation:
                active_task.cancel()
                self._active_streams.pop(session_id, None)
                logger.info(
                    "Cancelled stale stream for new message",
                    sid=sid,
                    session_id=session_id,
                )

        # Start streaming task
        task = asyncio.create_task(self._run_stream(sid, session_id, request))
        self._active_streams[session_id] = task

        def _done_cleanup(_t: asyncio.Task[None]) -> None:
            # Only pop if WE are still the active stream for this session.
            # A continuation may have already replaced us in _active_streams.
            if self._active_streams.get(session_id) is _t:
                self._active_streams.pop(session_id, None)

        task.add_done_callback(_done_cleanup)

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
        """Consume StreamingService generator and emit events to the client."""
        logger.info("[STREAM] _run_stream started", sid=sid, session_id=session_id)
        event_count = 0
        last_event_type: str | None = None
        last_socket_event: str | None = None
        released_session = False
        try:
            async for event in self._streaming_service.stream_message(request):
                event_type = event.get("type", "")
                last_event_type = event_type
                socket_event = _EVENT_TYPE_MAP.get(event_type)
                if socket_event is None:
                    # debug_* events → assistant:debug
                    if event_type.startswith("debug_"):
                        socket_event = "assistant:debug"
                    else:
                        socket_event = "assistant:unknown"
                last_socket_event = socket_event
                is_terminal_completed = (
                    event_type == "agent_status" and event.get("status") == "completed"
                )
                is_deferred_final = (
                    event_type == "final_response" and event.get("pending_tool_call") is not None
                )
                if not released_session and (is_terminal_completed or is_deferred_final):
                    # Release the session before notifying the client so an immediate
                    # frontend-tool continuation can start on the same socket/session.
                    # Only pop once — a continuation may have already registered a new
                    # task after the deferred final_response early-release.
                    self._active_streams.pop(session_id, None)
                    released_session = True
                await self.emit(socket_event, event, to=sid)
                event_count += 1
                if is_terminal_completed:
                    break
            logger.info(
                "[STREAM] _run_stream completed",
                sid=sid,
                session_id=session_id,
                events_emitted=event_count,
            )
        except asyncio.CancelledError:
            logger.info("[STREAM] _run_stream cancelled", sid=sid, session_id=session_id)
            await self.emit(
                "assistant:error",
                {"type": "cancelled", "message": "Stream cancelled"},
                to=sid,
            )
        except Exception as e:
            logger.exception(
                "[STREAM] _run_stream failed",
                sid=sid,
                session_id=session_id,
                events_emitted=event_count,
                last_event_type=last_event_type,
                last_socket_event=last_socket_event,
                error_type=type(e).__name__,
                error=str(e),
            )
            with contextlib.suppress(Exception):
                await self.emit(
                    "assistant:error",
                    {"type": "internal", "message": f"Stream failed: {e}"},
                    to=sid,
                )
        finally:
            if not released_session:
                self._active_streams.pop(session_id, None)

    def _try_get_streaming_service(self) -> StreamingService | None:
        """Best-effort accessor for tests and early lifecycle states."""
        server = getattr(self, "server", None)
        fastapi_app = getattr(server, "fastapi_app", None)
        if fastapi_app is None:
            return None
        state = getattr(fastapi_app, "state", None)
        if state is None:
            return None
        return getattr(state, "streaming_service", None)
