"""Socket.IO server — real-time transport layer for assistant streaming.

The AssistantNamespace consumes StreamingService's async generator and emits
events directly to the requesting client (to=sid).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

import socketio
from loguru import logger

from assistant_runtime.app.access.exceptions import AccessDeniedError, AuthenticationError
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.host_context import host_context_from_payload
from assistant_runtime.principal import Credentials, Principal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from assistant_runtime.app.streaming.interface import StreamingService

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


def socket_event_name(event: dict[str, Any]) -> str:
    """The Socket.IO event name for a stream event dict."""
    event_type = event.get("type", "")
    name = _EVENT_TYPE_MAP.get(event_type)
    if name is not None:
        return name
    return "assistant:debug" if event_type.startswith("debug_") else "assistant:unknown"


def create_sio() -> socketio.AsyncServer:
    """Create and configure the Socket.IO async server."""
    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
    sio.register_namespace(AssistantNamespace("/assistant"))
    return sio


@dataclass
class _StreamDelivery:
    sid: str
    started: bool = False


class AssistantNamespace(socketio.AsyncNamespace):
    """Socket.IO namespace for assistant streaming.

    Handles message streaming and cancellation. Events are emitted directly
    to the requesting client via to=sid (1:1 model, matching the backbone's
    terminal namespace pattern).
    """

    def __init__(self, namespace: str = "/assistant") -> None:
        super().__init__(namespace)
        self._active_streams: dict[str, asyncio.Task[None]] = {}
        self._stream_delivery: dict[asyncio.Task[None], _StreamDelivery] = {}
        # Running handlers and their waiters retain their lock; idle sessions
        # need no permanent namespace entry.
        self._session_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        # The principal established at connect time, per socket.
        self._principals: dict[str, Principal] = {}

    @property
    def _streaming_service(self) -> StreamingService:
        return self.server.fastapi_app.state.streaming_service

    async def on_connect(self, sid: str, environ: dict, auth: Any = None) -> bool:
        """Authenticate the connection; a rejected caller is never joined to anything."""
        access = getattr(self.server.fastapi_app.state, "access_service", None)
        if access is None:
            logger.warning("Socket.IO connection refused: access service missing", sid=sid)
            return False
        try:
            principal = await access.authenticate(Credentials.from_environ(environ, auth))
        except AuthenticationError as e:
            logger.info("Socket.IO connection refused", sid=sid, reason=str(e))
            return False
        self._principals[sid] = principal
        logger.info(
            "Socket.IO client connected",
            sid=sid,
            namespace=self.namespace,
            principal=principal.id,
        )
        return True

    async def on_disconnect(self, sid: str) -> None:
        self._principals.pop(sid, None)
        logger.info("Socket.IO client disconnected", sid=sid, namespace=self.namespace)

    def _principal(self, sid: str) -> Principal | None:
        return self._principals.get(sid)

    async def _forbid(self, sid: str, exc: Exception) -> None:
        await self.emit("assistant:error", {"type": "forbidden", "message": str(exc)}, to=sid)

    async def on_assistant_join_session(self, sid: str, data: dict[str, Any]) -> None:
        """Client joins a session room for receiving events."""
        session_id = data.get("session_id") if isinstance(data, dict) else None
        if not session_id:
            await self.emit(
                "assistant:error", {"type": "validation", "message": "Missing session_id"}, to=sid
            )
            return
        try:
            host_context = host_context_from_payload(data)
        except ValueError as e:
            await self.emit(
                "assistant:error",
                {"type": "validation", "message": f"Invalid host_context: {e}"},
                to=sid,
            )
            return
        principal = self._principal(sid)
        streaming_service = self._try_get_streaming_service()
        if streaming_service is not None:
            warm = getattr(streaming_service, "warm_session", None)
            if callable(warm):
                try:
                    maybe_awaitable = warm(session_id, host_context, principal=principal)
                    if inspect.isawaitable(maybe_awaitable):
                        await maybe_awaitable
                except AccessDeniedError as e:
                    await self._forbid(sid, e)
                    return
                except Exception as e:
                    logger.warning(
                        "Session warmup failed during join",
                        sid=sid,
                        session_id=session_id,
                        error=str(e),
                    )
        await self.enter_room(sid, f"session:{session_id}")
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

        async with self._session_lock(request.session_id):
            await self._start_request(sid, request)

    async def _start_request(self, sid: str, request: AssistantRequest) -> None:
        """Start a request while its session's transport ownership is locked."""
        session_id = request.session_id

        if request.is_steering:
            active_task = self._active_streams.get(session_id)
            has_live_stream = active_task is not None and not active_task.done()
            try:
                action = await self._streaming_service.accept_steering(
                    request,
                    has_live_stream=has_live_stream,
                    principal=self._principal(sid),
                )
            except AccessDeniedError as e:
                await self._forbid(sid, e)
                return
            except Exception as e:
                logger.error(
                    "Steering queueing failed",
                    sid=sid,
                    session_id=session_id,
                    error=str(e),
                )
                await self.emit(
                    "assistant:error",
                    {"type": "session", "message": str(e)},
                    to=sid,
                )
            else:
                if action == "promoted":
                    self._start_stream(sid, request)
            return

        # Let native cancellation persist the previous turn before replacing
        # its transport. A continuation waits for its producer centrally and
        # must not cancel the stream that dispatched the host tool.
        active_task = self._active_streams.get(session_id)
        if active_task is not None:
            if active_task.done():
                self._release_stream(session_id, active_task)
            elif not request.is_continuation:
                try:
                    await self._streaming_service.cancel_session(
                        session_id, principal=self._principal(sid)
                    )
                except AccessDeniedError as e:
                    await self._forbid(sid, e)
                    return
                await self._streaming_service.wait_for_session(session_id)
                # Its producer is drained, but socket emission may still be
                # blocked. Finish that consumer before exposing a replacement.
                if not active_task.done():
                    active_task.cancel()
                await asyncio.gather(active_task, return_exceptions=True)
                self._release_stream(session_id, active_task)
                logger.info(
                    "Cancelled stale stream for new message",
                    sid=sid,
                    session_id=session_id,
                )

        self._start_stream(sid, request)

    async def on_assistant_cancel(self, sid: str, data: dict[str, Any]) -> None:
        """Cancel an active stream for a session."""
        session_id = data.get("session_id") if isinstance(data, dict) else None
        if not session_id:
            return

        async with self._session_lock(session_id):
            task = self._active_streams.get(session_id)
            delivery = self._stream_delivery.get(task)
            try:
                cancelled = await self._streaming_service.cancel_session(
                    session_id, principal=self._principal(sid)
                )
            except AccessDeniedError as e:
                await self._forbid(sid, e)
                return
            if delivery is not None and not delivery.started:
                # The consumer may not have registered a producer yet, or may
                # still be waiting for a predecessor. Do not let that scheduled
                # turn start after cancellation. An already registered producer
                # is cancelled and drained by stream_message's protected cleanup.
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await self._streaming_service.wait_for_session(session_id)
                # The consumer was stopped before it could forward the native
                # terminal envelope (or before a producer existed at all).
                # Close the accepted transport lifecycle after cleanup, even
                # when cancellation made its producer finish without events.
                self._start_transport(
                    delivery.sid, session_id, self._cancelled_before_start_events(session_id)
                )
                cancelled = True
            if cancelled:
                logger.info("Stream cancelled by client", sid=sid, session_id=session_id)

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    def _start_stream(self, sid: str, request: AssistantRequest) -> None:
        self._start_transport(
            sid,
            request.session_id,
            self._streaming_service.stream_message(request, principal=self._principal(sid)),
        )

    async def _cancelled_before_start_events(
        self, session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        for event in self._streaming_service.cancelled_before_start_events(session_id):
            yield event

    def _start_transport(
        self, sid: str, session_id: str, events: AsyncIterator[dict[str, Any]]
    ) -> None:
        task = asyncio.create_task(self._run_stream(sid, session_id, events))
        self._active_streams[session_id] = task
        self._stream_delivery[task] = _StreamDelivery(sid)

        def cleanup(finished):
            self._stream_delivery.pop(finished, None)
            self._release_stream(session_id, finished)

        task.add_done_callback(cleanup)

    def _release_stream(self, session_id: str, task: asyncio.Task | None) -> None:
        if self._active_streams.get(session_id) is task:
            self._active_streams.pop(session_id, None)

    async def _run_stream(
        self, sid: str, session_id: str, stream: AsyncIterator[dict[str, Any]]
    ) -> None:
        """Consume StreamingService generator and emit events to the client."""
        logger.info("[STREAM] _run_stream started", sid=sid, session_id=session_id)
        event_count = 0
        last_event_type: str | None = None
        last_socket_event: str | None = None
        task = asyncio.current_task()
        try:
            async with contextlib.aclosing(stream) as events:
                async for event in events:
                    event_type = event.get("type", "")
                    last_event_type = event_type
                    socket_event = socket_event_name(event)
                    last_socket_event = socket_event
                    is_terminal_completed = (
                        event_type == "agent_status" and event.get("status") == "completed"
                    )
                    is_deferred_final = (
                        event_type == "final_response"
                        and event.get("pending_tool_call") is not None
                    )
                    if is_terminal_completed or is_deferred_final:
                        # Release the session before notifying the client so an immediate
                        # host-tool continuation can start on the same socket/session.
                        # A continuation may already own the session.
                        self._release_stream(session_id, task)
                    if event_type == "agent_status" and event.get("status") == "started":
                        # Claim this emission before awaiting the transport: a
                        # blocked emit may already have delivered to the client.
                        # Earlier debug events do not start the lifecycle.
                        delivery = self._stream_delivery.get(task)
                        if delivery is not None:
                            delivery.started = True
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
            # Cancellation outcomes come from the runtime's native run control.
            # A transport being replaced may itself be blocked in socket emit.
            raise
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
            self._release_stream(session_id, task)

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
