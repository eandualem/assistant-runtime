"""StreamingService — runs assistant turns and streams them as event dicts.

Every request kind (new message, host-tool continuation, promoted steering)
goes through the same pipeline: ``TurnPlanner`` validates the request against
the session and describes the run, ``TurnRunner`` executes it. Callers that
want one answer instead of a stream use ``run_message``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from assistant_runtime.app.assistant.exceptions import AgentRunError, SessionError
from assistant_runtime.app.assistant.models import AssistantRequest, AssistantResult
from assistant_runtime.app.streaming._event_builder import (
    make_agent_status_event,
    make_debug_error_event,
    make_error_event,
    make_final_response_event,
)
from assistant_runtime.app.streaming._runner import TurnRunner, format_error_message
from assistant_runtime.app.streaming._turn import TurnPlanner
from assistant_runtime.app.streaming.config import StreamingConfig
from assistant_runtime.app.streaming.exceptions import StreamingError, StreamSetupError

if TYPE_CHECKING:
    from assistant_runtime.app.assistant._session_store import SessionStore
    from assistant_runtime.app.assistant.interface import AssistantService
    from assistant_runtime.services.database.interface import DatabaseService
    from assistant_runtime.services.history.interface import HistoryService
    from assistant_runtime.services.tools.interface import ToolService


class StreamingService:
    """Turn pipeline facade. Implements LifecycleAware."""

    def __init__(
        self,
        config: StreamingConfig,
        history_service: HistoryService,
        tool_service: ToolService,
        assistant_service: AssistantService,
        database_service: DatabaseService | None = None,
    ) -> None:
        self._config = config
        self._history = history_service
        self._tools = tool_service
        self._assistant_service = assistant_service
        self._db = database_service
        self._ingress: Any | None = None
        self._started = False

    @property
    def _sessions(self) -> SessionStore:
        sessions = self._assistant_service.get_session_store()
        if sessions is None:
            raise StreamSetupError("Assistant service not started — no session store")
        return sessions

    @property
    def _runner(self) -> TurnRunner:
        return TurnRunner(
            config=self._config,
            sessions=self._sessions,
            tools=self._tools,
            history=self._history,
            assistant_service=self._assistant_service,
            database_service=self._db,
        )

    def attach_ingress(self, ingress: Any | None) -> None:
        """Attach the ingress service; its queue is drained into each new message turn."""
        self._ingress = ingress

    async def start(self) -> None:
        self._started = True
        logger.info("Streaming service started")

    async def stop(self) -> None:
        self._started = False
        logger.info("Streaming service stopped")

    async def health_check(self) -> dict:
        return {"healthy": self._started}

    async def warm_session(
        self, session_id: str, host_context: dict[str, Any] | None = None
    ) -> None:
        """Warm shared request-path state for a joined session."""
        if not self._started:
            return
        await self._assistant_service.warm_session(session_id, host_context)

    async def accept_steering(self, request: AssistantRequest, *, has_live_stream: bool) -> str:
        """Queue steering behind a live stream, or promote it to run now: ``queued``/``promoted``."""
        if not request.is_steering:
            raise SessionError("Only steering requests can be accepted")
        session_context = await self._sessions.get_context_if_exists_async(request.session_id)
        if session_context is None:
            raise SessionError(f"Steering rejected: session '{request.session_id}' does not exist")

        if has_live_stream or session_context.get("pending_tool_call_id"):
            await self._sessions.queue_steering(request.session_id, request)
            return "queued"

        active_leaf_id = session_context.get("active_leaf_id")
        if active_leaf_id is None or session_context["message_index"].get(active_leaf_id) is None:
            raise SessionError(
                f"Steering rejected: session '{request.session_id}' has no active conversation"
            )
        turn_in_flight = (
            session_context.get("current_assistant_message_id") is not None
            or session_context.get("pending_assistant_message_id") is not None
        )
        if turn_in_flight:
            await self._sessions.queue_steering(request.session_id, request)
            return "queued"
        await self._sessions.queue_steering(
            request.session_id, request, status="promoted", delivered_at=datetime.now(UTC)
        )
        return "promoted"

    async def stream_message(self, request: AssistantRequest) -> AsyncIterator[dict[str, Any]]:
        """Run one turn and yield its events; the last one is ``agent_status: completed``.

        Raises:
            StreamingError: If the service is not started.
        """
        if not self._started:
            raise StreamingError("Streaming service not started")
        try:
            plan = await TurnPlanner(self._sessions).plan(request)
        except (StreamSetupError, SessionError) as e:
            # Nothing was emitted yet: send a minimal lifecycle envelope so the
            # client can leave its "thinking" state.
            async for event in self._setup_failure_envelope(request, e):
                yield event
            return
        if self._ingress is not None and plan.kind == "message":
            # The session exists now; waiting messages ride along as steering.
            try:
                await self._ingress.drain(request.session_id)
            except Exception as e:
                logger.warning("Inbox drain failed", session_id=request.session_id, error=str(e))
        async for event in self._runner.run(plan):
            yield event

    async def run_message(self, request: AssistantRequest) -> AssistantResult:
        """Run one turn and return the final answer (the non-streaming form).

        Raises:
            SessionError: The request does not fit the session (unknown parent, ...).
            AgentRunError: The run failed.
        """
        text: list[str] = []
        final: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        async for event in self.stream_message(request):
            kind = event.get("type")
            if kind == "text_delta":
                text.append(event.get("content") or "")
            elif kind == "final_response":
                final = event
            elif kind == "error" and event.get("terminal"):
                error = event
        if final is None or final.get("error"):
            message = (error or {}).get("message") or "Agent execution failed"
            if (final or {}).get("error_type") == "session_error":
                raise SessionError(message)
            raise AgentRunError(message)
        turn_number = self._sessions.get_context(request.session_id).get("turn_number", 0)
        return AssistantResult(
            content=final.get("content") or "".join(text),
            model=final["model"],
            session_id=request.session_id,
            turn_number=turn_number,
            message_id=final.get("message_id"),
            pending_tool_call=final.get("pending_tool_call"),
        )

    async def _setup_failure_envelope(
        self, request: AssistantRequest, exc: Exception
    ) -> AsyncIterator[dict[str, Any]]:
        is_session_error = isinstance(exc, SessionError)
        error_type = "session_error" if is_session_error else "setup_error"
        (logger.warning if is_session_error else logger.error)(
            "[STREAM] Setup failed — emitting minimal lifecycle envelope",
            session_id=request.session_id,
            error_type=error_type,
            error=str(exc),
        )
        retry_allowed = not is_session_error
        trace_id = str(uuid.uuid4())
        message = f"Setup failed: {exc}"
        await self._runner.save_trace(
            request.session_id,
            [
                make_debug_error_event(
                    message,
                    error_type=error_type,
                    retry_allowed=retry_allowed,
                    trace_id=trace_id,
                    model="unknown",
                    phase="setup",
                )
            ],
            trace_id=trace_id,
            user_message=request.content,
            screenshot=request.images[0] if request.images else None,
        )
        yield make_agent_status_event("started")
        yield make_final_response_event(
            None,
            "unknown",
            session_id=request.session_id,
            trace_id=trace_id,
            error=True,
            error_type=error_type,
        )
        yield make_error_event(
            format_error_message(message, trace_id),
            error_type=error_type,
            trace_id=trace_id,
            terminal=True,
            retry_allowed=retry_allowed,
        )
        yield make_agent_status_event("completed")
