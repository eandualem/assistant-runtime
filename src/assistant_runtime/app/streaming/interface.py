"""StreamingService — runs assistant turns and streams them as event dicts.

Every request kind (new message, host-tool continuation, promoted steering)
goes through the same pipeline: ``TurnPlanner`` validates the request against
the session and describes the run, ``TurnRunner`` executes it. Callers that
want one answer instead of a stream use ``run_message``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai.exceptions import RunCancelled

from assistant_runtime.app.access.exceptions import AccessDeniedError
from assistant_runtime.app.access.interface import AccessService
from assistant_runtime.app.assistant.exceptions import AgentRunError, SessionError
from assistant_runtime.app.assistant.models import AssistantRequest, AssistantResult
from assistant_runtime.app.streaming._control import TurnControl
from assistant_runtime.app.streaming._event_builder import (
    make_agent_status_event,
    make_debug_error_event,
    make_error_event,
    make_final_response_event,
    make_voice_event,
)
from assistant_runtime.app.streaming._host_tool import clear_stale_pending_call
from assistant_runtime.app.streaming._runner import TurnRunner, format_error_message
from assistant_runtime.app.streaming._turn import TurnPlanner
from assistant_runtime.app.streaming.config import StreamingConfig
from assistant_runtime.app.streaming.exceptions import StreamingError, StreamSetupError
from assistant_runtime.principal import LOCAL_PRINCIPAL, Principal
from assistant_runtime.services.artifacts.exceptions import UnknownProfileError

if TYPE_CHECKING:
    from assistant_runtime.app.assistant import SessionStore
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
        self._active_turns: dict[str, TurnControl] = {}
        self._session_leases: dict[str, str] = {}
        self._session_mutations: set[str] = set()
        # Work a turn leaves behind after its terminal event (working memory);
        # ordered per session so a later turn's extraction never lands first.
        self._background: set[asyncio.Task[Any]] = set()
        self._background_chains: dict[str, asyncio.Task[Any]] = {}

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
            background=self._spawn_background,
        )

    def _spawn_background(self, coroutine: Any, *, session_id: str | None = None) -> None:
        """Run post-turn work (working memory) without delaying the terminal event.

        Work for the same session runs in submission order: the extraction of
        turn 2 waits for turn 1's, so the older result cannot overwrite the newer.
        """
        previous = self._background_chains.get(session_id) if session_id else None

        async def _ordered() -> None:
            if previous is not None:
                await asyncio.gather(previous, return_exceptions=True)
            await coroutine

        task = asyncio.create_task(_ordered())
        self._background.add(task)
        if session_id is not None:
            self._background_chains[session_id] = task

        def _done(finished: asyncio.Task[Any]) -> None:
            self._background.discard(finished)
            if session_id is not None and self._background_chains.get(session_id) is finished:
                self._background_chains.pop(session_id, None)

        task.add_done_callback(_done)

    async def wait_for_background(self) -> None:
        """Wait for post-turn work to finish (tests, shutdown)."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    def attach_ingress(self, ingress: Any | None) -> None:
        """Attach the ingress service; its queue is drained into each new message turn."""
        self._ingress = ingress

    async def start(self) -> None:
        self._started = True
        logger.info("Streaming service started")

    async def stop(self) -> None:
        self._started = False
        turns = list(self._active_turns.values())
        for turn in turns:
            turn.cancel()
        await asyncio.gather(*(turn.done.wait() for turn in turns))
        await self.wait_for_background()
        logger.info("Streaming service stopped")

    async def cancel_session(
        self,
        session_id: str,
        *,
        principal: Principal | None = None,
        session_lease: str | None = None,
    ) -> bool:
        """Request cancellation of the active turn; idle host actions are unchanged.

        Returns whether an active, cancellable turn was found. Its producer
        saves the native snapshot and emits its terminal events. Use
        ``wait_for_session`` when the caller needs finalization to finish.
        ``principal`` must own the session (``AccessDeniedError`` otherwise);
        in-process callers that pass none act as the local operator.
        """
        await self._authorize(session_id, principal)
        self.check_session_available(session_id, session_lease)
        turn = self._active_turns.get(session_id)
        return turn.cancel() if turn is not None else False

    def check_session_available(self, session_id: str, session_lease: str | None = None) -> None:
        """An active voice transport owns this session until it releases its lease."""
        if session_id in self._session_mutations:
            raise SessionError("Session administration is in progress")
        lease = self._session_leases.get(session_id)
        if lease is not None and lease != session_lease:
            raise SessionError("Session is reserved by a voice call; use the voice call endpoints")

    @contextmanager
    def session_mutation(self, session_id: str):
        """Exclude voice reservation throughout an asynchronous session mutation."""
        self.check_session_available(session_id)
        self._session_mutations.add(session_id)
        try:
            yield
        finally:
            self._session_mutations.discard(session_id)

    async def authorize_session(self, session_id: str, principal: Principal) -> None:
        """Check current session ownership without starting or cancelling a turn."""
        if await self._sessions.get_context_if_exists_async(session_id) is None:
            raise SessionError("Session not found")
        await self._authorize(session_id, principal)

    async def reserve_session(
        self, session_id: str, lease: str, principal: Principal
    ) -> list[dict]:
        """Reserve an idle session and return client-visible text history only."""
        await self._authorize(session_id, principal)
        self.check_session_available(session_id)
        if session_id in self._active_turns:
            raise SessionError("A turn is already running in this session")
        self._session_leases[session_id] = lease
        # The reserved session must stay cached for the call's lifetime.
        self._sessions.pin(session_id)
        try:
            ctx = await self._sessions.ensure_owned_session(session_id, principal.id)
            if ctx.get("pending_tool_call_id"):
                raise SessionError("Resolve the pending host action before starting voice")
            path = await self._sessions.get_message_path(session_id)
            return [{"role": m["role"], "content": m["content"]} for m in path if m["content"]]
        except BaseException:
            self.release_session(session_id, lease)
            raise

    def release_session(self, session_id: str, lease: str) -> None:
        if self._session_leases.get(session_id) == lease:
            self._session_leases.pop(session_id)
            self._sessions.unpin(session_id)

    async def cancel_reserved_work(self, session_id: str, lease: str, principal: Principal) -> None:
        """Drain a transport-owned turn and resolve pending host calls as interrupted."""
        await self.cancel_session(session_id, principal=principal, session_lease=lease)
        await self.wait_for_session(session_id)
        ctx = await self._sessions.get_context_if_exists_async(session_id)
        if ctx is not None:
            await clear_stale_pending_call(self._sessions, session_id, ctx, status="cancelled")

    @staticmethod
    def voice_event(call_id: str, event: str, data: dict) -> dict:
        """Build a voice lifecycle envelope through the common wire-event builder."""
        return make_voice_event(call_id, event, data)

    async def _authorize(self, session_id: str, principal: Principal | None) -> None:
        """Deny a principal that may not act on an existing session."""
        actor = principal or LOCAL_PRINCIPAL
        if actor.is_admin:
            return
        context = await self._sessions.get_context_if_exists_async(session_id)
        if context is not None:
            AccessService.check_session(actor, context.get("owner_id"), session_id)

    @staticmethod
    def cancelled_before_start_events(session_id: str) -> list[dict[str, Any]]:
        """Close a transport cancelled before it forwarded the turn lifecycle.

        This minimal envelope makes no claim about an accepted message or
        saved snapshot. Drain the producer before using it; a running stream
        normally forwards its own cancellation outcome instead.
        """
        return [
            make_agent_status_event("started"),
            make_final_response_event(
                None,
                "unknown",
                session_id=session_id,
                error=True,
                error_type="cancelled",
            ),
            make_error_event(
                "Request cancelled before stream delivery started",
                error_type="cancelled",
                terminal=True,
                retry_allowed=False,
            ),
            make_agent_status_event("completed"),
        ]

    async def wait_for_session(self, session_id: str) -> None:
        """Wait for the currently registered turn's persistence and cleanup."""
        turn = self._active_turns.get(session_id)
        if turn is not None:
            await turn.done.wait()

    async def health_check(self) -> dict:
        return {"healthy": self._started}

    async def cleanup_expired_traces(self) -> int:
        """Delete debug traces older than ``trace_retention_hours``; 0 without a database."""
        if self._db is None or not getattr(self._db, "healthy", False):
            return 0
        from assistant_runtime.services.database.repositories import TraceRepository

        cutoff = datetime.now(UTC) - timedelta(hours=self._config.trace_retention_hours)
        async with self._db.session_context() as db_session:
            return await TraceRepository(db_session).delete_older_than(cutoff)

    async def warm_session(
        self,
        session_id: str,
        host_context: dict[str, Any] | None = None,
        *,
        principal: Principal | None = None,
    ) -> None:
        """Warm shared request-path state for a joined session."""
        if not self._started:
            return
        await self._authorize(session_id, principal)
        await self._assistant_service.warm_session(session_id, host_context)

    def validate_profile(self, name: str | None) -> None:
        """Validate a registered profile without starting work or allocating a call."""
        self._assistant_service.validate_profile(name)

    async def accept_steering(
        self,
        request: AssistantRequest,
        *,
        has_live_stream: bool,
        principal: Principal | None = None,
    ) -> str:
        """Queue steering behind a live stream, or promote it to run now: ``queued``/``promoted``."""
        if not request.is_steering:
            raise SessionError("Only steering requests can be accepted")
        try:
            self.validate_profile(request.profile)
        except UnknownProfileError as exc:
            raise SessionError(str(exc)) from exc
        # A voice call owns its session: ordinary steering must not reach it.
        self.check_session_available(request.session_id)
        session_context = await self._sessions.get_context_if_exists_async(request.session_id)
        if session_context is None:
            raise SessionError(f"Steering rejected: session '{request.session_id}' does not exist")
        await self._authorize(request.session_id, principal)

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
        # Promotion chooses when to run; delivery is acknowledged only after
        # a native model request consumes this still-pending instruction.
        await self._sessions.queue_steering(request.session_id, request)
        return "promoted"

    async def stream_message(
        self,
        request: AssistantRequest,
        *,
        principal: Principal | None = None,
        native_sink: Callable[[Any], None] | None = None,
        session_lease: str | None = None,
        cancel_after_plan: Callable[[], bool] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one turn and yield events through ``agent_status: completed``.

        ``principal`` owns a session the request creates and must be allowed
        on one that exists; without one the caller is the local operator.
        ``native_sink`` receives every native Pydantic AI event of the turn as
        it happens, so a protocol adapter can encode the same run without a
        second execution path.

        A producer owns execution and persistence so a slow or disappearing
        consumer cannot interrupt finalization. Closing/cancelling this
        iterator requests native cancellation and waits for cleanup; external
        ``CancelledError`` still propagates to the caller. A new ordinary turn
        replaces the previous turn only after it has drained. Continuations
        and promoted steering wait without cancelling it.
        """
        if not self._started:
            raise StreamingError("Streaming service not started")
        try:
            self.validate_profile(request.profile)
        except UnknownProfileError as exc:
            raise SessionError(str(exc)) from exc
        self.check_session_available(request.session_id, session_lease)
        while (previous := self._active_turns.get(request.session_id)) is not None:
            if not request.is_continuation and not request.is_steering:
                previous.cancel()
            await previous.done.wait()
            self.check_session_available(request.session_id, session_lease)
            if not self._started:
                raise StreamingError("Streaming service not started")
        turn = TurnControl(native_sink=native_sink, cancel_after_plan=cancel_after_plan)
        self._active_turns[request.session_id] = turn
        self._sessions.pin(request.session_id)
        turn.task = asyncio.create_task(
            self._produce_turn(request, turn, principal or LOCAL_PRINCIPAL)
        )
        try:
            while (event := await turn.events.get()) is not None:
                yield event
            if turn.error is not None:
                raise turn.error
        finally:
            turn.cancel()
            # Consumer cancellation must not cancel the native snapshot writer.
            await asyncio.shield(turn.task)

    async def _produce_turn(
        self, request: AssistantRequest, turn: TurnControl, principal: Principal
    ) -> None:
        """Drive the one turn pipeline independently of transport backpressure."""
        try:
            async with aclosing(self._stream_turn(request, turn, principal)) as stream:
                async for event in stream:
                    turn.events.put_nowait(event)
        except BaseException as exc:
            turn.error = exc
        finally:
            if self._active_turns.get(request.session_id) is turn:
                self._active_turns.pop(request.session_id, None)
                self._sessions.unpin(request.session_id)
            turn.accepting_cancel = False
            turn.done.set()
            turn.events.put_nowait(None)

    async def _stream_turn(
        self, request: AssistantRequest, turn: TurnControl, principal: Principal
    ) -> AsyncIterator[dict[str, Any]]:
        """Plan and execute an accepted turn through the shared runner."""
        try:
            (plan,) = await turn.prepare(TurnPlanner(self._sessions).plan(request, principal))
        except (StreamSetupError, SessionError, AccessDeniedError, RunCancelled) as e:
            # Nothing was emitted yet: send a minimal lifecycle envelope so the
            # client can leave its "thinking" state.
            async for event in self._setup_failure_envelope(request, e):
                yield event
            return
        # Once admitted, the runner saves accepted host results even when the
        # native token is cancelled. Never cancel result validation halfway.
        if turn.cancel_after_plan is not None and turn.cancel_after_plan():
            turn.cancel()
        if self._ingress is not None and plan.kind == "message":
            # The session exists now; waiting messages ride along as steering.
            try:
                await turn.prepare(self._ingress.drain(request.session_id))
            except RunCancelled:
                # The accepted plan still needs its saved cancellation envelope.
                pass
            except Exception as e:
                logger.warning("Inbox drain failed", session_id=request.session_id, error=str(e))
        async for event in self._runner.run(plan, control=turn):
            yield event

    async def run_message(
        self, request: AssistantRequest, *, principal: Principal | None = None
    ) -> AssistantResult:
        """Run one turn and return the final answer (the non-streaming form).

        Raises:
            SessionError: The request does not fit the session (unknown parent, ...).
            AccessDeniedError: The principal may not act on the session.
            AgentRunError: The run failed.
        """
        text: list[str] = []
        final: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        async for event in self.stream_message(request, principal=principal):
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
            if (final or {}).get("error_type") == "forbidden":
                raise AccessDeniedError(message)
            raise AgentRunError(message)
        turn_number = self._sessions.get_context(request.session_id).get("turn_number", 0)
        return AssistantResult(
            content=None if final.get("decision") else final.get("content") or "".join(text),
            decision=final.get("decision"),
            model=final["model"],
            session_id=request.session_id,
            turn_number=turn_number,
            message_id=final.get("message_id"),
            pending_tool_call=final.get("pending_tool_call"),
            usage=final.get("usage"),
        )

    async def _setup_failure_envelope(
        self, request: AssistantRequest, exc: Exception
    ) -> AsyncIterator[dict[str, Any]]:
        is_session_error = isinstance(exc, SessionError | AccessDeniedError)
        is_cancelled = isinstance(exc, RunCancelled)
        error_type = (
            "cancelled"
            if is_cancelled
            else "forbidden"
            if isinstance(exc, AccessDeniedError)
            else "session_error"
            if is_session_error
            else "setup_error"
        )
        (logger.warning if is_session_error else logger.error)(
            "[STREAM] Setup failed — emitting minimal lifecycle envelope",
            session_id=request.session_id,
            error_type=error_type,
            error=str(exc),
        )
        retry_allowed = not is_session_error and not is_cancelled
        trace_id = str(uuid.uuid4())
        if is_cancelled:
            message = "Request cancelled before turn acceptance"
        elif is_session_error or self._config.client_error_detail:
            # A session or access error describes the client's own request.
            message = f"Setup failed: {exc}"
        else:
            message = "Setup failed"
        # A denied request leaves nothing under the session it could not reach.
        if not isinstance(exc, AccessDeniedError):
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
