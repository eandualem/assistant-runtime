"""Execute one ``TurnPlan`` as a stream of events.

The shape of every turn is the same:

1. ``agent_status: started``;
2. agent setup (system prompt, tools, model), with ``debug_*`` events
   describing it;
3. the agent run, streamed through ``_agent_run.iterate_run``; the history
   policy runs inside it as a native capability before each model request;
4. the assistant message persisted (created or extended);
5. the output: a text reply, or a deferred host-tool call the host must
   answer; steering queued during the run is delivered as follow-up runs
   on the same assistant message;
6. exactly one ``final_response`` and, whatever happened, exactly one
   ``agent_status: completed``.

Errors after ``started`` become ``final_response(error=True)`` + a terminal
``error`` event; the ``EventCoordinator`` enforces the one-of-each rule.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import DeferredToolRequests, capture_run_messages
from pydantic_ai.exceptions import RunCancelled, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
)
from pydantic_ai.usage import RunUsage

from assistant_runtime.app.access.exceptions import AccessDeniedError
from assistant_runtime.app.assistant.exceptions import SessionError
from assistant_runtime.app.assistant.models import HoldDecision
from assistant_runtime.app.assistant.usage import (
    usage_dict,
    with_auxiliary,
)
from assistant_runtime.app.streaming import _telemetry
from assistant_runtime.app.streaming._agent_run import TurnPolicy, iterate_run
from assistant_runtime.app.streaming._control import TurnControl
from assistant_runtime.app.streaming._coordinator import EventCoordinator
from assistant_runtime.app.streaming._event_builder import (
    make_debug_completed_event,
    make_debug_final_response_event,
    make_debug_request_event,
)
from assistant_runtime.app.streaming._host_tool import pending_call_payloads, queued_payload
from assistant_runtime.app.streaming._snapshot import (
    RunSnapshot,
    _IncompleteRun,
    capture_result,
    mark_unanswered_calls,
    persist_snapshot,
)
from assistant_runtime.app.streaming.exceptions import InvalidDecisionError
from assistant_runtime.services.tools.request_context import (
    assistant_request_context,
    get_current_telegram_chat_binding,
)
from assistant_runtime.services.tracing import create_request_trace

if TYPE_CHECKING:
    from assistant_runtime.app.assistant import SessionStore
    from assistant_runtime.app.assistant.interface import AssistantService
    from assistant_runtime.app.assistant.models import AgentSetupContext
    from assistant_runtime.app.streaming._turn import TurnPlan
    from assistant_runtime.app.streaming.config import StreamingConfig
    from assistant_runtime.services.database.interface import DatabaseService
    from assistant_runtime.services.history.interface import HistoryService
    from assistant_runtime.services.tools.interface import ToolService


class TurnRunner:
    """Run ``TurnPlan``s. One instance per ``StreamingService``."""

    def __init__(
        self,
        *,
        config: StreamingConfig,
        sessions: SessionStore,
        tools: ToolService,
        history: HistoryService,
        assistant_service: AssistantService,
        database_service: DatabaseService | None,
        background: Callable[..., None] | None = None,
    ) -> None:
        self._config = config
        self._sessions = sessions
        self._tools = tools
        self._history = history
        self._assistant = assistant_service
        self._db = database_service
        # Schedules post-turn work; without it the work runs inline before ``completed``.
        self._background = background

    async def run(
        self, plan: TurnPlan, *, control: TurnControl | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute a turn, persisting cancellation before its terminal envelope."""
        control = control or TurnControl()
        coordinator = EventCoordinator(self._config.max_events_per_stream)
        emit_debug = self._config.emit_debug_events and plan.request.output_mode != "host_tools"
        started_at = time.monotonic()
        trace_id = str(uuid.uuid4())
        request = plan.request
        session_id = plan.session_id
        session_context = plan.session_context
        state = RunSnapshot(
            assistant_messages=list(plan.prior_assistant_messages), usage=plan.prior_usage
        )
        resolved_model = "unknown"
        trace_cm: Any = None
        ctx: AgentSetupContext | None = None
        phase = "setup"
        started = coordinator.try_started()
        if started:
            yield started

        try:
            # A host has already performed this action. Save its accepted result
            # before dependencies/model setup can be cancelled, using the same row.
            if plan.accepted_tool_result is not None:
                state.assistant_messages.append(plan.accepted_tool_result)
                await persist_snapshot(self._sessions, plan, state)
                await self._clear_pending(plan)
            if plan.next_pending is not None:
                # More host calls from the same response wait; hand the next one
                # over and let the model resume once all results are in.
                await self._sessions.set_pending_action(
                    session_id,
                    tool_call_id=plan.next_pending["call_id"],
                    tool_name=plan.next_pending["tool_name"],
                    assistant_message_id=plan.assistant_message_id,
                    batch=plan.pending_batch,
                    output_mode=request.output_mode,
                )
                state.pending_tool_call = plan.next_pending
                control.accepting_cancel = False
                final = coordinator.try_final_response(
                    None,
                    resolved_model,
                    session_id=session_id,
                    message_id=plan.assistant_message_id,
                    usage=state.stored_usage or state.usage,
                    pending_tool_call=plan.next_pending,
                    decision="pending" if request.output_mode == "host_tools" else None,
                )
                if final:
                    yield final
                return
            if plan.receipt_only:
                control.accepting_cancel = False
                final = coordinator.try_final_response(
                    None,
                    resolved_model,
                    session_id=session_id,
                    message_id=plan.assistant_message_id,
                    usage=state.stored_usage or state.usage,
                    decision="completed",
                )
                if final:
                    yield final
                return
            control.check_cancelled()
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_request_event(
                        session_id=session_id,
                        message=plan.input_message,
                        is_continuation=plan.kind == "continuation",
                        has_host_context=request.host_context is not None,
                        host_context=request.host_context,
                        image_count=len(request.images),
                    )
                )
            [(ctx, agent_setup_ms)] = await control.prepare(
                _timed(lambda: self._assistant.prepare_agent_context(request, session_context)),
            )
            control.check_cancelled()
            state.prompt_record, state.prompt_snapshot = ctx.prompt_record, ctx.prompt_snapshot
            resolved_model = ctx.resolved_model
            state.history = (
                None
                if request.output_mode == "host_tools"
                else self._history.processor(session_context, collect_debug=emit_debug)
            )
            if emit_debug:
                for event in _telemetry._setup_debug_events(ctx, session_id):
                    yield coordinator.track_debug(event)
            logger.info(
                "[STREAM] Turn setup ready",
                kind=plan.kind,
                session_id=session_id,
                model=resolved_model,
                agent_setup_ms=agent_setup_ms,
                pre_stream_ms=(time.monotonic() - started_at) * 1000,
            )
            trace_cm = create_request_trace(
                session_id=session_id,
                model=resolved_model,
                is_continuation=plan.kind == "continuation",
                input_message=plan.input_message,
                metadata=plan.trace_metadata or None,
                set_current_observation=False,
            )
            trace_cm.__enter__()
            phase = "stream"
            async for event in self._run_agent(
                plan,
                ctx,
                coordinator,
                state,
                control=control,
                user_prompt=plan.user_prompt,
                message_history=plan.history,
                deferred_tool_results=plan.deferred_tool_results,
                usage=RunUsage(),
            ):
                yield event
            if emit_debug:
                history_event = _telemetry._history_debug_event(state)
                if history_event is not None:
                    yield coordinator.track_debug(history_event)
                yield coordinator.track_debug(_telemetry._usage_debug_event(state.run_usage))
            control.check_cancelled()
            await persist_snapshot(self._sessions, plan, state)
            await self._clear_pending(plan)
            async for event in self._handle_output(plan, ctx, coordinator, state, control=control):
                yield event
            control.check_cancelled()
            control.accepting_cancel = False
            final = coordinator.try_final_response(
                None if state.pending_tool_call is not None else state.final_output,
                resolved_model,
                session_id=session_id,
                message_id=plan.assistant_message_id,
                usage=state.stored_usage or state.usage,
                pending_tool_call=state.pending_tool_call,
                decision=("pending" if state.pending_tool_call else "hold")
                if request.output_mode == "host_tools"
                else None,
            )
            if final:
                yield final
        except (RunCancelled, TimeoutError) as cancellation:
            state.cancelled = True
            control.accepting_cancel = False
            try:
                await self._persist_cancelled(plan, state)
            except Exception as exc:
                logger.exception(
                    "Cancelled turn snapshot could not be saved", session_id=session_id
                )
                events = _telemetry._outcome_events(
                    coordinator,
                    session_id=session_id,
                    message=_telemetry._persistence_failure(
                        "Cancelled turn", exc, detail=self._config.client_error_detail
                    ),
                    error_type="persistence_error",
                    retry_allowed=False,
                    trace_id=trace_id,
                    model=resolved_model,
                    phase=phase,
                    emit_debug=emit_debug,
                )
            else:
                if isinstance(cancellation, TimeoutError):
                    events = _telemetry._outcome_events(
                        coordinator,
                        session_id=session_id,
                        message=f"Request timed out after {self._config.stream_timeout_seconds}s",
                        error_type="timeout",
                        retry_allowed=True,
                        trace_id=trace_id,
                        model=resolved_model,
                        phase=phase,
                        emit_debug=emit_debug,
                    )
                else:
                    events = _telemetry._outcome_events(
                        coordinator,
                        session_id=session_id,
                        message="Request cancelled",
                        error_type="cancelled",
                        retry_allowed=False,
                        trace_id=trace_id,
                        model=resolved_model,
                        phase=phase,
                        emit_debug=emit_debug,
                        message_id=plan.assistant_message_id,
                        content="",
                        usage=state.usage,
                    )
            for event in events:
                yield event
        except UsageLimitExceeded as exc:
            # Native limits stop the run before the request that would exceed
            # them; the work done so far was captured by _run_agent.
            control.accepting_cancel = False
            try:
                await self._persist_cancelled(plan, state, interrupted=False)
            except Exception as persist_exc:
                logger.exception("Usage-limited turn could not be saved", session_id=session_id)
                events = _telemetry._outcome_events(
                    coordinator,
                    session_id=session_id,
                    message=_telemetry._persistence_failure(
                        "Usage-limited turn", persist_exc, detail=self._config.client_error_detail
                    ),
                    error_type="persistence_error",
                    retry_allowed=False,
                    trace_id=trace_id,
                    model=resolved_model,
                    phase=phase,
                    emit_debug=emit_debug,
                )
            else:
                state.cancelled = False
                logger.info(
                    "[STREAM] Turn stopped by usage limit", session_id=session_id, error=str(exc)
                )
                events = _telemetry._outcome_events(
                    coordinator,
                    session_id=session_id,
                    message=str(exc),
                    error_type="usage_limit",
                    retry_allowed=False,
                    trace_id=trace_id,
                    model=resolved_model,
                    phase=phase,
                    emit_debug=emit_debug,
                    message_id=plan.assistant_message_id,
                    content="",
                    usage=state.stored_usage,
                )
            for event in events:
                yield event
        except Exception as exc:
            detail = self._config.client_error_detail
            if phase == "setup":
                is_session_error = isinstance(exc, SessionError)
                # A session or access error describes the client's own request; its
                # text stays whatever the detail setting (as in the setup envelope).
                own_request = isinstance(exc, SessionError | AccessDeniedError)
                message = f"Setup failed: {exc}" if detail or own_request else "Setup failed"
                error_type = (
                    "session_error"
                    if is_session_error
                    else "forbidden"
                    if isinstance(exc, AccessDeniedError)
                    else "setup_error"
                )
                retry_allowed = not own_request
            else:
                message, error_type, retry_allowed = _telemetry._describe_error(exc, detail=detail)
            # The log always carries the exception; only the client text is redacted.
            logger.warning(
                "[STREAM] Turn failed",
                session_id=session_id,
                error_type=error_type,
                error=f"{exc.__class__.__name__}: {exc}",
            )
            # Preserve completed effects as well as unfinished calls: retrying a
            # model request must not erase work the tools already performed.
            try:
                await self._resolve_unanswered_calls(plan, state)
            except Exception as persist_exc:
                logger.exception("Failed turn snapshot could not be saved", session_id=session_id)
                message = _telemetry._persistence_failure("Failed turn", persist_exc, detail=detail)
                error_type, retry_allowed = "persistence_error", False
            for event in _telemetry._outcome_events(
                coordinator,
                session_id=session_id,
                message=message,
                error_type=error_type,
                retry_allowed=retry_allowed,
                trace_id=trace_id,
                model=resolved_model,
                phase=phase,
                emit_debug=emit_debug,
                message_id=plan.assistant_message_id if state.persisted else None,
                usage=state.stored_usage if state.persisted else None,
            ):
                yield event
        finally:
            control.accepting_cancel = False
            session_context.pop("current_assistant_message_id", None)
            update_memory = (
                state.persisted
                and not state.cancelled
                and not control.token.cancelled
                and plan.update_working_memory
                and ctx is not None
                and ctx.effective_config.enable_working_memory
            )
            if update_memory and self._background is None:
                await self._update_working_memory(plan, state)
            if emit_debug:
                coordinator.flush_thinking()
                if coordinator.accumulated_response:
                    yield coordinator.track_debug(
                        make_debug_final_response_event(
                            coordinator.accumulated_response,
                            resolved_model,
                            usage=usage_dict(state.run_usage),
                        )
                    )
            duration_ms = (time.monotonic() - started_at) * 1000
            if emit_debug:
                yield coordinator.track_debug(make_debug_completed_event(duration_ms=duration_ms))
            if trace_cm is not None:
                with contextlib.suppress(Exception):
                    trace_cm.__exit__(None, None, None)
            await _telemetry.save_trace(
                self._db,
                session_id,
                coordinator.debug_events,
                trace_id=trace_id,
                user_message=plan.input_message,
                duration_ms=duration_ms,
            )
            completed = coordinator.try_completed()
            if completed:
                yield completed
            if update_memory and self._background is not None:
                # An extra model call; the client already has its answer and
                # the terminal event, so it runs after them. The session stays
                # pinned until it is done so its context is the cached one.
                self._sessions.pin(session_id)
                self._background(
                    self._update_working_memory(plan, state, release_pin=True),
                    session_id=session_id,
                )

    async def _update_working_memory(
        self, plan: TurnPlan, state: RunSnapshot, *, release_pin: bool = False
    ) -> None:
        """Extract the working-memory delta and account its usage on the assistant row."""
        try:
            memory_usage = await self._assistant.update_working_memory(
                plan.session_id,
                plan.session_context,
                plan.turn_number,
                leaf_id=plan.assistant_message_id,
            )
            if memory_usage is None:
                return
            # Merge into the row's usage as it is now: a continuation may have
            # added its own requests since this turn saved its snapshot.
            with contextlib.suppress(Exception):
                record = await self._sessions.update_message(
                    plan.session_id,
                    plan.assistant_message_id,
                    usage=lambda current: with_auxiliary(current, "working_memory", memory_usage),
                )
                state.stored_usage = record.get("usage")
        finally:
            if release_pin:
                self._sessions.unpin(plan.session_id)

    async def _clear_pending(self, plan: TurnPlan) -> None:
        """Clear the turn's pending host action, in memory and on the session row."""
        await self._sessions.clear_pending_action(plan.session_id)

    async def _persist_cancelled(
        self, plan: TurnPlan, state: RunSnapshot, *, interrupted: bool = True
    ) -> None:
        """Retain native work and explicitly mark unresolved calls interrupted."""
        state.cancelled = interrupted
        state.interrupted_status = "cancelled"
        mark_unanswered_calls(state)
        await persist_snapshot(self._sessions, plan, state)
        await self._clear_pending(plan)

    async def _resolve_unanswered_calls(self, plan: TurnPlan, state: RunSnapshot) -> None:
        """Save completed work and mark unanswered calls interrupted after a failure."""
        if not state.assistant_messages:
            return
        state.cancelled = True
        mark_unanswered_calls(state)
        state.interrupted_status = "cancelled"
        await persist_snapshot(self._sessions, plan, state)
        await self._clear_pending(plan)

    # --- pieces -----------------------------------------------------------------

    async def _run_agent(
        self,
        plan: TurnPlan,
        ctx: AgentSetupContext,
        coordinator: EventCoordinator,
        state: RunSnapshot,
        *,
        control: TurnControl,
        user_prompt: str | None,
        message_history: list[ModelMessage],
        deferred_tool_results: Any,
        usage: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """One native event stream; ``state`` receives its messages, usage and output."""
        session_id = plan.session_id
        telegram_chat_id: str | None = None
        run_id = str(uuid.uuid4())
        captured: list[ModelMessage] = []
        control.check_cancelled()
        try:
            async with asyncio.timeout(self._config.stream_timeout_seconds):
                with (
                    assistant_request_context(
                        session_id,
                        screenshot=plan.screenshot,
                        profile_name=plan.request.profile,
                        subject=plan.request.subject,
                        principal=plan.principal,
                        host_context=plan.session_context.get("last_host_context"),
                        subagent_config={
                            "subagent_model": ctx.effective_config.subagent_model,
                            "subagent_thinking_budget": (
                                ctx.effective_config.subagent_thinking_budget
                            ),
                        },
                    ),
                    capture_run_messages() as captured,
                ):
                    try:
                        async with ctx.agent.run_stream_events(
                            user_prompt,
                            run_id=run_id,
                            message_history=message_history or None,
                            usage_limits=ctx.usage_limits,
                            deferred_tool_results=deferred_tool_results,
                            usage=usage,
                            deps=ctx.deps,
                            cancellation_token=control.token,
                            capabilities=[]
                            if plan.request.output_mode == "host_tools"
                            else [
                                TurnPolicy(
                                    self._sessions,
                                    session_id,
                                    plan.session_context,
                                    profile_name=ctx.profile_name,
                                ),
                                *([state.history.capability()] if state.history else []),
                            ],
                        ) as run:
                            async for event in iterate_run(
                                run,
                                coordinator,
                                config=self._config,
                                tools=self._tools,
                                emit_debug=self._config.emit_debug_events
                                and plan.request.output_mode != "host_tools",
                                silent=plan.request.output_mode == "host_tools",
                                suppress_tool_call_ids=plan.suppress_tool_call_ids,
                                host_tool_names=_host_tool_names(ctx),
                                native_sink=control.native_sink,
                            ):
                                yield event
                    finally:
                        telegram_chat_id = get_current_telegram_chat_binding()
            capture_result(plan, state, run.result)
        except (RunCancelled, asyncio.CancelledError, TimeoutError) as exc:
            snapshot = RunCancelled.from_cancellation(exc)
            if snapshot is not None:
                capture_result(plan, state, snapshot)
            raise
        except Exception as exc:
            snapshot = RunCancelled.from_cancellation(exc)
            capture_result(plan, state, snapshot or _IncompleteRun(captured, usage, run_id))
            raise
        finally:
            if telegram_chat_id:
                plan.session_context["telegram_chat_id"] = telegram_chat_id
                plan.session_context["telegram_bound_at"] = datetime.now(UTC)
                try:
                    await self._sessions.save_session_state_async(session_id)
                except Exception:
                    # Optional binding metadata must not mask native cancellation
                    # or prevent the response snapshot from being persisted.
                    logger.exception("Failed to save Telegram binding", session_id=session_id)

    async def _handle_output(
        self,
        plan: TurnPlan,
        ctx: AgentSetupContext,
        coordinator: EventCoordinator,
        state: RunSnapshot,
        *,
        control: TurnControl,
    ) -> AsyncIterator[dict[str, Any]]:
        """Resolve the run output; deliver queued steering as follow-up runs."""
        session_context = plan.session_context
        if await self._take_output(plan, state, ctx):
            return
        if plan.request.output_mode == "host_tools":
            return
        while session_context.get("pending_steering_ids"):
            pending = await self._sessions.list_pending_steering(
                plan.session_id, profile_name=ctx.profile_name
            )
            if not pending:
                break
            async for event in self._run_agent(
                plan,
                ctx,
                coordinator,
                state,
                control=control,
                user_prompt=None,
                message_history=state.all_messages,
                deferred_tool_results=None,
                usage=state.run_usage,
            ):
                yield event
            control.check_cancelled()
            await persist_snapshot(self._sessions, plan, state)
            if await self._take_output(plan, state, ctx):
                return

    async def _take_output(
        self, plan: TurnPlan, state: RunSnapshot, ctx: AgentSetupContext
    ) -> bool:
        """Record the run output on ``state``; True when it is a deferred host-tool call."""
        session_context = plan.session_context
        if isinstance(state.output, DeferredToolRequests):
            # The tool_call event was already streamed by iterate_run. The
            # assistant row holds the call already; the session row now records
            # that it waits for this call, so a restart can still accept the result.
            payloads = pending_call_payloads(
                self._tools,
                state.output,
                assistant_segments=state.assistant_segments,
                host_tool_names=_host_tool_names(ctx),
            )
            if plan.request.output_mode == "host_tools" and len(payloads) != 1:
                raise InvalidDecisionError(
                    "host_tools requires exactly one host action per decision"
                )
            batch = [payload["call_id"] for payload in payloads]
            await self._sessions.set_pending_action(
                plan.session_id,
                tool_call_id=payloads[0]["call_id"],
                tool_name=payloads[0]["tool_name"],
                assistant_message_id=plan.assistant_message_id,
                batch=batch,
                output_mode=plan.request.output_mode,
            )
            state.pending_tool_call = queued_payload(payloads[0], batch[1:])
            state.final_output = None
            return True
        session_context.pop("pending_assistant_message_id", None)
        state.pending_tool_call = None
        if plan.request.output_mode == "host_tools":
            if not isinstance(state.output, HoldDecision):
                raise InvalidDecisionError("host_tools requires a host action or structured hold")
            state.final_output = None
        else:
            state.final_output = str(state.output)
        return False


def _host_tool_names(ctx: AgentSetupContext) -> set[str]:
    """The host tools of this turn: configured ones plus request-declared actions."""
    return {tool.name for tool in ctx.available_tools.host_tools}


async def _timed(operation: Callable[[], Awaitable[Any]]) -> tuple[Any, float]:
    """Await and return ``(result, elapsed_ms)``."""
    started = time.monotonic()
    result = await operation()
    return result, (time.monotonic() - started) * 1000
