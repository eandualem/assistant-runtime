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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import DeferredToolRequests
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart, ToolReturnPart

from assistant_runtime.app.assistant._serialization import (
    build_assistant_message_content,
)
from assistant_runtime.app.assistant.exceptions import SessionError
from assistant_runtime.app.streaming._agent_run import TurnPolicy, iterate_run
from assistant_runtime.app.streaming._control import TurnControl
from assistant_runtime.app.streaming._coordinator import EventCoordinator
from assistant_runtime.app.streaming._event_builder import (
    make_debug_agent_config_event,
    make_debug_completed_event,
    make_debug_error_event,
    make_debug_final_response_event,
    make_debug_history_event,
    make_debug_request_event,
    make_debug_system_prompt_event,
    make_debug_tool_selection_event,
    make_debug_usage_event,
    make_error_event,
)
from assistant_runtime.app.streaming._host_tool import store_pending_call
from assistant_runtime.app.streaming._usage import cache_counts, merge_usage, usage_dict
from assistant_runtime.app.streaming.exceptions import StreamingError
from assistant_runtime.services.llm.exceptions import LLMCallError, classify_llm_error
from assistant_runtime.services.tools._request_context import (
    assistant_request_context,
    get_current_telegram_chat_binding,
)
from assistant_runtime.services.tracing import create_request_trace

if TYPE_CHECKING:
    from assistant_runtime.app.assistant._session_store import SessionStore
    from assistant_runtime.app.assistant.interface import AssistantService
    from assistant_runtime.app.assistant.models import AgentSetupContext
    from assistant_runtime.app.streaming._turn import TurnPlan
    from assistant_runtime.app.streaming.config import StreamingConfig
    from assistant_runtime.services.database.interface import DatabaseService
    from assistant_runtime.services.history.interface import HistoryProcessor, HistoryService
    from assistant_runtime.services.tools.interface import ToolService

_LLM_ERROR_TYPES = {
    "RATE_LIMIT": "rate_limit",
    "SERVER_ERROR": "provider_error",
    "CONNECTION_ERROR": "connection_error",
    "TIMEOUT": "timeout",
    "AUTH_ERROR": "provider_auth",
    "CLIENT_ERROR": "provider_client_error",
}


def format_error_message(message: str, trace_id: str | None) -> str:
    """Append the trace id so client-visible errors can be correlated with traces."""
    if trace_id is None or "[trace_id:" in message:
        return message
    return f"{message} [trace_id: {trace_id}]"


def _describe_error(exc: Exception) -> tuple[str, str, bool]:
    """``(message, error_type, retry_allowed)`` for an exception raised by the run."""
    if isinstance(exc, LLMCallError):
        llm_error: LLMCallError | None = exc
    else:
        classified = classify_llm_error(exc)
        llm_error = None if classified.error_category == "UNKNOWN" else classified
    if llm_error is None:
        return f"Request failed: {exc.__class__.__name__}: {exc}", "internal", False
    error_type = _LLM_ERROR_TYPES.get(llm_error.error_category, "provider_error")
    return str(llm_error), error_type, llm_error.retry_allowed


@dataclass
class _RunState:
    """What one agent run left behind, carried across steering follow-ups."""

    all_messages: list[ModelMessage] = field(default_factory=list)
    assistant_messages: list[ModelMessage] = field(default_factory=list)
    assistant_segments: list[dict[str, Any]] | None = None
    run_usage: Any = None
    usage: dict[str, int] | None = None
    output: Any = None
    final_output: str | None = None
    pending_tool_call: dict[str, Any] | None = None
    persisted: bool = False
    cancelled: bool = False
    history: HistoryProcessor | None = None


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
    ) -> None:
        self._config = config
        self._sessions = sessions
        self._tools = tools
        self._history = history
        self._assistant = assistant_service
        self._db = database_service

    async def run(
        self, plan: TurnPlan, *, control: TurnControl | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute a turn, persisting cancellation before its terminal envelope."""
        control = control or TurnControl()
        coordinator = EventCoordinator(self._config.max_events_per_stream)
        emit_debug = self._config.emit_debug_events
        started_at = time.monotonic()
        trace_id = str(uuid.uuid4())
        request = plan.request
        session_id = plan.session_id
        session_context = plan.session_context
        state = _RunState(
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
                await self._persist(plan, state)
                self._clear_pending(plan)
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
            resolved_model = ctx.resolved_model
            state.history = self._history.processor(session_context)
            if emit_debug:
                for event in self._setup_debug_events(ctx, session_id):
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
                usage=None,
            ):
                yield event
            if emit_debug:
                history_event = self._history_debug_event(state)
                if history_event is not None:
                    yield coordinator.track_debug(history_event)
                yield coordinator.track_debug(self._usage_debug_event(state.run_usage))
            control.check_cancelled()
            await self._persist(plan, state)
            self._clear_pending(plan)
            async for event in self._handle_output(plan, ctx, coordinator, state, control=control):
                yield event
            control.check_cancelled()
            control.accepting_cancel = False
            final = coordinator.try_final_response(
                None if state.pending_tool_call is not None else state.final_output,
                resolved_model,
                session_id=session_id,
                message_id=plan.assistant_message_id,
                usage=state.usage,
                pending_tool_call=state.pending_tool_call,
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
                events = self._terminal_error_events(
                    coordinator,
                    session_id=session_id,
                    message=f"Cancelled turn could not be saved: {exc}",
                    error_type="persistence_error",
                    retry_allowed=False,
                    trace_id=trace_id,
                    model=resolved_model,
                    phase=phase,
                    emit_debug=emit_debug,
                )
            else:
                if isinstance(cancellation, TimeoutError):
                    events = self._terminal_error_events(
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
                    events = self._cancelled_events(
                        coordinator,
                        plan,
                        state,
                        model=resolved_model,
                        trace_id=trace_id,
                        phase=phase,
                        emit_debug=emit_debug,
                    )
            for event in events:
                yield event
        except StreamingError:
            raise
        except Exception as exc:
            if phase == "setup":
                is_session_error = isinstance(exc, SessionError)
                message = f"Setup failed: {exc}"
                error_type = "session_error" if is_session_error else "setup_error"
                retry_allowed = not is_session_error
            else:
                message, error_type, retry_allowed = _describe_error(exc)
            logger.warning("[STREAM] Turn failed", session_id=session_id, error=message)
            for event in self._terminal_error_events(
                coordinator,
                session_id=session_id,
                message=message,
                error_type=error_type,
                retry_allowed=retry_allowed,
                trace_id=trace_id,
                model=resolved_model,
                phase=phase,
                emit_debug=emit_debug,
            ):
                yield event
        finally:
            control.accepting_cancel = False
            session_context.pop("current_assistant_message_id", None)
            if (
                state.persisted
                and not state.cancelled
                and not control.token.cancelled
                and plan.update_working_memory
                and ctx is not None
                and ctx.effective_config.enable_working_memory
            ):
                await self._assistant.update_working_memory(
                    session_id, session_context, plan.turn_number
                )
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
            await self.save_trace(
                session_id,
                coordinator.debug_events,
                trace_id=trace_id,
                user_message=plan.input_message,
                duration_ms=duration_ms,
                screenshot=request.images[0] if request.images else None,
            )
            completed = coordinator.try_completed()
            if completed:
                yield completed

    @staticmethod
    def _clear_pending(plan: TurnPlan) -> None:
        """Clear only the completed turn's pending host frontier."""
        for key in ("pending_tool_call_id", "pending_tool_name", "pending_assistant_message_id"):
            plan.session_context.pop(key, None)

    async def _persist_cancelled(self, plan: TurnPlan, state: _RunState) -> None:
        """Retain native work and explicitly mark unresolved calls interrupted."""
        state.cancelled = True
        returned = {
            part.tool_call_id
            for message in state.assistant_messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        interrupted = [
            ToolReturnPart(
                tool_name=part.tool_name,
                tool_call_id=part.tool_call_id,
                content="[Tool execution was interrupted; its external outcome is unknown.]",
                outcome="interrupted",
            )
            for message in state.assistant_messages
            for part in message.parts
            if isinstance(part, ToolCallPart) and part.tool_call_id not in returned
        ]
        if interrupted:
            state.assistant_messages.append(ModelRequest(parts=interrupted))
        await self._persist(plan, state)
        self._clear_pending(plan)

    @staticmethod
    def _cancelled_events(
        coordinator: EventCoordinator,
        plan: TurnPlan,
        state: _RunState,
        *,
        model: str,
        trace_id: str,
        phase: str,
        emit_debug: bool,
    ) -> list[dict[str, Any]]:
        """A saved cancellation is one final/error/completed lifecycle."""
        final = coordinator.try_final_response(
            # Only native event middleware may decide what text reaches clients.
            # The saved snapshot can contain deliberately suppressed content.
            "",
            model,
            session_id=plan.session_id,
            message_id=plan.assistant_message_id,
            trace_id=trace_id,
            error=True,
            error_type="cancelled",
            usage=state.usage,
        )
        debug = make_debug_error_event(
            "Request cancelled",
            error_type="cancelled",
            retry_allowed=False,
            trace_id=trace_id,
            model=model,
            phase=phase,
        )
        coordinator.track_debug(debug)
        events = [debug] if emit_debug else []
        if final:
            events.append(final)
        events.append(
            make_error_event(
                "Request cancelled",
                error_type="cancelled",
                trace_id=trace_id,
                terminal=True,
                retry_allowed=False,
            )
        )
        return events

    # --- pieces -----------------------------------------------------------------

    async def _run_agent(
        self,
        plan: TurnPlan,
        ctx: AgentSetupContext,
        coordinator: EventCoordinator,
        state: _RunState,
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
        control.check_cancelled()
        try:
            async with asyncio.timeout(self._config.stream_timeout_seconds):
                with assistant_request_context(
                    session_id, screenshot=plan.screenshot, principal=plan.principal
                ):
                    try:
                        async with ctx.agent.run_stream_events(
                            user_prompt,
                            message_history=message_history or None,
                            usage_limits=ctx.usage_limits,
                            deferred_tool_results=deferred_tool_results,
                            usage=usage,
                            deps=ctx.deps,
                            cancellation_token=control.token,
                            capabilities=[
                                TurnPolicy(self._sessions, session_id, plan.session_context),
                                *([state.history.capability()] if state.history else []),
                            ],
                        ) as run:
                            async for event in iterate_run(
                                run,
                                coordinator,
                                config=self._config,
                                tools=self._tools,
                                emit_debug=self._config.emit_debug_events,
                                suppress_tool_call_ids=plan.suppress_tool_call_ids,
                                host_tool_names=_host_tool_names(ctx),
                            ):
                                yield event
                    finally:
                        telegram_chat_id = get_current_telegram_chat_binding()
            self._capture_result(plan, state, run.result)
        except (RunCancelled, asyncio.CancelledError, TimeoutError) as exc:
            snapshot = RunCancelled.from_cancellation(exc)
            if snapshot is not None:
                self._capture_result(plan, state, snapshot)
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

    @staticmethod
    def _capture_result(plan: TurnPlan, state: _RunState, result: Any) -> None:
        """Persist native history, including snapshots attached during teardown."""
        state.all_messages = list(result.all_messages())
        # new_messages(), not index slicing: pydantic-ai may merge consecutive
        # ModelRequests while cleaning the history, shrinking the list.
        new_messages = result.new_messages()
        if plan.accepted_tool_result is not None and any(
            isinstance(part, ToolReturnPart) and part.tool_call_id == plan.request.tool_call_id
            for message in new_messages
            for part in message.parts
        ):
            # The result was saved before setup; native resumption now includes it.
            state.assistant_messages = [
                message
                for message in state.assistant_messages
                if message is not plan.accepted_tool_result
            ]
        state.assistant_messages.extend(new_messages)
        state.run_usage = result.usage
        state.usage = merge_usage(plan.prior_usage, usage_dict(result))
        if not isinstance(result, RunCancelled):
            state.output = result.output

    async def _persist(self, plan: TurnPlan, state: _RunState) -> None:
        """Create or extend the assistant row with everything run so far."""
        content, segments, timestamp = build_assistant_message_content(state.assistant_messages)
        state.assistant_segments = segments
        if plan.assistant_parent_id is not None and not state.persisted:
            await self._sessions.register_assistant_message(
                plan.session_id,
                message_id=plan.assistant_message_id,
                parent_id=plan.assistant_parent_id,
                content=content,
                segments=segments,
                usage=state.usage,
                created_at=timestamp,
            )
        else:
            await self._sessions.update_message(
                plan.session_id,
                plan.assistant_message_id,
                content=content,
                segments=segments,
                usage=state.usage,
            )
        state.persisted = True

    async def _handle_output(
        self,
        plan: TurnPlan,
        ctx: AgentSetupContext,
        coordinator: EventCoordinator,
        state: _RunState,
        *,
        control: TurnControl,
    ) -> AsyncIterator[dict[str, Any]]:
        """Resolve the run output; deliver queued steering as follow-up runs."""
        session_context = plan.session_context
        if self._take_output(plan, state, ctx):
            return
        while session_context.get("pending_steering_ids"):
            pending = await self._sessions.list_pending_steering(plan.session_id)
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
            await self._persist(plan, state)
            if self._take_output(plan, state, ctx):
                return

    def _take_output(self, plan: TurnPlan, state: _RunState, ctx: AgentSetupContext) -> bool:
        """Record the run output on ``state``; True when it is a deferred host-tool call."""
        session_context = plan.session_context
        if isinstance(state.output, DeferredToolRequests):
            # The tool_call event was already streamed by iterate_run.
            state.pending_tool_call = store_pending_call(
                self._tools,
                state.output,
                session_context,
                assistant_segments=state.assistant_segments,
                host_tool_names=_host_tool_names(ctx),
            )
            session_context["pending_assistant_message_id"] = plan.assistant_message_id
            state.final_output = None
            return True
        session_context.pop("pending_assistant_message_id", None)
        state.pending_tool_call = None
        state.final_output = str(state.output)
        return False

    @staticmethod
    def _setup_debug_events(ctx: AgentSetupContext, session_id: str) -> list[dict[str, Any]]:
        tools = ctx.available_tools
        return [
            make_debug_tool_selection_event(
                page=tools.page,
                backend_count=len(tools.backend_tools),
                filtered_out=tools.filtered_out_count,
                tool_names=tools.tool_names,
                tools=[
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters_schema": t.parameters_schema,
                    }
                    for t in tools.backend_tools
                ],
            ),
            make_debug_system_prompt_event(
                total_length=len(ctx.prompt_result.content),
                fragment_count=len(ctx.prompt_result.fragments),
                fragments=ctx.prompt_result.fragments,
                content=ctx.prompt_result.content,
            ),
            make_debug_agent_config_event(
                model=ctx.resolved_model,
                output_type="str",
                thinking_budget=ctx.effective_config.thinking_budget,
                temperature=ctx.effective_config.temperature,
                session_id=session_id,
            ),
        ]

    @staticmethod
    def _history_debug_event(state: _RunState) -> dict[str, Any] | None:
        """Describe the model input the history policy last produced, if it ran."""
        result = state.history.result if state.history is not None else None
        if result is None:
            return None
        return make_debug_history_event(
            message_count=result.message_count,
            estimated_tokens=result.estimated_tokens,
            was_compacted=result.was_compacted,
            compacted_from=result.compacted_from,
            messages=result.message_summaries,
        )

    @staticmethod
    def _usage_debug_event(run_usage: Any) -> dict[str, Any]:
        snapshot = usage_dict(run_usage) or {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        cache_read, cache_write = cache_counts(run_usage)
        return make_debug_usage_event(
            input_tokens=snapshot["input_tokens"],
            output_tokens=snapshot["output_tokens"],
            cache_read=cache_read,
            cache_write=cache_write,
            requests=getattr(run_usage, "requests", 0) or 0,
            total=snapshot["total_tokens"],
        )

    @staticmethod
    def _terminal_error_events(
        coordinator: EventCoordinator,
        *,
        session_id: str,
        message: str,
        error_type: str,
        retry_allowed: bool,
        trace_id: str,
        model: str,
        phase: str,
        emit_debug: bool,
    ) -> list[dict[str, Any]]:
        """``final_response(error)`` + terminal ``error``; the debug error is always traced."""
        events: list[dict[str, Any]] = []
        final = coordinator.try_final_response(
            None,
            model,
            session_id=session_id,
            trace_id=trace_id,
            error=True,
            error_type=error_type,
        )
        if final:
            events.append(final)
        debug_error = coordinator.track_debug(
            make_debug_error_event(
                message,
                error_type=error_type,
                retry_allowed=retry_allowed,
                trace_id=trace_id,
                model=model,
                phase=phase,
            )
        )
        if emit_debug:
            events.append(debug_error)
        events.append(
            coordinator.track(
                make_error_event(
                    format_error_message(message, trace_id),
                    error_type=error_type,
                    trace_id=trace_id,
                    terminal=True,
                    retry_allowed=retry_allowed,
                )
            )
        )
        return events

    async def save_trace(
        self,
        session_id: str,
        trace_events: list[dict[str, Any]],
        *,
        trace_id: str | None = None,
        user_message: str | None = None,
        duration_ms: float | None = None,
        screenshot: str | None = None,
    ) -> None:
        """Persist collected debug events as a trace row. Best-effort."""
        if self._db is None or not trace_events:
            return
        try:
            from assistant_runtime.services.database.repositories import TraceRepository

            async with self._db.session_context() as db_session:
                await TraceRepository(db_session).create(
                    trace_id=trace_id or str(uuid.uuid4()),
                    session_id=session_id,
                    events=trace_events,
                    user_message=user_message,
                    duration_ms=duration_ms,
                    screenshot=screenshot,
                )
        except Exception as e:
            logger.warning("Failed to persist trace", session_id=session_id, error=str(e))


def _host_tool_names(ctx: AgentSetupContext) -> set[str]:
    """The host tools of this turn: configured ones plus request-declared actions."""
    return {tool.name for tool in ctx.available_tools.host_tools}


async def _timed(operation: Callable[[], Awaitable[Any]]) -> tuple[Any, float]:
    """Await and return ``(result, elapsed_ms)``."""
    started = time.monotonic()
    result = await operation()
    return result, (time.monotonic() - started) * 1000
