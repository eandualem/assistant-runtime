"""StreamingService — streaming orchestrator.

Wraps the same pipeline as AssistantService but uses agent.iter() to stream
responses as event dicts. The caller (Socket.IO layer) emits events to clients.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)
from pydantic_graph.nodes import End

from lovely_assistant.app.assistant._serialization import (
    SteeringRecord,
    build_assistant_message_content,
    build_steering_request,
    path_records_to_model_history,
    sanitize_image_tool_returns,
)
from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.exceptions import SessionError
from lovely_assistant.app.assistant.models import AssistantRequest, _build_user_prompt
from lovely_assistant.app.streaming._coordinator import EventCoordinator
from lovely_assistant.app.streaming._event_builder import (
    make_debug_agent_config_event,
    make_debug_completed_event,
    make_debug_final_response_event,
    make_debug_history_event,
    make_debug_request_event,
    make_debug_system_prompt_event,
    make_debug_tool_selection_event,
    make_debug_usage_event,
    make_error_event,
    make_tool_call_event,
    make_tool_error_event,
    make_tool_result_event,
)
from lovely_assistant.app.streaming.config import StreamingConfig
from lovely_assistant.app.streaming.exceptions import (
    StreamingError,
    StreamSetupError,
)
from lovely_assistant.services.llm.exceptions import LLMCallError, classify_llm_error
from lovely_assistant.services.tools._frontend_tools import FRONTEND_TOOL_SCHEMAS
from lovely_assistant.services.tools._registry import get_tool_invalidates
from lovely_assistant.services.tools._request_context import assistant_request_context
from lovely_assistant.services.tools._screen_tools import (
    clear_current_screenshot,
    extract_screenshot_data_uri,
    set_current_screenshot,
    strip_screenshot_from_tool_result,
)
from lovely_assistant.services.tracing import create_request_trace

if TYPE_CHECKING:
    from lovely_assistant.app.assistant.config import AssistantConfig
    from lovely_assistant.app.assistant.interface import AssistantService
    from lovely_assistant.app.settings import RuntimeSettings
    from lovely_assistant.services.database.interface import DatabaseService
    from lovely_assistant.services.history.interface import HistoryService
    from lovely_assistant.services.llm.interface import LlmService
    from lovely_assistant.services.tools.interface import ToolService


class StreamingService:
    """Streaming orchestrator. Implements LifecycleAware."""

    def __init__(
        self,
        config: StreamingConfig,
        llm_service: LlmService,
        history_service: HistoryService,
        tool_service: ToolService,
        assistant_service: AssistantService,
        runtime_settings: RuntimeSettings | None = None,
        assistant_config: AssistantConfig | None = None,
        database_service: DatabaseService | None = None,
    ) -> None:
        self._config = config
        self._llm = llm_service
        self._history = history_service
        self._tools = tool_service
        self._assistant_service = assistant_service
        self._runtime_settings = runtime_settings
        self._assistant_config = assistant_config
        self._db: DatabaseService | None = database_service
        self._started = False

    @property
    def _sessions(self) -> SessionStore:
        """Access sessions through assistant service (created during start())."""
        try:
            return self._assistant_service.get_session_store()
        except Exception as e:
            raise StreamSetupError("Assistant service not started — no session store") from e

    def set_runtime_settings(self, runtime_settings: RuntimeSettings | None) -> None:
        """Attach live runtime settings after service construction."""
        self._runtime_settings = runtime_settings

    async def start(self) -> None:
        """Initialize the streaming service."""
        self._started = True
        logger.info("Streaming service started")

    async def stop(self) -> None:
        """Shutdown the streaming service."""
        self._started = False
        logger.info("Streaming service stopped")

    async def health_check(self) -> dict:
        """Report health status."""
        return {"healthy": self._started}

    async def warm_session(
        self,
        session_id: str,
        machine_state: dict[str, Any] | None = None,
    ) -> None:
        """Warm shared request-path state for a joined session."""
        if not self._started:
            return
        await self._assistant_service.warm_session(session_id, machine_state)

    async def accept_steering(
        self,
        request: AssistantRequest,
        *,
        has_live_stream: bool,
    ) -> str:
        """Accept steering and return `queued` or `promoted`."""
        if not request.is_steering:
            raise SessionError("Only steering requests can be accepted")

        session_context = await self._sessions.get_context_if_exists_async(request.session_id)
        if session_context is None:
            raise SessionError(f"Steering rejected: session '{request.session_id}' does not exist")

        if has_live_stream or session_context.get("pending_tool_call_id"):
            await self._sessions.queue_steering(request.session_id, request)
            return "queued"

        active_leaf_id = session_context.get("active_leaf_id")
        if active_leaf_id is None:
            raise SessionError(
                f"Steering rejected: session '{request.session_id}' has no active conversation"
            )

        if session_context["message_index"].get(active_leaf_id) is None:
            raise SessionError(
                f"Steering rejected: session '{request.session_id}' has no active conversation"
            )

        if (
            session_context.get("current_assistant_message_id") is None
            and session_context.get("pending_assistant_message_id") is None
        ):
            await self._sessions.queue_steering(
                request.session_id,
                request,
                status="promoted",
                delivered_at=datetime.now(UTC),
            )
            return "promoted"

        await self._sessions.queue_steering(request.session_id, request)
        return "queued"

    async def stream_message(self, request: AssistantRequest) -> AsyncIterator[dict[str, Any]]:
        """Stream a response as event dicts.

        Orchestrates: tools → prompt → agent.iter() → stream events.

        Args:
            request: The assistant request.

        Yields:
            Event dicts for the caller to emit via Socket.IO.

        Raises:
            StreamingError: If the service is not started.
            StreamSetupError: If agent setup fails.
            StreamExecutionError: If streaming fails mid-stream.
        """
        if not self._started:
            raise StreamingError("Streaming service not started")

        try:
            if request.is_steering:
                async for event in self._stream_promoted_steering(request):
                    yield event
            elif request.is_continuation:
                async for event in self._stream_continuation(request):
                    yield event
            else:
                async for event in self._stream_new_message(request):
                    yield event
        except (StreamSetupError, SessionError) as e:
            # Setup failed before any lifecycle events were emitted.
            # Emit a minimal started → final_response(error) → completed
            # envelope so the frontend can exit the "thinking" state.
            from lovely_assistant.app.streaming._event_builder import (
                make_agent_status_event,
                make_error_event,
                make_final_response_event,
            )

            logger.error(
                "[STREAM] Setup failed — emitting minimal lifecycle envelope",
                session_id=request.session_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            error_type = "session_error" if isinstance(e, SessionError) else "setup_error"
            retry_allowed = not isinstance(e, SessionError)
            yield make_agent_status_event("started")
            yield make_final_response_event(
                None,
                "unknown",
                session_id=request.session_id,
                error=True,
                error_type=error_type,
            )
            yield make_error_event(
                f"Setup failed: {e}",
                error_type=error_type,
                terminal=True,
                retry_allowed=retry_allowed,
            )
            yield make_agent_status_event("completed")

    async def _stream_continuation(
        self, request: AssistantRequest
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a continuation — frontend returning a tool result."""
        coordinator = EventCoordinator(self._config.max_events_per_stream)
        emit_debug = self._config.emit_debug_events
        start_time = time.monotonic()

        try:
            session_id = request.session_id
            session_context = await self._sessions.get_context_if_exists_async(session_id)
            if session_context is None:
                raise SessionError(f"Continuation rejected: session '{session_id}' does not exist")

            pending_tool_call_id = session_context.get("pending_tool_call_id")
            if not pending_tool_call_id:
                # Pending state was cleared (new message arrived during slow tool execution).
                # The tool result is stale — emit a non-fatal lifecycle so the frontend
                # can exit its frontendToolExecution state cleanly.
                logger.info(
                    "Late continuation arrived after pending state cleared",
                    session_id=session_id,
                    tool_call_id=request.tool_call_id,
                )
                raise SessionError(
                    f"Continuation ignored: session '{session_id}' no longer has a pending "
                    "tool call (a new message was processed). The frontend tool result "
                    "arrived after the session moved on."
                )
            if pending_tool_call_id != request.tool_call_id:
                raise SessionError(
                    "Continuation rejected: "
                    f"tool_call_id '{request.tool_call_id}' does not match pending "
                    f"tool call '{pending_tool_call_id}' for session '{session_id}'"
                )

            assistant_message_id = session_context.get("pending_assistant_message_id")
            if not assistant_message_id:
                raise SessionError(
                    "Continuation rejected: "
                    f"session '{session_id}' is missing the pending assistant message id"
                )
            session_context["current_assistant_message_id"] = assistant_message_id

        except SessionError:
            raise
        except Exception as e:
            raise StreamSetupError(f"Continuation setup failed: {e}") from e

        started = coordinator.try_started()
        if started:
            yield started

        try:
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_request_event(
                        session_id=session_id,
                        message=request.message or "(continuation)",
                        is_continuation=True,
                        has_machine_state=request.machine_state is not None,
                        machine_state=request.machine_state,
                    )
                )

            history = self._sessions.get_history(session_id)
            assistant_record = session_context["message_index"].get(assistant_message_id)
            if assistant_record is None:
                raise SessionError(
                    "Continuation rejected: "
                    f"assistant message '{assistant_message_id}' is not cached for session '{session_id}'"
                )
            assistant_history = path_records_to_model_history([assistant_record])
            (ctx, agent_setup_ms), (history_result, history_prep_ms) = await asyncio.gather(
                self._timed_async(
                    self._assistant_service.prepare_agent_context(request, session_context)
                ),
                self._timed_async(
                    self._history.prepare_history_with_metadata(
                        history, session_context, is_continuation=True
                    )
                ),
            )
            resolved_model = ctx.resolved_model
            prepared_history = history_result.history

            # Extract screenshot from tool_result BEFORE passing to LLM.
            # The screenshot goes into the ContextVar (for look_at_screen);
            # the tool_result sent to the LLM gets the base64 blob stripped.
            screenshot = extract_screenshot_data_uri(
                images=request.images,
                tool_result=request.tool_result,
            )
            if screenshot:
                set_current_screenshot(screenshot)
                tool_result_for_llm = strip_screenshot_from_tool_result(request.tool_result)
            else:
                tool_result_for_llm = request.tool_result

            deferred = DeferredToolResults(calls={request.tool_call_id: tool_result_for_llm})

            logger.info(
                "[STREAM] Continuation setup ready",
                session_id=session_id,
                model=resolved_model,
                agent_setup_ms=agent_setup_ms,
                history_prep_ms=history_prep_ms,
                pre_stream_ms=(time.monotonic() - start_time) * 1000,
            )

            # Root trace — manual context manager (generator cannot use `with`)
            trace_cm = create_request_trace(
                session_id=session_id,
                model=resolved_model,
                is_continuation=True,
                input_message=request.message or "(continuation)",
                set_current_observation=False,
            )
            trace_cm.__enter__()
        except Exception as e:
            error_type = "session_error" if isinstance(e, SessionError) else "setup_error"
            retry_allowed = not isinstance(e, SessionError)
            logger.error(
                "[STREAM] Continuation setup failed after started",
                session_id=session_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            for event in self._iter_setup_failure_events(
                coordinator,
                session_id=session_id,
                message=f"Setup failed: {e}",
                error_type=error_type,
                retry_allowed=retry_allowed,
            ):
                yield event
            return

        # Streaming phase
        _assistant_persisted = False
        try:
            async with asyncio.timeout(self._config.stream_timeout_seconds):
                with assistant_request_context(session_id):
                    async with ctx.agent.iter(
                        None,
                        message_history=prepared_history if prepared_history else None,
                        usage_limits=ctx.usage_limits,
                        deferred_tool_results=deferred,
                    ) as run:
                        async for event in self._iterate_run(
                            run,
                            coordinator,
                            resolved_model,
                            emit_debug=emit_debug,
                            suppress_tool_call_ids={request.tool_call_id},
                            session_id=session_id,
                            session_context=session_context,
                        ):
                            yield event

            all_messages = list(run.result.all_messages())
            continuation_messages = all_messages[len(prepared_history) :]

            # Clear pending tool call state
            session_context.pop("pending_tool_call_id", None)
            session_context.pop("pending_tool_name", None)

            # Extract usage
            usage_dict = self._merge_usage(
                assistant_record.get("usage"),
                self._safe_usage_dict(run.result),
            )
            assistant_turn_messages = [*assistant_history, *continuation_messages]

            assistant_content, assistant_segments, assistant_timestamp = build_assistant_message_content(
                assistant_turn_messages
            )
            await self._sessions.update_message(
                session_id,
                assistant_message_id,
                content=assistant_content,
                segments=assistant_segments,
                usage=usage_dict,
            )
            _assistant_persisted = True

            # Handle output (same DeferredToolRequests check as new message path)
            output = run.result.output
            if isinstance(output, DeferredToolRequests):
                # tool_call already emitted during _iterate_run() — just store pending state
                pending_info = self._store_pending_frontend_tool(output, session_context)
                session_context["pending_assistant_message_id"] = assistant_message_id
                session_context.pop("current_assistant_message_id", None)

                final = coordinator.try_final_response(
                    None,
                    resolved_model,
                    session_id=session_id,
                    message_id=assistant_message_id,
                    usage=usage_dict,
                    pending_tool_call=pending_info,
                )
            else:
                session_context.pop("pending_assistant_message_id", None)
                pending_info = None
                final_output = str(output)
                if session_context.get("pending_steering_ids"):
                    followup_outcome: dict[str, Any] = {}
                    async for event in self._stream_steering_followups(
                        session_id=session_id,
                        session_context=session_context,
                        assistant_message_id=assistant_message_id,
                        agent_context=ctx,
                        coordinator=coordinator,
                        emit_debug=emit_debug,
                        resolved_model=resolved_model,
                        conversation_history=all_messages,
                        assistant_messages=assistant_turn_messages,
                        cumulative_usage=run.result.usage(),
                        outcome=followup_outcome,
                    ):
                        yield event
                    final_output = followup_outcome.get("final_output", final_output)
                    usage_dict = followup_outcome.get("usage", usage_dict)
                    pending_info = followup_outcome.get("pending_tool_call")
                session_context.pop("current_assistant_message_id", None)
                if pending_info is not None:
                    final = coordinator.try_final_response(
                        None,
                        resolved_model,
                        session_id=session_id,
                        message_id=assistant_message_id,
                        usage=usage_dict,
                        pending_tool_call=pending_info,
                    )
                else:
                    final = coordinator.try_final_response(
                        final_output,
                        resolved_model,
                        session_id=session_id,
                        message_id=assistant_message_id,
                        usage=usage_dict,
                    )
            if final:
                yield final

            logger.info(
                "[STREAM] Continuation completed",
                session_id=session_id,
                model=resolved_model,
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        except TimeoutError:
            logger.error(
                "[STREAM] Continuation timed out",
                session_id=session_id,
                timeout=self._config.stream_timeout_seconds,
            )
            final = coordinator.try_final_response(
                None,
                resolved_model if "resolved_model" in locals() else "unknown",
                session_id=session_id,
                error=True,
                error_type="timeout",
            )
            if final:
                yield final
            yield coordinator.track(
                make_error_event(
                    f"Continuation timed out after {self._config.stream_timeout_seconds}s",
                    error_type="timeout",
                    terminal=True,
                    retry_allowed=True,
                )
            )
        except StreamingError:
            raise
        except Exception as e:
            llm_error = self._classify_provider_error(e)
            error_type = self._llm_error_type(llm_error) if llm_error is not None else "internal"
            retry_allowed = llm_error.retry_allowed if llm_error is not None else False
            message = (
                str(llm_error)
                if llm_error is not None
                else f"Continuation failed: {e.__class__.__name__}: {e}"
            )

            if llm_error is None:
                logger.exception(
                    "[STREAM] Continuation failed",
                    session_id=session_id,
                    error_type=e.__class__.__name__,
                    error=str(e),
                )
            else:
                logger.warning(
                    "[STREAM] Continuation failed with provider error",
                    session_id=session_id,
                    error_type=error_type,
                    retry_allowed=retry_allowed,
                    error=str(llm_error),
                )
            final = coordinator.try_final_response(
                None,
                resolved_model if "resolved_model" in locals() else "unknown",
                session_id=session_id,
                error=True,
                error_type=error_type,
            )
            if final:
                yield final
            yield coordinator.track(
                make_error_event(
                    message,
                    error_type=error_type,
                    terminal=True,
                    retry_allowed=retry_allowed,
                )
            )
        finally:
            clear_current_screenshot()

            # Flush any unflushed thinking (error path safety net)
            if emit_debug:
                coordinator.flush_thinking()
            if emit_debug and coordinator.accumulated_response:
                yield coordinator.track_debug(
                    make_debug_final_response_event(
                        coordinator.accumulated_response,
                        resolved_model if "resolved_model" in locals() else "unknown",
                        usage=self._safe_usage_dict(run.result if "run" in locals() else None),
                    )
                )

            duration_ms = (time.monotonic() - start_time) * 1000
            if emit_debug:
                yield coordinator.track_debug(make_debug_completed_event(duration_ms=duration_ms))

            screenshot = request.images[0] if request.images else None
            with contextlib.suppress(Exception):
                trace_cm.__exit__(None, None, None)
            await self._save_trace(
                session_id,
                coordinator.debug_events,
                user_message=request.message or "(continuation)",
                duration_ms=duration_ms,
                screenshot=screenshot,
            )

            completed = coordinator.try_completed()
            if completed:
                yield completed

    async def _stream_new_message(self, request: AssistantRequest) -> AsyncIterator[dict[str, Any]]:
        """Stream a fresh user message."""
        coordinator = EventCoordinator(self._config.max_events_per_stream)
        emit_debug = self._config.emit_debug_events
        start_time = time.monotonic()

        # Setup phase
        try:
            session_id = request.session_id
            session_context, _user_record = await self._sessions.register_user_message(request)
            self._ensure_no_pending_frontend_tool(session_context, session_id)
            turn_number = session_context.get("turn_number", 0)
            assistant_message_id = str(uuid.uuid4())
            session_context["current_assistant_message_id"] = assistant_message_id

        except SessionError:
            raise
        except Exception as e:
            raise StreamSetupError(f"Stream setup failed: {e}") from e

        started = coordinator.try_started()
        if started:
            yield started

        try:
            # Debug: request
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_request_event(
                        session_id=session_id,
                        message=request.message,
                        is_continuation=False,
                        has_machine_state=request.machine_state is not None,
                        machine_state=request.machine_state,
                        image_count=len(request.images),
                    )
                )

            history = self._sessions.get_history(session_id, exclude_leaf=True)
            (ctx, agent_setup_ms), (history_result, history_prep_ms) = await asyncio.gather(
                self._timed_async(
                    self._assistant_service.prepare_agent_context(request, session_context)
                ),
                self._timed_async(
                    self._history.prepare_history_with_metadata(history, session_context)
                ),
            )
            available_tools = ctx.available_tools
            resolved_model = ctx.resolved_model
            prepared_history = history_result.history

            # Debug: tool selection
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_tool_selection_event(
                        page=available_tools.page,
                        backend_count=len(available_tools.backend_tools),
                        filtered_out=available_tools.filtered_out_count,
                        tool_names=available_tools.tool_names,
                        tools=[
                            {
                                "name": t.name,
                                "description": t.description,
                                "parameters_schema": t.parameters_schema,
                            }
                            for t in available_tools.backend_tools
                        ],
                    )
                )

            # Debug: system prompt
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_system_prompt_event(
                        total_length=len(ctx.prompt_result.content),
                        fragment_count=len(ctx.prompt_result.fragments),
                        fragments=ctx.prompt_result.fragments,
                        content=ctx.prompt_result.content,
                    )
                )

            logger.info(
                "[STREAM] Setup ready",
                session_id=session_id,
                model=resolved_model,
                has_images=bool(request.images),
                max_turns=ctx.effective_config.max_turns,
                thinking_budget=ctx.effective_config.thinking_budget,
                temperature=ctx.effective_config.temperature,
                agent_setup_ms=agent_setup_ms,
                history_prep_ms=history_prep_ms,
                pre_stream_ms=(time.monotonic() - start_time) * 1000,
            )

            # Debug: agent config
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_agent_config_event(
                        model=resolved_model,
                        output_type="str",
                        thinking_budget=ctx.effective_config.thinking_budget,
                        temperature=ctx.effective_config.temperature,
                        session_id=session_id,
                    )
                )

            # Debug: history
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_history_event(
                        message_count=history_result.message_count,
                        estimated_tokens=history_result.estimated_tokens,
                        was_compacted=history_result.was_compacted,
                        compacted_from=history_result.compacted_from,
                        messages=history_result.message_summaries,
                    )
                )

            # Root trace — manual context manager (generator cannot use `with`)
            trace_cm = create_request_trace(
                session_id=session_id,
                model=resolved_model,
                is_continuation=False,
                input_message=request.message,
                metadata={"has_images": bool(request.images)},
                set_current_observation=False,
            )
            trace_cm.__enter__()
        except Exception as e:
            error_type = "session_error" if isinstance(e, SessionError) else "setup_error"
            retry_allowed = not isinstance(e, SessionError)
            logger.error(
                "[STREAM] Setup failed after started",
                session_id=session_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            for event in self._iter_setup_failure_events(
                coordinator,
                session_id=session_id,
                message=f"Setup failed: {e}",
                error_type=error_type,
                retry_allowed=retry_allowed,
            ):
                yield event
            return

        # Streaming phase
        # Store screenshot for look_at_screen tool (request-scoped ContextVar)
        screenshot = extract_screenshot_data_uri(
            images=request.images,
            tool_result=request.tool_result,
        )
        if screenshot:
            set_current_screenshot(screenshot)

        user_prompt = _build_user_prompt(request.message)
        _assistant_persisted = False
        try:
            async with asyncio.timeout(self._config.stream_timeout_seconds):
                with assistant_request_context(session_id):
                    async with ctx.agent.iter(
                        user_prompt,
                        message_history=prepared_history if prepared_history else None,
                        usage_limits=ctx.usage_limits,
                    ) as run:
                        async for event in self._iterate_run(
                            run,
                            coordinator,
                            resolved_model,
                            emit_debug=emit_debug,
                            session_id=session_id,
                            session_context=session_context,
                        ):
                            yield event

            all_messages = list(run.result.all_messages())
            turn_messages = all_messages[len(prepared_history) :]

            # Debug: usage
            if emit_debug:
                try:
                    usage = run.result.usage()
                    usage_snapshot = self._safe_usage_dict(usage) or {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    }
                    yield coordinator.track_debug(
                        make_debug_usage_event(
                            input_tokens=usage_snapshot["input_tokens"],
                            output_tokens=usage_snapshot["output_tokens"],
                            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
                            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                            requests=getattr(usage, "requests", 0) or 0,
                            total=usage_snapshot["total_tokens"],
                        )
                    )
                except Exception as e:
                    logger.debug("Failed to extract usage stats", error=str(e))

            usage_dict = self._safe_usage_dict(run.result)

            assistant_content, assistant_segments, assistant_timestamp = build_assistant_message_content(
                turn_messages
            )
            await self._sessions.register_assistant_message(
                session_id,
                message_id=assistant_message_id,
                parent_id=request.id,
                content=assistant_content,
                segments=assistant_segments,
                usage=usage_dict,
                created_at=assistant_timestamp,
            )
            _assistant_persisted = True

            # Handle output
            output = run.result.output

            if isinstance(output, DeferredToolRequests):
                # Frontend tool call — store pending state for continuation.
                # The tool_call event was already emitted during
                # _iterate_run() (via _stream_node or CallToolsNode handler).
                pending_info = self._store_pending_frontend_tool(output, session_context)
                session_context["pending_assistant_message_id"] = assistant_message_id
                session_context.pop("current_assistant_message_id", None)

                final = coordinator.try_final_response(
                    None,
                    resolved_model,
                    session_id=session_id,
                    message_id=assistant_message_id,
                    usage=usage_dict,
                    pending_tool_call=pending_info,
                )
                if final:
                    yield final
                logger.info(
                    "[STREAM] Deferred tool request emitted",
                    session_id=session_id,
                    model=resolved_model,
                    tool_name=pending_info["tool_name"],
                    duration_ms=(time.monotonic() - start_time) * 1000,
                )
            else:
                session_context.pop("pending_assistant_message_id", None)
                pending_info = None
                final_output = str(output)
                if session_context.get("pending_steering_ids"):
                    followup_outcome: dict[str, Any] = {}
                    async for event in self._stream_steering_followups(
                        session_id=session_id,
                        session_context=session_context,
                        assistant_message_id=assistant_message_id,
                        agent_context=ctx,
                        coordinator=coordinator,
                        emit_debug=emit_debug,
                        resolved_model=resolved_model,
                        conversation_history=all_messages,
                        assistant_messages=turn_messages,
                        cumulative_usage=run.result.usage(),
                        outcome=followup_outcome,
                    ):
                        yield event
                    final_output = followup_outcome.get("final_output", final_output)
                    usage_dict = followup_outcome.get("usage", usage_dict)
                    pending_info = followup_outcome.get("pending_tool_call")
                session_context.pop("current_assistant_message_id", None)
                final = coordinator.try_final_response(
                    None if pending_info is not None else final_output,
                    resolved_model,
                    session_id=session_id,
                    message_id=assistant_message_id,
                    usage=usage_dict,
                    pending_tool_call=pending_info,
                )
                if final:
                    yield final
                logger.info(
                    "[STREAM] Streaming request completed",
                    session_id=session_id,
                    model=resolved_model,
                    duration_ms=(time.monotonic() - start_time) * 1000,
                    output_type=type(output).__name__,
                )

        except TimeoutError:
            logger.error(
                "[STREAM] Streaming request timed out",
                session_id=session_id,
                timeout=self._config.stream_timeout_seconds,
                duration_ms=(time.monotonic() - start_time) * 1000,
            )
            final = coordinator.try_final_response(
                None,
                resolved_model if "resolved_model" in locals() else "unknown",
                session_id=session_id,
                error=True,
                error_type="timeout",
            )
            if final:
                yield final
            yield coordinator.track(
                make_error_event(
                    f"Request timed out after {self._config.stream_timeout_seconds}s",
                    error_type="timeout",
                    terminal=True,
                    retry_allowed=True,
                )
            )
        except StreamingError:
            raise
        except Exception as e:
            llm_error = self._classify_provider_error(e)
            error_type = self._llm_error_type(llm_error) if llm_error is not None else "internal"
            retry_allowed = llm_error.retry_allowed if llm_error is not None else False
            message = (
                str(llm_error)
                if llm_error is not None
                else f"Streaming failed: {e.__class__.__name__}: {e}"
            )

            if llm_error is None:
                logger.exception(
                    "[STREAM] Streaming request failed",
                    session_id=session_id,
                    model=resolved_model if "resolved_model" in locals() else None,
                    duration_ms=(time.monotonic() - start_time) * 1000,
                    error_type=e.__class__.__name__,
                    error=str(e),
                )
            else:
                logger.warning(
                    "[STREAM] Streaming request failed with provider error",
                    session_id=session_id,
                    model=resolved_model if "resolved_model" in locals() else None,
                    duration_ms=(time.monotonic() - start_time) * 1000,
                    error_type=error_type,
                    retry_allowed=retry_allowed,
                    error=str(llm_error),
                )
            final = coordinator.try_final_response(
                None,
                resolved_model if "resolved_model" in locals() else "unknown",
                session_id=session_id,
                error=True,
                error_type=error_type,
            )
            if final:
                yield final
            yield coordinator.track(
                make_error_event(
                    message,
                    error_type=error_type,
                    terminal=True,
                    retry_allowed=retry_allowed,
                )
            )
        finally:
            clear_current_screenshot()

            # Working memory delta (mirrors non-streaming path, best-effort)
            try:
                if _assistant_persisted and ctx.effective_config.enable_working_memory:
                    await self._assistant_service._update_working_memory(
                        session_id, session_context, turn_number
                    )
            except Exception as e:
                logger.debug("Working memory update failed in streaming path", error=str(e))

            # Flush any unflushed thinking (error path safety net)
            if emit_debug:
                coordinator.flush_thinking()
            if emit_debug and coordinator.accumulated_response:
                yield coordinator.track_debug(
                    make_debug_final_response_event(
                        coordinator.accumulated_response,
                        resolved_model if "resolved_model" in locals() else "unknown",
                        usage=self._safe_usage_dict(run.result if "run" in locals() else None),
                    )
                )

            # Debug: completed
            duration_ms = (time.monotonic() - start_time) * 1000
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_completed_event(
                        duration_ms=duration_ms,
                    )
                )

            # Persist trace
            screenshot = request.images[0] if request.images else None
            with contextlib.suppress(Exception):
                trace_cm.__exit__(None, None, None)
            await self._save_trace(
                session_id,
                coordinator.debug_events,
                user_message=request.message,
                duration_ms=duration_ms,
                screenshot=screenshot,
            )

            completed = coordinator.try_completed()
            if completed:
                yield completed

    def _ensure_no_pending_frontend_tool(
        self, session_context: dict[str, Any], session_id: str
    ) -> None:
        """Clear stale pending frontend tool state if a new message arrives.

        Frontend tools (navigate, ui_send_event) can take 2-4s (html2canvas).
        If the user sends a new message during that window, clear the pending
        state so the new stream can proceed. The late tool_result will arrive
        as a continuation — _stream_continuation handles missing pending state
        gracefully.
        """
        pending_tool_call_id = session_context.get("pending_tool_call_id")
        if not pending_tool_call_id:
            return

        pending_tool_name = session_context.get("pending_tool_name") or "unknown"
        logger.warning(
            "Clearing pending frontend tool for new message",
            session_id=session_id,
            pending_tool=pending_tool_name,
            pending_call_id=pending_tool_call_id,
        )
        session_context.pop("pending_tool_call_id", None)
        session_context.pop("pending_tool_name", None)
        session_context.pop("pending_assistant_message_id", None)
        session_context.pop("current_assistant_message_id", None)

    def _store_pending_frontend_tool(
        self, output: DeferredToolRequests, session_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist the single supported frontend tool request and return final_response metadata."""
        calls = list(output.calls)
        if len(calls) != 1:
            raise ValueError(
                "Frontend tool protocol violation: "
                f"expected exactly 1 deferred tool call, got {len(calls)}"
            )

        first = calls[0]
        if first.tool_name not in FRONTEND_TOOL_SCHEMAS:
            raise ValueError(
                "Frontend tool protocol violation: "
                f"unknown deferred frontend tool '{first.tool_name}'"
            )

        session_context["pending_tool_call_id"] = first.tool_call_id
        session_context["pending_tool_name"] = first.tool_name
        try:
            args = first.args_as_dict()
        except Exception:
            args = {}
        return {
            "tool_name": first.tool_name,
            "call_id": first.tool_call_id,
            "arguments": args,
        }

    @staticmethod
    def _iter_setup_failure_events(
        coordinator: EventCoordinator,
        *,
        session_id: str,
        message: str,
        error_type: str,
        retry_allowed: bool,
        model: str = "unknown",
    ) -> list[dict[str, Any]]:
        """Build the terminal lifecycle envelope for failures after started was emitted."""
        events: list[dict[str, Any]] = []
        final = coordinator.try_final_response(
            None,
            model,
            session_id=session_id,
            error=True,
            error_type=error_type,
        )
        if final:
            events.append(final)
        events.append(
            coordinator.track(
                make_error_event(
                    message,
                    error_type=error_type,
                    terminal=True,
                    retry_allowed=retry_allowed,
                )
            )
        )
        completed = coordinator.try_completed()
        if completed:
            events.append(completed)
        return events

    @staticmethod
    def _classify_provider_error(exc: Exception) -> LLMCallError | None:
        """Return a classified LLM/provider error when the exception matches a known shape."""
        if isinstance(exc, LLMCallError):
            return exc
        classified = classify_llm_error(exc)
        if classified.error_category == "UNKNOWN":
            return None
        return classified

    @staticmethod
    def _llm_error_type(error: LLMCallError | None) -> str:
        """Map LLM error categories onto the streaming protocol error types."""
        if error is None:
            return "internal"
        return {
            "RATE_LIMIT": "rate_limit",
            "SERVER_ERROR": "provider_error",
            "CONNECTION_ERROR": "connection_error",
            "TIMEOUT": "timeout",
            "AUTH_ERROR": "provider_auth",
            "CLIENT_ERROR": "provider_client_error",
        }.get(error.error_category, "provider_error")

    @staticmethod
    async def _timed_async(awaitable: Any) -> tuple[Any, float]:
        """Await an operation and return its result plus elapsed milliseconds."""
        started_at = time.monotonic()
        result = await awaitable
        return result, (time.monotonic() - started_at) * 1000

    @staticmethod
    def _usage_value(usage: Any, *names: str) -> int:
        """Read a usage field while tolerating provider-specific attribute names."""
        for name in names:
            value = getattr(usage, name, None)
            if isinstance(value, int):
                return value
        return 0

    def _safe_usage_dict(self, result_or_usage: Any) -> dict[str, int] | None:
        """Extract usage stats from a run result or RunUsage object."""
        try:
            usage = (
                result_or_usage.usage()
                if callable(getattr(result_or_usage, "usage", None))
                else result_or_usage
            )
        except Exception as e:
            logger.debug("Failed to extract usage stats", error=str(e))
            return None

        if usage is None:
            return None

        input_tokens = self._usage_value(usage, "input_tokens", "request_tokens")
        output_tokens = self._usage_value(usage, "output_tokens", "response_tokens")
        total_tokens = self._usage_value(usage, "total_tokens")
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _merge_usage(*usage_dicts: dict[str, int] | None) -> dict[str, int] | None:
        """Sum usage snapshots across multiple runs."""
        merged = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        saw_usage = False
        for usage in usage_dicts:
            if not usage:
                continue
            saw_usage = True
            for key in merged:
                value = usage.get(key)
                if isinstance(value, int):
                    merged[key] += value
        return merged if saw_usage else None

    async def _deliver_steering_into_request(
        self,
        *,
        session_id: str | None,
        session_context: dict[str, Any] | None,
        next_node: ModelRequestNode,
    ) -> list[SteeringRecord]:
        """Append queued steering to the next model request and mark it delivered."""
        if session_id is None or session_context is None:
            return []
        if not session_context.get("pending_steering_ids"):
            return []

        delivered_steering = await self._sessions.deliver_pending_steering(session_id)
        if not delivered_steering:
            return []

        steering_request = build_steering_request(delivered_steering)
        next_node.request.parts.extend(steering_request.parts)
        return delivered_steering

    async def _stream_steering_followups(
        self,
        *,
        session_id: str,
        session_context: dict[str, Any],
        assistant_message_id: str,
        agent_context: Any,
        coordinator: EventCoordinator,
        emit_debug: bool,
        resolved_model: str,
        conversation_history: list[Any],
        assistant_messages: list[Any],
        cumulative_usage: Any,
        outcome: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Deliver queued steering after a turn ends, updating the same assistant row."""
        latest_output = ""
        pending_info: dict[str, Any] | None = None
        current_history = list(conversation_history)
        accumulated_messages = list(assistant_messages)
        usage_state = cumulative_usage
        usage_dict = self._safe_usage_dict(usage_state)

        while session_context.get("pending_steering_ids"):
            delivered_steering = await self._sessions.deliver_pending_steering(session_id)
            if not delivered_steering:
                break

            followup_history = [*current_history, build_steering_request(delivered_steering)]
            async with asyncio.timeout(self._config.stream_timeout_seconds):
                with assistant_request_context(session_id):
                    async with agent_context.agent.iter(
                        None,
                        message_history=followup_history,
                        usage_limits=agent_context.usage_limits,
                        usage=usage_state,
                    ) as run:
                        async for event in self._iterate_run(
                            run,
                            coordinator,
                            resolved_model,
                            emit_debug=emit_debug,
                            session_id=session_id,
                            session_context=session_context,
                        ):
                            yield event

            current_history = list(run.result.all_messages())
            accumulated_messages.extend(run.result.new_messages())
            usage_state = run.result.usage()
            usage_dict = self._safe_usage_dict(usage_state)

            assistant_content, assistant_segments, _assistant_timestamp = build_assistant_message_content(
                accumulated_messages
            )
            await self._sessions.update_message(
                session_id,
                assistant_message_id,
                content=assistant_content,
                segments=assistant_segments,
                usage=usage_dict,
            )

            output = run.result.output
            latest_output = str(output)
            if isinstance(output, DeferredToolRequests):
                pending_info = self._store_pending_frontend_tool(output, session_context)
                session_context["pending_assistant_message_id"] = assistant_message_id
                break

            session_context.pop("pending_assistant_message_id", None)
            pending_info = None

        outcome["final_output"] = latest_output
        outcome["usage"] = usage_dict
        outcome["pending_tool_call"] = pending_info

    async def _stream_promoted_steering(
        self, request: AssistantRequest
    ) -> AsyncIterator[dict[str, Any]]:
        """Run a promoted steering turn immediately when the session is idle."""
        coordinator = EventCoordinator(self._config.max_events_per_stream)
        emit_debug = self._config.emit_debug_events
        start_time = time.monotonic()

        try:
            session_id = request.session_id
            session_context = await self._sessions.get_context_if_exists_async(session_id)
            if session_context is None:
                raise SessionError(f"Steering rejected: session '{session_id}' does not exist")

            active_leaf_id = session_context.get("active_leaf_id")
            if active_leaf_id is None:
                raise SessionError(
                    f"Steering rejected: session '{session_id}' has no active conversation"
                )

            active_leaf = session_context["message_index"].get(active_leaf_id)
            if active_leaf is None:
                raise SessionError(
                    f"Steering rejected: session '{session_id}' has no active conversation"
                )

            steering_record = session_context["steering_index"].get(request.id)
            if steering_record is None:
                steering_record = await self._sessions.queue_steering(
                    session_id,
                    request,
                    status="promoted",
                    delivered_at=datetime.now(UTC),
                )
            elif steering_record.get("status") == "pending":
                steering_record = await self._sessions.mark_steering_promoted(session_id, request.id)

            create_new_assistant = active_leaf.get("role") == "user"
            if active_leaf.get("role") not in {"user", "assistant"}:
                raise SessionError(
                    f"Steering rejected: session '{session_id}' has no active assistant context"
                )

            assistant_message_id = active_leaf_id if not create_new_assistant else str(uuid.uuid4())
            assistant_messages = (
                []
                if create_new_assistant
                else path_records_to_model_history([active_leaf])
            )
            assistant_parent_id = active_leaf_id if create_new_assistant else active_leaf.get("parent_id")
            session_context["current_assistant_message_id"] = assistant_message_id

            history = self._sessions.get_history(session_id)
            (ctx, agent_setup_ms), (history_result, history_prep_ms) = await asyncio.gather(
                self._timed_async(
                    self._assistant_service.prepare_agent_context(request, session_context)
                ),
                self._timed_async(
                    self._history.prepare_history_with_metadata(history, session_context)
                ),
            )
            resolved_model = ctx.resolved_model
            prepared_history = history_result.history
            promoted_history = [*prepared_history, build_steering_request([steering_record])]
            turn_number = session_context.get("turn_number", 0)

            trace_cm = create_request_trace(
                session_id=session_id,
                model=resolved_model,
                is_continuation=False,
                input_message=request.message,
                metadata={"steering": True},
                set_current_observation=False,
            )
            trace_cm.__enter__()
        except SessionError:
            raise
        except Exception as e:
            raise StreamSetupError(f"Promoted steering setup failed: {e}") from e

        started = coordinator.try_started()
        if started:
            yield started

        try:
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_request_event(
                        session_id=session_id,
                        message=request.message,
                        is_continuation=False,
                        has_machine_state=request.machine_state is not None,
                        machine_state=request.machine_state,
                    )
                )

            logger.info(
                "[STREAM] Promoted steering setup ready",
                session_id=session_id,
                model=resolved_model,
                agent_setup_ms=agent_setup_ms,
                history_prep_ms=history_prep_ms,
                pre_stream_ms=(time.monotonic() - start_time) * 1000,
            )
        except Exception as e:
            error_type = "session_error" if isinstance(e, SessionError) else "setup_error"
            retry_allowed = not isinstance(e, SessionError)
            for event in self._iter_setup_failure_events(
                coordinator,
                session_id=session_id,
                message=f"Setup failed: {e}",
                error_type=error_type,
                retry_allowed=retry_allowed,
            ):
                yield event
            return

        _assistant_persisted = False
        try:
            async with asyncio.timeout(self._config.stream_timeout_seconds):
                with assistant_request_context(session_id):
                    async with ctx.agent.iter(
                        None,
                        message_history=promoted_history,
                        usage_limits=ctx.usage_limits,
                    ) as run:
                        async for event in self._iterate_run(
                            run,
                            coordinator,
                            resolved_model,
                            emit_debug=emit_debug,
                            session_id=session_id,
                            session_context=session_context,
                        ):
                            yield event

            all_messages = list(run.result.all_messages())
            promoted_messages = list(run.result.new_messages())
            usage_dict = self._safe_usage_dict(run.result)
            assistant_turn_messages = [*assistant_messages, *promoted_messages]

            assistant_content, assistant_segments, assistant_timestamp = build_assistant_message_content(
                assistant_turn_messages
            )
            if create_new_assistant:
                await self._sessions.register_assistant_message(
                    session_id,
                    message_id=assistant_message_id,
                    parent_id=assistant_parent_id,
                    content=assistant_content,
                    segments=assistant_segments,
                    usage=usage_dict,
                    created_at=assistant_timestamp,
                )
            else:
                await self._sessions.update_message(
                    session_id,
                    assistant_message_id,
                    content=assistant_content,
                    segments=assistant_segments,
                    usage=usage_dict,
                )
            _assistant_persisted = True

            output = run.result.output
            if isinstance(output, DeferredToolRequests):
                pending_info = self._store_pending_frontend_tool(output, session_context)
                session_context["pending_assistant_message_id"] = assistant_message_id
                final_output = None
            else:
                session_context.pop("pending_assistant_message_id", None)
                pending_info = None
                final_output = str(output)
                if session_context.get("pending_steering_ids"):
                    followup_outcome: dict[str, Any] = {}
                    async for event in self._stream_steering_followups(
                        session_id=session_id,
                        session_context=session_context,
                        assistant_message_id=assistant_message_id,
                        agent_context=ctx,
                        coordinator=coordinator,
                        emit_debug=emit_debug,
                        resolved_model=resolved_model,
                        conversation_history=all_messages,
                        assistant_messages=assistant_turn_messages,
                        cumulative_usage=run.result.usage(),
                        outcome=followup_outcome,
                    ):
                        yield event
                    final_output = followup_outcome.get("final_output", final_output)
                    usage_dict = followup_outcome.get("usage", usage_dict)
                    pending_info = followup_outcome.get("pending_tool_call")

            session_context.pop("current_assistant_message_id", None)

            final = coordinator.try_final_response(
                final_output,
                resolved_model,
                session_id=session_id,
                message_id=assistant_message_id,
                usage=usage_dict,
                pending_tool_call=pending_info,
            )
            if final:
                yield final
        except TimeoutError:
            final = coordinator.try_final_response(
                None,
                resolved_model if "resolved_model" in locals() else "unknown",
                session_id=session_id,
                error=True,
                error_type="timeout",
            )
            if final:
                yield final
            yield coordinator.track(
                make_error_event(
                    f"Request timed out after {self._config.stream_timeout_seconds}s",
                    error_type="timeout",
                    terminal=True,
                    retry_allowed=True,
                )
            )
        except StreamingError:
            raise
        except Exception as e:
            llm_error = self._classify_provider_error(e)
            error_type = self._llm_error_type(llm_error) if llm_error is not None else "internal"
            retry_allowed = llm_error.retry_allowed if llm_error is not None else False
            message = (
                str(llm_error)
                if llm_error is not None
                else f"Streaming failed: {e.__class__.__name__}: {e}"
            )
            final = coordinator.try_final_response(
                None,
                resolved_model if "resolved_model" in locals() else "unknown",
                session_id=session_id,
                error=True,
                error_type=error_type,
            )
            if final:
                yield final
            yield coordinator.track(
                make_error_event(
                    message,
                    error_type=error_type,
                    terminal=True,
                    retry_allowed=retry_allowed,
                )
            )
        finally:
            session_context = locals().get("session_context")
            if isinstance(session_context, dict):
                session_context.pop("current_assistant_message_id", None)

            try:
                if (
                    _assistant_persisted
                    and "ctx" in locals()
                    and ctx.effective_config.enable_working_memory
                ):
                    await self._assistant_service._update_working_memory(
                        session_id, session_context, turn_number
                    )
            except Exception as e:
                logger.debug("Working memory update failed in streaming path", error=str(e))

            if emit_debug:
                coordinator.flush_thinking()
            if emit_debug and coordinator.accumulated_response:
                yield coordinator.track_debug(
                    make_debug_final_response_event(
                        coordinator.accumulated_response,
                        resolved_model if "resolved_model" in locals() else "unknown",
                        usage=self._safe_usage_dict(run.result if "run" in locals() else None),
                    )
                )

            duration_ms = (time.monotonic() - start_time) * 1000
            if emit_debug:
                yield coordinator.track_debug(make_debug_completed_event(duration_ms=duration_ms))

            with contextlib.suppress(Exception):
                trace_cm.__exit__(None, None, None)
            await self._save_trace(
                session_id,
                coordinator.debug_events,
                user_message=request.message,
                duration_ms=duration_ms,
            )

            completed = coordinator.try_completed()
            if completed:
                yield completed

    async def _iterate_run(
        self,
        run: Any,
        coordinator: EventCoordinator,
        model: str,
        *,
        emit_debug: bool = False,
        suppress_tool_call_ids: set[str] | None = None,
        session_id: str | None = None,
        session_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate over agent run nodes, yielding streaming events."""
        _pending_image_sanitize = False
        suppressed_tool_call_ids = suppress_tool_call_ids or set()
        while True:
            node = run.next_node
            if isinstance(node, End):
                break
            if isinstance(node, ModelRequestNode):
                async for event in self._stream_node(node, run, coordinator):
                    yield event
                # Flush this iteration's thinking into the trace (chronological position)
                if emit_debug:
                    coordinator.flush_thinking()
                coordinator.clear_content_segment()
                # Deferred sanitization: strip image after the LLM has seen it.
                # look_at_screen sets the flag at CallToolsNode time; we wait
                # until this ModelRequestNode finishes streaming (LLM consumed
                # the image) before replacing BinaryContent with a placeholder.
                if _pending_image_sanitize:
                    sanitize_image_tool_returns(run.ctx.state.message_history)
                    _pending_image_sanitize = False
                # Advance to next node
                node = await run.next(node)
            elif isinstance(node, CallToolsNode):
                # Emit tool_call events with complete arguments.
                # Tool calls are NOT emitted during _stream_node() because
                # at PartStartEvent time the args are still empty. Here,
                # model_response.tool_calls has the fully accumulated args.
                all_calls = list(node.model_response.tool_calls)
                for tc in all_calls:
                    if tc.tool_call_id in suppressed_tool_call_ids:
                        continue
                    try:
                        args = tc.args_as_dict()
                    except Exception:
                        args = {}
                    category = "frontend" if tc.tool_name in FRONTEND_TOOL_SCHEMAS else "backend"
                    evt = make_tool_call_event(
                        tc.tool_name, args, tc.tool_call_id, category=category
                    )
                    coordinator.track_debug(evt)
                    yield coordinator.track(evt)

                # Execute tools (timed)
                tool_start = time.monotonic()
                next_node = await run.next(node)
                tool_duration_ms = (time.monotonic() - tool_start) * 1000

                # Emit tool_result (always) and tool_error (on failure) events.
                # Dashboard needs tool_result to transition tool cards out of
                # "running" state — even when the tool errored.
                if isinstance(next_node, ModelRequestNode) and all_calls:
                    for tc in all_calls:
                        raw_content = self._extract_tool_result_raw(next_node, tc.tool_call_id)
                        is_error, error_msg = self._is_tool_error(raw_content)
                        if is_error:
                            err_evt = make_tool_error_event(
                                tc.tool_name, error_msg, tc.tool_call_id
                            )
                            coordinator.track_debug(err_evt)
                            yield coordinator.track(err_evt)
                        result_str = (
                            error_msg
                            if is_error
                            else str(raw_content)
                            if raw_content is not None
                            else ""
                        )
                        res_evt = make_tool_result_event(
                            tc.tool_name,
                            result_str,
                            tc.tool_call_id,
                            duration_ms=tool_duration_ms,
                            invalidates=get_tool_invalidates(tc.tool_name),
                        )
                        coordinator.track_debug(res_evt)
                        yield coordinator.track(res_evt)

                    # Check for RetryPromptPart (Pydantic AI validation failures)
                    for part in next_node.request.parts:
                        if isinstance(part, RetryPromptPart):
                            retry_tool = getattr(part, "tool_name", None) or "unknown"
                            retry_id = getattr(part, "tool_call_id", None) or ""
                            retry_evt = make_tool_error_event(
                                retry_tool,
                                str(part.content) if part.content else "Validation failed",
                                retry_id,
                            )
                            coordinator.track_debug(retry_evt)
                            yield coordinator.track(retry_evt)

                    await self._deliver_steering_into_request(
                        session_id=session_id,
                        session_context=session_context,
                        next_node=next_node,
                    )

                # Flag deferred sanitization: the image must survive until
                # the next ModelRequestNode streams (LLM sees it), then gets
                # stripped so subsequent tool rounds don't resend the image.
                if any(tc.tool_name == "look_at_screen" for tc in all_calls):
                    _pending_image_sanitize = True
            else:
                # Unknown node type — advance
                node = await run.next(node)

    async def _stream_node(
        self,
        node: ModelRequestNode,
        run: Any,
        coordinator: EventCoordinator,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream events from a single ModelRequestNode.

        Handles PartStartEvent (first chunk of a new part) and PartDeltaEvent
        (incremental updates) for text and thinking content.

        ToolCallPart events are NOT emitted here — at PartStartEvent time the
        arguments are still empty (streaming deltas haven't arrived yet).
        Tool calls are emitted from the CallToolsNode handler in _iterate_run()
        where model_response.tool_calls has complete arguments.

        Large PartStartEvent payloads (common with Gemini thinking) are chunked
        into smaller events to avoid buffering delays at the browser.
        """
        threshold = self._config.part_start_chunk_threshold
        chunk_size = self._config.part_start_chunk_size

        async with node.stream(run.ctx) as stream:
            async for event in stream:
                if isinstance(event, PartStartEvent):
                    if isinstance(event.part, ThinkingPart) and event.part.content:
                        async for e in self._yield_chunked(
                            event.part.content,
                            coordinator.emit_thinking_delta,
                            threshold,
                            chunk_size,
                        ):
                            yield e
                    elif isinstance(event.part, TextPart) and event.part.content:
                        async for e in self._yield_chunked(
                            event.part.content,
                            coordinator.emit_text_delta,
                            threshold,
                            chunk_size,
                        ):
                            yield e
                    # ToolCallPart deliberately skipped — see docstring
                elif isinstance(event, PartDeltaEvent):
                    if isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta:
                        yield coordinator.emit_thinking_delta(event.delta.content_delta)
                    elif isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
                        yield coordinator.emit_text_delta(event.delta.content_delta)

    @staticmethod
    async def _yield_chunked(
        content: str,
        emit_fn: Any,
        threshold: int,
        chunk_size: int,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield content as one or more events, chunking if above threshold."""
        if len(content) <= threshold:
            yield emit_fn(content)
            return

        for i in range(0, len(content), chunk_size):
            yield emit_fn(content[i : i + chunk_size])
            await asyncio.sleep(0)

    @staticmethod
    def _extract_tool_result(next_node: ModelRequestNode, tool_call_id: str) -> str:
        """Extract tool result content from the next ModelRequestNode's request parts."""
        try:
            for part in next_node.request.parts:
                if isinstance(part, ToolReturnPart) and part.tool_call_id == tool_call_id:
                    return str(part.content) if part.content is not None else ""
        except Exception:
            pass
        return ""

    @staticmethod
    def _extract_tool_result_raw(next_node: ModelRequestNode, tool_call_id: str) -> Any:
        """Extract raw tool result content (preserving type) from request parts."""
        try:
            for part in next_node.request.parts:
                if isinstance(part, ToolReturnPart) and part.tool_call_id == tool_call_id:
                    return part.content
        except Exception:
            pass
        return None

    @staticmethod
    def _is_tool_error(content: Any) -> tuple[bool, str]:
        """Check if tool result content represents an error.

        Returns (is_error, error_message).
        """
        if isinstance(content, dict) and ("error" in content or "error_code" in content):
            error_msg = str(content.get("error", content.get("error_code", "Unknown error")))
            return True, error_msg
        return False, ""

    async def _save_trace(
        self,
        session_id: str,
        trace_events: list[dict[str, Any]],
        *,
        user_message: str | None = None,
        duration_ms: float | None = None,
        screenshot: str | None = None,
    ) -> None:
        """Persist collected debug events as a trace row. Best-effort."""
        if self._db is None or not trace_events:
            return

        try:
            async with self._db.session_context() as db_session:
                from lovely_assistant.services.database.repositories import TraceRepository

                repo = TraceRepository(db_session)
                await repo.create(
                    trace_id=str(uuid.uuid4()),
                    session_id=session_id,
                    events=trace_events,
                    user_message=user_message,
                    duration_ms=duration_ms,
                    screenshot=screenshot,
                )
        except Exception as e:
            logger.warning("Failed to persist trace", session_id=session_id, error=str(e))
