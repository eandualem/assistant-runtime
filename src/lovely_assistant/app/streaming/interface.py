"""StreamingService — SSE streaming orchestrator.

Wraps the same pipeline as AssistantService but uses agent.iter() to stream
responses as SSE event dicts. The caller (HTTP layer) serializes to SSE format.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import DeferredToolResults
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.result import DeferredToolRequests
from pydantic_graph.nodes import End

from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.exceptions import (
    REJECTED_TOOL_RESULT,
    ContinuationMismatchError,
)
from lovely_assistant.app.assistant.models import AssistantRequest, _build_user_prompt
from lovely_assistant.app.streaming._coordinator import EventCoordinator
from lovely_assistant.app.streaming._event_builder import (
    make_debug_agent_config_event,
    make_debug_completed_event,
    make_debug_history_event,
    make_debug_request_event,
    make_debug_system_prompt_event,
    make_debug_tool_execution_event,
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

if TYPE_CHECKING:
    from lovely_assistant.app.assistant.config import AssistantConfig
    from lovely_assistant.app.assistant.interface import AssistantService
    from lovely_assistant.app.settings import RuntimeSettings
    from lovely_assistant.services.database.interface import DatabaseService
    from lovely_assistant.services.history.interface import HistoryService
    from lovely_assistant.services.llm.interface import LlmService
    from lovely_assistant.services.tools.interface import ToolService


class StreamingService:
    """SSE streaming orchestrator. Implements LifecycleAware."""

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

    async def stream_message(self, request: AssistantRequest) -> AsyncIterator[dict[str, Any]]:
        """Stream a response as SSE event dicts.

        Orchestrates: tools → prompt → agent.iter() → stream events.

        Args:
            request: The assistant request.

        Yields:
            SSE event dicts for the caller to serialize.

        Raises:
            StreamingError: If the service is not started.
            StreamSetupError: If agent setup fails.
            StreamExecutionError: If streaming fails mid-stream.
        """
        if not self._started:
            raise StreamingError("Streaming service not started")

        is_continuation = request.tool_call_id is not None

        if is_continuation:
            async for event in self._stream_continuation(request):
                yield event
        else:
            async for event in self._stream_new_message(request):
                yield event

    async def _stream_new_message(self, request: AssistantRequest) -> AsyncIterator[dict[str, Any]]:
        """Stream a fresh user message."""
        coordinator = EventCoordinator(self._config.max_events_per_stream)
        emit_debug = self._config.emit_debug_events
        start_time = time.monotonic()
        has_deferred = False

        # Setup phase
        try:
            session_id = request.session_id
            session_context = await self._sessions.get_context_async(session_id)
            self._sessions.increment_turn(session_id)

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

            # Build agent setup context (single source of truth for agent setup)
            ctx = await self._assistant_service.prepare_agent_context(request, session_context)
            available_tools = ctx.available_tools
            resolved_model = ctx.resolved_model

            # Debug: tool selection
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_tool_selection_event(
                        page=available_tools.page,
                        backend_count=len(available_tools.backend_tools),
                        frontend_count=len(available_tools.frontend_tools),
                        filtered_out=available_tools.filtered_out_count,
                        tool_names=available_tools.tool_names,
                        tools=[
                            {"name": t.name, "description": t.description}
                            for t in available_tools.backend_tools + available_tools.frontend_tools
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
                "[STREAM] Executing streaming request",
                session_id=session_id,
                continuation=False,
                model=resolved_model,
                has_images=bool(request.images),
                max_turns=ctx.effective_config.max_turns,
                thinking_budget=ctx.effective_config.thinking_budget,
                temperature=ctx.effective_config.temperature,
            )

            # Debug: agent config
            if emit_debug:
                output_type_str = (
                    "union[str, DeferredToolRequests]" if ctx.has_frontend_tools else "str"
                )
                yield coordinator.track_debug(
                    make_debug_agent_config_event(
                        model=resolved_model,
                        output_type=output_type_str,
                        has_frontend_tools=ctx.has_frontend_tools,
                        thinking_budget=ctx.effective_config.thinking_budget,
                        temperature=ctx.effective_config.temperature,
                        session_id=session_id,
                    )
                )

            history = self._sessions.get_history(session_id)
            history_result = await self._history.prepare_history_with_metadata(
                history, session_context
            )
            prepared_history = history_result.history

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
        except Exception as e:
            raise StreamSetupError(f"Stream setup failed: {e}") from e

        # Streaming phase
        started = coordinator.try_started()
        if started:
            yield started

        user_prompt = _build_user_prompt(request.message, request.images, resolved_model)
        _history_saved = False
        try:
            frontend_tool_names = frozenset(t.name for t in available_tools.frontend_tools)
            async with asyncio.timeout(self._config.stream_timeout_seconds):
                async with ctx.agent.iter(
                    user_prompt,
                    message_history=prepared_history if prepared_history else None,
                    usage_limits=ctx.usage_limits,
                ) as run:
                    async for event in self._iterate_run(
                        run, coordinator, resolved_model, frontend_tool_names
                    ):
                        yield event

            # Save history after run completes (async = persists to DB)
            await self._sessions.save_history_async(session_id, list(run.result.all_messages()))
            _history_saved = True

            # Debug: tool execution
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_tool_execution_event(
                        tool_names=available_tools.tool_names,
                        backend_count=len(available_tools.backend_tools),
                        frontend_count=len(available_tools.frontend_tools),
                    )
                )

            # Debug: usage
            if emit_debug:
                try:
                    usage = run.result.usage()
                    yield coordinator.track_debug(
                        make_debug_usage_event(
                            input_tokens=usage.request_tokens or 0,
                            output_tokens=usage.response_tokens or 0,
                            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
                            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                            requests=usage.requests,
                            total=usage.total_tokens or 0,
                        )
                    )
                except Exception as e:
                    logger.debug("Failed to extract usage stats", error=str(e))

            # Handle output
            output = run.result.output
            if isinstance(output, DeferredToolRequests):
                has_deferred = True
                async for event in self._handle_deferred_stream(
                    output, coordinator, session_id, resolved_model
                ):
                    yield event
            else:
                final = coordinator.try_final_response(
                    str(output), resolved_model, session_id=session_id
                )
                if final:
                    yield final
            logger.info(
                "[STREAM] Streaming request completed",
                session_id=session_id,
                continuation=False,
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
            logger.error(
                "[STREAM] Streaming request failed",
                session_id=session_id,
                continuation=False,
                model=resolved_model if "resolved_model" in locals() else None,
                duration_ms=(time.monotonic() - start_time) * 1000,
                error_type=e.__class__.__name__,
                error=str(e),
            )
            yield coordinator.track(
                make_error_event(
                    f"Streaming failed: {e.__class__.__name__}: {e}",
                    error_type="internal",
                    terminal=True,
                    retry_allowed=False,
                )
            )
        finally:
            # Backup history save if normal path didn't execute
            if not _history_saved:
                try:
                    if "run" in dir() and hasattr(run, "result") and run.result is not None:
                        await self._sessions.save_history_async(
                            session_id, list(run.result.all_messages())
                        )
                        logger.debug("Backup history save succeeded", session_id=session_id)
                except Exception:
                    logger.debug(
                        "Backup history save failed (expected if run did not complete)",
                        session_id=session_id,
                    )

            completed = coordinator.try_completed()
            if completed:
                yield completed

            # Debug: completed
            duration_ms = (time.monotonic() - start_time) * 1000
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_completed_event(
                        has_deferred=has_deferred,
                        duration_ms=duration_ms,
                    )
                )

            # Persist trace
            screenshot = request.images[0] if request.images else None
            await self._save_trace(
                session_id,
                coordinator.debug_events,
                user_message=request.message,
                is_continuation=False,
                duration_ms=duration_ms,
                screenshot=screenshot,
            )

    async def _stream_continuation(
        self, request: AssistantRequest
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a continuation (frontend returning tool result)."""
        coordinator = EventCoordinator(self._config.max_events_per_stream)
        emit_debug = self._config.emit_debug_events
        start_time = time.monotonic()
        has_deferred = False

        # Setup phase
        try:
            session_id = request.session_id

            if not self._sessions.has_session(session_id):
                raise StreamSetupError(f"No session found for continuation: {session_id}")

            pending = self._sessions.clear_pending_tool_call(session_id)
            if pending is None:
                raise StreamSetupError(f"No pending tool call for session {session_id}")

            # Validate continuation ID matches stored pending ID
            assert request.tool_call_id is not None
            if pending["tool_call_id"] != request.tool_call_id:
                raise ContinuationMismatchError(
                    expected=pending["tool_call_id"],
                    received=request.tool_call_id,
                    tool_name=pending.get("tool_name", "unknown"),
                )

            session_context = await self._sessions.get_context_async(session_id)
            self._sessions.increment_turn(session_id)

            # Debug: request
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_request_event(
                        session_id=session_id,
                        message=request.message or "",
                        is_continuation=True,
                        has_machine_state=request.machine_state is not None,
                        machine_state=request.machine_state,
                        image_count=0,
                    )
                )

            # Build deferred tool results (include rejected call results if any)
            calls_map: dict[str, Any] = {request.tool_call_id: request.tool_result}
            rejected_ids: list[str] = pending.get("rejected_call_ids") or []
            for rid in rejected_ids:
                calls_map[rid] = REJECTED_TOOL_RESULT
            deferred_results = DeferredToolResults(calls=calls_map)

            # Build agent setup context (single source of truth for agent setup)
            ctx = await self._assistant_service.prepare_agent_context(request, session_context)
            available_tools = ctx.available_tools
            resolved_model = ctx.resolved_model

            # Debug: tool selection
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_tool_selection_event(
                        page=available_tools.page,
                        backend_count=len(available_tools.backend_tools),
                        frontend_count=len(available_tools.frontend_tools),
                        filtered_out=available_tools.filtered_out_count,
                        tool_names=available_tools.tool_names,
                        tools=[
                            {"name": t.name, "description": t.description}
                            for t in available_tools.backend_tools + available_tools.frontend_tools
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
                "[STREAM] Executing streaming request",
                session_id=session_id,
                continuation=True,
                model=resolved_model,
                max_turns=ctx.effective_config.max_turns,
                thinking_budget=ctx.effective_config.thinking_budget,
                temperature=ctx.effective_config.temperature,
            )

            # Debug: agent config
            if emit_debug:
                output_type_str = (
                    "union[str, DeferredToolRequests]" if ctx.has_frontend_tools else "str"
                )
                yield coordinator.track_debug(
                    make_debug_agent_config_event(
                        model=resolved_model,
                        output_type=output_type_str,
                        has_frontend_tools=ctx.has_frontend_tools,
                        thinking_budget=ctx.effective_config.thinking_budget,
                        temperature=ctx.effective_config.temperature,
                        session_id=session_id,
                    )
                )

            history = self._sessions.get_history(session_id)
            history_result = await self._history.prepare_history_with_metadata(
                history, session_context, is_continuation=True
            )
            prepared_history = history_result.history

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
        except (StreamingError, ContinuationMismatchError):
            raise
        except Exception as e:
            raise StreamSetupError(f"Continuation setup failed: {e}") from e

        # Streaming phase
        started = coordinator.try_started()
        if started:
            yield started

        _history_saved = False
        try:
            frontend_tool_names = frozenset(t.name for t in available_tools.frontend_tools)
            async with asyncio.timeout(self._config.stream_timeout_seconds):
                async with ctx.agent.iter(
                    None,
                    message_history=prepared_history if prepared_history else None,
                    deferred_tool_results=deferred_results,
                    usage_limits=ctx.usage_limits,
                ) as run:
                    async for event in self._iterate_run(
                        run, coordinator, resolved_model, frontend_tool_names
                    ):
                        yield event

            await self._sessions.save_history_async(session_id, list(run.result.all_messages()))
            _history_saved = True

            # Debug: tool execution
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_tool_execution_event(
                        tool_names=available_tools.tool_names,
                        backend_count=len(available_tools.backend_tools),
                        frontend_count=len(available_tools.frontend_tools),
                    )
                )

            # Debug: usage
            if emit_debug:
                try:
                    usage = run.result.usage()
                    yield coordinator.track_debug(
                        make_debug_usage_event(
                            input_tokens=usage.request_tokens or 0,
                            output_tokens=usage.response_tokens or 0,
                            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
                            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                            requests=usage.requests,
                            total=usage.total_tokens or 0,
                        )
                    )
                except Exception as e:
                    logger.debug("Failed to extract usage stats", error=str(e))

            output = run.result.output
            if isinstance(output, DeferredToolRequests):
                has_deferred = True
                async for event in self._handle_deferred_stream(
                    output, coordinator, session_id, resolved_model
                ):
                    yield event
            else:
                final = coordinator.try_final_response(
                    str(output), resolved_model, session_id=session_id
                )
                if final:
                    yield final
            logger.info(
                "[STREAM] Streaming request completed",
                session_id=session_id,
                continuation=True,
                model=resolved_model,
                duration_ms=(time.monotonic() - start_time) * 1000,
                output_type=type(output).__name__,
            )

        except TimeoutError:
            logger.error(
                "[STREAM] Streaming continuation timed out",
                session_id=session_id,
                timeout=self._config.stream_timeout_seconds,
                duration_ms=(time.monotonic() - start_time) * 1000,
            )
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
            logger.error(
                "[STREAM] Streaming continuation failed",
                session_id=session_id,
                continuation=True,
                model=resolved_model if "resolved_model" in locals() else None,
                duration_ms=(time.monotonic() - start_time) * 1000,
                error_type=e.__class__.__name__,
                error=str(e),
            )
            yield coordinator.track(
                make_error_event(
                    f"Continuation streaming failed: {e.__class__.__name__}: {e}",
                    error_type="internal",
                    terminal=True,
                    retry_allowed=False,
                )
            )
        finally:
            # Backup history save if normal path didn't execute
            if not _history_saved:
                try:
                    if "run" in dir() and hasattr(run, "result") and run.result is not None:
                        await self._sessions.save_history_async(
                            session_id, list(run.result.all_messages())
                        )
                        logger.debug("Backup history save succeeded", session_id=session_id)
                except Exception:
                    logger.debug(
                        "Backup history save failed (expected if run did not complete)",
                        session_id=session_id,
                    )

            completed = coordinator.try_completed()
            if completed:
                yield completed

            # Debug: completed
            duration_ms = (time.monotonic() - start_time) * 1000
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_completed_event(
                        has_deferred=has_deferred,
                        duration_ms=duration_ms,
                    )
                )

            # Persist trace
            await self._save_trace(
                session_id,
                coordinator.debug_events,
                user_message=request.message,
                is_continuation=True,
                duration_ms=duration_ms,
                screenshot=None,
            )

    async def _iterate_run(
        self,
        run: Any,
        coordinator: EventCoordinator,
        model: str,
        frontend_tool_names: frozenset[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate over agent run nodes, yielding SSE events."""
        _frontend = frontend_tool_names or frozenset()
        _streamed_tool_ids: set[str] = set()
        while True:
            node = run.next_node
            if isinstance(node, End):
                break
            if isinstance(node, ModelRequestNode):
                async for event in self._stream_node(
                    node, run, coordinator, _frontend, _streamed_tool_ids
                ):
                    yield event
                # Advance to next node
                node = await run.next(node)
            elif isinstance(node, CallToolsNode):
                # Emit tool_call events for backend tools not already streamed
                all_backend_calls = [
                    tc for tc in node.model_response.tool_calls if tc.tool_name not in _frontend
                ]
                new_calls = [
                    tc for tc in all_backend_calls if tc.tool_call_id not in _streamed_tool_ids
                ]
                for tc in new_calls:
                    try:
                        args = tc.args_as_dict()
                    except Exception:
                        args = {}
                    yield coordinator.track(
                        make_tool_call_event(tc.tool_name, args, tc.tool_call_id)
                    )

                # Execute tools
                next_node = await run.next(node)

                # Emit tool_result or tool_error events for ALL backend tools
                if isinstance(next_node, ModelRequestNode) and all_backend_calls:
                    for tc in all_backend_calls:
                        raw_content = self._extract_tool_result_raw(next_node, tc.tool_call_id)
                        is_error, error_msg = self._is_tool_error(raw_content)
                        if is_error:
                            yield coordinator.track(
                                make_tool_error_event(tc.tool_name, error_msg, tc.tool_call_id)
                            )
                        else:
                            result_str = str(raw_content) if raw_content is not None else ""
                            yield coordinator.track(
                                make_tool_result_event(tc.tool_name, result_str, tc.tool_call_id)
                            )

                    # Check for RetryPromptPart (Pydantic AI validation failures)
                    for part in next_node.request.parts:
                        if isinstance(part, RetryPromptPart):
                            # Find the tool name for this retry (if identifiable)
                            retry_tool = getattr(part, "tool_name", None) or "unknown"
                            retry_id = getattr(part, "tool_call_id", None) or ""
                            if retry_tool not in _frontend:
                                yield coordinator.track(
                                    make_tool_error_event(
                                        retry_tool,
                                        str(part.content) if part.content else "Validation failed",
                                        retry_id,
                                    )
                                )
            else:
                # Unknown node type — advance
                node = await run.next(node)

    async def _stream_node(
        self,
        node: ModelRequestNode,
        run: Any,
        coordinator: EventCoordinator,
        frontend_tool_names: frozenset[str] | None = None,
        streamed_tool_ids: set[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream events from a single ModelRequestNode.

        Handles PartStartEvent (first chunk of a new part), PartDeltaEvent
        (incremental updates), and ToolCallPart (inline tool calls).

        ToolCallPart events are emitted during streaming to preserve the
        positional interleaving of text and tool calls (e.g., [text, tool,
        text, tool] instead of [text, text, tool, tool]). Frontend tools
        are skipped — they follow a separate DeferredToolRequests path.

        Large PartStartEvent payloads (common with Gemini thinking) are chunked
        into smaller SSE events to avoid buffering delays at the browser.
        """
        threshold = self._config.part_start_chunk_threshold
        chunk_size = self._config.part_start_chunk_size
        _frontend = frontend_tool_names or frozenset()
        _streamed = streamed_tool_ids if streamed_tool_ids is not None else set()

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
                    elif (
                        isinstance(event.part, ToolCallPart)
                        and event.part.tool_name not in _frontend
                    ):
                        try:
                            args = event.part.args_as_dict()
                        except Exception:
                            args = {}
                        yield coordinator.track(
                            make_tool_call_event(
                                event.part.tool_name, args, event.part.tool_call_id
                            )
                        )
                        _streamed.add(event.part.tool_call_id)
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
        """Yield content as one or more SSE events, chunking if above threshold."""
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

    async def _handle_deferred_stream(
        self,
        deferred: DeferredToolRequests,
        coordinator: EventCoordinator,
        session_id: str,
        model: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Handle DeferredToolRequests in streaming context."""
        calls: list[ToolCallPart] = deferred.calls
        if not calls:
            return

        call = calls[0]
        tool_call_id = call.tool_call_id

        # Store pending state (with rejected IDs if multiple calls)
        rejected_call_ids = [c.tool_call_id for c in calls[1:]] if len(calls) > 1 else None
        self._sessions.set_pending_tool_call(
            session_id, tool_call_id, call.tool_name, rejected_call_ids=rejected_call_ids
        )

        # Parse args
        args: dict[str, Any] = {}
        if isinstance(call.args, dict):
            args = call.args
        elif isinstance(call.args, str):
            try:
                args = json.loads(call.args)
            except (json.JSONDecodeError, TypeError):
                args = {}

        yield coordinator.track(make_tool_call_event(call.tool_name, args, tool_call_id))

        # Final response is null content for deferred tool calls
        final = coordinator.try_final_response(None, model, session_id=session_id)
        if final:
            yield final

    async def _save_trace(
        self,
        session_id: str,
        trace_events: list[dict[str, Any]],
        *,
        user_message: str | None = None,
        is_continuation: bool = False,
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
                    is_continuation=is_continuation,
                    duration_ms=duration_ms,
                    screenshot=screenshot,
                )
        except Exception as e:
            logger.warning("Failed to persist trace", session_id=session_id, error=str(e))
