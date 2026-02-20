"""StreamingService — SSE streaming orchestrator.

Wraps the same pipeline as AssistantService but uses agent.iter() to stream
responses as SSE event dicts. The caller (HTTP layer) serializes to SSE format.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import DeferredToolResults
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.messages import (
    PartDeltaEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.result import DeferredToolRequests
from pydantic_graph.nodes import End

from lovely_assistant.app.assistant._prompt_builder import build_system_prompt
from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.models import AssistantRequest, _build_user_prompt
from lovely_assistant.app.settings import RuntimeSettings, resolve_effective_config
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
    make_text_delta_event,
    make_thinking_delta_event,
    make_tool_call_event,
    make_tool_result_event,
)
from lovely_assistant.app.streaming.config import StreamingConfig
from lovely_assistant.app.streaming.exceptions import (
    StreamExecutionError,
    StreamingError,
    StreamSetupError,
)

if TYPE_CHECKING:
    from lovely_assistant.app.assistant.config import AssistantConfig
    from lovely_assistant.app.assistant.interface import AssistantService
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
    ) -> None:
        self._config = config
        self._llm = llm_service
        self._history = history_service
        self._tools = tool_service
        self._assistant_service = assistant_service
        self._runtime_settings = runtime_settings
        self._assistant_config = assistant_config
        self._started = False

    @property
    def _sessions(self) -> SessionStore:
        """Access sessions through assistant service (created during start())."""
        sessions = self._assistant_service._sessions
        if sessions is None:
            raise StreamSetupError("Assistant service not started — no session store")
        return sessions

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
            session_context = self._sessions.get_context(session_id)
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

            available_tools = self._tools.get_available_tools(request.machine_state)
            toolsets = self._tools.build_toolset(request.machine_state)

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

            prompt_result = build_system_prompt(
                available_tools=available_tools,
                session_context=session_context,
                machine_state=request.machine_state,
            )
            system_prompt = prompt_result.content

            # Debug: system prompt
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_system_prompt_event(
                        total_length=len(system_prompt),
                        fragment_count=len(prompt_result.fragments),
                        fragments=prompt_result.fragments,
                        content=system_prompt,
                    )
                )

            has_frontend_tools = len(available_tools.frontend_tools) > 0
            output_type: Any = [str, DeferredToolRequests] if has_frontend_tools else str

            # Resolve effective config (per-request > runtime > frozen)
            frozen = self._assistant_config
            if frozen is None:
                from lovely_assistant.app.assistant.config import AssistantConfig

                frozen = AssistantConfig()
            effective = resolve_effective_config(frozen, self._runtime_settings, request.config)
            model = effective.default_model
            resolved_model = model or self._llm._config.primary_model
            agent = self._llm.build_agent(
                system_prompt=system_prompt,
                toolsets=toolsets,
                model=model,
                output_type=output_type,
                thinking_budget=effective.thinking_budget,
                temperature=effective.temperature,
            )

            # Debug: agent config
            if emit_debug:
                output_type_str = (
                    "union[str, DeferredToolRequests]" if has_frontend_tools else "str"
                )
                yield coordinator.track_debug(
                    make_debug_agent_config_event(
                        model=resolved_model,
                        output_type=output_type_str,
                        has_frontend_tools=has_frontend_tools,
                        thinking_budget=effective.thinking_budget,
                        temperature=effective.temperature,
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

        user_prompt = _build_user_prompt(request.message, request.images, model)
        try:
            frontend_tool_names = frozenset(t.name for t in available_tools.frontend_tools)
            async with agent.iter(
                user_prompt,
                message_history=prepared_history if prepared_history else None,
            ) as run:
                async for event in self._iterate_run(
                    run, coordinator, resolved_model, frontend_tool_names
                ):
                    yield event

            # Save history after run completes
            self._sessions.save_history(session_id, list(run.result.all_messages()))

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
                except Exception:
                    pass

            # Handle output
            output = run.result.output
            if isinstance(output, DeferredToolRequests):
                has_deferred = True
                async for event in self._handle_deferred_stream(
                    output, coordinator, session_id, resolved_model
                ):
                    yield event
            else:
                final = coordinator.try_final_response(str(output), resolved_model)
                if final:
                    yield final

        except StreamingError:
            raise
        except Exception as e:
            raise StreamExecutionError(f"Streaming failed: {e}") from e
        finally:
            completed = coordinator.try_completed()
            if completed:
                yield completed

            # Debug: completed
            if emit_debug:
                duration_ms = (time.monotonic() - start_time) * 1000
                yield coordinator.track_debug(
                    make_debug_completed_event(
                        has_deferred=has_deferred,
                        duration_ms=duration_ms,
                    )
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

            session_context = self._sessions.get_context(session_id)
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

            assert request.tool_call_id is not None
            deferred_results = DeferredToolResults(
                calls={request.tool_call_id: request.tool_result}
            )

            available_tools = self._tools.get_available_tools(request.machine_state)
            toolsets = self._tools.build_toolset(request.machine_state)

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

            prompt_result = build_system_prompt(
                available_tools=available_tools,
                session_context=session_context,
                machine_state=request.machine_state,
            )
            system_prompt = prompt_result.content

            # Debug: system prompt
            if emit_debug:
                yield coordinator.track_debug(
                    make_debug_system_prompt_event(
                        total_length=len(system_prompt),
                        fragment_count=len(prompt_result.fragments),
                        fragments=prompt_result.fragments,
                        content=system_prompt,
                    )
                )

            has_frontend_tools = len(available_tools.frontend_tools) > 0
            output_type: Any = [str, DeferredToolRequests] if has_frontend_tools else str

            # Resolve effective config (per-request > runtime > frozen)
            frozen = self._assistant_config
            if frozen is None:
                from lovely_assistant.app.assistant.config import AssistantConfig

                frozen = AssistantConfig()
            effective = resolve_effective_config(frozen, self._runtime_settings, request.config)
            model = effective.default_model
            resolved_model = model or self._llm._config.primary_model
            agent = self._llm.build_agent(
                system_prompt=system_prompt,
                toolsets=toolsets,
                model=model,
                output_type=output_type,
                thinking_budget=effective.thinking_budget,
                temperature=effective.temperature,
            )

            # Debug: agent config
            if emit_debug:
                output_type_str = (
                    "union[str, DeferredToolRequests]" if has_frontend_tools else "str"
                )
                yield coordinator.track_debug(
                    make_debug_agent_config_event(
                        model=resolved_model,
                        output_type=output_type_str,
                        has_frontend_tools=has_frontend_tools,
                        thinking_budget=effective.thinking_budget,
                        temperature=effective.temperature,
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
        except StreamingError:
            raise
        except Exception as e:
            raise StreamSetupError(f"Continuation setup failed: {e}") from e

        # Streaming phase
        started = coordinator.try_started()
        if started:
            yield started

        try:
            frontend_tool_names = frozenset(t.name for t in available_tools.frontend_tools)
            async with agent.iter(
                None,
                message_history=prepared_history if prepared_history else None,
                deferred_tool_results=deferred_results,
            ) as run:
                async for event in self._iterate_run(
                    run, coordinator, resolved_model, frontend_tool_names
                ):
                    yield event

            self._sessions.save_history(session_id, list(run.result.all_messages()))

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
                except Exception:
                    pass

            output = run.result.output
            if isinstance(output, DeferredToolRequests):
                has_deferred = True
                async for event in self._handle_deferred_stream(
                    output, coordinator, session_id, resolved_model
                ):
                    yield event
            else:
                final = coordinator.try_final_response(str(output), resolved_model)
                if final:
                    yield final

        except StreamingError:
            raise
        except Exception as e:
            raise StreamExecutionError(f"Continuation streaming failed: {e}") from e
        finally:
            completed = coordinator.try_completed()
            if completed:
                yield completed

            # Debug: completed
            if emit_debug:
                duration_ms = (time.monotonic() - start_time) * 1000
                yield coordinator.track_debug(
                    make_debug_completed_event(
                        has_deferred=has_deferred,
                        duration_ms=duration_ms,
                    )
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
        while True:
            node = run.next_node
            if isinstance(node, End):
                break
            if isinstance(node, ModelRequestNode):
                async for event in self._stream_node(node, run, coordinator):
                    yield event
                # Advance to next node
                node = await run.next(node)
            elif isinstance(node, CallToolsNode):
                # Emit tool_call events for backend tools before execution
                backend_calls = [
                    tc for tc in node.model_response.tool_calls if tc.tool_name not in _frontend
                ]
                for tc in backend_calls:
                    try:
                        args = tc.args_as_dict()
                    except Exception:
                        args = {}
                    yield coordinator.track(
                        make_tool_call_event(tc.tool_name, args, tc.tool_call_id)
                    )

                # Execute tools
                next_node = await run.next(node)

                # Emit tool_result events for completed backend tools
                if isinstance(next_node, ModelRequestNode) and backend_calls:
                    for tc in backend_calls:
                        result_content = self._extract_tool_result(next_node, tc.tool_call_id)
                        yield coordinator.track(
                            make_tool_result_event(tc.tool_name, result_content, tc.tool_call_id)
                        )
            else:
                # Unknown node type — advance
                node = await run.next(node)

    async def _stream_node(
        self,
        node: ModelRequestNode,
        run: Any,
        coordinator: EventCoordinator,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream events from a single ModelRequestNode."""
        async with node.stream(run.ctx) as stream:
            # Stream thinking deltas first
            async for event in stream:
                if (
                    isinstance(event, PartDeltaEvent)
                    and isinstance(event.delta, ThinkingPartDelta)
                    and event.delta.content_delta
                ):
                    yield coordinator.track(make_thinking_delta_event(event.delta.content_delta))
                elif (
                    isinstance(event, PartDeltaEvent)
                    and isinstance(event.delta, TextPartDelta)
                    and event.delta.content_delta
                ):
                    yield coordinator.track(make_text_delta_event(event.delta.content_delta))

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

        # Store pending state
        self._sessions.set_pending_tool_call(session_id, tool_call_id, call.tool_name)

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
        final = coordinator.try_final_response(None, model)
        if final:
            yield final
