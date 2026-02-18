"""StreamingService — SSE streaming orchestrator.

Wraps the same pipeline as AssistantService but uses agent.iter() to stream
responses as SSE event dicts. The caller (HTTP layer) serializes to SSE format.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import DeferredToolResults
from pydantic_ai._agent_graph import ModelRequestNode
from pydantic_ai.messages import (
    PartDeltaEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPart,
)
from pydantic_ai.result import DeferredToolRequests
from pydantic_graph.nodes import End

from lovely_assistant.app.assistant._prompt_builder import build_system_prompt
from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.models import AssistantRequest
from lovely_assistant.app.streaming._coordinator import EventCoordinator
from lovely_assistant.app.streaming._event_builder import (
    make_text_delta_event,
    make_thinking_delta_event,
    make_tool_call_event,
)
from lovely_assistant.app.streaming.config import StreamingConfig
from lovely_assistant.app.streaming.exceptions import (
    StreamExecutionError,
    StreamingError,
    StreamSetupError,
)

if TYPE_CHECKING:
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
        sessions: SessionStore,
        assistant_config: Any = None,
    ) -> None:
        self._config = config
        self._llm = llm_service
        self._history = history_service
        self._tools = tool_service
        self._sessions = sessions
        self._assistant_config = assistant_config
        self._started = False

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

        # Setup phase
        try:
            session_id = request.session_id
            session_context = self._sessions.get_context(session_id)
            self._sessions.increment_turn(session_id)

            available_tools = self._tools.get_available_tools(request.machine_state)
            toolsets = self._tools.build_toolset(request.machine_state)

            system_prompt = build_system_prompt(
                available_tools=available_tools,
                session_context=session_context,
                machine_state=request.machine_state,
            )

            has_frontend_tools = len(available_tools.frontend_tools) > 0
            output_type: Any = [str, DeferredToolRequests] if has_frontend_tools else str

            default_model = self._assistant_config.default_model if self._assistant_config else None
            thinking_budget = (
                self._assistant_config.thinking_budget if self._assistant_config else None
            )
            model = default_model
            agent = self._llm.build_agent(
                system_prompt=system_prompt,
                toolsets=toolsets,
                model=model,
                output_type=output_type,
                thinking_budget=thinking_budget,
            )

            history = self._sessions.get_history(session_id)
            prepared_history, _ = await self._history.prepare_history(history, session_context)
        except Exception as e:
            raise StreamSetupError(f"Stream setup failed: {e}") from e

        # Streaming phase
        started = coordinator.try_started()
        if started:
            yield started

        resolved_model = model or self._llm._config.primary_model

        try:
            async with agent.iter(
                request.message,
                message_history=prepared_history if prepared_history else None,
            ) as run:
                async for event in self._iterate_run(run, coordinator, resolved_model):
                    yield event

            # Save history after run completes
            self._sessions.save_history(session_id, list(run.result.all_messages()))

            # Handle output
            output = run.result.output
            if isinstance(output, DeferredToolRequests):
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

    async def _stream_continuation(
        self, request: AssistantRequest
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a continuation (frontend returning tool result)."""
        coordinator = EventCoordinator(self._config.max_events_per_stream)

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

            assert request.tool_call_id is not None
            deferred_results = DeferredToolResults(
                calls={request.tool_call_id: request.tool_result}
            )

            available_tools = self._tools.get_available_tools(request.machine_state)
            toolsets = self._tools.build_toolset(request.machine_state)

            system_prompt = build_system_prompt(
                available_tools=available_tools,
                session_context=session_context,
                machine_state=request.machine_state,
            )

            has_frontend_tools = len(available_tools.frontend_tools) > 0
            output_type: Any = [str, DeferredToolRequests] if has_frontend_tools else str

            default_model = self._assistant_config.default_model if self._assistant_config else None
            thinking_budget = (
                self._assistant_config.thinking_budget if self._assistant_config else None
            )
            model = default_model
            agent = self._llm.build_agent(
                system_prompt=system_prompt,
                toolsets=toolsets,
                model=model,
                output_type=output_type,
                thinking_budget=thinking_budget,
            )

            history = self._sessions.get_history(session_id)
            prepared_history, _ = await self._history.prepare_history(
                history, session_context, is_continuation=True
            )
        except StreamingError:
            raise
        except Exception as e:
            raise StreamSetupError(f"Continuation setup failed: {e}") from e

        # Streaming phase
        started = coordinator.try_started()
        if started:
            yield started

        resolved_model = model or self._llm._config.primary_model

        try:
            async with agent.iter(
                None,
                message_history=prepared_history if prepared_history else None,
                deferred_tool_results=deferred_results,
            ) as run:
                async for event in self._iterate_run(run, coordinator, resolved_model):
                    yield event

            self._sessions.save_history(session_id, list(run.result.all_messages()))

            output = run.result.output
            if isinstance(output, DeferredToolRequests):
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

    async def _iterate_run(
        self,
        run: Any,
        coordinator: EventCoordinator,
        model: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate over agent run nodes, yielding SSE events."""
        while True:
            node = run.next_node
            if isinstance(node, End):
                break
            if isinstance(node, ModelRequestNode):
                async for event in self._stream_node(node, run, coordinator):
                    yield event
                # Advance to next node
                node = await run.next(node)
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
