"""Tests for StreamingService — the SSE streaming orchestrator."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_graph.nodes import End

from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.config import AssistantConfig
from lovely_assistant.app.assistant.exceptions import ContinuationMismatchError
from lovely_assistant.app.assistant.models import AssistantRequest
from lovely_assistant.app.settings import RuntimeSettings
from lovely_assistant.app.streaming.config import StreamingConfig
from lovely_assistant.app.streaming.exceptions import (
    StreamingError,
    StreamSetupError,
)
from lovely_assistant.app.streaming.interface import StreamingService
from lovely_assistant.services.history.models import HistoryPreparationResult
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet

# --- Helpers ---


def _make_service(
    *,
    streaming_config: StreamingConfig | None = None,
    assistant_config: AssistantConfig | None = None,
    sessions: SessionStore | None = None,
    database_service: Any | None = None,
) -> StreamingService:
    """Create a StreamingService with mocked dependencies."""
    llm_service = MagicMock()
    llm_service._config.primary_model = "test-model"
    llm_service.resolve_model = MagicMock(
        side_effect=lambda model=None: (
            model.lower().strip().replace("/", ":", 1) if model else "test-model"
        )
    )

    history_service = AsyncMock()
    history_service.prepare_history = AsyncMock(return_value=([], False))
    history_service.prepare_history_with_metadata = AsyncMock(
        return_value=HistoryPreparationResult(
            history=[], was_compacted=False, message_count=0, estimated_tokens=0
        )
    )

    tool_service = MagicMock()
    tool_service.get_available_tools.return_value = ToolSet()
    tool_service.build_toolset.return_value = []

    # Mock assistant service with _sessions attribute
    mock_assistant_service = MagicMock()
    mock_assistant_service._sessions = sessions or SessionStore()

    config = assistant_config or AssistantConfig()
    return StreamingService(
        config=streaming_config or StreamingConfig(emit_debug_events=False),
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
        assistant_service=mock_assistant_service,
        runtime_settings=RuntimeSettings(frozen_config=config),
        assistant_config=config,
        database_service=database_service,
    )


def _make_request(**kwargs) -> AssistantRequest:
    """Create a test AssistantRequest."""
    defaults = {
        "session_id": "test-session",
        "message": "Hello",
    }
    defaults.update(kwargs)
    return AssistantRequest(**defaults)


@asynccontextmanager
async def _empty_stream():
    """Async context manager that yields an empty async iterator."""

    async def _empty_iter():
        return
        yield  # noqa: F541

    yield _empty_iter()


@asynccontextmanager
async def _events_stream(events: list):
    """Async context manager that yields events from a list."""

    async def _iter():
        for e in events:
            yield e

    yield _iter()


def _make_call_tools_node(tool_calls: list[ToolCallPart]) -> MagicMock:
    """Create a mock that isinstance(x, CallToolsNode) recognizes."""
    mock_response = MagicMock(spec=ModelResponse)
    mock_response.tool_calls = tool_calls

    node = MagicMock(spec=CallToolsNode)
    node.model_response = mock_response
    return node


def _make_model_request_node(parts: list) -> MagicMock:
    """Create a mock ModelRequestNode with request parts and working stream()."""
    mock_request = MagicMock(spec=ModelRequest)
    mock_request.parts = parts

    node = MagicMock(spec=ModelRequestNode)
    node.request = mock_request
    # stream() must be an async context manager yielding an async iterator
    node.stream = MagicMock(side_effect=lambda *a, **kw: _empty_stream())
    return node


class _MockAgentRun:
    """Mock agent run context manager."""

    def __init__(self, nodes: list, output: Any = "Test response"):
        self._nodes = list(nodes)
        self._node_index = 0
        self._output = output
        self.ctx = MagicMock()

        # Build result mock
        self.result = MagicMock()
        self.result.output = output
        self.result.all_messages.return_value = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    @property
    def next_node(self):
        if self._node_index < len(self._nodes):
            return self._nodes[self._node_index]
        return End(data=self._output)

    async def next(self, node):
        self._node_index += 1
        if self._node_index < len(self._nodes):
            return self._nodes[self._node_index]
        return End(data=self._output)


# --- Lifecycle Tests ---


class TestStreamingServiceLifecycle:
    @pytest.mark.asyncio
    async def test_start(self):
        service = _make_service()
        await service.start()
        assert service._started is True

    @pytest.mark.asyncio
    async def test_stop(self):
        service = _make_service()
        await service.start()
        await service.stop()
        assert service._started is False

    @pytest.mark.asyncio
    async def test_health_check_started(self):
        service = _make_service()
        await service.start()
        health = await service.health_check()
        assert health["healthy"] is True

    @pytest.mark.asyncio
    async def test_health_check_not_started(self):
        service = _make_service()
        health = await service.health_check()
        assert health["healthy"] is False


class TestStreamingServiceNotStarted:
    @pytest.mark.asyncio
    async def test_stream_message_raises_when_not_started(self):
        service = _make_service()
        request = _make_request()
        with pytest.raises(StreamingError, match="not started"):
            async for _ in service.stream_message(request):
                pass


class TestStreamNewMessage:
    @pytest.mark.asyncio
    async def test_emits_started_and_completed(self):
        service = _make_service()
        await service.start()
        request = _make_request()

        # Mock agent.iter() to return a run with just an End node
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        types = [e["type"] for e in events]
        assert types[0] == "agent_status"
        assert events[0]["status"] == "started"
        assert types[-1] == "agent_status"
        assert events[-1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_emits_final_response(self):
        service = _make_service()
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Hello world")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"]
        assert len(final) == 1
        assert final[0]["content"] == "Hello world"
        assert final[0]["streamed"] is False

    @pytest.mark.asyncio
    async def test_saves_history_after_run(self):
        sessions = SessionStore()
        service = _make_service(sessions=sessions)
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.all_messages.return_value = [MagicMock()]
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        async for _ in service.stream_message(request):
            pass

        history = sessions.get_history("test-session")
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_increments_turn(self):
        sessions = SessionStore()
        service = _make_service(sessions=sessions)
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        async for _ in service.stream_message(request):
            pass

        ctx = sessions.get_context("test-session")
        assert ctx["turn_number"] == 1


class TestStreamWithImages:
    @pytest.mark.asyncio
    async def test_images_passed_to_agent_iter(self):
        """When images are provided, agent.iter() receives a list prompt."""
        service = _make_service()
        await service.start()
        request = _make_request(
            message="What's here?",
            images=["data:image/jpeg;base64,/9j/4AAQ"],
        )

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        async for _ in service.stream_message(request):
            pass

        call_args = mock_agent.iter.call_args
        user_prompt = call_args[0][0]
        assert isinstance(user_prompt, list)
        assert user_prompt[0] == "What's here?"

    @pytest.mark.asyncio
    async def test_no_images_passes_plain_string(self):
        """When no images, agent.iter() receives plain string."""
        service = _make_service()
        await service.start()
        request = _make_request(message="Hello there")

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        async for _ in service.stream_message(request):
            pass

        call_args = mock_agent.iter.call_args
        assert call_args[0][0] == "Hello there"


class TestStreamWithTools:
    @pytest.mark.asyncio
    async def test_builds_agent_with_frontend_tools(self):
        service = _make_service()
        await service.start()

        service._tools.get_available_tools.return_value = ToolSet(
            frontend_tools=[
                ToolDefinition(
                    name="navigate",
                    description="Notify UI",
                    parameters_schema={},
                    category=ToolCategory.FRONTEND,
                )
            ]
        )

        request = _make_request()
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        async for _ in service.stream_message(request):
            pass

        # Verify output_type includes DeferredToolRequests
        call_kwargs = service._llm.build_agent.call_args[1]
        assert isinstance(call_kwargs["output_type"], list)


class TestStreamDeferredToolRequests:
    @pytest.mark.asyncio
    async def test_deferred_tool_emits_tool_call_event(self):
        sessions = SessionStore()
        service = _make_service(sessions=sessions)
        await service.start()

        # Create a mock DeferredToolRequests output
        mock_call = MagicMock()
        mock_call.tool_call_id = "call_abc"
        mock_call.tool_name = "navigate"
        mock_call.args = {"message": "hello", "level": "info"}

        mock_deferred = MagicMock()
        mock_deferred.calls = [mock_call]

        # Patch isinstance to recognize our mock as DeferredToolRequests
        mock_run = _MockAgentRun(nodes=[], output=mock_deferred)
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        # We need to patch the isinstance check for DeferredToolRequests
        from pydantic_ai.result import DeferredToolRequests

        mock_deferred.__class__ = DeferredToolRequests

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 1
        assert tool_call_events[0]["tool_name"] == "navigate"
        assert tool_call_events[0]["call_id"] == "call_abc"

    @pytest.mark.asyncio
    async def test_deferred_tool_stores_pending(self):
        sessions = SessionStore()
        service = _make_service(sessions=sessions)
        await service.start()

        mock_call = MagicMock()
        mock_call.tool_call_id = "call_xyz"
        mock_call.tool_name = "ui_navigate"
        mock_call.args = {"path": "/agents"}

        mock_deferred = MagicMock()
        mock_deferred.calls = [mock_call]

        from pydantic_ai.result import DeferredToolRequests

        mock_deferred.__class__ = DeferredToolRequests

        mock_run = _MockAgentRun(nodes=[], output=mock_deferred)
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        async for _ in service.stream_message(request):
            pass

        ctx = sessions.get_context("test-session")
        assert ctx["pending_tool_call"]["tool_call_id"] == "call_xyz"
        assert "created_at" in ctx["pending_tool_call"]

    @pytest.mark.asyncio
    async def test_deferred_tool_final_response_null_content(self):
        sessions = SessionStore()
        service = _make_service(sessions=sessions)
        await service.start()

        mock_call = MagicMock()
        mock_call.tool_call_id = "call_123"
        mock_call.tool_name = "navigate"
        mock_call.args = {}

        mock_deferred = MagicMock()
        mock_deferred.calls = [mock_call]

        from pydantic_ai.result import DeferredToolRequests

        mock_deferred.__class__ = DeferredToolRequests

        mock_run = _MockAgentRun(nodes=[], output=mock_deferred)
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"]
        assert len(final) == 1
        assert final[0]["content"] is None


class TestStreamContinuation:
    @pytest.mark.asyncio
    async def test_continuation_requires_session(self):
        service = _make_service()
        await service.start()

        request = _make_request(
            tool_call_id="call_123",
            tool_result="ok",
        )
        with pytest.raises(StreamSetupError, match="No session found"):
            async for _ in service.stream_message(request):
                pass

    @pytest.mark.asyncio
    async def test_continuation_requires_pending_tool_call(self):
        sessions = SessionStore()
        # Create session but no pending tool call
        sessions.get_context("test-session")

        service = _make_service(sessions=sessions)
        await service.start()

        request = _make_request(
            tool_call_id="call_123",
            tool_result="ok",
        )
        with pytest.raises(StreamSetupError, match="No pending tool call"):
            async for _ in service.stream_message(request):
                pass

    @pytest.mark.asyncio
    async def test_continuation_streams_successfully(self):
        sessions = SessionStore()
        sessions.get_context("test-session")
        sessions.set_pending_tool_call("test-session", "call_123", "navigate")

        service = _make_service(sessions=sessions)
        await service.start()

        mock_run = _MockAgentRun(nodes=[], output="Continued response")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request(
            tool_call_id="call_123",
            tool_result="Tool executed successfully",
        )

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        types = [e["type"] for e in events]
        assert "agent_status" in types
        assert "final_response" in types

        final = [e for e in events if e["type"] == "final_response"]
        assert final[0]["content"] == "Continued response"


class TestStreamContinuationValidation:
    @pytest.mark.asyncio
    async def test_stream_continuation_id_mismatch_raises(self):
        """Streaming continuation with wrong tool_call_id raises ContinuationMismatchError."""
        sessions = SessionStore()
        sessions.get_context("test-session")
        sessions.set_pending_tool_call("test-session", "call_correct", "navigate")

        service = _make_service(sessions=sessions)
        await service.start()

        request = _make_request(
            tool_call_id="call_wrong",
            tool_result="ok",
        )
        with pytest.raises(ContinuationMismatchError) as exc_info:
            async for _ in service.stream_message(request):
                pass
        assert exc_info.value.expected == "call_correct"
        assert exc_info.value.received == "call_wrong"

    @pytest.mark.asyncio
    async def test_stream_continuation_with_rejected_ids(self):
        """Streaming continuation with rejected IDs completes successfully."""
        sessions = SessionStore()
        sessions.get_context("test-session")
        sessions.set_pending_tool_call(
            "test-session", "call_a", "navigate", rejected_call_ids=["call_b", "call_c"]
        )

        service = _make_service(sessions=sessions)
        await service.start()

        mock_run = _MockAgentRun(nodes=[], output="Done with rejections")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request(
            tool_call_id="call_a",
            tool_result={"status": "ok"},
        )

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"]
        assert len(final) == 1
        assert final[0]["content"] == "Done with rejections"


class TestStreamSetupErrors:
    @pytest.mark.asyncio
    async def test_tool_service_failure_raises_setup_error(self):
        service = _make_service()
        await service.start()

        service._tools.get_available_tools.side_effect = RuntimeError("tool fail")
        request = _make_request()

        with pytest.raises(StreamSetupError, match="Stream setup failed"):
            async for _ in service.stream_message(request):
                pass

    @pytest.mark.asyncio
    async def test_history_failure_raises_setup_error(self):
        service = _make_service()
        await service.start()

        service._history.prepare_history_with_metadata.side_effect = RuntimeError("history fail")
        request = _make_request()

        with pytest.raises(StreamSetupError, match="Stream setup failed"):
            async for _ in service.stream_message(request):
                pass


class TestStreamExecutionErrors:
    @pytest.mark.asyncio
    async def test_agent_iter_failure_emits_error_event(self):
        service = _make_service()
        await service.start()
        request = _make_request()

        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(side_effect=RuntimeError("iter fail"))
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        types = [e["type"] for e in events]
        assert "error" in types
        error_event = [e for e in events if e["type"] == "error"][0]
        assert error_event["error_type"] == "internal"
        assert error_event["terminal"] is True
        assert error_event["retry_allowed"] is False
        assert types[-1] == "agent_status"
        assert events[-1]["status"] == "completed"


class TestStreamModelResolution:
    @pytest.mark.asyncio
    async def test_uses_configured_model(self):
        config = AssistantConfig(default_model="custom-model")
        service = _make_service(assistant_config=config)
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"]
        assert final[0]["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_normalizes_configured_model(self):
        config = AssistantConfig(default_model="Anthropic/Claude-Sonnet-4-6")
        service = _make_service(assistant_config=config)
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"]
        assert final[0]["model"] == "anthropic:claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_falls_back_to_primary_model(self):
        service = _make_service()
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"]
        assert final[0]["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_applies_max_turns_usage_limit(self):
        config = AssistantConfig(max_turns=6)
        service = _make_service(assistant_config=config)
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        async for _ in service.stream_message(request):
            pass

        call_kwargs = mock_agent.iter.call_args.kwargs
        assert call_kwargs["usage_limits"].request_limit == 6


class TestStreamDebugEvents:
    @pytest.mark.asyncio
    async def test_debug_events_emitted_when_enabled(self):
        service = _make_service(streaming_config=StreamingConfig(emit_debug_events=True))
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=100,
            response_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            requests=1,
            total_tokens=150,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        types = [e["type"] for e in events]
        assert "debug_request" in types
        assert "debug_system_prompt" in types
        assert "debug_tool_selection" in types
        assert "debug_agent_config" in types
        assert "debug_history" in types
        assert "debug_tool_execution" in types
        assert "debug_usage" in types
        assert "debug_completed" in types

    @pytest.mark.asyncio
    async def test_debug_events_not_emitted_when_disabled(self):
        service = _make_service(streaming_config=StreamingConfig(emit_debug_events=False))
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        debug_events = [e for e in events if e["type"].startswith("debug_")]
        assert len(debug_events) == 0

    @pytest.mark.asyncio
    async def test_debug_request_event_content(self):
        service = _make_service(streaming_config=StreamingConfig(emit_debug_events=True))
        await service.start()
        request = _make_request(
            message="Test message",
            machine_state={"active_page": {"name": "home"}},
        )

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        debug_req = [e for e in events if e["type"] == "debug_request"][0]
        assert debug_req["session_id"] == "test-session"
        assert debug_req["message"] == "Test message"
        assert debug_req["is_continuation"] is False
        assert debug_req["has_machine_state"] is True
        assert debug_req["machine_state"] == {"active_page": {"name": "home"}}

    @pytest.mark.asyncio
    async def test_debug_completed_has_duration(self):
        service = _make_service(streaming_config=StreamingConfig(emit_debug_events=True))
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        completed = [e for e in events if e["type"] == "debug_completed"][0]
        assert "duration_ms" in completed
        assert completed["duration_ms"] >= 0
        assert completed["has_deferred"] is False


class TestStreamBackendToolCalls:
    """Tests for backend tool call SSE event emission."""

    @pytest.mark.asyncio
    async def test_backend_tool_emits_tool_call_event(self):
        """Backend tool calls should emit tool_call SSE events."""
        service = _make_service()
        await service.start()

        tc = ToolCallPart(
            tool_name="list_agents", args={"filter": "active"}, tool_call_id="call_001"
        )

        call_node = _make_call_tools_node([tc])
        # After tool execution, next node is ModelRequestNode with tool return
        tr = ToolReturnPart(
            tool_name="list_agents",
            content="[agent1, agent2]",
            tool_call_id="call_001",
            timestamp=datetime.now(UTC),
        )
        next_model_node = _make_model_request_node([tr])

        # Nodes: CallToolsNode -> ModelRequestNode (with results) -> End
        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="Here are the agents")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 1
        assert tool_call_events[0]["tool_name"] == "list_agents"
        assert tool_call_events[0]["arguments"] == {"filter": "active"}
        assert tool_call_events[0]["call_id"] == "call_001"

    @pytest.mark.asyncio
    async def test_backend_tool_emits_tool_result_event(self):
        """Backend tools should emit tool_result with the execution result."""
        service = _make_service()
        await service.start()

        tc = ToolCallPart(tool_name="list_agents", args={}, tool_call_id="call_002")
        call_node = _make_call_tools_node([tc])

        tr = ToolReturnPart(
            tool_name="list_agents",
            content="agent1, agent2",
            tool_call_id="call_002",
            timestamp=datetime.now(UTC),
        )
        next_model_node = _make_model_request_node([tr])

        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_result_events) == 1
        assert tool_result_events[0]["tool_name"] == "list_agents"
        assert tool_result_events[0]["result"] == "agent1, agent2"
        assert tool_result_events[0]["call_id"] == "call_002"

    @pytest.mark.asyncio
    async def test_frontend_tools_not_emitted_as_backend(self):
        """Frontend (deferred) tools should NOT emit tool_call from _iterate_run."""
        service = _make_service()
        await service.start()

        # Set up a frontend tool in available tools
        service._tools.get_available_tools.return_value = ToolSet(
            frontend_tools=[
                ToolDefinition(
                    name="navigate",
                    description="Notify UI",
                    parameters_schema={},
                    category=ToolCategory.FRONTEND,
                )
            ]
        )

        # CallToolsNode has a frontend tool call
        tc = ToolCallPart(tool_name="navigate", args={"message": "hello"}, tool_call_id="call_003")
        call_node = _make_call_tools_node([tc])

        # For frontend-only calls, the run may end after tool execution
        mock_run = _MockAgentRun(nodes=[call_node], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        # No tool_call events from _iterate_run (frontend tools are filtered)
        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 0

    @pytest.mark.asyncio
    async def test_multiple_backend_tools_emit_multiple_events(self):
        """Multiple backend tool calls in one response should each get events."""
        service = _make_service()
        await service.start()

        tc1 = ToolCallPart(tool_name="list_agents", args={}, tool_call_id="call_a")
        tc2 = ToolCallPart(tool_name="check_status", args={"agent": "leo"}, tool_call_id="call_b")
        call_node = _make_call_tools_node([tc1, tc2])

        tr1 = ToolReturnPart(
            tool_name="list_agents",
            content="[leo, ike]",
            tool_call_id="call_a",
            timestamp=datetime.now(UTC),
        )
        tr2 = ToolReturnPart(
            tool_name="check_status",
            content="idle",
            tool_call_id="call_b",
            timestamp=datetime.now(UTC),
        )
        next_model_node = _make_model_request_node([tr1, tr2])

        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        tool_result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_call_events) == 2
        assert len(tool_result_events) == 2
        assert tool_call_events[0]["tool_name"] == "list_agents"
        assert tool_call_events[1]["tool_name"] == "check_status"
        assert tool_result_events[0]["result"] == "[leo, ike]"
        assert tool_result_events[1]["result"] == "idle"

    @pytest.mark.asyncio
    async def test_tool_call_before_tool_result_ordering(self):
        """tool_call events should appear before tool_result events."""
        service = _make_service()
        await service.start()

        tc = ToolCallPart(tool_name="list_agents", args={}, tool_call_id="call_ord")
        call_node = _make_call_tools_node([tc])
        tr = ToolReturnPart(
            tool_name="list_agents",
            content="ok",
            tool_call_id="call_ord",
            timestamp=datetime.now(UTC),
        )
        next_model_node = _make_model_request_node([tr])

        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_types = [e["type"] for e in events if e["type"] in ("tool_call", "tool_result")]
        assert tool_types == ["tool_call", "tool_result"]

    @pytest.mark.asyncio
    async def test_tool_result_empty_when_not_found(self):
        """If tool return part not found, result should be empty string."""
        service = _make_service()
        await service.start()

        tc = ToolCallPart(tool_name="list_agents", args={}, tool_call_id="call_missing")
        call_node = _make_call_tools_node([tc])
        # Next node has no matching ToolReturnPart
        next_model_node = _make_model_request_node([])

        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_result_events) == 1
        assert tool_result_events[0]["result"] == ""

    @pytest.mark.asyncio
    async def test_tool_call_with_string_args(self):
        """Tool calls with JSON string args should be parsed to dict."""
        service = _make_service()
        await service.start()

        # ToolCallPart with string args (JSON)
        tc = ToolCallPart(
            tool_name="send_message",
            args='{"to": "leo", "msg": "hello"}',
            tool_call_id="call_str",
        )
        call_node = _make_call_tools_node([tc])
        tr = ToolReturnPart(
            tool_name="send_message",
            content="sent",
            tool_call_id="call_str",
            timestamp=datetime.now(UTC),
        )
        next_model_node = _make_model_request_node([tr])

        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 1
        # args_as_dict() should parse the JSON string
        assert tool_call_events[0]["arguments"] == {"to": "leo", "msg": "hello"}


class TestStreamPartStartEvents:
    """Tests for PartStartEvent handling — initial content from new parts."""

    @pytest.mark.asyncio
    async def test_text_part_start_emits_text_delta(self):
        """PartStartEvent with TextPart should yield a text_delta event."""
        service = _make_service()
        await service.start()

        # ModelRequestNode that streams a PartStartEvent with TextPart
        stream_events = [
            PartStartEvent(index=0, part=TextPart(content="Hello from Google")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output="Hello from Google")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(text_deltas) >= 1
        assert text_deltas[0]["content"] == "Hello from Google"

    @pytest.mark.asyncio
    async def test_thinking_part_start_emits_thinking_delta(self):
        """PartStartEvent with ThinkingPart should yield a thinking_delta event."""
        service = _make_service()
        await service.start()

        stream_events = [
            PartStartEvent(index=0, part=ThinkingPart(content="Let me reason...")),
            PartStartEvent(index=1, part=TextPart(content="Here is the answer.")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output="Here is the answer.")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        thinking_deltas = [e for e in events if e["type"] == "thinking_delta"]
        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(thinking_deltas) >= 1
        assert thinking_deltas[0]["content"] == "Let me reason..."
        assert len(text_deltas) >= 1
        assert text_deltas[0]["content"] == "Here is the answer."

    @pytest.mark.asyncio
    async def test_part_start_followed_by_part_delta(self):
        """Both PartStartEvent and PartDeltaEvent should produce events."""
        service = _make_service()
        await service.start()

        stream_events = [
            PartStartEvent(index=0, part=TextPart(content="Hello")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" world")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output="Hello world")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(text_deltas) == 2
        assert text_deltas[0]["content"] == "Hello"
        assert text_deltas[1]["content"] == " world"

    @pytest.mark.asyncio
    async def test_empty_part_start_content_skipped(self):
        """PartStartEvent with empty content should not yield an event."""
        service = _make_service()
        await service.start()

        stream_events = [
            PartStartEvent(index=0, part=TextPart(content="")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="Actual content")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output="Actual content")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(text_deltas) == 1
        assert text_deltas[0]["content"] == "Actual content"


class TestPartStartChunking:
    """Tests for chunking large PartStartEvent content into smaller SSE events."""

    @pytest.mark.asyncio
    async def test_large_thinking_part_is_chunked(self):
        """ThinkingPart content above threshold is split into multiple events."""
        service = _make_service(
            streaming_config=StreamingConfig(
                part_start_chunk_threshold=50,
                part_start_chunk_size=20,
            ),
        )
        await service.start()

        # 100-char thinking content — above 50-char threshold
        thinking_content = "A" * 100
        stream_events = [
            PartStartEvent(index=0, part=ThinkingPart(content=thinking_content)),
            PartStartEvent(index=1, part=TextPart(content="Answer")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output="Answer")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        thinking_deltas = [e for e in events if e["type"] == "thinking_delta"]
        # 100 chars / 20 chunk_size = 5 chunks
        assert len(thinking_deltas) == 5
        assert all(len(e["content"]) == 20 for e in thinking_deltas)
        assert "".join(e["content"] for e in thinking_deltas) == thinking_content

    @pytest.mark.asyncio
    async def test_large_text_part_is_chunked(self):
        """TextPart content above threshold is split into multiple events."""
        service = _make_service(
            streaming_config=StreamingConfig(
                part_start_chunk_threshold=50,
                part_start_chunk_size=30,
            ),
        )
        await service.start()

        # 90-char text — above 50-char threshold, 30-char chunks → 3 events
        text_content = "B" * 90
        stream_events = [
            PartStartEvent(index=0, part=TextPart(content=text_content)),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output=text_content)
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(text_deltas) == 3
        assert "".join(e["content"] for e in text_deltas) == text_content

    @pytest.mark.asyncio
    async def test_small_content_not_chunked(self):
        """Content at or below threshold is emitted as a single event."""
        service = _make_service(
            streaming_config=StreamingConfig(
                part_start_chunk_threshold=200,
                part_start_chunk_size=100,
            ),
        )
        await service.start()

        # 150 chars — below 200-char threshold
        text_content = "C" * 150
        stream_events = [
            PartStartEvent(index=0, part=TextPart(content=text_content)),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output=text_content)
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(text_deltas) == 1
        assert text_deltas[0]["content"] == text_content

    @pytest.mark.asyncio
    async def test_exact_threshold_not_chunked(self):
        """Content exactly at threshold is NOT chunked (uses <= check)."""
        service = _make_service(
            streaming_config=StreamingConfig(
                part_start_chunk_threshold=100,
                part_start_chunk_size=50,
            ),
        )
        await service.start()

        text_content = "D" * 100  # exactly at threshold
        stream_events = [
            PartStartEvent(index=0, part=TextPart(content=text_content)),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output=text_content)
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(text_deltas) == 1

    @pytest.mark.asyncio
    async def test_uneven_chunk_remainder(self):
        """Content that doesn't divide evenly produces a shorter final chunk."""
        service = _make_service(
            streaming_config=StreamingConfig(
                part_start_chunk_threshold=50,
                part_start_chunk_size=30,
            ),
        )
        await service.start()

        text_content = "E" * 70  # 30 + 30 + 10
        stream_events = [
            PartStartEvent(index=0, part=TextPart(content=text_content)),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output=text_content)
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(text_deltas) == 3
        assert len(text_deltas[0]["content"]) == 30
        assert len(text_deltas[1]["content"]) == 30
        assert len(text_deltas[2]["content"]) == 10


class TestStreamTracePersistence:
    """Tests for _save_trace — debug event collection and DB persistence."""

    @pytest.mark.asyncio
    async def test_save_trace_called_after_new_message(self):
        """When DB and debug events are present, trace is persisted."""
        mock_db = MagicMock()
        mock_db_session = AsyncMock()
        mock_db.session_context = MagicMock(return_value=mock_db_session)
        mock_db_session.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock_db_session.__aexit__ = AsyncMock(return_value=None)
        # Mock the repository
        mock_db_session.add = MagicMock()
        mock_db_session.flush = AsyncMock()

        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=True),
            database_service=mock_db,
        )
        await service.start()
        request = _make_request(message="Hello trace test")

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        async for _ in service.stream_message(request):
            pass

        # session_context should have been called (trace persistence)
        mock_db.session_context.assert_called()

    @pytest.mark.asyncio
    async def test_save_trace_skipped_when_no_db(self):
        """When no DB, _save_trace is a no-op — no error."""
        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=True),
            database_service=None,
        )
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        # Should complete without error
        types = [e["type"] for e in events]
        assert "agent_status" in types

    @pytest.mark.asyncio
    async def test_save_trace_skipped_when_debug_disabled(self):
        """When debug events are disabled, no trace events to save."""
        mock_db = MagicMock()

        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=False),
            database_service=mock_db,
        )
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        async for _ in service.stream_message(request):
            pass

        # No trace events → _save_trace returns early
        mock_db.session_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_trace_failure_does_not_break_stream(self):
        """DB failure in _save_trace is logged, not raised."""
        mock_db = MagicMock()
        mock_db.session_context = MagicMock(side_effect=RuntimeError("DB down"))

        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=True),
            database_service=mock_db,
        )
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        # Stream should still complete normally (completed status present)
        completed = [
            e for e in events if e.get("type") == "agent_status" and e.get("status") == "completed"
        ]
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_save_trace_for_continuation(self):
        """Continuations also persist traces."""
        mock_db = MagicMock()
        mock_db_session = AsyncMock()
        mock_db.session_context = MagicMock(return_value=mock_db_session)
        mock_db_session.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock_db_session.__aexit__ = AsyncMock(return_value=None)
        mock_db_session.add = MagicMock()
        mock_db_session.flush = AsyncMock()

        sessions = SessionStore()
        sessions.get_context("test-session")
        sessions.set_pending_tool_call("test-session", "call_123", "navigate")

        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=True),
            sessions=sessions,
            database_service=mock_db,
        )
        await service.start()

        mock_run = _MockAgentRun(nodes=[], output="Continued")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request(
            tool_call_id="call_123",
            tool_result="Tool done",
        )
        async for _ in service.stream_message(request):
            pass

        mock_db.session_context.assert_called()


class TestStreamTimeout:
    """Tests for streaming timeout enforcement via asyncio.timeout()."""

    @pytest.mark.asyncio
    async def test_timeout_emits_error_event(self):
        """When agent.iter() hangs beyond timeout, an error event is emitted."""
        import asyncio

        config = StreamingConfig(emit_debug_events=False)
        service = _make_service(streaming_config=config)
        # Use a mock config with a fast timeout for testing
        mock_config = MagicMock()
        mock_config.stream_timeout_seconds = 0.1
        mock_config.emit_debug_events = False
        mock_config.max_events_per_stream = 10000
        mock_config.part_start_chunk_threshold = 200
        mock_config.part_start_chunk_size = 100
        service._config = mock_config
        await service.start()
        request = _make_request()

        # Mock agent.iter() to hang indefinitely
        class _HangingRun:
            def __init__(self):
                self.ctx = MagicMock()
                self.result = MagicMock()
                self.result.output = "never reached"
                self.result.all_messages.return_value = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            @property
            def next_node(self):
                return MagicMock(spec=ModelRequestNode)

            async def next(self, node):
                await asyncio.sleep(10)  # Will be cancelled by timeout
                return End(data="never reached")

        hanging_run = _HangingRun()
        # The stream() method also needs to hang
        hanging_run.next_node.stream = MagicMock(side_effect=lambda *a, **kw: _hanging_stream())

        @asynccontextmanager
        async def _hanging_stream():
            async def _iter():
                await asyncio.sleep(10)  # Will be cancelled
                yield  # never reached

            yield _iter()

        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=hanging_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "timed out" in error_events[0]["message"]
        assert error_events[0]["error_type"] == "timeout"
        assert error_events[0]["terminal"] is True
        assert error_events[0]["retry_allowed"] is True

        # Should still emit completed status
        completed = [
            e for e in events if e.get("type") == "agent_status" and e.get("status") == "completed"
        ]
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_no_timeout_when_fast_enough(self):
        """Normal fast responses complete without timeout issues."""
        config = StreamingConfig(stream_timeout_seconds=10.0, emit_debug_events=False)
        service = _make_service(streaming_config=config)
        await service.start()
        request = _make_request()

        mock_run = _MockAgentRun(nodes=[], output="Fast response")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 0
        final = [e for e in events if e["type"] == "final_response"]
        assert final[0]["content"] == "Fast response"


class TestStreamToolCallInterleaving:
    """Tests that tool_call events are emitted during streaming (_stream_node)
    at the correct position, rather than being batched by _iterate_run."""

    @pytest.mark.asyncio
    async def test_tool_call_emitted_during_streaming_at_correct_position(self):
        """Tool call events interleave with text deltas in stream order."""
        service = _make_service()
        await service.start()

        tc = ToolCallPart(tool_name="get_status", args={"agent": "leo"}, tool_call_id="call_s1")
        stream_events = [
            PartStartEvent(index=0, part=TextPart(content="Let me check.")),
            PartStartEvent(index=1, part=tc),
            PartStartEvent(index=2, part=TextPart(content="Now checking Ada.")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        call_node = _make_call_tools_node([tc])

        tr = ToolReturnPart(
            tool_name="get_status",
            content="idle",
            tool_call_id="call_s1",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        next_model_node = _make_model_request_node([tr])

        mock_run = _MockAgentRun(nodes=[model_node, call_node, next_model_node], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        relevant = [e for e in events if e["type"] in ("text_delta", "tool_call", "tool_result")]
        assert len(relevant) == 4
        assert relevant[0]["type"] == "text_delta"
        assert relevant[0]["content"] == "Let me check."
        assert relevant[1]["type"] == "tool_call"
        assert relevant[1]["tool_name"] == "get_status"
        assert relevant[1]["call_id"] == "call_s1"
        assert relevant[2]["type"] == "text_delta"
        assert relevant[2]["content"] == "Now checking Ada."
        assert relevant[3]["type"] == "tool_result"
        assert relevant[3]["tool_name"] == "get_status"

    @pytest.mark.asyncio
    async def test_tool_call_not_duplicated_in_call_tools_node(self):
        """A tool call streamed from _stream_node is NOT re-emitted by CallToolsNode."""
        service = _make_service()
        await service.start()

        tc = ToolCallPart(tool_name="get_status", args={"agent": "leo"}, tool_call_id="call_s1")
        stream_events = [
            PartStartEvent(index=0, part=TextPart(content="Let me check.")),
            PartStartEvent(index=1, part=tc),
            PartStartEvent(index=2, part=TextPart(content="Now checking Ada.")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        call_node = _make_call_tools_node([tc])

        tr = ToolReturnPart(
            tool_name="get_status",
            content="idle",
            tool_call_id="call_s1",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        next_model_node = _make_model_request_node([tr])

        mock_run = _MockAgentRun(nodes=[model_node, call_node, next_model_node], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 1

    @pytest.mark.asyncio
    async def test_frontend_tool_not_emitted_during_streaming(self):
        """Frontend tool calls in _stream_node are skipped (not emitted)."""
        service = _make_service()
        await service.start()

        service._tools.get_available_tools.return_value = ToolSet(
            frontend_tools=[
                ToolDefinition(
                    name="navigate",
                    description="Nav",
                    parameters_schema={},
                    category=ToolCategory.FRONTEND,
                )
            ]
        )

        stream_events = [
            PartStartEvent(
                index=0,
                part=ToolCallPart(
                    tool_name="navigate",
                    args={"path": "/agents"},
                    tool_call_id="call_f1",
                ),
            ),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 0

    @pytest.mark.asyncio
    async def test_interleaved_multi_tool_streaming(self):
        """Multiple tool calls interleave correctly with text deltas."""
        service = _make_service()
        await service.start()

        tc_a = ToolCallPart(tool_name="tool_a", args={}, tool_call_id="call_a1")
        tc_b = ToolCallPart(tool_name="tool_b", args={"x": 1}, tool_call_id="call_b1")

        stream_events = [
            PartStartEvent(index=0, part=TextPart(content="First")),
            PartStartEvent(index=1, part=tc_a),
            PartStartEvent(index=2, part=TextPart(content="Second")),
            PartStartEvent(index=3, part=tc_b),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        call_node = _make_call_tools_node([tc_a, tc_b])

        tr_a = ToolReturnPart(
            tool_name="tool_a",
            content="result_a",
            tool_call_id="call_a1",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        tr_b = ToolReturnPart(
            tool_name="tool_b",
            content="result_b",
            tool_call_id="call_b1",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        next_model_node = _make_model_request_node([tr_a, tr_b])

        mock_run = _MockAgentRun(nodes=[model_node, call_node, next_model_node], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        relevant = [e for e in events if e["type"] in ("text_delta", "tool_call", "tool_result")]
        assert len(relevant) == 6
        assert relevant[0]["type"] == "text_delta"
        assert relevant[0]["content"] == "First"
        assert relevant[1]["type"] == "tool_call"
        assert relevant[1]["tool_name"] == "tool_a"
        assert relevant[2]["type"] == "text_delta"
        assert relevant[2]["content"] == "Second"
        assert relevant[3]["type"] == "tool_call"
        assert relevant[3]["tool_name"] == "tool_b"
        assert relevant[4]["type"] == "tool_result"
        assert relevant[4]["tool_name"] == "tool_a"
        assert relevant[4]["result"] == "result_a"
        assert relevant[5]["type"] == "tool_result"
        assert relevant[5]["tool_name"] == "tool_b"
        assert relevant[5]["result"] == "result_b"

        # Verify no duplicate tool_call events from CallToolsNode
        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 2


# --- Tool Error Detection Tests ---


class TestToolErrorDetection:
    """Tests for _is_tool_error and _extract_tool_result_raw static methods."""

    def test_is_tool_error_detects_error_key(self):
        is_error, msg = StreamingService._is_tool_error({"error": "Something failed"})
        assert is_error is True
        assert msg == "Something failed"

    def test_is_tool_error_detects_error_code_key(self):
        is_error, msg = StreamingService._is_tool_error({"error_code": "TOOL_EXECUTION_ERROR"})
        assert is_error is True
        assert msg == "TOOL_EXECUTION_ERROR"

    def test_is_tool_error_false_for_success_dict(self):
        is_error, msg = StreamingService._is_tool_error({"result": "ok", "count": 5})
        assert is_error is False
        assert msg == ""

    def test_is_tool_error_false_for_string(self):
        is_error, msg = StreamingService._is_tool_error("just a string result")
        assert is_error is False
        assert msg == ""

    def test_is_tool_error_false_for_none(self):
        is_error, msg = StreamingService._is_tool_error(None)
        assert is_error is False
        assert msg == ""

    def test_extract_tool_result_raw_returns_content(self):
        tr = ToolReturnPart(
            tool_name="get_time",
            content={"time": "12:00"},
            tool_call_id="call_1",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        node = _make_model_request_node([tr])
        result = StreamingService._extract_tool_result_raw(node, "call_1")
        assert result == {"time": "12:00"}

    def test_extract_tool_result_raw_returns_none_for_missing(self):
        node = _make_model_request_node([])
        result = StreamingService._extract_tool_result_raw(node, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_tool_error_emitted_for_error_result(self):
        """When a tool returns an error dict, tool_error event is emitted instead of tool_result."""
        service = _make_service()
        await service.start()

        tc = ToolCallPart(tool_name="get_time", args={}, tool_call_id="call_err")
        call_node = _make_call_tools_node([tc])

        error_content = {"error": "API unavailable", "error_code": "SERVICE_DOWN"}
        tr = ToolReturnPart(
            tool_name="get_time",
            content=error_content,
            tool_call_id="call_err",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        next_model_node = _make_model_request_node([tr])

        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="Error noted")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        events = []
        async for event in service.stream_message(_make_request()):
            events.append(event)

        error_events = [e for e in events if e["type"] == "tool_error"]
        assert len(error_events) == 1
        assert error_events[0]["tool_name"] == "get_time"
        assert error_events[0]["error"] == "API unavailable"

        # No tool_result events for this tool
        result_events = [
            e for e in events if e["type"] == "tool_result" and e["tool_name"] == "get_time"
        ]
        assert len(result_events) == 0


class TestStreamFinalResponseMetadata:
    """Tests that final_response reflects actual streaming state."""

    @pytest.mark.asyncio
    async def test_streaming_text_deltas_clears_content(self):
        """When text deltas are streamed, final_response has empty content and streamed=True."""
        service = _make_service()
        await service.start()

        stream_events = [
            PartStartEvent(index=0, part=TextPart(content="Hello")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" world")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output="Hello world")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"][0]
        assert final["content"] == ""
        assert final["streamed"] is True

    @pytest.mark.asyncio
    async def test_no_streaming_preserves_content(self):
        """When no deltas are emitted, final_response has full content and streamed=False."""
        service = _make_service()
        await service.start()

        mock_run = _MockAgentRun(nodes=[], output="Direct response")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"][0]
        assert final["content"] == "Direct response"
        assert final["streamed"] is False

    @pytest.mark.asyncio
    async def test_thinking_deltas_set_thinking_streamed(self):
        """When thinking deltas are streamed, final_response has thinking_streamed=True."""
        service = _make_service()
        await service.start()

        stream_events = [
            PartStartEvent(index=0, part=ThinkingPart(content="Let me think")),
            PartStartEvent(index=1, part=TextPart(content="Answer")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output="Answer")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service._llm.build_agent.return_value = mock_agent

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"][0]
        assert final["thinking_streamed"] is True
