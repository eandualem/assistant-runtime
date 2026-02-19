"""Tests for StreamingService — the SSE streaming orchestrator."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_graph.nodes import End

from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.config import AssistantConfig
from lovely_assistant.app.assistant.models import AssistantRequest
from lovely_assistant.app.streaming.config import StreamingConfig
from lovely_assistant.app.streaming.exceptions import (
    StreamExecutionError,
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
) -> StreamingService:
    """Create a StreamingService with mocked dependencies."""
    llm_service = MagicMock()
    llm_service._config.primary_model = "test-model"

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

    return StreamingService(
        config=streaming_config or StreamingConfig(emit_debug_events=False),
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
        assistant_service=mock_assistant_service,
        assistant_config=assistant_config or AssistantConfig(),
    )


def _make_request(**kwargs) -> AssistantRequest:
    """Create a test AssistantRequest."""
    defaults = {
        "session_id": "test-session",
        "message": "Hello",
    }
    defaults.update(kwargs)
    return AssistantRequest(**defaults)


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
        assert final[0]["streamed"] is True

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


class TestStreamWithTools:
    @pytest.mark.asyncio
    async def test_builds_agent_with_frontend_tools(self):
        service = _make_service()
        await service.start()

        service._tools.get_available_tools.return_value = ToolSet(
            frontend_tools=[
                ToolDefinition(
                    name="ui_notify",
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
        mock_call.tool_name = "ui_notify"
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
        assert tool_call_events[0]["tool_name"] == "ui_notify"
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

    @pytest.mark.asyncio
    async def test_deferred_tool_final_response_null_content(self):
        sessions = SessionStore()
        service = _make_service(sessions=sessions)
        await service.start()

        mock_call = MagicMock()
        mock_call.tool_call_id = "call_123"
        mock_call.tool_name = "ui_notify"
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
        sessions.set_pending_tool_call("test-session", "call_123", "ui_notify")

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
    async def test_agent_iter_failure_raises_execution_error(self):
        service = _make_service()
        await service.start()
        request = _make_request()

        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(side_effect=RuntimeError("iter fail"))
        service._llm.build_agent.return_value = mock_agent

        with pytest.raises(StreamExecutionError, match="Streaming failed"):
            async for _ in service.stream_message(request):
                pass


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
