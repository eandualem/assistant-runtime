"""Tests for StreamingService — the SSE streaming orchestrator."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.usage import UsageLimits
from pydantic_graph.nodes import End

from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.config import AssistantConfig
from lovely_assistant.app.assistant.models import AgentSetupContext, AssistantRequest, PromptResult
from lovely_assistant.app.settings import EffectiveConfig, RuntimeSettings
from lovely_assistant.app.streaming._coordinator import EventCoordinator
from lovely_assistant.app.streaming.config import StreamingConfig
from lovely_assistant.app.streaming.exceptions import StreamingError
from lovely_assistant.app.streaming.interface import StreamingService
from lovely_assistant.services.history.models import HistoryPreparationResult
from lovely_assistant.services.tools.models import ToolSet

# --- Helpers ---


def _make_default_agent_context(
    *,
    available_tools: ToolSet | None = None,
    resolved_model: str = "test-model",
    effective_config: EffectiveConfig | None = None,
    agent: Any = None,
) -> AgentSetupContext:
    """Build a default AgentSetupContext for tests."""
    tools = available_tools or ToolSet()
    config = effective_config or EffectiveConfig(
        default_model=None,
        thinking_budget=None,
        temperature=None,
        max_turns=10,
        enable_working_memory=True,
        summarization_model=None,
        working_memory_model=None,
        default_image_model=None,
        default_video_model=None,
        subagent_model=None,
        subagent_thinking_budget=None,
    )
    mock_agent = agent or MagicMock()
    return AgentSetupContext(
        agent=mock_agent,
        available_tools=tools,
        toolsets=[],
        prompt_result=PromptResult(content="You are Jarvis.", fragments=[]),
        resolved_model=resolved_model,
        usage_limits=UsageLimits(request_limit=config.max_turns),
        output_type=str,
        effective_config=config,
        mcp_summary=None,
    )


def _make_service(
    *,
    streaming_config: StreamingConfig | None = None,
    assistant_config: AssistantConfig | None = None,
    sessions: SessionStore | None = None,
    database_service: Any | None = None,
    agent_context: AgentSetupContext | None = None,
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
    tool_service.get_mcp_summary = AsyncMock(return_value=None)

    # Mock assistant service with public session store accessor and prepare_agent_context
    mock_assistant_service = MagicMock()
    mock_assistant_service.get_session_store = MagicMock(return_value=sessions or SessionStore())
    ctx = agent_context or _make_default_agent_context()
    mock_assistant_service.prepare_agent_context = AsyncMock(return_value=ctx)

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

    async def test_sessions_accesses_assistant_public_accessor(self):
        service = _make_service()
        store = service._assistant_service.get_session_store.return_value
        assert service._sessions is store
        service._assistant_service.get_session_store.assert_called()


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
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()
        request = _make_request()

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
        mock_run = _MockAgentRun(nodes=[], output="Hello world")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()
        request = _make_request()

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
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.all_messages.return_value = [MagicMock()]
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(
            sessions=sessions, agent_context=_make_default_agent_context(agent=mock_agent)
        )
        await service.start()
        request = _make_request()

        async for _ in service.stream_message(request):
            pass

        history = sessions.get_history("test-session")
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_increments_turn(self):
        sessions = SessionStore()
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(
            sessions=sessions, agent_context=_make_default_agent_context(agent=mock_agent)
        )
        await service.start()
        request = _make_request()

        async for _ in service.stream_message(request):
            pass

        ctx = sessions.get_context("test-session")
        assert ctx["turn_number"] == 1


class TestStreamWithImages:
    @pytest.mark.asyncio
    async def test_images_not_auto_attached_to_prompt(self):
        """Images are NOT auto-attached — prompt is always plain string."""
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()
        request = _make_request(
            message="What's here?",
            images=["data:image/jpeg;base64,/9j/4AAQ"],
        )

        async for _ in service.stream_message(request):
            pass

        call_args = mock_agent.iter.call_args
        user_prompt = call_args[0][0]
        assert isinstance(user_prompt, str)
        assert user_prompt == "What's here?"

    @pytest.mark.asyncio
    async def test_no_images_passes_plain_string(self):
        """When no images, agent.iter() receives plain string."""
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()
        request = _make_request(message="Hello there")

        async for _ in service.stream_message(request):
            pass

        call_args = mock_agent.iter.call_args
        assert call_args[0][0] == "Hello there"


class TestStreamSetupErrors:
    @pytest.mark.asyncio
    async def test_prepare_agent_context_failure_emits_lifecycle_envelope(self):
        """Setup failures emit started → final_response(error) → error → completed."""
        service = _make_service()
        await service.start()

        service._assistant_service.prepare_agent_context = AsyncMock(
            side_effect=RuntimeError("tool fail")
        )
        request = _make_request()

        events = [e async for e in service.stream_message(request)]

        # Must contain lifecycle envelope so frontend exits "thinking"
        assert any(e.get("status") == "started" for e in events)
        assert any(e.get("status") == "completed" for e in events)
        assert any(e.get("type") == "final_response" for e in events)
        assert any(e.get("type") == "error" for e in events)

    @pytest.mark.asyncio
    async def test_history_failure_emits_lifecycle_envelope(self):
        """History setup failure emits full lifecycle envelope."""
        service = _make_service()
        await service.start()

        service._history.prepare_history_with_metadata.side_effect = RuntimeError("history fail")
        request = _make_request()

        events = [e async for e in service.stream_message(request)]

        assert any(e.get("status") == "started" for e in events)
        assert any(e.get("status") == "completed" for e in events)
        assert any(e.get("type") == "error" for e in events)

    @pytest.mark.asyncio
    async def test_continuation_missing_session_rejected_without_creating_session(self):
        sessions = SessionStore()
        service = _make_service(sessions=sessions)
        await service.start()

        request = _make_request(
            message="",
            tool_call_id="call_missing_session",
            tool_result={"status": "ok"},
        )

        events = [e async for e in service.stream_message(request)]

        assert sessions.session_count() == 0
        assert any(
            e.get("type") == "final_response" and e.get("error_type") == "session_error"
            for e in events
        )
        assert any("does not exist" in e.get("message", "") for e in events if e["type"] == "error")
        assert any(e.get("status") == "completed" for e in events)

    @pytest.mark.asyncio
    async def test_continuation_missing_pending_tool_call_rejected(self):
        sessions = SessionStore()
        sessions.get_context("test-session")
        service = _make_service(sessions=sessions)
        await service.start()

        request = _make_request(
            message="",
            tool_call_id="call_missing_pending",
            tool_result={"status": "ok"},
        )

        events = [e async for e in service.stream_message(request)]

        assert any(
            "has no pending tool call" in e.get("message", "")
            for e in events
            if e["type"] == "error"
        )
        assert any(
            e.get("type") == "final_response" and e.get("error_type") == "session_error"
            for e in events
        )

    @pytest.mark.asyncio
    async def test_continuation_mismatched_pending_tool_call_rejected(self):
        sessions = SessionStore()
        session_context = sessions.get_context("test-session")
        session_context["pending_tool_call_id"] = "call_expected"
        session_context["pending_tool_name"] = "navigate"
        service = _make_service(sessions=sessions)
        await service.start()

        request = _make_request(
            message="",
            tool_call_id="call_actual",
            tool_result={"status": "ok"},
        )

        events = [e async for e in service.stream_message(request)]

        assert any(
            "does not match pending tool call" in e.get("message", "")
            for e in events
            if e["type"] == "error"
        )
        assert any(
            e.get("type") == "final_response" and e.get("error_type") == "session_error"
            for e in events
        )


class TestStreamExecutionErrors:
    @pytest.mark.asyncio
    async def test_agent_iter_failure_emits_error_event(self):
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(side_effect=RuntimeError("iter fail"))
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()
        request = _make_request()

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
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        ctx = _make_default_agent_context(resolved_model="custom-model", agent=mock_agent)
        service = _make_service(agent_context=ctx)
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"]
        assert final[0]["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_normalizes_configured_model(self):
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        ctx = _make_default_agent_context(
            resolved_model="anthropic:claude-sonnet-4-6", agent=mock_agent
        )
        service = _make_service(agent_context=ctx)
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"]
        assert final[0]["model"] == "anthropic:claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_falls_back_to_primary_model(self):
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"]
        assert final[0]["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_applies_max_turns_usage_limit(self):
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        config = EffectiveConfig(
            default_model=None,
            thinking_budget=None,
            temperature=None,
            max_turns=6,
            enable_working_memory=True,
            summarization_model=None,
            working_memory_model=None,
            default_image_model=None,
            default_video_model=None,
            subagent_model=None,
            subagent_thinking_budget=None,
        )
        ctx = _make_default_agent_context(effective_config=config, agent=mock_agent)
        service = _make_service(agent_context=ctx)
        await service.start()
        request = _make_request()

        async for _ in service.stream_message(request):
            pass

        call_kwargs = mock_agent.iter.call_args.kwargs
        assert call_kwargs["usage_limits"].request_limit == 6


class TestStreamDebugEvents:
    @pytest.mark.asyncio
    async def test_debug_events_emitted_when_enabled(self):
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
        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=True),
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        types = [e["type"] for e in events]
        assert "debug_request" in types
        assert "debug_system_prompt" in types
        assert "debug_tool_selection" in types
        assert "debug_agent_config" in types
        assert "debug_history" in types
        assert "debug_usage" in types
        assert "debug_completed" in types

    @pytest.mark.asyncio
    async def test_debug_events_not_emitted_when_disabled(self):
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=False),
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        debug_events = [e for e in events if e["type"].startswith("debug_")]
        assert len(debug_events) == 0

    @pytest.mark.asyncio
    async def test_debug_request_event_content(self):
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=True),
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()
        request = _make_request(
            message="Test message",
            machine_state={"active_page": {"name": "home"}},
        )

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
    async def test_debug_agent_config_includes_session_id(self):
        """debug_agent_config event includes session_id when debug events enabled."""
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=True),
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()
        request = _make_request(session_id="sess-debug")

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        agent_config = [e for e in events if e["type"] == "debug_agent_config"][0]
        assert agent_config["session_id"] == "sess-debug"

    @pytest.mark.asyncio
    async def test_debug_completed_has_duration(self):
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=True),
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        completed = [e for e in events if e["type"] == "debug_completed"][0]
        assert "duration_ms" in completed
        assert completed["duration_ms"] >= 0


class TestStreamBackendToolCalls:
    """Tests for backend tool call SSE event emission."""

    def _make_service_with_agent(self, mock_agent, **kwargs):
        return _make_service(agent_context=_make_default_agent_context(agent=mock_agent), **kwargs)

    @pytest.mark.asyncio
    async def test_backend_tool_emits_tool_call_event(self):
        """Backend tool calls should emit tool_call SSE events."""
        tc = ToolCallPart(
            tool_name="list_agents", args={"filter": "active"}, tool_call_id="call_001"
        )
        call_node = _make_call_tools_node([tc])
        tr = ToolReturnPart(
            tool_name="list_agents",
            content="[agent1, agent2]",
            tool_call_id="call_001",
            timestamp=datetime.now(UTC),
        )
        next_model_node = _make_model_request_node([tr])

        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="Here are the agents")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = self._make_service_with_agent(mock_agent)
        await service.start()

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
        service = self._make_service_with_agent(mock_agent)
        await service.start()

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_result_events) == 1
        assert tool_result_events[0]["tool_name"] == "list_agents"
        assert tool_result_events[0]["output"] == "agent1, agent2"
        assert tool_result_events[0]["call_id"] == "call_002"

    @pytest.mark.asyncio
    async def test_multiple_backend_tools_emit_multiple_events(self):
        """Multiple backend tool calls in one response should each get events."""
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
        service = self._make_service_with_agent(mock_agent)
        await service.start()

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
        assert tool_result_events[0]["output"] == "[leo, ike]"
        assert tool_result_events[1]["output"] == "idle"

    @pytest.mark.asyncio
    async def test_tool_call_before_tool_result_ordering(self):
        """tool_call events should appear before tool_result events."""
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
        service = self._make_service_with_agent(mock_agent)
        await service.start()

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_types = [e["type"] for e in events if e["type"] in ("tool_call", "tool_result")]
        assert tool_types == ["tool_call", "tool_result"]

    @pytest.mark.asyncio
    async def test_tool_result_empty_when_not_found(self):
        """If tool return part not found, result should be empty string."""
        tc = ToolCallPart(tool_name="list_agents", args={}, tool_call_id="call_missing")
        call_node = _make_call_tools_node([tc])
        next_model_node = _make_model_request_node([])

        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = self._make_service_with_agent(mock_agent)
        await service.start()

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_result_events) == 1
        assert tool_result_events[0]["output"] == ""

    @pytest.mark.asyncio
    async def test_tool_call_with_string_args(self):
        """Tool calls with JSON string args should be parsed to dict."""
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
        service = self._make_service_with_agent(mock_agent)
        await service.start()

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 1
        assert tool_call_events[0]["arguments"] == {"to": "leo", "msg": "hello"}


class TestStreamPartStartEvents:
    """Tests for PartStartEvent handling — initial content from new parts."""

    @pytest.mark.asyncio
    async def test_text_part_start_emits_text_delta(self):
        """PartStartEvent with TextPart should yield a text_delta event."""
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
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

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
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

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
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

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
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

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
        service = _make_service(
            streaming_config=StreamingConfig(
                part_start_chunk_threshold=50,
                part_start_chunk_size=20,
            ),
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()

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
        service = _make_service(
            streaming_config=StreamingConfig(
                part_start_chunk_threshold=50,
                part_start_chunk_size=30,
            ),
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()

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
        service = _make_service(
            streaming_config=StreamingConfig(
                part_start_chunk_threshold=200,
                part_start_chunk_size=100,
            ),
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()

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
        service = _make_service(
            streaming_config=StreamingConfig(
                part_start_chunk_threshold=100,
                part_start_chunk_size=50,
            ),
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(text_deltas) == 1

    @pytest.mark.asyncio
    async def test_uneven_chunk_remainder(self):
        """Content that doesn't divide evenly produces a shorter final chunk."""
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
        service = _make_service(
            streaming_config=StreamingConfig(
                part_start_chunk_threshold=50,
                part_start_chunk_size=30,
            ),
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()

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

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=True),
            database_service=mock_db,
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()
        request = _make_request(message="Hello trace test")

        async for _ in service.stream_message(request):
            pass

        # session_context should have been called (trace persistence)
        mock_db.session_context.assert_called()

    @pytest.mark.asyncio
    async def test_save_trace_skipped_when_no_db(self):
        """When no DB, _save_trace is a no-op — no error."""
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=True),
            database_service=None,
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        # Should complete without error
        types = [e["type"] for e in events]
        assert "agent_status" in types

    @pytest.mark.asyncio
    async def test_save_trace_skipped_when_debug_disabled(self):
        """When debug events are disabled, no trace events to save."""
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=False),
            database_service=None,
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        # No DB → no trace saved, no artifact loading — completes cleanly
        types = [e["type"] for e in events]
        assert "agent_status" in types

    @pytest.mark.asyncio
    async def test_save_trace_failure_does_not_break_stream(self):
        """DB failure in _save_trace is logged, not raised."""
        mock_db = MagicMock()
        mock_db.session_context = MagicMock(side_effect=RuntimeError("DB down"))

        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_run.result.usage.return_value = MagicMock(
            request_tokens=0,
            response_tokens=0,
            requests=0,
            total_tokens=0,
        )
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(
            streaming_config=StreamingConfig(emit_debug_events=True),
            database_service=mock_db,
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        # Stream should still complete normally (completed status present)
        completed = [
            e for e in events if e.get("type") == "agent_status" and e.get("status") == "completed"
        ]
        assert len(completed) == 1


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
        service._assistant_service.prepare_agent_context.return_value = _make_default_agent_context(
            agent=mock_agent
        )

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
        mock_run = _MockAgentRun(nodes=[], output="Fast response")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        config = StreamingConfig(stream_timeout_seconds=10.0, emit_debug_events=False)
        service = _make_service(
            streaming_config=config,
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 0
        final = [e for e in events if e["type"] == "final_response"]
        assert final[0]["content"] == "Fast response"


class TestStreamToolCallTiming:
    """Tests that tool_call events are emitted from CallToolsNode with complete
    arguments, NOT during _stream_node (where args are still empty)."""

    @pytest.mark.asyncio
    async def test_tool_call_emitted_after_streaming_with_complete_args(self):
        """Tool call events appear after text deltas, with full arguments."""
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
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        relevant = [e for e in events if e["type"] in ("text_delta", "tool_call", "tool_result")]
        assert len(relevant) == 4
        # Text deltas come first (from streaming)
        assert relevant[0]["type"] == "text_delta"
        assert relevant[0]["content"] == "Let me check."
        assert relevant[1]["type"] == "text_delta"
        assert relevant[1]["content"] == "Now checking Ada."
        # tool_call after streaming with complete args (from CallToolsNode)
        assert relevant[2]["type"] == "tool_call"
        assert relevant[2]["tool_name"] == "get_status"
        assert relevant[2]["arguments"] == {"agent": "leo"}
        assert relevant[2]["call_id"] == "call_s1"
        # tool_result last
        assert relevant[3]["type"] == "tool_result"
        assert relevant[3]["tool_name"] == "get_status"

    @pytest.mark.asyncio
    async def test_tool_call_not_duplicated(self):
        """Each tool call emits exactly one tool_call event."""
        tc = ToolCallPart(tool_name="get_status", args={"agent": "leo"}, tool_call_id="call_s1")
        stream_events = [
            PartStartEvent(index=0, part=TextPart(content="Let me check.")),
            PartStartEvent(index=1, part=tc),
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
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 1

    @pytest.mark.asyncio
    async def test_multi_tool_calls_emitted_with_complete_args(self):
        """Multiple tool calls all get complete arguments from CallToolsNode."""
        tc_a = ToolCallPart(tool_name="tool_a", args={"key": "val_a"}, tool_call_id="call_a1")
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
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        relevant = [e for e in events if e["type"] in ("text_delta", "tool_call", "tool_result")]
        assert len(relevant) == 6
        # Text deltas from streaming
        assert relevant[0]["type"] == "text_delta"
        assert relevant[0]["content"] == "First"
        assert relevant[1]["type"] == "text_delta"
        assert relevant[1]["content"] == "Second"
        # tool_call events from CallToolsNode with complete args
        assert relevant[2]["type"] == "tool_call"
        assert relevant[2]["tool_name"] == "tool_a"
        assert relevant[2]["arguments"] == {"key": "val_a"}
        assert relevant[3]["type"] == "tool_call"
        assert relevant[3]["tool_name"] == "tool_b"
        assert relevant[3]["arguments"] == {"x": 1}
        # tool_result events
        assert relevant[4]["type"] == "tool_result"
        assert relevant[4]["tool_name"] == "tool_a"
        assert relevant[4]["output"] == "result_a"
        assert relevant[5]["type"] == "tool_result"
        assert relevant[5]["tool_name"] == "tool_b"
        assert relevant[5]["output"] == "result_b"

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
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

        events = []
        async for event in service.stream_message(_make_request()):
            events.append(event)

        error_events = [e for e in events if e["type"] == "tool_error"]
        assert len(error_events) == 1
        assert error_events[0]["tool_name"] == "get_time"
        assert error_events[0]["error"] == "API unavailable"

        # tool_result also emitted (dashboard needs it to close the tool card)
        result_events = [
            e for e in events if e["type"] == "tool_result" and e["tool_name"] == "get_time"
        ]
        assert len(result_events) == 1
        assert result_events[0]["output"] == "API unavailable"


class TestStreamFinalResponseMetadata:
    """Tests that final_response reflects actual streaming state."""

    @pytest.mark.asyncio
    async def test_streaming_text_deltas_clears_content(self):
        """When text deltas are streamed, final_response has empty content and streamed=True."""
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
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

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
        mock_run = _MockAgentRun(nodes=[], output="Direct response")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"][0]
        assert final["content"] == "Direct response"
        assert final["streamed"] is False

    @pytest.mark.asyncio
    async def test_final_response_includes_session_id(self):
        """final_response event includes the session_id."""
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

        request = _make_request(session_id="sess-abc")
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"][0]
        assert final["session_id"] == "sess-abc"

    @pytest.mark.asyncio
    async def test_thinking_deltas_set_thinking_streamed(self):
        """When thinking deltas are streamed, final_response has thinking_streamed=True."""
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
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

        request = _make_request()
        events = []
        async for event in service.stream_message(request):
            events.append(event)

        final = [e for e in events if e["type"] == "final_response"][0]
        assert final["thinking_streamed"] is True


class TestLookAtScreenDeferredSanitization:
    """Image sanitization is deferred until after the next ModelRequestNode streams.

    The bug: sanitize_image_tool_returns() was called immediately after CallToolsNode,
    stripping the image BEFORE the LLM saw it. The fix defers sanitization to after
    the next ModelRequestNode completes streaming.
    """

    @pytest.mark.asyncio
    async def test_image_sanitized_after_model_request(self):
        """After _iterate_run, look_at_screen images are sanitized from history."""
        from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

        tc = ToolCallPart(tool_name="look_at_screen", args={}, tool_call_id="call_screen")
        call_node = _make_call_tools_node([tc])
        tr = ToolReturnPart(
            tool_name="look_at_screen",
            content="screenshot bytes",
            tool_call_id="call_screen",
            timestamp=datetime.now(UTC),
        )
        next_model_node = _make_model_request_node([tr])

        # Build message_history with BinaryContent (what Pydantic AI puts there)
        image = BinaryContent(data=b"fake-png", media_type="image/png")
        history_request = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="look_at_screen",
                    content=image,
                    tool_call_id="call_screen",
                    timestamp=datetime.now(UTC),
                ),
                UserPromptPart(content=["Reference: look_at_screen", image]),
            ]
        )
        message_history = [history_request]

        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="I see a dashboard")
        mock_run.ctx.state.message_history = message_history

        service = _make_service()
        await service.start()
        coordinator = EventCoordinator(100)

        _ = [e async for e in service._iterate_run(mock_run, coordinator, "test-model")]

        # After _iterate_run, ToolReturnPart content should be the placeholder
        tool_return = history_request.parts[0]
        assert tool_return.content == "[Inspected current screen]"
        # Synthetic UserPromptPart with BinaryContent should be removed
        assert len(history_request.parts) == 1

    @pytest.mark.asyncio
    async def test_image_survives_until_model_streams(self):
        """The image stays in history while ModelRequestNode streams (LLM sees it)."""
        from unittest.mock import patch

        from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

        tc = ToolCallPart(tool_name="look_at_screen", args={}, tool_call_id="call_screen")
        call_node = _make_call_tools_node([tc])
        tr = ToolReturnPart(
            tool_name="look_at_screen",
            content="screenshot bytes",
            tool_call_id="call_screen",
            timestamp=datetime.now(UTC),
        )
        next_model_node = _make_model_request_node([tr])

        image = BinaryContent(data=b"fake-png", media_type="image/png")
        history_request = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="look_at_screen",
                    content=image,
                    tool_call_id="call_screen",
                    timestamp=datetime.now(UTC),
                ),
                UserPromptPart(content=["Reference: look_at_screen", image]),
            ]
        )
        message_history = [history_request]

        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="I see a dashboard")
        mock_run.ctx.state.message_history = message_history

        # Track call ordering: stream_node vs sanitize
        call_order: list[str] = []
        original_stream_node = StreamingService._stream_node

        async def recording_stream_node(node, run, coordinator):
            call_order.append("stream_start")
            async for event in original_stream_node(service, node, run, coordinator):
                yield event
            call_order.append("stream_end")

        service = _make_service()
        await service.start()
        coordinator = EventCoordinator(100)

        with (
            patch.object(service, "_stream_node", recording_stream_node),
            patch(
                "lovely_assistant.app.streaming.interface.sanitize_image_tool_returns",
                side_effect=lambda msgs: (call_order.append("sanitize"), msgs)[1],
            ),
        ):
            _ = [e async for e in service._iterate_run(mock_run, coordinator, "test-model")]

        # Sanitization must happen AFTER the model streams (LLM saw the image)
        assert call_order == ["stream_start", "stream_end", "sanitize"]

    @pytest.mark.asyncio
    async def test_no_sanitization_without_look_at_screen(self):
        """Regular tool calls don't trigger deferred sanitization."""
        from unittest.mock import patch

        tc = ToolCallPart(tool_name="list_agents", args={}, tool_call_id="call_001")
        call_node = _make_call_tools_node([tc])
        tr = ToolReturnPart(
            tool_name="list_agents",
            content="[agent1]",
            tool_call_id="call_001",
            timestamp=datetime.now(UTC),
        )
        next_model_node = _make_model_request_node([tr])

        mock_run = _MockAgentRun(nodes=[call_node, next_model_node], output="Done")
        mock_run.ctx.state.message_history = []

        service = _make_service()
        await service.start()
        coordinator = EventCoordinator(100)

        with patch(
            "lovely_assistant.app.streaming.interface.sanitize_image_tool_returns"
        ) as mock_sanitize:
            _ = [e async for e in service._iterate_run(mock_run, coordinator, "test-model")]

        mock_sanitize.assert_not_called()


class TestContinuationHistory:
    """Verify that the continuation path passes is_continuation=True to history preparation."""

    @pytest.mark.asyncio
    async def test_continuation_passes_is_continuation_flag(self):
        """_stream_continuation must pass is_continuation=True so dangling tool calls are preserved."""
        mock_run = _MockAgentRun(nodes=[], output="Continued response")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        sessions = SessionStore()
        sessions.get_context("test-session")["pending_tool_call_id"] = "call_ui_send_event_001"
        service = _make_service(
            sessions=sessions,
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()

        request = _make_request(
            message="",
            tool_call_id="call_ui_send_event_001",
            tool_result={"status": "ok"},
        )

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        # Verify prepare_history_with_metadata was called with is_continuation=True
        history_mock = service._history
        history_mock.prepare_history_with_metadata.assert_called_once()
        call_kwargs = history_mock.prepare_history_with_metadata.call_args
        assert call_kwargs.kwargs.get("is_continuation") is True

    @pytest.mark.asyncio
    async def test_new_message_does_not_pass_is_continuation(self):
        """_stream_new_message must NOT pass is_continuation=True (default False)."""
        mock_run = _MockAgentRun(nodes=[], output="Fresh response")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        await service.start()

        request = _make_request(message="Hello")

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        history_mock = service._history
        history_mock.prepare_history_with_metadata.assert_called_once()
        call_kwargs = history_mock.prepare_history_with_metadata.call_args
        # Should not have is_continuation=True (either absent or False)
        assert call_kwargs.kwargs.get("is_continuation", False) is False


class TestContinuationScreenshots:
    """Continuation requests can supply fresh screenshots via tool_result payloads."""

    @pytest.mark.asyncio
    async def test_continuation_uses_screenshot_from_tool_result_payload(self):
        mock_run = _MockAgentRun(nodes=[], output="Continued response")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        sessions = SessionStore()
        sessions.get_context("test-session")["pending_tool_call_id"] = "call_navigate_001"
        service = _make_service(
            sessions=sessions,
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()

        request = _make_request(
            message="",
            tool_call_id="call_navigate_001",
            tool_result={
                "status": "ok",
                "result": {"imageDataUri": "data:image/png;base64,fresh-screen"},
            },
        )

        with patch("lovely_assistant.app.streaming.interface.set_current_screenshot") as mock_set:
            async for _event in service.stream_message(request):
                pass

        mock_set.assert_called_once_with("data:image/png;base64,fresh-screen")

    @pytest.mark.asyncio
    async def test_continuation_prefers_explicit_request_images(self):
        mock_run = _MockAgentRun(nodes=[], output="Continued response")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        sessions = SessionStore()
        sessions.get_context("test-session")["pending_tool_call_id"] = "call_navigate_001"
        service = _make_service(
            sessions=sessions,
            agent_context=_make_default_agent_context(agent=mock_agent),
        )
        await service.start()

        request = _make_request(
            message="",
            images=["data:image/png;base64,top-level"],
            tool_call_id="call_navigate_001",
            tool_result={"screenshot": "data:image/png;base64,from-result"},
        )

        with patch("lovely_assistant.app.streaming.interface.set_current_screenshot") as mock_set:
            async for _event in service.stream_message(request):
                pass

        mock_set.assert_called_once_with("data:image/png;base64,top-level")


class TestStreamTraceChronologicalOrder:
    """Verify thinking is flushed per iteration for chronological trace ordering."""

    @pytest.mark.asyncio
    async def test_thinking_flushed_per_iteration(self):
        """Two model iterations produce two separate debug_thinking events at correct positions.

        Simulates: ModelRequestNode(thinking) → CallToolsNode → ModelRequestNode(thinking + text).
        Expected trace order: debug_thinking(iter1), tool_call, tool_result, debug_thinking(iter2).
        """
        # Iteration 1: ModelRequestNode with thinking
        stream_events_1 = [
            PartStartEvent(index=0, part=ThinkingPart(content="First thought")),
        ]
        model_node_1 = MagicMock(spec=ModelRequestNode)
        model_node_1.request = MagicMock(spec=ModelRequest, parts=[])
        model_node_1.stream = MagicMock(
            side_effect=lambda *a, **kw: _events_stream(stream_events_1)
        )

        # CallToolsNode
        tc = ToolCallPart(tool_name="get_status", args={"agent": "leo"}, tool_call_id="call_1")
        call_node = _make_call_tools_node([tc])

        # Tool result node
        tr = ToolReturnPart(
            tool_name="get_status",
            content="idle",
            tool_call_id="call_1",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )

        # Iteration 2: ModelRequestNode with thinking + text
        stream_events_2 = [
            PartStartEvent(index=0, part=ThinkingPart(content="Second thought")),
            PartStartEvent(index=1, part=TextPart(content="Final answer")),
        ]
        model_node_2 = MagicMock(spec=ModelRequestNode)
        model_node_2.request = MagicMock(spec=ModelRequest, parts=[tr])
        model_node_2.stream = MagicMock(
            side_effect=lambda *a, **kw: _events_stream(stream_events_2)
        )

        mock_run = _MockAgentRun(
            nodes=[model_node_1, call_node, model_node_2], output="Final answer"
        )
        service = _make_service()
        await service.start()
        coordinator = EventCoordinator(1000)

        # Collect all events from _iterate_run with emit_debug=True
        events = [
            e
            async for e in service._iterate_run(
                mock_run, coordinator, "test-model", emit_debug=True
            )
        ]

        # SSE events should include thinking_delta, tool_call, tool_result, text_delta
        sse_types = [e["type"] for e in events]
        assert "thinking_delta" in sse_types
        assert "tool_call" in sse_types
        assert "tool_result" in sse_types
        assert "text_delta" in sse_types

        # Debug events (trace) should have two separate debug_thinking events
        debug_events = coordinator.debug_events
        debug_thinking = [e for e in debug_events if e["type"] == "debug_thinking"]
        assert len(debug_thinking) == 2
        assert debug_thinking[0]["content"] == "First thought"
        assert debug_thinking[1]["content"] == "Second thought"

        # Verify chronological order in the trace:
        # debug_thinking(iter1) should come before tool_call
        # tool_call/tool_result should come before debug_thinking(iter2)
        debug_types = [e["type"] for e in debug_events]
        idx_think_1 = debug_types.index("debug_thinking")
        idx_tool_call = debug_types.index("tool_call")
        idx_tool_result = debug_types.index("tool_result")
        # Find second debug_thinking
        idx_think_2 = len(debug_types) - 1 - debug_types[::-1].index("debug_thinking")
        assert idx_think_1 < idx_tool_call < idx_tool_result < idx_think_2

    @pytest.mark.asyncio
    async def test_no_thinking_no_debug_event(self):
        """When a ModelRequestNode has no thinking, no debug_thinking event is added."""
        stream_events = [
            PartStartEvent(index=0, part=TextPart(content="Direct answer")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output="Direct answer")
        service = _make_service()
        await service.start()
        coordinator = EventCoordinator(100)

        _ = [
            e
            async for e in service._iterate_run(
                mock_run, coordinator, "test-model", emit_debug=True
            )
        ]

        debug_thinking = [e for e in coordinator.debug_events if e["type"] == "debug_thinking"]
        assert len(debug_thinking) == 0

    @pytest.mark.asyncio
    async def test_emit_debug_false_skips_flush(self):
        """When emit_debug=False, flush_thinking is not called (backward compat)."""
        stream_events = [
            PartStartEvent(index=0, part=ThinkingPart(content="Some reasoning")),
            PartStartEvent(index=1, part=TextPart(content="Answer")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output="Answer")
        service = _make_service()
        await service.start()
        coordinator = EventCoordinator(100)

        _ = [
            e
            async for e in service._iterate_run(
                mock_run, coordinator, "test-model", emit_debug=False
            )
        ]

        # No debug events should be in the trace
        debug_thinking = [e for e in coordinator.debug_events if e["type"] == "debug_thinking"]
        assert len(debug_thinking) == 0
        # With emit_debug=False, flush_thinking() is NOT called, so buffer retains content
        assert coordinator.accumulated_thinking == "Some reasoning"

    @pytest.mark.asyncio
    async def test_thinking_deltas_accumulated_across_chunks(self):
        """Multiple thinking deltas within one iteration are concatenated in the flush."""
        stream_events = [
            PartStartEvent(index=0, part=ThinkingPart(content="Start ")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="middle ")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="end")),
            PartStartEvent(index=1, part=TextPart(content="Answer")),
        ]
        model_node = MagicMock(spec=ModelRequestNode)
        model_node.request = MagicMock(spec=ModelRequest, parts=[])
        model_node.stream = MagicMock(side_effect=lambda *a, **kw: _events_stream(stream_events))

        mock_run = _MockAgentRun(nodes=[model_node], output="Answer")
        service = _make_service()
        await service.start()
        coordinator = EventCoordinator(100)

        _ = [
            e
            async for e in service._iterate_run(
                mock_run, coordinator, "test-model", emit_debug=True
            )
        ]

        debug_thinking = [e for e in coordinator.debug_events if e["type"] == "debug_thinking"]
        assert len(debug_thinking) == 1
        assert debug_thinking[0]["content"] == "Start middle end"


# --- Working memory in streaming path tests ---


class TestStreamingWorkingMemory:
    @pytest.mark.asyncio
    async def test_working_memory_called_after_stream(self):
        """_update_working_memory is called in the finally block of _stream_new_message."""
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        service._assistant_service._update_working_memory = AsyncMock()
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        service._assistant_service._update_working_memory.assert_called_once()
        call_args = service._assistant_service._update_working_memory.call_args
        assert call_args[0][0] == "test-session"  # session_id
        assert isinstance(call_args[0][1], dict)  # session_context
        assert isinstance(call_args[0][2], int)  # turn_number

    @pytest.mark.asyncio
    async def test_working_memory_skipped_when_disabled(self):
        """_update_working_memory is NOT called when enable_working_memory=False."""
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        config = EffectiveConfig(
            default_model=None,
            thinking_budget=None,
            temperature=None,
            max_turns=10,
            enable_working_memory=False,
            summarization_model=None,
            working_memory_model=None,
            default_image_model=None,
            default_video_model=None,
            subagent_model=None,
            subagent_thinking_budget=None,
        )
        service = _make_service(
            agent_context=_make_default_agent_context(agent=mock_agent, effective_config=config)
        )
        service._assistant_service._update_working_memory = AsyncMock()
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        service._assistant_service._update_working_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_working_memory_failure_does_not_break_stream(self):
        """_update_working_memory failure is best-effort — stream still completes."""
        mock_run = _MockAgentRun(nodes=[], output="Done")
        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)
        service = _make_service(agent_context=_make_default_agent_context(agent=mock_agent))
        service._assistant_service._update_working_memory = AsyncMock(
            side_effect=RuntimeError("memory extraction boom")
        )
        await service.start()
        request = _make_request()

        events = []
        async for event in service.stream_message(request):
            events.append(event)

        # Stream should still complete normally
        types = [e["type"] for e in events]
        assert "agent_status" in types
        assert "final_response" in types
