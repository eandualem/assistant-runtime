"""Tests for AssistantService — orchestration logic with mocked services."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.result import DeferredToolRequests

from lovely_assistant.app.assistant.config import AssistantConfig
from lovely_assistant.app.assistant.exceptions import AgentRunError, AssistantError, SessionError
from lovely_assistant.app.assistant.interface import AssistantService
from lovely_assistant.app.assistant.models import AssistantRequest
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet

# --- Fixtures ---


def _make_tool_set(*, with_frontend: bool = False) -> ToolSet:
    """Create a ToolSet for testing."""
    backend = [
        ToolDefinition(
            name="get_time",
            description="Get time",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )
    ]
    frontend = []
    if with_frontend:
        frontend = [
            ToolDefinition(
                name="ui_notify",
                description="Notify UI",
                parameters_schema={},
                category=ToolCategory.FRONTEND,
            )
        ]
    return ToolSet(backend_tools=backend, frontend_tools=frontend)


def _make_mock_run_result(output: Any) -> MagicMock:
    """Create a mock AgentRunResult."""
    result = MagicMock()
    result.output = output
    result.all_messages.return_value = []
    return result


@pytest.fixture
def config():
    return AssistantConfig()


@pytest.fixture
def llm_service():
    svc = MagicMock()
    svc._config = MagicMock()
    svc._config.primary_model = "anthropic:claude-haiku-4-5"
    # build_agent returns a mock agent whose run() returns a mock result
    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=_make_mock_run_result("Hello!"))
    svc.build_agent = MagicMock(return_value=mock_agent)
    return svc


@pytest.fixture
def history_service():
    svc = AsyncMock()
    svc.prepare_history = AsyncMock(return_value=([], False))
    svc.extract_memory_delta = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def tool_service():
    svc = MagicMock()
    svc.get_available_tools = MagicMock(return_value=_make_tool_set())
    svc.build_toolset = MagicMock(return_value=[MagicMock()])
    return svc


@pytest.fixture
def service(config, llm_service, history_service, tool_service):
    return AssistantService(
        config=config,
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
    )


@pytest.fixture
def request_msg():
    return AssistantRequest(session_id="test-session", message="Hello")


# --- Lifecycle Tests ---


class TestLifecycle:
    async def test_start(self, service):
        await service.start()
        assert service._started is True

    async def test_stop(self, service):
        await service.start()
        await service.stop()
        assert service._started is False

    async def test_health_before_start(self, service):
        health = await service.health_check()
        assert health["healthy"] is False

    async def test_health_after_start(self, service):
        await service.start()
        health = await service.health_check()
        assert health["healthy"] is True
        assert health["active_sessions"] == 0

    async def test_health_after_stop(self, service):
        await service.start()
        await service.stop()
        health = await service.health_check()
        assert health["healthy"] is False


# --- Not Started Guard ---


class TestNotStartedGuard:
    async def test_process_message_before_start(self, service, request_msg):
        with pytest.raises(AssistantError, match="not started"):
            await service.process_message(request_msg)


# --- New Message Processing ---


class TestProcessNewMessage:
    async def test_returns_text_result(self, service, request_msg):
        await service.start()
        result = await service.process_message(request_msg)
        assert result.content == "Hello!"
        assert result.session_id == "test-session"
        assert result.turn_number == 1
        assert result.is_tool_call is False

    async def test_increments_turn_number(self, service, llm_service):
        await service.start()
        req1 = AssistantRequest(session_id="s1", message="first")
        req2 = AssistantRequest(session_id="s1", message="second")

        r1 = await service.process_message(req1)
        r2 = await service.process_message(req2)

        assert r1.turn_number == 1
        assert r2.turn_number == 2

    async def test_different_sessions_independent(self, service):
        await service.start()
        r1 = await service.process_message(AssistantRequest(session_id="s1", message="hi"))
        r2 = await service.process_message(AssistantRequest(session_id="s2", message="hi"))
        assert r1.turn_number == 1
        assert r2.turn_number == 1

    async def test_passes_machine_state_to_tools(self, service, tool_service):
        await service.start()
        state = {"current_state": "agents"}
        req = AssistantRequest(session_id="s1", message="hi", machine_state=state)
        await service.process_message(req)

        tool_service.get_available_tools.assert_called_with(state)
        tool_service.build_toolset.assert_called_with(state)

    async def test_calls_build_agent_with_system_prompt(self, service, llm_service):
        await service.start()
        await service.process_message(AssistantRequest(session_id="s1", message="hi"))
        llm_service.build_agent.assert_called_once()
        call_kwargs = llm_service.build_agent.call_args[1]
        assert "system_prompt" in call_kwargs
        assert "Lovely Console" in call_kwargs["system_prompt"]

    async def test_calls_build_agent_with_toolsets(self, service, llm_service, tool_service):
        await service.start()
        await service.process_message(AssistantRequest(session_id="s1", message="hi"))
        call_kwargs = llm_service.build_agent.call_args[1]
        assert "toolsets" in call_kwargs

    async def test_calls_prepare_history(self, service, history_service):
        await service.start()
        await service.process_message(AssistantRequest(session_id="s1", message="hi"))
        history_service.prepare_history.assert_called_once()

    async def test_saves_history_after_run(self, service, llm_service):
        await service.start()
        mock_messages = [MagicMock(), MagicMock()]
        mock_result = _make_mock_run_result("ok")
        mock_result.all_messages.return_value = mock_messages
        agent = llm_service.build_agent.return_value
        agent.run = AsyncMock(return_value=mock_result)

        await service.process_message(AssistantRequest(session_id="s1", message="hi"))
        # History should now have the messages
        assert service._sessions.get_history("s1") == mock_messages

    async def test_agent_run_failure_raises(self, service, llm_service):
        await service.start()
        agent = llm_service.build_agent.return_value
        agent.run = AsyncMock(side_effect=RuntimeError("LLM exploded"))

        with pytest.raises(AgentRunError, match="Agent execution failed"):
            await service.process_message(AssistantRequest(session_id="s1", message="hi"))

    async def test_model_in_result(self, service):
        await service.start()
        result = await service.process_message(AssistantRequest(session_id="s1", message="hi"))
        # default_model is None, so falls back to primary_model
        assert result.model == "anthropic:claude-haiku-4-5"

    async def test_model_override(self, llm_service, history_service, tool_service):
        config = AssistantConfig(default_model="openai:gpt-4o")
        svc = AssistantService(
            config=config,
            llm_service=llm_service,
            history_service=history_service,
            tool_service=tool_service,
        )
        await svc.start()
        result = await svc.process_message(AssistantRequest(session_id="s1", message="hi"))
        assert result.model == "openai:gpt-4o"


# --- Deferred Tool Call Handling ---


class TestDeferredToolCall:
    async def test_returns_deferred_result(self, service, llm_service, tool_service):
        """When agent returns DeferredToolRequests, result is a tool call."""
        await service.start()

        # Setup: frontend tools available so output_type includes DeferredToolRequests
        tool_service.get_available_tools.return_value = _make_tool_set(with_frontend=True)

        # Agent returns DeferredToolRequests
        deferred = DeferredToolRequests(
            calls=[
                ToolCallPart(
                    tool_name="ui_notify",
                    args={"message": "Hello!"},
                    tool_call_id="tc-42",
                )
            ]
        )
        mock_result = _make_mock_run_result(deferred)
        agent = llm_service.build_agent.return_value
        agent.run = AsyncMock(return_value=mock_result)

        result = await service.process_message(
            AssistantRequest(session_id="s1", message="notify the user")
        )

        assert result.is_tool_call is True
        assert result.content is None
        assert result.deferred_tool_request is not None
        assert result.deferred_tool_request.tool_name == "ui_notify"
        assert result.deferred_tool_request.request_id == "tc-42"
        assert result.deferred_tool_request.arguments == {"message": "Hello!"}

    async def test_stores_pending_tool_call(self, service, llm_service, tool_service):
        """DeferredToolRequests stores pending state for continuation."""
        await service.start()
        tool_service.get_available_tools.return_value = _make_tool_set(with_frontend=True)

        deferred = DeferredToolRequests(
            calls=[
                ToolCallPart(
                    tool_name="ui_notify",
                    args={},
                    tool_call_id="tc-99",
                )
            ]
        )
        mock_result = _make_mock_run_result(deferred)
        agent = llm_service.build_agent.return_value
        agent.run = AsyncMock(return_value=mock_result)

        await service.process_message(AssistantRequest(session_id="s1", message="do it"))

        # Pending tool call should be stored
        ctx = service._sessions.get_context("s1")
        assert ctx["pending_tool_call"] == {
            "tool_call_id": "tc-99",
            "tool_name": "ui_notify",
        }

    async def test_output_type_includes_deferred_when_frontend_tools(
        self, service, llm_service, tool_service
    ):
        """output_type should be [str, DeferredToolRequests] when frontend tools exist."""
        await service.start()
        tool_service.get_available_tools.return_value = _make_tool_set(with_frontend=True)

        await service.process_message(AssistantRequest(session_id="s1", message="hi"))

        call_kwargs = llm_service.build_agent.call_args[1]
        assert call_kwargs["output_type"] == [str, DeferredToolRequests]

    async def test_output_type_str_when_no_frontend_tools(self, service, llm_service, tool_service):
        """output_type should be str when no frontend tools."""
        await service.start()
        tool_service.get_available_tools.return_value = _make_tool_set(with_frontend=False)

        await service.process_message(AssistantRequest(session_id="s1", message="hi"))

        call_kwargs = llm_service.build_agent.call_args[1]
        assert call_kwargs["output_type"] is str


# --- Continuation Handling ---


class TestContinuation:
    async def test_continuation_with_pending(self, service, llm_service, tool_service):
        """Continuation resumes agent with deferred tool results."""
        await service.start()
        tool_service.get_available_tools.return_value = _make_tool_set(with_frontend=True)

        # Step 1: Initial request returns deferred tool call
        deferred = DeferredToolRequests(
            calls=[
                ToolCallPart(
                    tool_name="ui_notify",
                    args={},
                    tool_call_id="tc-100",
                )
            ]
        )
        mock_result1 = _make_mock_run_result(deferred)
        agent = llm_service.build_agent.return_value
        agent.run = AsyncMock(return_value=mock_result1)

        await service.process_message(AssistantRequest(session_id="s1", message="do it"))

        # Step 2: Continuation with tool result
        mock_result2 = _make_mock_run_result("Done! Notification sent.")
        agent.run = AsyncMock(return_value=mock_result2)

        result = await service.process_message(
            AssistantRequest(
                session_id="s1",
                message="",
                tool_call_id="tc-100",
                tool_result={"status": "displayed"},
            )
        )

        assert result.content == "Done! Notification sent."
        assert result.is_tool_call is False

        # Verify agent.run was called with deferred_tool_results
        call_kwargs = agent.run.call_args[1]
        assert "deferred_tool_results" in call_kwargs

    async def test_continuation_passes_is_continuation_to_history(
        self, service, llm_service, history_service, tool_service
    ):
        """Continuation passes is_continuation=True to prepare_history."""
        await service.start()
        tool_service.get_available_tools.return_value = _make_tool_set(with_frontend=True)

        # Setup pending
        deferred = DeferredToolRequests(
            calls=[ToolCallPart(tool_name="ui_notify", args={}, tool_call_id="tc-1")]
        )
        agent = llm_service.build_agent.return_value
        agent.run = AsyncMock(return_value=_make_mock_run_result(deferred))
        await service.process_message(AssistantRequest(session_id="s1", message="go"))

        # Continuation
        agent.run = AsyncMock(return_value=_make_mock_run_result("ok"))
        await service.process_message(
            AssistantRequest(session_id="s1", message="", tool_call_id="tc-1", tool_result="ok")
        )

        # Second call to prepare_history should have is_continuation=True
        calls = history_service.prepare_history.call_args_list
        assert len(calls) == 2
        assert calls[1][1].get("is_continuation") is True

    async def test_continuation_no_session_raises(self, service):
        """Continuation for nonexistent session raises."""
        await service.start()
        with pytest.raises(SessionError, match="No session found"):
            await service.process_message(
                AssistantRequest(
                    session_id="nonexistent",
                    message="",
                    tool_call_id="tc-1",
                    tool_result="ok",
                )
            )

    async def test_continuation_no_pending_raises(self, service):
        """Continuation without pending tool call raises."""
        await service.start()
        # Create session but no pending tool call
        service._sessions.get_context("s1")
        with pytest.raises(SessionError, match="No pending tool call"):
            await service.process_message(
                AssistantRequest(
                    session_id="s1",
                    message="",
                    tool_call_id="tc-1",
                    tool_result="ok",
                )
            )

    async def test_continuation_failure_raises(self, service, llm_service, tool_service):
        """Agent failure during continuation raises AgentRunError."""
        await service.start()
        tool_service.get_available_tools.return_value = _make_tool_set(with_frontend=True)

        # Setup pending
        deferred = DeferredToolRequests(
            calls=[ToolCallPart(tool_name="ui_notify", args={}, tool_call_id="tc-1")]
        )
        agent = llm_service.build_agent.return_value
        agent.run = AsyncMock(return_value=_make_mock_run_result(deferred))
        await service.process_message(AssistantRequest(session_id="s1", message="go"))

        # Continuation fails
        agent.run = AsyncMock(side_effect=RuntimeError("crash"))
        with pytest.raises(AgentRunError, match="Continuation failed"):
            await service.process_message(
                AssistantRequest(session_id="s1", message="", tool_call_id="tc-1", tool_result="ok")
            )


# --- Working Memory ---


class TestWorkingMemory:
    async def test_extracts_working_memory(self, service, history_service):
        """Working memory extraction is called after successful run."""
        await service.start()
        await service.process_message(AssistantRequest(session_id="s1", message="hi"))
        history_service.extract_memory_delta.assert_called_once()

    async def test_working_memory_disabled(self, llm_service, history_service, tool_service):
        """Working memory extraction skipped when disabled."""
        config = AssistantConfig(enable_working_memory=False)
        svc = AssistantService(
            config=config,
            llm_service=llm_service,
            history_service=history_service,
            tool_service=tool_service,
        )
        await svc.start()
        await svc.process_message(AssistantRequest(session_id="s1", message="hi"))
        history_service.extract_memory_delta.assert_not_called()

    async def test_working_memory_failure_does_not_fail_request(self, service, history_service):
        """Working memory extraction failure is swallowed."""
        await service.start()
        history_service.extract_memory_delta = AsyncMock(
            side_effect=RuntimeError("memory extraction failed")
        )

        # Should not raise
        result = await service.process_message(AssistantRequest(session_id="s1", message="hi"))
        assert result.content == "Hello!"
