"""Tests for AssistantService — orchestration logic with mocked services."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lovely_assistant.app.assistant.config import AssistantConfig
from lovely_assistant.app.assistant.exceptions import (
    AgentRunError,
    AssistantError,
)
from lovely_assistant.app.assistant.interface import AssistantService
from lovely_assistant.app.assistant.models import AssistantRequest, RequestConfigOverride
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet

_MOCK_ARTIFACTS = {
    "persona": "You are Jarvis, the operational assistant.",
    "communication_protocol": "Messages may arrive with envelope tags.",
    "ecosystem": "The Lovely Universe agents: Leo, Ike, Feynman.",
}


@pytest.fixture(autouse=True)
def _mock_artifact_loading():
    """Patch artifact loading for all tests — no DB in unit tests."""
    with patch.object(
        AssistantService,
        "_load_active_artifacts",
        new=AsyncMock(return_value=_MOCK_ARTIFACTS),
    ):
        yield

# --- Fixtures ---


def _make_tool_set() -> ToolSet:
    """Create a ToolSet for testing."""
    backend = [
        ToolDefinition(
            name="get_time",
            description="Get time",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )
    ]
    return ToolSet(backend_tools=backend)


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
    svc.resolve_model = MagicMock(
        side_effect=lambda model=None: (
            model.lower().strip().replace("/", ":", 1) if model else "anthropic:claude-haiku-4-5"
        )
    )
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
    svc.get_mcp_summary = AsyncMock(return_value=None)
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
        assert "Jarvis" in call_kwargs["system_prompt"]

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

    async def test_model_override_is_normalized(self, llm_service, history_service, tool_service):
        config = AssistantConfig(default_model="Anthropic/Claude-Sonnet-4-6")
        svc = AssistantService(
            config=config,
            llm_service=llm_service,
            history_service=history_service,
            tool_service=tool_service,
        )
        await svc.start()
        result = await svc.process_message(AssistantRequest(session_id="s1", message="hi"))
        assert result.model == "anthropic:claude-sonnet-4-6"

    async def test_applies_max_turns_usage_limit(self, llm_service, history_service, tool_service):
        config = AssistantConfig(max_turns=7)
        svc = AssistantService(
            config=config,
            llm_service=llm_service,
            history_service=history_service,
            tool_service=tool_service,
        )
        await svc.start()
        await svc.process_message(AssistantRequest(session_id="s1", message="hi"))
        agent = llm_service.build_agent.return_value
        call_kwargs = agent.run.call_args.kwargs
        assert call_kwargs["usage_limits"].request_limit == 7

    async def test_images_not_auto_attached_to_prompt(self, service, llm_service):
        """Images are NOT auto-attached — prompt is always plain string."""
        await service.start()
        req = AssistantRequest(
            session_id="s1",
            message="What's in this image?",
            images=["data:image/jpeg;base64,/9j/4AAQ"],
        )
        await service.process_message(req)
        agent = llm_service.build_agent.return_value
        call_args = agent.run.call_args
        user_prompt = call_args[0][0]
        assert isinstance(user_prompt, str)
        assert user_prompt == "What's in this image?"

    async def test_no_images_passes_plain_string(self, service, llm_service):
        """When no images, agent.run() receives plain string."""
        await service.start()
        req = AssistantRequest(session_id="s1", message="Hello")
        await service.process_message(req)
        agent = llm_service.build_agent.return_value
        call_args = agent.run.call_args
        assert call_args[0][0] == "Hello"


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


# --- Session Cleanup ---


class TestCleanupExpiredSessions:
    async def test_cleanup_returns_zero_when_not_started(self, service):
        """cleanup_expired_sessions returns 0 before start."""
        result = await service.cleanup_expired_sessions()
        assert result == 0

    async def test_cleanup_delegates_to_session_store(self, service):
        """cleanup_expired_sessions delegates to SessionStore.cleanup_expired."""
        await service.start()
        service._sessions.cleanup_expired = AsyncMock(return_value=3)

        result = await service.cleanup_expired_sessions()

        assert result == 3
        service._sessions.cleanup_expired.assert_awaited_once()


# --- prepare_agent_context ---


class TestPrepareAgentContext:
    async def test_returns_agent_setup_context(self, service, request_msg):
        """prepare_agent_context returns an AgentSetupContext with all fields."""
        from lovely_assistant.app.assistant.models import AgentSetupContext

        await service.start()
        session_context = service._sessions.get_context(request_msg.session_id)

        ctx = await service.prepare_agent_context(request_msg, session_context)

        assert isinstance(ctx, AgentSetupContext)
        assert ctx.agent is not None
        assert ctx.available_tools is not None
        assert ctx.toolsets is not None
        assert ctx.prompt_result is not None
        assert ctx.resolved_model == "anthropic:claude-haiku-4-5"
        assert ctx.usage_limits is not None
        assert ctx.effective_config is not None

    async def test_uses_tool_service(self, service, tool_service, request_msg):
        """prepare_agent_context calls tool_service for tools and MCP."""
        await service.start()
        session_context = service._sessions.get_context(request_msg.session_id)

        await service.prepare_agent_context(request_msg, session_context)

        tool_service.get_available_tools.assert_called_once()
        tool_service.build_toolset.assert_called_once()
        tool_service.get_mcp_summary.assert_awaited_once()

    async def test_mcp_summary_included(self, service, tool_service, request_msg):
        """MCP summary from tool_service is included in the context."""
        await service.start()
        mcp_data = [{"name": "test-mcp", "tools": ["tool1"]}]
        tool_service.get_mcp_summary = AsyncMock(return_value=mcp_data)
        session_context = service._sessions.get_context(request_msg.session_id)

        ctx = await service.prepare_agent_context(request_msg, session_context)

        assert ctx.mcp_summary == mcp_data

    async def test_output_type_always_str(self, service, tool_service, request_msg):
        """output_type is always str regardless of available tools."""
        await service.start()
        tool_service.get_available_tools.return_value = _make_tool_set()
        session_context = service._sessions.get_context(request_msg.session_id)

        ctx = await service.prepare_agent_context(request_msg, session_context)

        assert ctx.output_type is str

    async def test_output_type_str_without_frontend_tools(self, service, tool_service, request_msg):
        """Without frontend tools, output_type is str."""
        await service.start()
        tool_service.get_available_tools.return_value = _make_tool_set()
        session_context = service._sessions.get_context(request_msg.session_id)

        ctx = await service.prepare_agent_context(request_msg, session_context)

        assert ctx.output_type is str

    async def test_per_request_config_override(self, llm_service, history_service, tool_service):
        """Per-request config overrides are applied via resolve_effective_config."""
        config = AssistantConfig(max_turns=5)
        svc = AssistantService(
            config=config,
            llm_service=llm_service,
            history_service=history_service,
            tool_service=tool_service,
        )
        await svc.start()

        req = AssistantRequest(
            session_id="s1",
            message="hi",
            config=RequestConfigOverride(max_turns=12),
        )
        session_context = svc._sessions.get_context("s1")
        ctx = await svc.prepare_agent_context(req, session_context)

        assert ctx.usage_limits.request_limit == 12

    async def test_builds_agent_via_llm_service(self, service, llm_service, request_msg):
        """prepare_agent_context delegates agent creation to llm_service.build_agent."""
        await service.start()
        session_context = service._sessions.get_context(request_msg.session_id)

        ctx = await service.prepare_agent_context(request_msg, session_context)

        llm_service.build_agent.assert_called_once()
        assert ctx.agent is llm_service.build_agent.return_value

    async def test_artifacts_loaded_when_db_available(
        self, llm_service, history_service, tool_service
    ):
        """Artifacts are loaded from DB when database_service is available."""
        db_service = MagicMock()
        mock_session = AsyncMock()
        db_service.session_context.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        db_service.session_context.return_value.__aexit__ = AsyncMock(return_value=None)

        svc = AssistantService(
            config=AssistantConfig(),
            llm_service=llm_service,
            history_service=history_service,
            tool_service=tool_service,
            database_service=db_service,
        )
        await svc.start()

        # Patch _load_active_artifacts to confirm it's called
        svc._load_active_artifacts = AsyncMock(return_value=_MOCK_ARTIFACTS)
        session_context = svc._sessions.get_context("s1")
        req = AssistantRequest(session_id="s1", message="hi")

        await svc.prepare_agent_context(req, session_context)

        svc._load_active_artifacts.assert_awaited_once()

    async def test_no_db_raises_assistant_error(self, llm_service, history_service, tool_service):
        """Without database_service, _load_active_artifacts returns None → AssistantError."""
        svc = AssistantService(
            config=AssistantConfig(),
            llm_service=llm_service,
            history_service=history_service,
            tool_service=tool_service,
        )
        await svc.start()

        # Override instance to simulate DB unavailable (returns None)
        svc._load_active_artifacts = AsyncMock(return_value=None)

        session_context = svc._sessions.get_context("s1")
        req = AssistantRequest(session_id="s1", message="hi")
        with pytest.raises(AssistantError, match="artifact store unavailable"):
            await svc.prepare_agent_context(req, session_context)
