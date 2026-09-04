"""Tests for subagent executor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.usage import RunUsage, UsageLimits

from assistant_runtime.services.tools.builtin._subagent_executor import execute_subagent
from assistant_runtime.services.tools.builtin.subagent import SubagentDefinition

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def subagent_definition() -> SubagentDefinition:
    """A minimal SubagentDefinition for testing."""
    return SubagentDefinition(
        id="researcher",
        name="Research Agent",
        description="Deep research agent for testing",
        system_prompt="You are a research assistant.",
        default_model="anthropic:claude-sonnet",
        default_thinking_budget=None,
        max_iterations=10,
    )


@pytest.fixture
def mock_agent_result() -> MagicMock:
    """A mock agent run result with sensible defaults."""
    result = MagicMock()
    result.output = "Research findings here"
    result.all_messages.return_value = []
    return result


@pytest.fixture
def mock_agent(mock_agent_result) -> MagicMock:
    """A mock Pydantic AI Agent whose run() returns mock_agent_result."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value=mock_agent_result)
    return agent


@pytest.fixture
def mock_llm_service(mock_agent) -> MagicMock:
    """A mock LlmService with resolve_model and build_agent configured."""
    service = MagicMock()
    service.resolve_model.return_value = "anthropic:claude-sonnet"
    service.build_agent.return_value = mock_agent
    return service


# ---------------------------------------------------------------------------
# TestExecuteSubagent
# ---------------------------------------------------------------------------


class TestExecuteSubagent:
    """Tests for the execute_subagent function."""

    async def test_success_returns_result_dict(self, subagent_definition, mock_llm_service):
        """Successful execution returns dict with result, model, and subagent_id."""
        result = await execute_subagent(
            definition=subagent_definition,
            task="Research topic X",
            llm_service=mock_llm_service,
            backend_toolsets=[],
        )

        assert "result" in result
        assert "model" in result
        assert "subagent_id" in result
        assert result["result"] == "Research findings here"
        assert result["model"] == "anthropic:claude-sonnet"
        assert result["subagent_id"] == "researcher"

    async def test_success_includes_metadata(self, subagent_definition, mock_llm_service):
        """Successful execution includes _metadata with expected keys."""
        result = await execute_subagent(
            definition=subagent_definition,
            task="Research topic X",
            llm_service=mock_llm_service,
            backend_toolsets=[],
        )

        assert "_metadata" in result
        metadata = result["_metadata"]
        expected_keys = {
            "iterations",
            "tool_calls_count",
            "tools_used",
            "input_tokens",
            "output_tokens",
            "duration_seconds",
        }
        assert set(metadata.keys()) == expected_keys

    async def test_exception_returns_error_dict(
        self, subagent_definition, mock_llm_service, mock_agent
    ):
        """When agent.run raises, execute_subagent returns an error dict."""
        mock_agent.run = AsyncMock(side_effect=RuntimeError("LLM failed"))

        result = await execute_subagent(
            definition=subagent_definition,
            task="This will fail",
            llm_service=mock_llm_service,
            backend_toolsets=[],
        )

        assert "error" in result
        assert result["error"] == "LLM failed"
        assert result["error_code"] == "SUBAGENT_EXECUTION_ERROR"
        assert result["subagent_id"] == "researcher"
        assert "result" not in result

    async def test_model_override_used(self, subagent_definition, mock_llm_service):
        """When model_override is provided, resolve_model is called with that override."""
        mock_llm_service.resolve_model.return_value = "anthropic:claude-haiku"

        await execute_subagent(
            definition=subagent_definition,
            task="Research with haiku",
            llm_service=mock_llm_service,
            backend_toolsets=[],
            model_override="anthropic:claude-haiku",
        )

        mock_llm_service.resolve_model.assert_called_once_with("anthropic:claude-haiku")

    async def test_default_model_used_without_override(self, subagent_definition, mock_llm_service):
        """Without model_override, resolve_model uses the definition's default_model."""
        await execute_subagent(
            definition=subagent_definition,
            task="Research with default",
            llm_service=mock_llm_service,
            backend_toolsets=[],
        )

        mock_llm_service.resolve_model.assert_called_once_with("anthropic:claude-sonnet")

    async def test_context_appended_to_prompt(self, subagent_definition, mock_llm_service):
        """When context is provided, build_agent receives a system_prompt with Additional Context."""
        await execute_subagent(
            definition=subagent_definition,
            task="Research with context",
            llm_service=mock_llm_service,
            backend_toolsets=[],
            context="Additional info here",
        )

        build_call = mock_llm_service.build_agent.call_args
        system_prompt = build_call.kwargs["system_prompt"]
        assert "## Additional Context" in system_prompt
        assert "Additional info here" in system_prompt
        # Original prompt should also be present
        assert "You are a research assistant." in system_prompt

    async def test_no_context_uses_original_prompt(self, subagent_definition, mock_llm_service):
        """Without context, build_agent receives the definition's system_prompt unmodified."""
        await execute_subagent(
            definition=subagent_definition,
            task="Research without context",
            llm_service=mock_llm_service,
            backend_toolsets=[],
        )

        build_call = mock_llm_service.build_agent.call_args
        system_prompt = build_call.kwargs["system_prompt"]
        assert system_prompt == "You are a research assistant."
        assert "Additional Context" not in system_prompt

    async def test_max_iterations_override(self, subagent_definition, mock_llm_service, mock_agent):
        """When max_iterations_override is provided, agent.run uses it as request_limit."""
        await execute_subagent(
            definition=subagent_definition,
            task="Research with iteration limit",
            llm_service=mock_llm_service,
            backend_toolsets=[],
            max_iterations_override=5,
        )

        run_call = mock_agent.run.call_args
        usage_limits = run_call.kwargs["usage_limits"]
        assert isinstance(usage_limits, UsageLimits)
        assert usage_limits.request_limit == 5

    async def test_default_max_iterations_from_definition(
        self, subagent_definition, mock_llm_service, mock_agent
    ):
        """Without override, agent.run uses the definition's max_iterations."""
        await execute_subagent(
            definition=subagent_definition,
            task="Research with default iterations",
            llm_service=mock_llm_service,
            backend_toolsets=[],
        )

        run_call = mock_agent.run.call_args
        usage_limits = run_call.kwargs["usage_limits"]
        assert usage_limits.request_limit == 10  # definition default

    async def test_usage_tracker_passed_through(
        self, subagent_definition, mock_llm_service, mock_agent
    ):
        """When usage is provided, it is forwarded to agent.run for aggregation."""
        parent_usage = RunUsage()

        await execute_subagent(
            definition=subagent_definition,
            task="Research with usage tracking",
            llm_service=mock_llm_service,
            backend_toolsets=[],
            usage=parent_usage,
        )

        run_call = mock_agent.run.call_args
        assert run_call.kwargs["usage"] is parent_usage

    async def test_build_agent_receives_correct_kwargs(self, subagent_definition, mock_llm_service):
        """Verify build_agent is called with the expected keyword arguments."""
        await execute_subagent(
            definition=subagent_definition,
            task="Check build_agent args",
            llm_service=mock_llm_service,
            backend_toolsets=[],
        )

        mock_llm_service.build_agent.assert_called_once_with(
            model="anthropic:claude-sonnet",
            system_prompt="You are a research assistant.",
            toolsets=None,
            output_type=str,
            thinking_budget=None,
        )

    async def test_build_agent_with_toolsets(self, subagent_definition, mock_llm_service):
        """When backend_toolsets is non-empty, build_agent receives them."""
        fake_toolset = MagicMock()

        await execute_subagent(
            definition=subagent_definition,
            task="Check toolsets",
            llm_service=mock_llm_service,
            backend_toolsets=[fake_toolset],
        )

        build_call = mock_llm_service.build_agent.call_args
        assert build_call.kwargs["toolsets"] == [fake_toolset]

    async def test_thinking_budget_override(self, subagent_definition, mock_llm_service):
        """When thinking_budget_override is provided, build_agent receives it."""
        await execute_subagent(
            definition=subagent_definition,
            task="Think deeply",
            llm_service=mock_llm_service,
            backend_toolsets=[],
            thinking_budget_override=2000,
        )

        build_call = mock_llm_service.build_agent.call_args
        assert build_call.kwargs["thinking_budget"] == 2000

    async def test_resolve_model_failure_returns_error(self, subagent_definition, mock_llm_service):
        """When resolve_model raises, execute_subagent returns an error dict."""
        mock_llm_service.resolve_model.side_effect = ValueError("Unknown model")

        result = await execute_subagent(
            definition=subagent_definition,
            task="Bad model",
            llm_service=mock_llm_service,
            backend_toolsets=[],
        )

        assert "error" in result
        assert result["error_code"] == "SUBAGENT_EXECUTION_ERROR"
        assert "Unknown model" in result["error"]
