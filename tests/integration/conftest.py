"""Shared fixtures for integration tests.

Integration tests wire real services together, mocking only the LLM boundary
(Agent.run / Agent.iter). This catches cross-module wiring issues that unit
tests miss.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_graph.nodes import End

from lovely_assistant.app.assistant.interface import AssistantService
from lovely_assistant.app.settings import RuntimeSettings
from lovely_assistant.app.streaming.interface import StreamingService
from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.config import AppSettings
from lovely_assistant.services.history.interface import HistoryService
from lovely_assistant.services.llm.interface import LlmService
from lovely_assistant.services.tools.interface import ToolService

# Required artifacts for integration tests (no DB available)
_INTEGRATION_ARTIFACTS = {
    "soul": "Jarvis exists to increase Elias's leverage inside a live operating environment.",
    "persona": "You are Jarvis, the operational assistant for the Lovely Universe.",
    "communication_protocol": "Messages may arrive with envelope tags indicating their source.",
    "ecosystem": "The Lovely Universe agents: Leo, Ike, Feynman.",
}


@pytest.fixture(autouse=True)
def _mock_artifact_loading():
    """Patch artifact loading for all integration tests — no DB available."""
    with patch.object(
        AssistantService,
        "_load_active_artifacts",
        new=AsyncMock(return_value=_INTEGRATION_ARTIFACTS),
    ):
        yield


def _make_mock_agent_result(output: Any = "Test response") -> MagicMock:
    """Create a mock AgentRunResult."""
    result = MagicMock()
    result.output = output
    result.all_messages.return_value = []
    # Usage mock for debug events
    usage = MagicMock()
    usage.request_tokens = 100
    usage.response_tokens = 50
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0
    usage.requests = 1
    usage.total_tokens = 150
    result.usage.return_value = usage
    return result


def _make_mock_agent(output: Any = "Test response") -> MagicMock:
    """Create a mock Agent with .run() returning a canned result."""
    agent = MagicMock()
    mock_result = _make_mock_agent_result(output)
    agent.run = AsyncMock(return_value=mock_result)
    return agent


def _make_mock_agent_run(output: Any = "Test response") -> MagicMock:
    """Create a mock agent.iter() context manager returning End immediately."""

    class _MockRun:
        def __init__(self):
            self.ctx = MagicMock()
            self.result = _make_mock_agent_result(output)
            self._done = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        @property
        def next_node(self):
            return End(data=output)

        async def next(self, node):
            return End(data=output)

    return _MockRun()


@pytest.fixture
def app_state():
    """Simple namespace to mimic FastAPI app.state."""
    return SimpleNamespace()


@pytest.fixture
async def lifecycle():
    """Create a fresh LifecycleManager."""
    return LifecycleManager()


@pytest.fixture
async def wired_services(monkeypatch):
    """Wire all real services together, with LLM API key mocked.

    Returns a dict with all services and a lifecycle manager.
    Services are started and ready to use.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-integration")

    settings = AppSettings()
    lm = LifecycleManager()

    # Create services in dependency order
    runtime_settings = RuntimeSettings(frozen_config=settings.assistant)

    llm_service = LlmService(config=settings.llm)
    history_service = HistoryService(config=settings.history, llm_service=llm_service)
    tool_service = ToolService(config=settings.tools)
    assistant_service = AssistantService(
        config=settings.assistant,
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
        runtime_settings=runtime_settings,
    )

    # Register all
    await lm.register("llm_service", llm_service)
    await lm.register("history_service", history_service)
    await lm.register("tool_service", tool_service)
    await lm.register("assistant_service", assistant_service)

    # Start all
    await lm.start_all()

    # Create streaming service (accesses sessions via assistant_service)
    streaming_service = StreamingService(
        config=settings.streaming,
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
        assistant_service=assistant_service,
        runtime_settings=runtime_settings,
        assistant_config=settings.assistant,
    )
    await lm.register("streaming_service", streaming_service)
    await streaming_service.start()

    yield {
        "lifecycle": lm,
        "llm_service": llm_service,
        "history_service": history_service,
        "tool_service": tool_service,
        "assistant_service": assistant_service,
        "streaming_service": streaming_service,
    }

    await lm.stop_all()
