"""Shared fixtures for integration tests.

Integration tests wire real services together, mocking only the LLM boundary
(Agent.run / Agent.run_stream_events). This catches cross-module wiring issues that unit
tests miss.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.run import AgentRunResultEvent

from assistant_runtime.app.assistant.interface import AssistantService
from assistant_runtime.app.settings import RuntimeSettings
from assistant_runtime.app.streaming.interface import StreamingService
from assistant_runtime.artifacts import technical_operator_profile
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.services.artifacts.config import ArtifactsConfig
from assistant_runtime.services.artifacts.interface import ArtifactService
from assistant_runtime.services.history.interface import HistoryService
from assistant_runtime.services.llm.interface import LlmService
from assistant_runtime.services.tools.interface import ToolService


def make_artifact_service() -> ArtifactService:
    """The example profile on an in-memory store — no DB available."""
    return ArtifactService(ArtifactsConfig(), technical_operator_profile())


def _make_mock_agent_result(output: Any = "Test response") -> MagicMock:
    """Create a mock AgentRunResult."""
    result = MagicMock()
    result.output = output
    messages = [ModelResponse(parts=[TextPart(content=output)])] if isinstance(output, str) else []
    result.all_messages.return_value = messages
    result.new_messages.return_value = messages
    # Usage mock for debug events
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_read_tokens = 0
    usage.cache_write_tokens = 0
    usage.requests = 1
    usage.total_tokens = 150
    result.usage = usage
    return result


def _make_mock_agent(output: Any = "Test response") -> MagicMock:
    """Create a mock Agent whose event stream yields one final result."""
    agent = MagicMock()
    agent.run_stream_events = MagicMock(side_effect=lambda *_a, **_k: _make_mock_agent_run(output))
    return agent


def _make_mock_agent_run(output: Any = "Test response") -> Any:
    """Create an event-stream context manager yielding one final result."""

    class _MockRun:
        def __init__(self):
            self.result = _make_mock_agent_result(output)
            self._done = False

        async def __aenter__(self):
            self._done = False
            return self

        async def __aexit__(self, *args):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return AgentRunResultEvent(self.result)

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
    artifact_service = make_artifact_service()

    tool_service = ToolService(config=settings.tools, artifact_service=artifact_service)
    assistant_service = AssistantService(
        config=settings.assistant,
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
        artifact_service=artifact_service,
        runtime_settings=runtime_settings,
    )

    # Register all
    await lm.register("llm_service", llm_service)
    await lm.register("history_service", history_service)

    await lm.register("artifact_service", artifact_service)
    await lm.register("tool_service", tool_service)
    await lm.register("assistant_service", assistant_service)

    # Start all
    await lm.start_all()

    # Create streaming service (accesses sessions via assistant_service)
    streaming_service = StreamingService(
        config=settings.streaming,
        history_service=history_service,
        tool_service=tool_service,
        assistant_service=assistant_service,
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
