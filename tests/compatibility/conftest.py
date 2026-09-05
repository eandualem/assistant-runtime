"""Real Pydantic AI execution with a scripted, offline model boundary."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingPart,
    FunctionModel,
)

from assistant_runtime.app.assistant.config import AssistantConfig
from assistant_runtime.app.assistant.interface import AssistantService
from assistant_runtime.app.streaming.config import StreamingConfig
from assistant_runtime.app.streaming.interface import StreamingService
from assistant_runtime.artifacts import neutral_profile
from assistant_runtime.config import AppSettings
from assistant_runtime.services.artifacts.config import ArtifactsConfig
from assistant_runtime.services.artifacts.interface import ArtifactService
from assistant_runtime.services.database.interface import DatabaseService
from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.history.interface import HistoryService
from assistant_runtime.services.llm.config import LLMConfig
from assistant_runtime.services.llm.interface import LlmService
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.interface import ToolService
from assistant_runtime.services.tools.providers.config import ProvidersConfig


@pytest.fixture(autouse=True)
def no_model_requests(monkeypatch):
    """A missed stub must fail instead of contacting a real provider."""
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


@pytest.fixture
def isolated_services(monkeypatch, tmp_path):
    # The model and database are the only service boundaries replaced. No
    # Agent methods, tools, factories, turn services or HTTP handlers are mocked.
    monkeypatch.setattr(DatabaseService, "start", AsyncMock())
    monkeypatch.setattr("assistant_runtime.main.load_dotenv", lambda: None)
    monkeypatch.setattr("assistant_runtime.main.initialize_tracing", lambda: False)
    monkeypatch.setattr("assistant_runtime.main.setup_logging", lambda **_: None)
    monkeypatch.setenv("MCP_CONFIG_PATH", str(tmp_path / "absent-mcp.json"))
    return AppSettings(
        _env_file=None,
        assistant=AssistantConfig(enable_working_memory=False, max_turns=4),
        tools=ToolConfig(builtin_tools=frozenset(), provider_capabilities=frozenset()),
        providers=ProvidersConfig(),
    )


@dataclass
class ModelScript:
    steps: list = field(default_factory=list)
    requests: list[list[ModelMessage]] = field(default_factory=list)

    def _frames(self, messages: list[ModelMessage]) -> list:
        index = len(self.requests)
        self.requests.append(copy.deepcopy(messages))
        assert index < len(self.steps), "Agent made an unexpected model request"
        return self.steps[index]

    async def stream(self, messages: list[ModelMessage], info: AgentInfo):
        for frame in self._frames(messages):
            yield frame

    async def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """The same script for non-streamed requests, such as the summarizer's."""
        parts: list = []
        for frame in self._frames(messages):
            if isinstance(frame, str):
                parts.append(TextPart(frame))
                continue
            for delta in frame.values():
                if isinstance(delta, DeltaThinkingPart):
                    parts.append(ThinkingPart(delta.content or ""))
                else:
                    parts.append(
                        ToolCallPart(delta.name or "", delta.json_args, delta.tool_call_id)
                    )
        return ModelResponse(parts=parts)

    def model(self):
        return FunctionModel(function=self.respond, stream_function=self.stream)


@pytest.fixture
def script():
    return ModelScript()


@pytest.fixture
def host_schema():
    return {
        "select_item": {
            "description": "Select an item in the host application.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
                "additionalProperties": False,
            },
        }
    }


@pytest.fixture
def history_config(request):
    """Override with ``pytest.mark.parametrize("history_config", [...], indirect=True)``."""
    return getattr(request, "param", HistoryConfig())


@pytest.fixture
async def runtime(monkeypatch, script, host_schema, history_config):
    llm = LlmService(LLMConfig())
    # Keep build_agent, Agent.iter, graph execution and toolsets real.
    monkeypatch.setattr(llm, "_resolve_agent_model", lambda _model: script.model())
    history = HistoryService(history_config, llm_service=llm)
    artifacts = ArtifactService(ArtifactsConfig(), neutral_profile())
    tools = ToolService(ToolConfig(host_tools=host_schema), artifact_service=artifacts)
    assistant = AssistantService(
        AssistantConfig(enable_working_memory=False),
        llm_service=llm,
        history_service=history,
        tool_service=tools,
        artifact_service=artifacts,
    )
    streaming = StreamingService(StreamingConfig(), history, tools, assistant)
    services = [llm, history, artifacts, tools, assistant, streaming]
    started = []
    try:
        for service in services:
            await service.start()
            started.append(service)
        yield SimpleNamespace(
            streaming=streaming, tools=tools, sessions=assistant.get_session_store()
        )
    finally:
        for service in reversed(started):
            await service.stop()
