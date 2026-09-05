"""Real Pydantic AI execution with a scripted, offline model boundary."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from assistant_runtime.app.assistant.config import AssistantConfig
from assistant_runtime.app.assistant.interface import AssistantService
from assistant_runtime.app.streaming.config import StreamingConfig
from assistant_runtime.app.streaming.interface import StreamingService
from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.history.interface import HistoryService
from assistant_runtime.services.llm.config import LLMConfig
from assistant_runtime.services.llm.interface import LlmService
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.interface import ToolService


@pytest.fixture(autouse=True)
def no_model_requests(monkeypatch):
    """A missed stub must fail instead of contacting a real provider."""
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


@dataclass
class ModelScript:
    steps: list = field(default_factory=list)
    requests: list[list[ModelMessage]] = field(default_factory=list)

    async def stream(self, messages: list[ModelMessage], info: AgentInfo):
        index = len(self.requests)
        self.requests.append(copy.deepcopy(messages))
        assert index < len(self.steps), "Agent made an unexpected model request"
        for frame in self.steps[index]:
            yield frame

    def model(self):
        return FunctionModel(stream_function=self.stream)


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
async def runtime(monkeypatch, script, host_schema):
    llm = LlmService(LLMConfig())
    # Keep build_agent, Agent.iter, graph execution and toolsets real.
    monkeypatch.setattr(llm, "_resolve_agent_model", lambda _model: script.model())
    history = HistoryService(HistoryConfig(), llm_service=llm)
    tools = ToolService(ToolConfig(host_tools=host_schema))
    assistant = AssistantService(
        AssistantConfig(enable_working_memory=False),
        llm_service=llm,
        history_service=history,
        tool_service=tools,
    )
    streaming = StreamingService(StreamingConfig(), history, tools, assistant)
    services = [llm, history, tools, assistant, streaming]
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
