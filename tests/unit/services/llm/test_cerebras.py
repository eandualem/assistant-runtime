"""Cerebras models are built with uniform, non-strict tool definitions."""

from __future__ import annotations

import pytest
from pydantic_ai.models.cerebras import CerebrasModel

from assistant_runtime.services.llm.config import LLMConfig
from assistant_runtime.services.llm.interface import LlmService


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-key")
    return LlmService(config=LLMConfig())


def test_cerebras_models_send_every_tool_non_strict(service):
    """Cerebras rejects mixed ``strict`` flags; the provider profile turns them all off."""
    model = service._resolve_agent_model("cerebras:gpt-oss-120b")
    assert isinstance(model, CerebrasModel)
    assert model.model_name == "gpt-oss-120b"
    assert model.profile["openai_supports_strict_tool_definition"] is False
    # The provider's own profile is kept underneath the override.
    assert model.profile["json_schema_transformer"] is not None


def test_build_agent_uses_the_cerebras_model(service):
    agent = service.build_agent(model="cerebras:qwen-3.8-27b", system_prompt="t")
    assert isinstance(agent.model, CerebrasModel)
    assert agent.model.profile["openai_supports_strict_tool_definition"] is False
