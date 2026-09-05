"""Request state shared with native agent tasks remains isolated by turn."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.main import AssistantDefinition, create_runtime
from assistant_runtime.services.llm.interface import LlmService
from assistant_runtime.services.tools._request_context import (
    assistant_request_context,
    get_current_assistant_session_id,
    get_current_screenshot,
    get_current_telegram_chat_binding,
    record_current_telegram_chat_binding,
)

from . import test_composition

isolated_services = test_composition.isolated_services


def test_binding_scope_resets_after_nested_failure_and_ignores_outside_writes():
    def fail_in_nested_context():
        with assistant_request_context("inner", screenshot="data:inner"):
            assert get_current_telegram_chat_binding() is None
            record_current_telegram_chat_binding("inner-chat")
            raise ValueError("Tool failed")

    record_current_telegram_chat_binding("outside")
    assert get_current_telegram_chat_binding() is None
    with assistant_request_context("outer", screenshot="data:outer"):
        record_current_telegram_chat_binding("outer-chat")
        with pytest.raises(ValueError, match="Tool failed"):
            fail_in_nested_context()
        assert get_current_assistant_session_id() == "outer"
        assert get_current_screenshot() == "data:outer"
        assert get_current_telegram_chat_binding() == "outer-chat"
    assert get_current_assistant_session_id() is None
    assert get_current_screenshot() is None
    assert get_current_telegram_chat_binding() is None


async def test_child_task_binding_is_visible_to_parent():
    async def record_in_child():
        assert get_current_assistant_session_id() == "session-1"
        assert get_current_screenshot() == "data:screen"
        record_current_telegram_chat_binding("chat-1")

    with assistant_request_context("session-1", screenshot="data:screen"):
        await asyncio.create_task(record_in_child())
        assert get_current_telegram_chat_binding() == "chat-1"
    assert get_current_telegram_chat_binding() is None


async def test_native_tools_bind_their_concurrent_sessions(isolated_services, monkeypatch):
    tool_sessions = set()
    both_tools_started = asyncio.Event()

    async def bind_chat(ctx: RunContext[str]) -> str:
        assert get_current_assistant_session_id() == ctx.deps
        record_current_telegram_chat_binding(f"chat-{ctx.deps}")
        tool_sessions.add(ctx.deps)
        if len(tool_sessions) == 2:
            both_tools_started.set()
        await both_tools_started.wait()
        return f"Bound {ctx.deps}"

    async def model_response(messages, info):
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if returns:
            yield str(returns[-1].content)
        else:
            yield {0: DeltaToolCall("bind_chat", "{}", tool_call_id="bind-1")}

    model = FunctionModel(stream_function=model_response)
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: model)
    definition = AssistantDefinition(
        tools=[bind_chat], deps_type=str, deps_factory=lambda request: request.session_id
    )
    async with create_runtime(assistant=definition, settings=isolated_services) as runtime:
        async with asyncio.timeout(5):
            results = await asyncio.gather(
                *(
                    runtime.run_message(
                        AssistantRequest(id=f"user-{i}", session_id=f"session-{i}", content="Bind")
                    )
                    for i in range(2)
                )
            )
        assert [result.content for result in results] == ["Bound session-0", "Bound session-1"]
        sessions = runtime._assistant_service.get_session_store()
        for i in range(2):
            assert await sessions.get_session_id_for_telegram_chat_async(f"chat-session-{i}") == (
                f"session-{i}"
            )
            assert sessions.get_context(f"session-{i}")["telegram_bound_at"] is not None
        assert get_current_telegram_chat_binding() is None
