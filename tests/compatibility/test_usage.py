"""Usage limits and accounting through the real turn pipeline."""

from __future__ import annotations

import pytest
from pydantic_ai.usage import UsageLimits

from assistant_runtime.app.assistant.config import AssistantConfig, UsageBudget
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.main import AssistantDefinition, create_runtime
from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.history.models import CompactionResult
from assistant_runtime.services.llm.interface import LlmService

from .test_execution import assert_terminal, calls, request
from .test_history import LONG_TEXT, SMALL_BUDGET


async def test_tool_call_limit_ends_the_turn_with_saved_partial_work(
    isolated_services, script, monkeypatch
):
    """A native limit stops the run; text and completed tools so far are persisted."""
    script.steps = [
        ["Looking... ", calls(("get_time", "{}", "t1"))],
        [calls(("get_time", "{}", "t2"))],
    ]
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: script.model())
    settings = isolated_services.model_copy(
        update={
            "assistant": AssistantConfig(
                enable_working_memory=False, max_turns=4, budget=UsageBudget(tool_calls=1)
            ),
            "tools": isolated_services.tools.model_copy(
                update={"builtin_tools": frozenset({"time"})}
            ),
        }
    )
    async with create_runtime(settings=settings) as runtime:
        events = [
            e
            async for e in runtime.stream_message(
                AssistantRequest(id="m1", session_id="s", content="What time is it?")
            )
        ]
        final = assert_terminal(events, error=True)
        assert final["error_type"] == "usage_limit"
        assert final["message_id"]
        error = next(e for e in events if e["type"] == "error")
        assert error["retry_allowed"] is False
        assert "tool_calls_limit" in error["message"]
        assert final["usage"]["tool_calls"] == 1
        assert final["usage"]["requests"] == 2
        assert final["usage"]["cost_usd"] is None
        path = await runtime._sessions.get_message_path("s")
        assert [m["role"] for m in path] == ["user", "assistant"]
        assert path[-1]["content"].startswith("Looking...")
        kinds = [s["kind"] for s in path[-1]["segments"]]
        assert "tool_group" in kinds
        assert path[-1]["usage"]["tool_calls"] == 1
        # The session is usable again afterwards.
        script.steps.append(["Fine."])
        result = await runtime.run_message(
            AssistantRequest(id="m2", session_id="s", parent_id=final["message_id"], content="ok")
        )
        assert result.content == "Fine."


async def test_definition_limits_and_budget_settings_take_the_stricter_value(
    isolated_services, script, monkeypatch
):
    script.steps = [["One."], ["Two."]]
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: script.model())
    seen: list[UsageLimits] = []
    original = LlmService.build_agent

    def spy(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LlmService, "build_agent", spy)
    settings = isolated_services.model_copy(
        update={
            "assistant": AssistantConfig(
                enable_working_memory=False,
                max_turns=5,
                budget=UsageBudget(output_tokens=500, tool_calls=3),
            )
        }
    )
    definition = AssistantDefinition(
        usage_limits=UsageLimits(tool_calls_limit=2, total_tokens_limit=9000, request_limit=9)
    )
    async with create_runtime(assistant=definition, settings=settings) as runtime:
        assistant = runtime._assistant_service
        ctx = await assistant.prepare_agent_context(
            AssistantRequest(id="m1", session_id="s", content="Hi", config={"max_turns": 50}),
            {},
        )
        seen.append(ctx.usage_limits)
    limits = seen[0]
    assert limits.request_limit == 5  # the request asked for 50; the host allows 5
    assert limits.tool_calls_limit == 2
    assert limits.output_tokens_limit == 500
    assert limits.total_tokens_limit == 9000
    assert limits.cost_limit is None


@pytest.mark.parametrize("history_config", [HistoryConfig(**SMALL_BUDGET)], indirect=True)
async def test_summarisation_usage_is_attributed_to_the_turn(runtime, script, monkeypatch):
    script.steps = [[LONG_TEXT]]
    first = assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    script.steps.extend(
        [
            [calls(("final_result", '{"summary":"Earlier: a long answer"}', "sum-1"))],
            ["Done."],
        ]
    )
    events = [
        e
        async for e in runtime.streaming.stream_message(
            request(id="user-2", parent_id=first["message_id"], content="Again")
        )
    ]
    final = assert_terminal(events)
    auxiliary = final["usage"]["auxiliary"]["summarization"]
    assert auxiliary["requests"] == 1
    assert auxiliary["output_tokens"] > 0
    # The run's own numbers do not include the auxiliary call.
    assert final["usage"]["requests"] == 1
    path = await runtime.sessions.get_message_path("compat")
    assert path[-1]["usage"]["auxiliary"]["summarization"]["requests"] == 1


async def test_working_memory_usage_lands_on_the_stored_message(
    isolated_services, script, monkeypatch
):
    script.steps = [["Hello."]]
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: script.model())
    settings = isolated_services.model_copy(
        update={"assistant": AssistantConfig(enable_working_memory=True, max_turns=4)}
    )
    async with create_runtime(settings=settings) as runtime:
        from assistant_runtime.services.history.models import WorkingMemory

        async def extract(current, recent, turn, *, usage=None):
            if usage is not None:
                usage.incr(
                    __import__("pydantic_ai.usage").usage.RunUsage(requests=1, output_tokens=7)
                )
            return WorkingMemory(active_goal="greet")

        monkeypatch.setattr(runtime._runner._history, "extract_memory_delta", extract)
        result = await runtime.run_message(AssistantRequest(id="m1", session_id="s", content="Hi"))
        path = await runtime._sessions.get_message_path("s")
        stored = path[-1]["usage"]
        assert stored["auxiliary"]["working_memory"] == {
            "input_tokens": 0,
            "output_tokens": 7,
            "total_tokens": 7,
            "requests": 1,
            "tool_calls": 0,
            "cost_usd": None,
        }
        assert result.content == "Hello."


def test_compaction_result_import_keeps_the_summariser_contract():
    assert CompactionResult(summary="x").summary == "x"
