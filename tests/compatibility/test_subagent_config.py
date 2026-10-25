"""A request's subagent settings reach the tools of its turn."""

from assistant_runtime.services.tools.request_context import get_current_subagent_config

from .test_execution import assert_terminal, calls, register_lookup, request


async def test_request_config_reaches_the_turn_tools(runtime, script):
    seen = []

    async def lookup(item):
        seen.append(get_current_subagent_config())
        return {"value": "ready"}

    register_lookup(runtime, lookup)
    script.steps = [[calls(("lookup", '{"item":"x"}', "lookup-1"))], ["Done."]]
    turn = request(config={"subagent_model": "openai:gpt-5.4", "subagent_thinking_budget": 512})
    assert_terminal([e async for e in runtime.streaming.stream_message(turn)])
    # The model is the request's; a thinking budget is a ceiling a request can only
    # lower, and this host set none, so it stays unset.
    assert seen == [{"subagent_model": "openai:gpt-5.4", "subagent_thinking_budget": None}]


async def test_subagent_inherits_current_parent_request_limit(runtime, script):
    from types import SimpleNamespace

    from pydantic_ai import Agent
    from pydantic_ai.models.function import FunctionModel

    from assistant_runtime.services.tools.builtin.subagent import register_subagent_tools

    child_requests = []

    def child_model(messages, info):
        child_requests.append(messages)
        raise AssertionError("A child request exceeded the parent's resolved ceiling")

    llm = SimpleNamespace(
        resolve_model=lambda _: "test",
        build_agent=lambda **_: Agent(FunctionModel(child_model)),
    )
    register_subagent_tools(
        runtime.tools._registry,
        llm,
        backend_toolsets=lambda: [],
        runtime_settings=lambda: None,
    )
    script.steps = [[calls(("run_subagent", '{"task":"research"}', "child-1"))]]
    events = [e async for e in runtime.streaming.stream_message(request(config={"max_turns": 1}))]
    assert_terminal(events, error=True)
    assert child_requests == []
    assert any(e.get("error_type") == "usage_limit" for e in events)
