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
