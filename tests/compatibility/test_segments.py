"""Model retries preserve separate wire segments without a tool boundary."""

from __future__ import annotations

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import RetryPromptPart

from assistant_runtime.main import AssistantDefinition, create_runtime
from assistant_runtime.services.llm.interface import LlmService

from .test_composition import isolated_services as isolated_services
from .test_execution import assert_terminal, request


async def test_output_retry_starts_a_new_text_segment(isolated_services, monkeypatch, script):
    rejected = []

    class RequireFinalAnswer(AbstractCapability):
        async def after_output_process(self, ctx, *, output_context, output):
            if not ctx.partial_output and output == "Draft answer":
                rejected.append(output)
                raise ModelRetry("Provide the final answer")
            return output

    script.steps = [["Draft ", "answer"], ["Final ", "answer"]]
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: script.model())
    definition = AssistantDefinition(capabilities=[RequireFinalAnswer()])
    async with create_runtime(assistant=definition, settings=isolated_services) as runtime:
        events = [event async for event in runtime.stream_message(request())]

    assert_terminal(events)
    assert rejected == ["Draft answer"]
    assert len(script.requests) == 2
    assert any(
        isinstance(part, RetryPromptPart) and "Provide the final answer" in str(part.content)
        for message in script.requests[1]
        for part in message.parts
    )
    assert not any(event["type"].startswith("tool_") for event in events)
    text = [event for event in events if event["type"] == "text_delta"]
    assert [event["content"] for event in text] == ["Draft ", "answer", "Final ", "answer"]
    assert text[0]["segment_id"] == text[1]["segment_id"]
    assert text[2]["segment_id"] == text[3]["segment_id"]
    assert text[0]["segment_id"] != text[2]["segment_id"]
    assert [event["segment_index"] for event in text] == [0, 0, 1, 1]
    assert [event["delta_index"] for event in text] == [0, 1, 0, 1]
    assert [event["segment_started"] for event in text] == [True, False, True, False]
