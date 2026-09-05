"""Steering survives interruption until a model request finishes consuming it."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import Agent, CancellationToken, RunCancelled
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import EnqueuedMessagesEvent, PartStartEvent, TextPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from assistant_runtime.app.assistant._serialization import build_assistant_message_content
from assistant_runtime.app.assistant._session_store import SessionStore
from assistant_runtime.app.streaming._agent_run import TurnPolicy

from .test_execution import request


def steering_prompts(messages):
    return [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
        and isinstance(part.content, str)
        and "Use the blue item" in part.content
    ]


@pytest.mark.parametrize(
    "cancel_before_model", [True, False], ids=["before-model", "partial-response"]
)
async def test_interrupted_steering_remains_pending_for_the_next_turn(cancel_before_model):
    sessions = SessionStore()
    await sessions.register_user_message(request())
    context = sessions.get_context("compat")
    token = CancellationToken()
    observed_requests = []
    enqueued = []
    closed = asyncio.Event()

    async def lookup() -> str:
        await sessions.queue_steering(
            "compat", request(id="steer-1", message_type="steering", content="Use the blue item")
        )
        return "Found an item"

    class CancelAtRequest(AbstractCapability):
        async def before_model_request(self, ctx, request_context):
            if cancel_before_model and steering_prompts(request_context.messages):
                token.cancel()
                await asyncio.sleep(0)
            return request_context

    async def respond(messages, info):
        observed_requests.append(steering_prompts(messages))
        if len(observed_requests) == 1:
            yield {0: DeltaToolCall("lookup", "{}", tool_call_id="lookup-1")}
        else:
            try:
                yield "Partially adjusted"
                await asyncio.Event().wait()
            finally:
                closed.set()

    agent = Agent(
        FunctionModel(stream_function=respond),
        tools=[lookup],
        capabilities=[TurnPolicy(sessions, "compat", context), CancelAtRequest()],
    )

    async def consume():
        async with (
            asyncio.timeout(5),
            agent.run_stream_events("Find an item", cancellation_token=token) as stream,
        ):
            async for event in stream:
                if isinstance(event, EnqueuedMessagesEvent):
                    enqueued.append(event.enqueue_id)
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    token.cancel()

    with pytest.raises(RunCancelled) as caught:
        await consume()
    assert len(observed_requests) == (1 if cancel_before_model else 2)
    if not cancel_before_model:
        assert closed.is_set()
        assert len(enqueued) == 1
        assert len(observed_requests[-1]) == 1
    assert context["pending_steering_ids"] == ["steer-1"]
    assert context["steering_index"]["steer-1"]["status"] == "pending"

    # The runtime stores message segments rather than native request parts.
    # Acknowledging at enqueue insertion would lose this instruction here.
    content, segments, timestamp = build_assistant_message_content(caught.value.new_messages())
    await sessions.register_assistant_message(
        "compat",
        message_id="assistant-1",
        parent_id="user-1",
        content=content,
        segments=segments,
        usage=None,
        created_at=timestamp,
    )
    assert not steering_prompts(sessions.get_history("compat"))
    await sessions.register_user_message(
        request(id="user-2", parent_id="assistant-1", content="Continue")
    )

    retry_requests = []

    async def retry(messages, info):
        retry_requests.append(steering_prompts(messages))
        yield "Adjusted successfully"

    retry_agent = Agent(
        FunctionModel(stream_function=retry), capabilities=[TurnPolicy(sessions, "compat", context)]
    )
    async with retry_agent.run_stream_events(
        "Continue", message_history=sessions.get_history("compat", exclude_leaf=True)
    ) as stream:
        async for _ in stream:
            pass
    assert stream.result.output == "Adjusted successfully"
    assert len(retry_requests) == len(retry_requests[0]) == 1
    assert context["pending_steering_ids"] == []
    assert context["steering_index"]["steer-1"]["status"] == "delivered"


async def test_acknowledging_consumed_steering_keeps_later_records_pending():
    sessions = SessionStore()
    await sessions.register_user_message(request())
    for index in range(2):
        await sessions.queue_steering(
            "compat",
            request(id=f"steer-{index}", message_type="steering", content=f"Change {index}"),
        )
    acknowledged = await sessions.mark_steering_delivered("compat", ["steer-0"])
    assert [record["id"] for record in acknowledged] == ["steer-0"]
    assert [record["id"] for record in await sessions.list_pending_steering("compat")] == [
        "steer-1"
    ]
    assert await sessions.mark_steering_delivered("compat", ["steer-0"]) == []
