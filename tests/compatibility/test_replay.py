"""Later turns replay exactly what the model saw, so a provider's prefix cache holds."""

from __future__ import annotations

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessagesTypeAdapter,
    ModelResponse,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall

from .test_execution import assert_terminal, calls, register_lookup, request

PNG = "data:image/png;base64,cmVm"
VIEW = {"version": 1, "view": {"name": "Orders"}}


def sent(messages):
    """What a provider receives: message kinds and parts, without bookkeeping fields."""
    dumped = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    return [
        {
            "kind": message["kind"],
            "parts": [
                {k: v for k, v in part.items() if k != "timestamp"} for part in message["parts"]
            ],
        }
        for message in dumped
    ]


def images(messages):
    return [
        item
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, list)
        for item in part.content
        if isinstance(item, BinaryContent)
    ]


def reference(data_uri=PNG):
    return [{"kind": "image", "purpose": "reference", "dataUri": data_uri}]


async def test_later_turns_start_with_what_the_model_saw(runtime, script):
    script.steps = [
        [
            {0: DeltaThinkingPart(content="Check the item.", signature="sig-1")},
            {1: DeltaToolCall(name="lookup", json_args='{"item":"sample"}', tool_call_id="l-1")},
        ],
        [calls(("select_item", '{"item":"sample"}', "host-1"))],
        ["Selected."],
        ["Next answer."],
    ]

    async def lookup(item):
        # Steering that arrives mid-turn, with an attachment of its own.
        await runtime.streaming.accept_steering(
            request(
                id="steer-1",
                content="Use the blue one",
                message_type="steering",
                attachments=reference("data:image/png;base64,Ymx1ZQ=="),
            ),
            has_live_stream=True,
        )
        return {"value": item}

    register_lookup(runtime, lookup)
    assert_terminal(
        [
            e
            async for e in runtime.streaming.stream_message(
                request(host_context=VIEW, attachments=reference())
            )
        ]
    )
    assert [image.data for image in images(script.requests[1])] == [b"ref", b"blue"]

    assert_terminal(
        [
            e
            async for e in runtime.streaming.stream_message(
                request(
                    id="continuation-1",
                    content="",
                    tool_call_id="host-1",
                    tool_result={"selected": True},
                    host_context=VIEW,
                    attachments=reference("data:image/png;base64,YWZ0ZXI="),
                )
            )
        ]
    )
    # The continuation sends its own attachment with the result.
    assert b"after" in [image.data for image in images(script.requests[2][-1:])]

    assert_terminal(
        [e async for e in runtime.streaming.stream_message(request(id="user-2", content="Next"))]
    )
    replayed, seen = script.requests[3], script.requests[2]
    # The earlier turn replays as sent, with its final answer after it: the
    # user's attachment, the steering, native thinking, the tool arguments as
    # the model wrote them, and the continuation's context and attachment.
    assert sent(replayed[: len(seen)]) == sent(seen)
    assert isinstance(replayed[len(seen)], ModelResponse)
    assert replayed[len(seen)].text == "Selected."
    thinking = next(p for m in replayed for p in m.parts if isinstance(p, ThinkingPart))
    assert thinking.signature == "sig-1"


async def test_look_at_screen_keeps_the_users_own_prompt(runtime, script):
    script.steps = [
        [calls(("look_at_screen", "{}", "screen-1"))],
        [calls(("lookup", '{"item":"sample"}', "lookup-1"))],
        ["Inspected."],
    ]

    async def lookup(item):
        return {"value": item}

    register_lookup(runtime, lookup)
    assert_terminal(
        [
            e
            async for e in runtime.streaming.stream_message(
                request(images=["data:image/png;base64,c2NyZWVu"], attachments=reference())
            )
        ]
    )
    # After the screenshot was used, the user's text and attachment remain.
    prompt = script.requests[2][0].parts[0]
    assert isinstance(prompt, UserPromptPart)
    assert prompt.content[0] == "Help"
    assert [image.data for image in images(script.requests[2])] == [b"ref"]
