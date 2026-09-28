"""The host contract through the real turn pipeline: attachments and declared actions."""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient
from pydantic_ai.messages import BinaryContent, ToolReturnPart, UserPromptPart

from assistant_runtime.main import create_app

from .test_execution import assert_terminal, calls, request

PNG = "data:image/png;base64,iVBORw0KGgo="


async def test_reference_attachments_reach_the_model_as_native_content(runtime, script):
    script.steps = [["Seen."]]
    events = [
        e
        async for e in runtime.streaming.stream_message(
            request(
                content="What is this?",
                attachments=[
                    {"kind": "image", "dataUri": PNG, "name": "photo"},
                    {"kind": "text", "text": "line one", "name": "notes.txt"},
                ],
                images=[PNG.replace("iVBOR", "SCREEN")],
            )
        )
    ]
    assert_terminal(events)
    prompt = next(p for m in script.requests[0] for p in m.parts if isinstance(p, UserPromptPart))
    assert isinstance(prompt.content, list)
    assert prompt.content[0] == "What is this?"
    assert isinstance(prompt.content[1], BinaryContent)
    assert prompt.content[1].media_type == "image/png"
    assert prompt.content[2] == "[notes.txt]\nline one"
    # The screenshot stays out of the message until look_at_screen asks for it.
    assert not any("SCREEN" in str(part) for part in prompt.content)
    path = await runtime.sessions.get_message_path("compat")
    assert path[0]["content"] == "What is this?"


async def test_host_context_attachments_reach_the_turn(runtime, script):
    script.steps = [["Seen."]]
    events = [
        e
        async for e in runtime.streaming.stream_message(
            request(
                content="Look",
                host_context={
                    "view": {"name": "gallery"},
                    "attachments": [
                        {"kind": "image", "dataUri": PNG, "name": "photo"},
                        {"kind": "image", "purpose": "screenshot", "dataUri": PNG},
                    ],
                },
            )
        )
    ]
    assert_terminal(events)
    prompt = next(p for m in script.requests[0] for p in m.parts if isinstance(p, UserPromptPart))
    assert isinstance(prompt.content, list)
    assert prompt.content[0].startswith("<host_context>")
    assert "call look_at_screen to see it" in prompt.content[0]
    assert "photo (attached to the message)" in prompt.content[0]
    assert prompt.content[1] == "Look"
    assert isinstance(prompt.content[2], BinaryContent)
    assert len(prompt.content) == 3
    assert "gallery" not in (script.requests[0][-1].instructions or "")


async def test_host_context_goes_with_its_message_and_is_replayed(runtime, script):
    script.steps = [["One."], ["Two."], ["Three."]]
    turns = [
        ("First", {"view": {"name": "board"}}),
        ("Second", {"view": {"name": "list"}}),
        ("Third", None),
    ]
    for index, (content, host_context) in enumerate(turns):
        events = [
            e
            async for e in runtime.streaming.stream_message(
                request(id=f"user-{index}", content=content, host_context=host_context)
            )
        ]
        assert_terminal(events)
    board = "<host_context>\nThe host is showing: board.\n</host_context>"
    listing = "<host_context>\nThe host is showing: list.\n</host_context>"
    prompts = [
        [p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart)]
        for messages in script.requests
    ]
    assert prompts[0] == [[board, "First"]]
    # Each message is replayed as it was sent; one without a context carries the last one.
    assert prompts[1] == [[board, "First"], [listing, "Second"]]
    assert prompts[2] == [[board, "First"], [listing, "Second"], [listing, "Third"]]
    assert not any("showing" in (m[-1].instructions or "") for m in script.requests)


async def test_promoted_steering_sends_its_host_context_with_it(runtime, script):
    script.steps = [["First."], ["Adjusted."]]
    assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    steering = request(
        id="steering-1",
        message_type="steering",
        content="Use the blue item",
        host_context={"view": {"name": "blue"}},
    )
    assert_terminal([e async for e in runtime.streaming.stream_message(steering)])
    prompts = [
        p.content for m in script.requests[1][2:] for p in m.parts if isinstance(p, UserPromptPart)
    ]
    assert prompts[0] == "<host_context>\nThe host is showing: blue.\n</host_context>"
    assert "Use the blue item" in prompts[1]


async def test_request_declared_action_is_called_and_continued(runtime, script):
    host_context = {
        "host": {"name": "orders-app", "kind": "desktop"},
        "view": {"name": "orders", "data": {"selected": 7}},
        "actions": [
            {
                "name": "open_order",
                "description": "Open an order in the host.",
                "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}},
            }
        ],
    }
    script.steps = [[calls(("open_order", '{"id":7}', "host-1"))], ["Opened it."]]
    events = [
        e
        async for e in runtime.streaming.stream_message(
            request(content="Open the selected order", host_context=host_context)
        )
    ]
    final = assert_terminal(events)
    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["category"] == "host"
    assert tool_call["tool_name"] == "open_order"
    assert final["pending_tool_call"]["call_id"] == "host-1"
    assert final["pending_tool_call"]["tool_name"] == "open_order"
    assert final["pending_tool_call"]["arguments"] == {"id": 7}
    prompt = next(p for m in script.requests[0] for p in m.parts if isinstance(p, UserPromptPart))
    assert "orders-app (desktop)" in prompt.content[0]
    assert "open_order: Open an order in the host." in prompt.content[0]

    after = {**host_context, "view": {"name": "order", "data": {"opened": 7}}}
    events = [
        e
        async for e in runtime.streaming.stream_message(
            request(
                id="cont-1",
                content="",
                tool_call_id="host-1",
                tool_result={"opened": 7},
                host_context=after,
            )
        )
    ]
    assert_terminal(events)
    assert "".join(e["content"] for e in events if e["type"] == "text_delta") == "Opened it."
    returns = [p for m in script.requests[1] for p in m.parts if isinstance(p, ToolReturnPart)]
    assert returns[-1].tool_call_id == "host-1"
    assert returns[-1].content == {"opened": 7}
    # The context sent with the result follows it in the same request.
    result_request = script.requests[1][-1]
    assert isinstance(result_request.parts[0], ToolReturnPart)
    assert isinstance(result_request.parts[-1], UserPromptPart)
    assert "The host is showing: order." in result_request.parts[-1].content


async def test_invalid_host_context_is_rejected_at_the_edge(isolated_services):
    app = create_app(settings=isolated_services)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/api/chat",
            json={
                "id": "m1",
                "session_id": "s1",
                "content": "hi",
                "host_context": {"version": 1, "view": {"name": "x"}, "surprise": True},
            },
        )
    assert response.status_code == 422
    assert "surprise" in response.text
