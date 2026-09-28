"""A host message joins the conversation: the model reads its text, never its components."""

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart

from assistant_runtime.app.assistant.models import AssistantRequest

from .test_execution import assert_terminal

CARD = {"kind": "component", "type": "proposal", "data": {"secret_detail": "not for models"}}


def request(message_id, content):
    return AssistantRequest.model_validate(
        {"id": message_id, "session_id": "compat", "parent_id": None, "content": content}
    )


async def test_the_next_turn_reads_the_host_text_and_continues_after_it(runtime, script):
    script.steps = [["First reply."], ["Second reply."]]
    assert_terminal([e async for e in runtime.streaming.stream_message(request("u1", "Hi"))])

    host = await runtime.streaming.append_host_message(
        "compat", content="Shown to the owner: a proposal to shorten replies.", segments=[CARD]
    )
    assert_terminal([e async for e in runtime.streaming.stream_message(request("u2", "Ok"))])

    received = script.requests[-1]
    shapes = [(type(m).__name__, [getattr(p, "content", None) for p in m.parts]) for m in received]
    # Upstream merges the host text into the next request, so roles still alternate.
    assert shapes == [
        ("ModelRequest", ["Hi"]),
        ("ModelResponse", ["First reply."]),
        ("ModelRequest", ["Shown to the owner: a proposal to shorten replies.", "Ok"]),
    ]
    assert all(isinstance(p, UserPromptPart) for p in received[-1].parts)
    assert isinstance(received[-1], ModelRequest)
    assert isinstance(received[-2], ModelResponse)
    assert "not for models" not in repr(received)

    path = await runtime.sessions.get_message_path("compat")
    assert [(m["role"], m["parent_id"]) for m in path] == [
        ("user", None),
        ("assistant", "u1"),
        ("host", path[1]["id"]),
        ("user", host["id"]),
        ("assistant", "u2"),
    ]


async def test_steering_after_a_card_gets_its_own_reply_and_leaves_the_card_intact(runtime, script):
    script.steps = [["First reply."], ["Noted."]]
    assert_terminal([e async for e in runtime.streaming.stream_message(request("u1", "Hi"))])
    host = await runtime.streaming.append_host_message(
        "compat", content="Shown to the owner: findings.", segments=[CARD]
    )
    steering = AssistantRequest.model_validate(
        {
            "id": "s1",
            "session_id": "compat",
            "message_type": "steering",
            "content": "[via:hub from:agent-a] build_failed (warning)",
        }
    )
    assert await runtime.streaming.accept_steering(steering, has_live_stream=False) == "promoted"
    assert_terminal([e async for e in runtime.streaming.stream_message(steering)])

    path = await runtime.sessions.get_message_path("compat")
    assert [(m["role"], m["parent_id"]) for m in path[-2:]] == [
        ("host", path[1]["id"]),
        ("assistant", host["id"]),
    ]
    assert path[-2]["segments"] == [CARD]  # the card was not rewritten by the turn
    assert path[-1]["content"] == "Noted."


async def test_a_card_waits_for_an_idle_session(runtime, script):
    from assistant_runtime.app.assistant.exceptions import SessionError
    from assistant_runtime.principal import LOCAL_PRINCIPAL

    script.steps = [["First reply."]]
    assert_terminal([e async for e in runtime.streaming.stream_message(request("u1", "Hi"))])
    context = await runtime.sessions.get_context_if_exists_async("compat")
    context["pending_tool_call_id"] = "call-1"
    with pytest.raises(SessionError, match="in progress"):
        await runtime.streaming.append_host_message("compat", content="x", segments=[CARD])
    context["pending_tool_call_id"] = None

    await runtime.streaming.reserve_session("compat", "voice-lease", LOCAL_PRINCIPAL)
    with pytest.raises(SessionError, match="voice"):
        await runtime.streaming.append_host_message("compat", content="x", segments=[CARD])
    runtime.streaming.release_session("compat", "voice-lease")
    await runtime.streaming.append_host_message("compat", content="x", segments=[CARD])


async def test_a_card_starts_a_missing_session_and_binds_it_as_a_first_message_would(
    runtime, script
):
    from assistant_runtime.app.assistant.exceptions import SessionError
    from assistant_runtime.principal import Principal

    script.steps = [["Noted."]]
    owner = Principal(id="alice", roles=frozenset())
    card = await runtime.streaming.append_host_message(
        "fresh", content="Spoken first.", segments=[CARD], principal=owner, subject="agent-a"
    )
    context = await runtime.sessions.get_context_if_exists_async("fresh")
    assert (card["parent_id"], context["owner_id"]) == (None, "alice")
    assert (context["profile"], context["subject"]) == ("neutral", "agent-a")

    with pytest.raises(SessionError, match="bound"):
        await runtime.streaming.append_host_message(
            "fresh", content="x", segments=[CARD], principal=owner, subject="agent-b"
        )
    typed = AssistantRequest.model_validate(
        {"id": "u1", "session_id": "fresh", "parent_id": None, "content": "Hi"}
    )
    assert_terminal([e async for e in runtime.streaming.stream_message(typed, principal=owner)])
    path = await runtime.sessions.get_message_path("fresh")
    assert [(m["role"], m["parent_id"]) for m in path][:2] == [("host", None), ("user", card["id"])]
    assert "Spoken first." in repr(script.requests[-1])

    # Without a subject the started session stays unbound: turns pick their profile.
    await runtime.streaming.append_host_message(
        "plain", content="", segments=[CARD], principal=owner
    )
    plain = await runtime.sessions.get_context_if_exists_async("plain")
    assert (plain["owner_id"], plain.get("profile"), plain.get("subject")) == ("alice", None, None)


async def test_a_turn_arriving_during_the_append_waits_and_continues_after_the_card(
    runtime, script
):
    import asyncio

    script.steps = [["First reply."], ["Second reply."]]
    assert_terminal([e async for e in runtime.streaming.stream_message(request("u1", "Hi"))])
    sessions, gate = runtime.sessions, asyncio.Event()
    write = sessions.register_host_message

    async def slow_write(*args, **kwargs):
        await gate.wait()  # the host message is still being saved
        return await write(*args, **kwargs)

    sessions.register_host_message = slow_write
    append = asyncio.create_task(
        runtime.streaming.append_host_message("compat", content="Card", segments=[CARD])
    )
    await asyncio.sleep(0)
    turn = asyncio.create_task(
        _collect(runtime.streaming.stream_message(request("u2", "Meanwhile")))
    )
    await asyncio.sleep(0.01)
    waited = not turn.done()
    gate.set()
    host = await append
    assert_terminal(await turn)
    assert waited  # the turn waited for the card to be saved
    path = await sessions.get_message_path("compat")
    assert [(m["role"], m["parent_id"]) for m in path[-3:]] == [
        ("host", path[1]["id"]),
        ("user", host["id"]),
        ("assistant", "u2"),
    ]


async def _collect(stream):
    return [event async for event in stream]
