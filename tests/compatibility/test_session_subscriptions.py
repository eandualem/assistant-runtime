"""Socket subscriptions retain access only while current ownership permits it."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from assistant_runtime.app.socketio_server import create_sio, forward_ingress_events
from assistant_runtime.principal import Principal

from .test_execution import request


async def _subscriber(runtime, principal):
    sio = create_sio()
    sio.fastapi_app = SimpleNamespace(state=SimpleNamespace(streaming_service=runtime.streaming))
    ns = sio.namespace_handlers["/assistant"]
    sid = await sio.manager.connect("transport", "/assistant")
    ns._principals[sid] = principal
    ns.emit = AsyncMock()
    return sio, ns, sid


@pytest.mark.parametrize("session_id", [123, True, ["123"], {"id": "123"}])
async def test_join_rejects_non_string_identifiers(runtime, session_id):
    sio, ns, sid = await _subscriber(runtime, Principal(id="bob"))
    await runtime.sessions.register_user_message(request(session_id="123"), owner_id="alice")
    await ns.on_assistant_join_session(sid, {"session_id": session_id})
    assert not list(sio.manager.get_participants("/assistant", "session:123"))
    assert ns.emit.await_args.args[1]["type"] == "validation"


async def test_join_requires_authorization_even_when_warmup_is_unavailable(runtime, monkeypatch):
    sio, ns, sid = await _subscriber(runtime, Principal(id="bob"))
    await runtime.sessions.register_user_message(request(session_id="private"), owner_id="alice")
    monkeypatch.setattr(runtime.streaming, "warm_session", AsyncMock())
    await ns.on_assistant_join_session(sid, {"session_id": "private"})
    assert not list(sio.manager.get_participants("/assistant", "session:private"))
    runtime.streaming.warm_session.assert_not_awaited()
    assert ns.emit.await_args.args[1]["type"] == "forbidden"


async def test_reassignment_revokes_existing_subscriptions(runtime):
    sio, ns, sid = await _subscriber(runtime, Principal(id="alice"))
    await runtime.sessions.register_user_message(request(session_id="s1"), owner_id="alice")
    await ns.on_assistant_join_session(sid, {"session_id": "s1"})
    forward = forward_ingress_events(sio)
    first = {"type": "text_delta", "content": "Before reassignment"}
    await forward("s1", first)
    ns.emit.assert_awaited_once_with("assistant:text_delta", first, to=sid)
    ns.emit.reset_mock()
    await runtime.streaming.set_session_owner("s1", "bob")
    await forward("s1", {"type": "text_delta", "content": "New owner's private data"})
    ns.emit.assert_not_awaited()
    assert not list(sio.manager.get_participants("/assistant", "session:s1"))


async def test_owner_update_waits_for_authorized_delivery(runtime):
    sio, ns, sid = await _subscriber(runtime, Principal(id="alice"))
    await runtime.sessions.register_user_message(request(session_id="s1"), owner_id="alice")
    await ns.on_assistant_join_session(sid, {"session_id": "s1"})
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed_emit(*args, **kwargs):
        started.set()
        await release.wait()

    ns.emit = delayed_emit
    delivery = asyncio.create_task(forward_ingress_events(sio)("s1", {"type": "text_delta"}))
    await started.wait()
    transfer = asyncio.create_task(runtime.streaming.set_session_owner("s1", "bob"))
    await asyncio.sleep(0)
    assert not transfer.done()
    release.set()
    await delivery
    await transfer
    assert runtime.sessions.get_context("s1")["owner_id"] == "bob"


async def test_join_redacts_unexpected_authorization_failures(runtime, monkeypatch):
    from assistant_runtime.services.database.exceptions import DatabaseError

    sio, ns, sid = await _subscriber(runtime, Principal(id="bob"))
    monkeypatch.setattr(
        runtime.sessions,
        "get_context_if_exists_async",
        AsyncMock(side_effect=DatabaseError("SQL parameters contain private data")),
    )
    await ns.on_assistant_join_session(sid, {"session_id": "private"})
    assert not list(sio.manager.get_participants("/assistant", "session:private"))
    assert ns.emit.await_args.args[1] == {
        "type": "forbidden",
        "message": "Session authorization unavailable",
    }


@pytest.mark.parametrize("failure", ["authorization", "delivery"])
async def test_subscriber_failure_does_not_block_other_recipients(runtime, monkeypatch, failure):
    from assistant_runtime.services.database.exceptions import DatabaseError

    sio, ns, sid = await _subscriber(runtime, Principal(id="alice"))
    await runtime.sessions.register_user_message(request(session_id="s1"), owner_id="alice")
    await ns.on_assistant_join_session(sid, {"session_id": "s1"})
    second = await sio.manager.connect("second-transport", "/assistant")
    ns._principals[second] = Principal(id="operator", roles={"admin"})
    await ns.on_assistant_join_session(second, {"session_id": "s1"})
    if failure == "authorization":
        context = runtime.sessions.get_context("s1")
        monkeypatch.setattr(
            runtime.sessions,
            "get_context_if_exists_async",
            AsyncMock(side_effect=[DatabaseError("lookup failed"), context]),
        )
    else:
        ns.emit.side_effect = [RuntimeError("socket failed"), None]
    event = {"type": "text_delta", "content": "reply"}
    await forward_ingress_events(sio)("s1", event)
    ns.emit.assert_awaited_with("assistant:text_delta", event, to=second)
    assert {client for client, _ in sio.manager.get_participants("/assistant", "session:s1")} == {
        sid,
        second,
    }
