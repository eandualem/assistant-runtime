"""Socket.IO establishes the principal at connect time and applies it to every handler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant_runtime.app.access.config import AccessConfig
from assistant_runtime.app.access.exceptions import AccessDeniedError
from assistant_runtime.app.access.interface import AccessService
from assistant_runtime.app.socketio_server import AssistantNamespace
from assistant_runtime.principal import LOCAL_PRINCIPAL, Principal


def _namespace(access: AccessService | None, streaming=None) -> AssistantNamespace:
    namespace = AssistantNamespace("/assistant")
    server = MagicMock()
    server.fastapi_app.state.access_service = access
    server.fastapi_app.state.streaming_service = streaming if streaming is not None else MagicMock()
    namespace.server = server
    namespace.emit = AsyncMock()
    namespace.enter_room = AsyncMock()
    return namespace


def _environ(**headers: str) -> dict:
    environ = {"REMOTE_ADDR": "127.0.0.1"}
    for name, value in headers.items():
        environ["HTTP_" + name.upper().replace("-", "_")] = value
    return environ


class TestConnect:
    async def test_trusted_local_accepts_and_remembers_the_operator(self):
        namespace = _namespace(AccessService(AccessConfig()))
        assert await namespace.on_connect("sid-1", _environ()) is True
        assert namespace._principal("sid-1") is LOCAL_PRINCIPAL
        await namespace.on_disconnect("sid-1")
        assert namespace._principal("sid-1") is None

    async def test_header_mode_refuses_unidentified_callers(self):
        namespace = _namespace(AccessService(AccessConfig(mode="header")))
        assert await namespace.on_connect("sid-1", _environ()) is False
        assert namespace._principal("sid-1") is None
        assert await namespace.on_connect("sid-2", _environ(x_assistant_principal="alice")) is True
        assert namespace._principal("sid-2") == Principal(id="alice")

    async def test_host_mode_sees_the_connect_auth_payload(self):
        seen = []

        def authenticate(credentials):
            seen.append(credentials)
            token = (credentials.auth or {}).get("token")
            return Principal(id="alice") if token == "ok" else None

        namespace = _namespace(AccessService(AccessConfig(mode="host"), authenticator=authenticate))
        assert await namespace.on_connect("sid-1", _environ(), {"token": "no"}) is False
        assert await namespace.on_connect("sid-2", _environ(), {"token": "ok"}) is True
        assert seen[-1].transport == "socketio"
        assert seen[-1].client == "127.0.0.1"

    async def test_missing_access_service_refuses(self):
        namespace = _namespace(None)
        assert await namespace.on_connect("sid-1", _environ()) is False


class TestHandlersCarryThePrincipal:
    @pytest.fixture
    async def connected(self):
        streaming = MagicMock()
        streaming.warm_session = AsyncMock()
        streaming.cancel_session = AsyncMock(return_value=False)
        streaming.accept_steering = AsyncMock(return_value="queued")
        namespace = _namespace(AccessService(AccessConfig(mode="header")), streaming)
        await namespace.on_connect("sid-1", _environ(x_assistant_principal="alice"))
        return namespace, streaming

    async def test_join_passes_the_principal_and_reports_denial(self, connected):
        namespace, streaming = connected
        await namespace.on_assistant_join_session("sid-1", {"session_id": "s1"})
        streaming.warm_session.assert_awaited_once_with("s1", None, principal=Principal(id="alice"))
        namespace.enter_room.assert_awaited_once_with("sid-1", "session:s1")

        namespace.enter_room.reset_mock()
        streaming.warm_session.side_effect = AccessDeniedError(
            "Session 's2' belongs to another principal"
        )
        await namespace.on_assistant_join_session("sid-1", {"session_id": "s2"})
        namespace.enter_room.assert_not_awaited()
        event, payload = namespace.emit.await_args.args[:2]
        assert event == "assistant:error"
        assert payload["type"] == "forbidden"
        assert "s2" in payload["message"]

    async def test_cancel_and_steering_pass_the_principal(self, connected):
        namespace, streaming = connected
        await namespace.on_assistant_cancel("sid-1", {"session_id": "s1"})
        streaming.cancel_session.assert_awaited_once_with("s1", principal=Principal(id="alice"))

        await namespace.on_assistant_message(
            "sid-1", {"id": "st-1", "session_id": "s1", "content": "x", "message_type": "steering"}
        )
        assert streaming.accept_steering.await_args.kwargs["principal"] == Principal(id="alice")

    async def test_denied_cancel_emits_forbidden(self, connected):
        namespace, streaming = connected
        streaming.cancel_session.side_effect = AccessDeniedError("nope")
        await namespace.on_assistant_cancel("sid-1", {"session_id": "s1"})
        event, payload = namespace.emit.await_args.args[:2]
        assert (event, payload["type"]) == ("assistant:error", "forbidden")

    @pytest.mark.parametrize("operation", ["cancel", "message", "steering"])
    async def test_queued_handler_keeps_its_principal_after_disconnect(self, connected, operation):
        namespace, streaming = connected
        observed = []

        async def stream(request, *, principal):
            observed.append(principal)
            yield {"type": "agent_status", "status": "completed"}

        streaming.stream_message = stream
        lock = namespace._session_lock("s1")
        await lock.acquire()
        payload = {"id": "m1", "session_id": "s1", "content": "hello"}
        if operation == "steering":
            payload["message_type"] = "steering"
        handler = (
            namespace.on_assistant_cancel
            if operation == "cancel"
            else namespace.on_assistant_message
        )
        pending = asyncio.create_task(handler("sid-1", payload))
        await asyncio.sleep(0)
        await namespace.on_disconnect("sid-1")
        lock.release()
        await pending
        if operation == "message":
            await asyncio.gather(*namespace._active_streams.values())
            assert observed == [Principal(id="alice")]
        else:
            call = streaming.cancel_session if operation == "cancel" else streaming.accept_steering
            assert call.await_args.kwargs["principal"] == Principal(id="alice")

    @pytest.mark.parametrize("operation", ["join_session", "message", "cancel"])
    async def test_handler_without_authenticated_socket_never_calls_service(
        self, connected, operation
    ):
        namespace, streaming = connected
        await namespace.on_disconnect("sid-1")
        handler = getattr(namespace, f"on_assistant_{operation}")
        await handler("sid-1", {"id": "m1", "session_id": "s1", "content": "hello"})
        assert streaming.mock_calls == []
        namespace.enter_room.assert_not_awaited()
        event, payload = namespace.emit.await_args.args[:2]
        assert (event, payload["type"]) == ("assistant:error", "forbidden")

    async def test_steering_promotion_keeps_identity_during_disconnect(self, connected):
        namespace, streaming = connected
        accepted, release = asyncio.Event(), asyncio.Event()
        observed = []

        async def promote(*args, **kwargs):
            accepted.set()
            await release.wait()
            return "promoted"

        async def stream(request, *, principal):
            observed.append(principal)
            yield {"type": "agent_status", "status": "completed"}

        streaming.accept_steering = promote
        streaming.stream_message = stream
        pending = asyncio.create_task(
            namespace.on_assistant_message(
                "sid-1",
                {"id": "m1", "session_id": "s1", "content": "hello", "message_type": "steering"},
            )
        )
        await accepted.wait()
        await namespace.on_disconnect("sid-1")
        release.set()
        await pending
        await asyncio.gather(*namespace._active_streams.values())
        assert observed == [Principal(id="alice")]
