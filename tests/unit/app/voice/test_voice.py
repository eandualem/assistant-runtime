"""Protocol and lifecycle tests: no provider requests, audio devices or database."""

import asyncio
import json
from contextlib import aclosing
from unittest.mock import AsyncMock

import pytest

from assistant_runtime.app.access.exceptions import AccessDeniedError
from assistant_runtime.app.assistant.exceptions import SessionError
from assistant_runtime.app.streaming.interface import StreamingService
from assistant_runtime.app.voice.config import VoiceConfig
from assistant_runtime.app.voice.exceptions import VoiceError
from assistant_runtime.app.voice.interface import VoiceService
from assistant_runtime.app.voice.models import VoiceOffer, VoiceToolResult
from assistant_runtime.principal import Principal
from tests.voice_helpers import Transport, delegate, until

OWNER = Principal("owner")
OTHER = Principal("other")


class Backend:
    voice_event = staticmethod(StreamingService.voice_event)

    def __init__(self):
        self.requests = []
        self.leases = {}
        self.owners = {}
        self.runner = None
        self.cancel_reserved_work = AsyncMock()

    async def authorize_session(self, session_id, principal):
        if self.owners.get(session_id) != principal.id and not principal.is_admin:
            raise AccessDeniedError("Session denied")

    async def reserve_session(self, session_id, lease, principal):
        if session_id in self.leases:
            raise SessionError("Session reserved")
        if session_id in self.owners:
            await self.authorize_session(session_id, principal)
        self.owners[session_id] = principal.id
        self.leases[session_id] = lease
        return []

    def release_session(self, session_id, lease):
        if self.leases.get(session_id) == lease:
            self.leases.pop(session_id)

    async def stream_message(self, request, *, principal, session_lease, cancel_after_plan=None):
        assert self.leases[request.session_id] == session_lease
        assert principal == OWNER
        self.requests.append(request)
        if self.runner:
            async for event in self.runner(request):
                yield event
        else:
            yield {
                "type": "final_response",
                "content": "Confirmed",
                "model": "test",
                "error": False,
            }


@pytest.fixture
async def setup(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    backend, transport = Backend(), Transport()
    service = VoiceService(
        VoiceConfig(enabled=True, close_timeout_seconds=0.05), backend, transport=transport
    )
    await service.start()
    yield service, backend, transport
    await service.stop()


async def create(setup):
    service, backend, transport = setup
    response = await service.create(VoiceOffer(session_id="conversation", sdp="offer"), OWNER)
    return response["call_id"], transport.connection


async def test_creation_uses_live_client_delegation_and_never_exposes_key(setup):
    service, backend, transport = setup
    call_id, _ = await create(setup)
    record = await service.get(call_id, OWNER)
    assert record["status"] == "connected"  # Attached sideband need not replay session.started.
    assert transport.created[0][1]["model"] == "gpt-live-1"
    assert transport.created[0][1]["delegation"] == {"type": "client"}
    assert transport.created[0][1]["store"] is False
    assert "test-key-never-sent" not in json.dumps(record)
    transport.attach.assert_awaited_once_with("test-key-never-sent", "live_provider")


async def test_disabled_or_missing_key_does_not_allocate(setup, monkeypatch):
    service, backend, transport = setup
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(VoiceError, match="Codex login"):
        await create(setup)
    assert not transport.created
    assert not backend.leases
    service.config = VoiceConfig()
    with pytest.raises(VoiceError, match="disabled"):
        await create(setup)


async def test_attach_failure_releases_reservation(setup):
    service, backend, transport = setup
    transport.attach.side_effect = RuntimeError("secret provider payload")
    with pytest.raises(VoiceError, match="unconfirmed") as failure:
        await create(setup)
    assert "secret" not in str(failure.value)
    assert not backend.leases


async def test_ownership_and_single_call_reservation(setup):
    service, backend, transport = setup
    call_id, _ = await create(setup)
    for method in (service.get, service.close, service.cancel_work):
        with pytest.raises(AccessDeniedError):
            await method(call_id, OTHER)
    with pytest.raises(SessionError):
        await create(setup)
    assert len(transport.created) == 1


async def test_delegation_uses_transcript_and_original_principal_once(setup):
    service, backend, _ = setup
    call_id, connection = await create(setup)
    delegate(connection)
    delegate(connection, text=None)
    await until(lambda: any(e["type"] == "session.commentary.append" for e in connection.sent))
    assert len(backend.requests) == 1
    assert "Check availability" in backend.requests[0].content
    assert '"role": "user"' in backend.requests[0].content
    result = connection.sent[0]
    assert result["delegation_id"] == "item_1"
    assert result["content"] == "Confirmed"
    connection.feed("session.commentary.appended", client_event_id=result["event_id"])
    await until(
        lambda: service._calls[call_id].delegations["item_1"]["status"] == "result_accepted"
    )


async def test_no_transcript_does_not_invent_task(setup):
    service, backend, _ = setup
    _, connection = await create(setup)
    delegate(connection, text=None)
    await until(lambda: connection.sent)
    assert not backend.requests
    assert "repeat" in connection.sent[0]["content"]


async def pending_runner(request):
    if not request.is_continuation:
        yield {
            "type": "final_response",
            "model": "test",
            "pending_tool_call": {
                "call_id": "host_1",
                "tool_name": "navigate",
                "arguments": {"view": "home"},
            },
        }
    else:
        yield {"type": "final_response", "model": "test", "content": "Navigation completed"}


async def test_host_result_round_trip_and_duplicate_rejection(setup):
    service, backend, _ = setup
    backend.runner = pending_runner
    call_id, connection = await create(setup)
    delegate(connection)
    await until(lambda: service._calls[call_id].pending is not None)
    result = VoiceToolResult(tool_call_id="host_1", tool_result={"ok": True})
    assert await service.tool_result(call_id, "item_1", result, OWNER) == {"accepted": True}
    assert backend.requests[-1].tool_call_id == "host_1"
    assert backend.requests[-1].tool_result == {"ok": True}
    with pytest.raises(VoiceError):
        await service.tool_result(call_id, "item_1", result, OWNER)
    assert connection.sent[0]["content"] == "Navigation completed"


@pytest.mark.parametrize("transition", ["supersede", "close", "cancel"])
async def test_submitted_host_result_drains_before_replacement_or_shutdown(setup, transition):
    service, backend, _ = setup
    started, release, recorded = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def runner(request):
        if request.is_continuation:
            started.set()
            await release.wait()
            recorded.set()
        async for event in pending_runner(request):
            yield event

    backend.runner = runner
    call_id, connection = await create(setup)
    delegate(connection)
    await until(lambda: service._calls[call_id].pending is not None)
    delivery = asyncio.create_task(
        service.tool_result(
            call_id, "item_1", VoiceToolResult(tool_call_id="host_1", tool_result="done"), OWNER
        )
    )
    await started.wait()
    transition_task = None
    if transition == "supersede":
        delegate(connection, "item_2", "Now check another view")
    elif transition == "close":
        transition_task = asyncio.create_task(service.close(call_id, OWNER))
    else:
        transition_task = asyncio.create_task(service.cancel_work(call_id, OWNER))
    await asyncio.sleep(0.01)
    assert not delivery.done()
    assert backend.leases
    release.set()
    assert await asyncio.wait_for(delivery, 1) == {"accepted": True}
    assert recorded.is_set()
    if transition_task:
        await asyncio.wait_for(transition_task, 1)
    assert not any(e.get("delegation_id") == "item_1" for e in connection.sent)


async def test_cancellation_drains_snapshot_before_releasing_lease(setup):
    service, backend, _ = setup
    started, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def runner(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup.set()
            await release.wait()
        yield {}  # async generator

    backend.runner = runner
    call_id, connection = await create(setup)
    delegate(connection)
    await started.wait()
    closing = asyncio.create_task(service.close(call_id, OWNER))
    await cleanup.wait()
    await asyncio.sleep(0.02)
    assert backend.leases
    assert not closing.done()
    release.set()
    await asyncio.wait_for(closing, 1)
    assert not backend.leases


async def test_usage_is_cumulative_and_close_preserves_finalization(setup):
    service, backend, _ = setup
    call_id, connection = await create(setup)
    connection.feed("session.usage.updated", usage={"seconds": 3})
    connection.feed("session.usage.updated", usage={"seconds": 8})
    await until(lambda: service._calls[call_id].usage.get("seconds") == 8)
    record = await service.close(call_id, OWNER)
    assert record["usage"] == {"seconds": 12}
    assert record["finalized"] is True
    assert connection.closed
    assert not backend.leases


async def test_missing_final_event_is_not_reported_successful(setup):
    service, _, _ = setup
    call_id, connection = await create(setup)
    connection.finalize = False
    record = await service.close(call_id, OWNER)
    assert record["status"] == "interrupted"
    assert record["finalized"] is False
    assert connection.closed


async def test_reconnect_event_replay_does_not_rerun_work(setup):
    service, backend, _ = setup
    call_id, connection = await create(setup)
    delegate(connection)
    await until(lambda: connection.sent)
    await service.close(call_id, OWNER)
    first = [e async for e in service.events(call_id, OWNER)]
    cursor = int(first[-1].splitlines()[0].split(": ")[1])
    assert not [e async for e in service.events(call_id, OWNER, cursor)]
    assert len(backend.requests) == 1


async def test_sse_disconnect_keeps_voice_running(setup):
    service, _, _ = setup
    call_id, _ = await create(setup)
    async with aclosing(service.events(call_id, OWNER)) as events:
        assert "connected" in await anext(events)
    assert not service._calls[call_id].done.is_set()


async def test_speech_fragments_and_duplicate_event_ids(setup):
    service, _, _ = setup
    call_id, connection = await create(setup)
    for _ in range(2):
        connection.feed(
            "session.input_transcript.delta", event_id="one", delta="Hello", start_ms=0, end_ms=100
        )
    connection.feed("session.output_transcript.delta", delta="Hi", start_ms=50, end_ms=200)
    await until(lambda: len(service._calls[call_id].transcript) == 2)
    record = await service.get(call_id, OWNER)
    assert [f["role"] for f in record["transcript"]] == ["user", "assistant"]
    assert record["transcript"][1]["start_ms"] == 50


async def test_transcript_limit_closes_without_dropping_final_usage(setup):
    service, _, _ = setup
    service.config = service.config.model_copy(update={"max_transcript_chars": 4})
    call_id, connection = await create(setup)
    connection.feed("session.input_transcript.delta", delta="too long")
    await until(lambda: service._calls[call_id].done.is_set())
    record = await service.get(call_id, OWNER)
    assert record["transcript"] == []
    assert record["finalized"]


async def test_worker_failure_closes_session(setup):
    service, _, _ = setup
    call_id, connection = await create(setup)
    connection.send = AsyncMock(side_effect=OSError("gone"))
    delegate(connection, text=None)
    await until(lambda: service._calls[call_id].done.is_set())
    assert (await service.get(call_id, OWNER))["status"] == "interrupted"


async def test_spoken_result_respects_append_byte_budget(setup):
    service, backend, _ = setup

    async def runner(request):
        yield {"type": "final_response", "content": "界" * 1000}

    backend.runner = runner
    _, connection = await create(setup)
    delegate(connection)
    await until(lambda: connection.sent)
    assert len(connection.sent[0]["content"].encode()) < 500


async def test_restored_checkpoint_is_interrupted_and_sse_never_restarts_work(setup):
    service, backend, transport = setup
    call_id, connection = await create(setup)
    delegate(connection)
    await until(
        lambda: service._calls[call_id].delegations.get("item_1", {}).get("status") == "result_sent"
    )
    checkpoint = service._calls[call_id].snapshot()
    await service.close(call_id, OWNER)
    service._calls.pop(call_id)
    service._persistence.load = AsyncMock(return_value=checkpoint)
    record = await service.get(call_id, OWNER)
    assert record["status"] == "interrupted"
    assert record["reason"] == "runtime_restarted"
    assert record["finalized"] is False
    before = len(backend.requests)
    frames = [frame async for frame in service.events(call_id, OWNER)]
    assert len(frames) == 1
    assert '"snapshot"' in frames[0]
    assert len(backend.requests) == before
    with pytest.raises(VoiceError, match="not active"):
        await service.cancel_work(call_id, OWNER)


async def test_reservation_is_held_until_final_checkpoint_is_complete(setup):
    service, backend, _ = setup
    call_id, _ = await create(setup)
    entered, release = asyncio.Event(), asyncio.Event()

    async def save(record):
        if record["status"] == "closed":
            entered.set()
            await release.wait()
        return True

    service._persistence.save = save
    closing = asyncio.create_task(service.close(call_id, OWNER))
    await asyncio.wait_for(entered.wait(), 2)
    assert "conversation" in backend.leases
    assert not service._calls[call_id].done.is_set()
    release.set()
    await closing
    assert not backend.leases
    service.forget_session("conversation")
    assert call_id not in service._calls


async def test_empty_provider_delegation_is_ignored_without_stopping_worker(setup):
    service, backend, _ = setup
    call_id, connection = await create(setup)
    delegate(connection, ident="")
    delegate(connection, ident="valid")
    await until(lambda: connection.sent)
    assert not service._calls[call_id].done.is_set()
    assert "" not in service._calls[call_id].delegations
    assert len(backend.requests) == 1
    assert connection.sent[0]["delegation_id"] == "valid"


@pytest.mark.parametrize("provider_closes", [False, True])
async def test_cancel_cleanup_does_not_block_sideband_or_lose_guard_on_http_disconnect(
    setup, provider_closes
):
    service, backend, _ = setup
    started, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def runner(request):
        if len(backend.requests) == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup.set()
                await release.wait()
        yield {"type": "final_response", "model": "test", "content": "Second request completed"}

    backend.runner = runner
    call_id, connection = await create(setup)
    call = service._calls[call_id]
    delegate(connection)
    await asyncio.wait_for(started.wait(), 2)
    cancelling = asyncio.create_task(service.cancel_work(call_id, OWNER))
    await asyncio.wait_for(cleanup.wait(), 2)
    cancelling.cancel()  # Simulate the HTTP caller disappearing during cleanup.
    with pytest.raises(asyncio.CancelledError):
        await cancelling
    delegate(connection, ident="item_2", text="Check a second item")
    connection.feed("session.usage.updated", usage={"seconds": 9})
    await until(lambda: call.usage.get("seconds") == 9)
    assert call.cancelling
    assert call.delegations["item_2"]["status"] == "waiting"
    assert len(backend.requests) == 1
    if provider_closes:
        connection.feed("session.closed", usage={"seconds": 10}, reason="provider_closed")
        await until(lambda: call.finalized)
        assert not call.done.is_set()
    release.set()
    if provider_closes:
        await asyncio.wait_for(call.done.wait(), 2)
        assert len(backend.requests) == 1
        assert call.snapshot()["finalized"] is True
    else:
        await until(lambda: any(e.get("delegation_id") == "item_2" for e in connection.sent))
        assert len(backend.requests) == 2
    assert not call.cancelling


async def test_second_cancel_discards_delegation_waiting_for_cleanup(setup):
    service, backend, _ = setup
    call_id, connection = await create(setup)
    entered, release = asyncio.Event(), asyncio.Event()

    async def cleanup(*args):
        entered.set()
        await release.wait()

    backend.cancel_reserved_work.side_effect = cleanup
    first = asyncio.create_task(service.cancel_work(call_id, OWNER))
    await asyncio.wait_for(entered.wait(), 2)
    delegate(connection, ident="waiting")
    call = service._calls[call_id]
    await until(lambda: call.deferred_delegation == "waiting")
    second = asyncio.create_task(service.cancel_work(call_id, OWNER))
    await until(lambda: call.deferred_delegation is None)
    release.set()
    assert await first == {"cancelled": False}
    assert await second == {"cancelled": True}
    assert call.delegations["waiting"]["status"] == "cancelled"
    assert not backend.requests
