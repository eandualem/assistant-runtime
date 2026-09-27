"""EventLogService: stored once, steered only when targeted, notices move forward."""

from __future__ import annotations

import asyncio

import pytest

from assistant_runtime.app.event_log.config import EventLogConfig
from assistant_runtime.app.event_log.exceptions import (
    EventConflictError,
    EventLogError,
    EventNotFoundError,
)
from assistant_runtime.app.event_log.interface import EventLogService


class FakeIngress:
    def __init__(self, outcome=None, error=None):
        self.calls = []
        self.outcome = outcome or {
            "status": "delivered",
            "session_id": "chat",
            "delivery": "queued",
        }
        self.error = error

    async def deliver(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.outcome


async def _service(ingress=None) -> EventLogService:
    service = EventLogService(EventLogConfig(), ingress_service=ingress or FakeIngress())
    await service.start()
    return service


def _event(**overrides):
    values = {
        "event_id": "e1",
        "direction": "inbound",
        "source": "hub",
        "kind": "build_failed",
        "agent": "agent-a",
        "summary": "Tests fail on main",
    }
    return {**values, **overrides}


class TestInbound:
    async def test_an_event_is_stored_once_and_only_steered_when_targeted(self):
        ingress = FakeIngress()
        service = await _service(ingress)
        stored, created = await service.record(**_event())
        assert (stored.id, stored.status, created) == (1, "received", True)
        assert not ingress.calls  # store-only by default
        again, created = await service.record(**_event(summary="changed"))
        assert (again.id, again.summary, created) == (1, "Tests fail on main", False)
        other_source, created = await service.record(**_event(source="mail"))
        assert (other_source.id, created) == (2, True)

    async def test_a_targeted_event_is_steered_into_its_session(self):
        ingress = FakeIngress()
        service = await _service(ingress)
        record, _ = await service.record(
            **_event(event_id="e2", severity="critical", target_session_id="chat")
        )
        assert (record.status, record.delivery) == (
            "delivered",
            {"session_id": "chat", "how": "queued"},
        )
        assert record.delivered_at is not None
        [call] = ingress.calls
        assert call["message"] == "build_failed (critical): Tests fail on main"
        assert (call["via"], call["from_agent"], call["session_id"]) == ("hub", "agent-a", "chat")
        assert call["severity"] == "urgent"  # the inbox's own ordering name
        await service.record(**_event(event_id="e2", target_session_id="chat"))
        assert len(ingress.calls) == 1  # a repeat is never delivered again

    async def test_a_repeat_delivers_an_event_only_if_no_request_started_to(self):
        ingress = FakeIngress()
        service = await _service(ingress)
        # The process stopped between storing the event and delivering it.
        async with service._store.transaction() as tx:
            await tx.create_if_new(
                **_event(target_session_id="chat"), severity="info", status="received"
            )
        record, created = await service.record(**_event(target_session_id="chat"))
        assert (created, record.status, len(ingress.calls)) == (False, "delivered", 1)

        started, gate = asyncio.Event(), asyncio.Event()
        slow = FakeIngress()
        original = slow.deliver

        async def deliver(**kwargs):
            started.set()
            await gate.wait()
            return await original(**kwargs)

        slow.deliver = deliver
        service = await _service(slow)
        first = asyncio.create_task(service.record(**_event(target_session_id="chat")))
        try:
            await asyncio.wait_for(started.wait(), 5)
            repeat, _ = await service.record(**_event(target_session_id="chat"))
            assert repeat.delivery == {"session_id": "chat"}  # started, outcome not known yet
        finally:
            gate.set()
        delivered, _ = await first
        assert (delivered.status, len(slow.calls)) == ("delivered", 1)

    async def test_held_history_and_failed_deliveries_are_recorded_as_such(self):
        held = FakeIngress({"status": "queued", "inbox_id": "i1", "session_id": "chat"})
        service = await _service(held)
        record, _ = await service.record(**_event(target_session_id="chat"))
        assert record.delivery == {"session_id": "chat", "how": "inbox"}
        imported, _ = await service.record(
            **_event(event_id="old", target_session_id="chat", history=True)
        )
        assert (imported.status, len(held.calls)) == ("received", 1)
        broken = await _service(FakeIngress(error=RuntimeError("no active conversation")))
        failed, _ = await broken.record(**_event(target_session_id="chat"))
        assert (failed.status, failed.delivery["error"]) == ("received", "no active conversation")

    async def test_listing_continues_after_an_id_and_can_leave_out_history(self):
        service = await _service()
        await service.record(**_event(event_id="old", history=True))
        await service.record(**_event(event_id="new-a"))
        await service.record(**_event(event_id="new-b", agent="agent-b"))
        assert [e.event_id for e in await service.list(after=1)] == ["new-a", "new-b"]
        assert [e.event_id for e in await service.list(news_only=True, agent="agent-b")] == [
            "new-b"
        ]
        assert [e.id for e in await service.list(limit=1)] == [1]


class TestNotices:
    async def test_a_notice_only_moves_forward(self):
        service = await _service()
        notice, _ = await service.record(**_event(event_id="n1", direction="outbound"))
        assert notice.status == "pending"
        heard = await service.update_notice(notice.id, "heard")
        assert heard.status == "heard"
        assert heard.delivered_at is not None
        assert heard.heard_at is not None
        with pytest.raises(EventConflictError, match="heard already"):
            await service.update_notice(notice.id, "delivered")
        assert (await service.update_notice(notice.id, "heard")).heard_at == heard.heard_at

    async def test_notices_are_outbound_and_have_no_target(self):
        service = await _service()
        inbound, _ = await service.record(**_event())
        with pytest.raises(EventConflictError, match="inbound"):
            await service.update_notice(inbound.id, "heard")
        with pytest.raises(EventLogError, match="target"):
            await service.record(**_event(direction="outbound", target_session_id="chat"))
        with pytest.raises(EventNotFoundError):
            await service.update_notice(99, "heard")
