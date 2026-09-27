"""/events routes against a real EventLogService on an in-memory store."""

from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.event_log.config import EventLogConfig
from assistant_runtime.app.event_log.interface import EventLogService
from assistant_runtime.app.routes.events import router


async def _client():
    service = EventLogService(EventLogConfig())
    await service.start()
    app = FastAPI()
    app.include_router(router)
    app.state.event_log_service = service
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


EVENT = {"event_id": "e1", "direction": "inbound", "source": "hub", "kind": "build_failed"}


async def test_an_event_is_created_once_and_listed_in_order():
    async with await _client() as c:
        first = await c.post("/events", json=EVENT)
        repeat = await c.post("/events", json=EVENT)
        await c.post("/events", json={**EVENT, "event_id": "e2", "history": True})
        listed = (await c.get("/events", params={"news_only": True})).json()
        after = (await c.get("/events", params={"after": 1})).json()
    assert (first.status_code, repeat.status_code) == (201, 200)
    assert repeat.json()["id"] == first.json()["id"]
    assert [e["event_id"] for e in listed["events"]] == ["e1"]
    assert (listed["next_after"], [e["id"] for e in after["events"]]) == (1, [2])


async def test_a_notice_moves_forward_and_bad_requests_are_refused():
    async with await _client() as c:
        notice = (await c.post("/events", json={**EVENT, "direction": "outbound"})).json()
        heard = await c.patch(f"/events/{notice['id']}", json={"status": "heard"})
        back = await c.patch(f"/events/{notice['id']}", json={"status": "delivered"})
        missing = await c.patch("/events/99", json={"status": "heard"})
        naive = await c.post("/events", json={**EVENT, "occurred_at": "2026-09-27T08:00:00"})
    assert heard.json()["status"] == "heard"
    assert (back.status_code, missing.status_code, naive.status_code) == (409, 404, 422)
