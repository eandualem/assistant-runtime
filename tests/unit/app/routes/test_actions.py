"""/actions routes against a real ActionService on an in-memory store."""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.actions import confirmations_router, router
from assistant_runtime.services.actions.config import ActionsConfig
from assistant_runtime.services.actions.interface import ActionService


async def _client():
    service = ActionService(ActionsConfig())
    await service.start()
    app = FastAPI()
    app.include_router(router)
    app.include_router(confirmations_router)
    app.state.action_service = service
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _confirmation(**overrides):
    return {
        "id": str(uuid.uuid4()),
        "recipient": "agent-a",
        "kind": "message",
        "revision": 1,
        "text_sha256": "b" * 64,
        "source": "voice",
        "confirmed_at": "2026-09-27T08:00:00+02:00",
        **overrides,
    }


async def test_an_action_changes_only_as_the_host_expects():
    async with await _client() as c:
        created = await c.post("/actions", json={"kind": "message", "text": "Draft"})
        action_id = created.json()["id"]
        edited = await c.patch(f"/actions/{action_id}", json={"text": "Better"})
        stale = await c.patch(
            f"/actions/{action_id}", json={"status": "sending", "expected_revision": 1}
        )
        sent = await c.patch(
            f"/actions/{action_id}",
            json={"status": "sent", "expected_status": ["proposed"], "confirmed_by": "local"},
        )
        cleared = await c.patch(f"/actions/{action_id}", json={"confirmed_by": None})
        listed = (await c.get("/actions", params={"status": "sent"})).json()
    assert created.status_code == 201
    assert (edited.json()["revision"], stale.status_code) == (2, 409)
    assert [h["status"] for h in sent.json()["history"]] == ["proposed", "sent"]
    assert sent.json()["history"][-1]["by"] == "local"
    assert cleared.json()["confirmed_by"] is None  # an explicit null clears the field
    assert [a["id"] for a in listed["actions"]] == [action_id]


async def test_confirmations_are_written_before_the_send_and_read_in_order():
    async with await _client() as c:
        action_id = (await c.post("/actions", json={"kind": "message"})).json()["id"]
        body = _confirmation()
        first = await c.post(f"/actions/{action_id}/confirmations", json=body)
        duplicate = await c.post(f"/actions/{action_id}/confirmations", json=body)
        naive = await c.post(
            f"/actions/{action_id}/confirmations",
            json=_confirmation(confirmed_at="2026-09-27T08:00:00"),
        )
        settled = await c.patch(
            f"/actions/{action_id}/confirmations/{body['id']}",
            json={"status": "failed", "result": "refused"},
        )
        elsewhere = await c.patch(
            f"/actions/{action_id + 1}/confirmations/{body['id']}", json={"reconciled": "missing"}
        )
        detail = (await c.get(f"/actions/{action_id}")).json()
        page = (await c.get("/action-confirmations", params={"after": 0})).json()
    assert (first.status_code, duplicate.status_code, naive.status_code) == (201, 409, 422)
    assert (settled.json()["status"], elsewhere.status_code) == ("failed", 404)
    assert [c["id"] for c in detail["confirmations"]] == [body["id"]]
    assert (page["next_after"], page["confirmations"][0]["recipient"]) == (1, "agent-a")


async def test_before_lists_the_next_older_actions():
    async with await _client() as c:
        ids = [(await c.post("/actions", json={"kind": "message"})).json()["id"] for _ in range(4)]
        page = (await c.get("/actions", params={"before": ids[2], "limit": 1})).json()
        rest = (await c.get("/actions", params={"before": ids[1]})).json()
        unknown = await c.get("/actions", params={"before": 999})
    assert [a["id"] for a in page["actions"]] == [ids[1]]
    assert [a["id"] for a in rest["actions"]] == [ids[0]]
    assert (unknown.status_code, unknown.json()["detail"]) == (404, "No action 999")
