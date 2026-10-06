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


async def test_updated_after_lists_changed_actions_oldest_first():
    async with await _client() as c:
        ids = [(await c.post("/actions", json={"kind": "message"})).json()["id"] for _ in range(3)]
        everything = (await c.get("/actions", params={"updated_after": "2000-01-01"})).json()
        cursor = everything["next_cursor"]
        idle = (await c.get("/actions", params={"updated_after": cursor})).json()
        await c.patch(f"/actions/{ids[0]}", json={"status": "sent"})
        changed = (await c.get("/actions", params={"updated_after": cursor})).json()
        plain = (await c.get("/actions")).json()
        both = await c.get("/actions", params={"updated_after": cursor, "before": ids[1]})
        not_an_id = await c.get("/actions", params={"updated_after": "2026-10-01T00:00|abc"})
    assert [a["id"] for a in everything["actions"]] == ids
    last = everything["actions"][-1]
    assert cursor == f"{last['updated_at']}|{last['id']}"
    assert idle == {"actions": [], "next_cursor": cursor}
    assert [(a["id"], a["status"]) for a in changed["actions"]] == [(ids[0], "sent")]
    assert "next_cursor" not in plain  # only with updated_after
    assert (both.status_code, not_an_id.status_code) == (422, 422)


async def test_a_confirmation_moves_its_action_past_the_cursor():
    async with await _client() as c:
        action_id = (await c.post("/actions", json={"kind": "message"})).json()["id"]
        cursor = (await c.get("/actions", params={"updated_after": "2000-01-01"})).json()[
            "next_cursor"
        ]
        body = _confirmation()
        await c.post(f"/actions/{action_id}/confirmations", json=body)
        confirmed = (await c.get("/actions", params={"updated_after": cursor})).json()
        await c.patch(f"/actions/{action_id}/confirmations/{body['id']}", json={"status": "sent"})
        settled = await c.get("/actions", params={"updated_after": confirmed["next_cursor"]})
    assert [a["id"] for a in confirmed["actions"]] == [action_id]
    assert [a["id"] for a in settled.json()["actions"]] == [action_id]
