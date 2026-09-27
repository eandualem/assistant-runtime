"""ActionService: host-written actions, conditional changes, and owner confirmations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from assistant_runtime.services.actions.config import ActionsConfig
from assistant_runtime.services.actions.exceptions import (
    ActionConflictError,
    ActionError,
    ActionNotFoundError,
)
from assistant_runtime.services.actions.interface import ActionService

SHA = "a" * 64
WHEN = datetime(2026, 9, 27, 8, 0, tzinfo=UTC)


async def _service() -> ActionService:
    service = ActionService(ActionsConfig())
    await service.start()
    return service


def _confirmation(**overrides):
    values = {
        "confirmation_id": str(uuid.uuid4()),
        "recipient": "agent-a",
        "kind": "message",
        "revision": 1,
        "text_sha256": SHA,
        "source": "button",
        "confirmed_at": WHEN,
    }
    return {**values, **overrides}


class TestActions:
    async def test_an_action_keeps_its_status_history_and_text_revisions(self):
        service = await _service()
        action = await service.create(kind="message", text="Draft", by="local", profile="lead")
        assert (action.id, action.status, action.revision) == (1, "proposed", 1)
        edited = await service.update(action.id, {"text": "Better"}, by="local:owner")
        assert (edited.text, edited.revision, edited.proposed_text) == ("Better", 2, "Draft")
        again = await service.update(action.id, {"text": "Best"}, by="local:owner")
        assert (again.revision, again.proposed_text) == (3, "Draft")  # the first text stays
        sent = await service.update(
            action.id,
            {"status": "sent", "results": {"agent-a": {"ok": True}}},
            by="local",
            expected_status=["proposed", "scheduled"],
            expected_revision=3,
        )
        assert [h["status"] for h in sent.history] == ["proposed", "sent"]
        merged = await service.update(action.id, {"results": {"agent-b": "late"}}, by="local")
        assert merged.results == {"agent-a": {"ok": True}, "agent-b": "late"}

    async def test_a_change_waits_for_the_status_and_revision_it_expects(self):
        service = await _service()
        action = await service.create(kind="message", text="Draft", by="local")
        await service.update(action.id, {"status": "discarded"}, by="local")
        with pytest.raises(ActionConflictError, match="discarded"):
            await service.update(
                action.id, {"status": "sending"}, by="local", expected_status=["proposed"]
            )
        with pytest.raises(ActionConflictError, match="revision 1"):
            await service.update(action.id, {"text": "x"}, by="local", expected_revision=2)
        assert (await service.get(action.id))[0].status == "discarded"

    async def test_unknown_values_and_ids_are_refused(self):
        service = await _service()
        with pytest.raises(ActionError, match="status"):
            await service.create(kind="message", by="local", status="done")
        action = await service.create(kind="message", by="local")
        with pytest.raises(ActionError, match="created_by"):
            await service.update(action.id, {"created_by": "someone"}, by="local")
        with pytest.raises(ActionNotFoundError):
            await service.update(99, {"status": "sent"}, by="local")

    async def test_listing_is_newest_first_and_filtered(self):
        service = await _service()
        await service.create(kind="message", by="local", subject="agent-a")
        await service.create(kind="steer", by="local", subject="agent-b")
        await service.create(kind="message", by="local", subject="agent-b")
        assert [a.id for a in await service.list()] == [3, 2, 1]
        assert [a.id for a in await service.list(kind="message", subject="agent-b")] == [3]


class TestConfirmations:
    async def test_what_was_confirmed_never_changes_and_settles_once(self):
        service = await _service()
        action = await service.create(kind="message", text="Hi", by="local")
        values = _confirmation()
        confirmed = await service.confirm(action.id, **values)
        assert (confirmed.seq, confirmed.status, confirmed.key_epoch) == (1, "confirmed", None)
        signed = await service.update_confirmation(
            confirmed.id, {"key_epoch": 3, "sender": "app", "audience": "hub"}
        )
        assert (signed.key_epoch, signed.sender) == (3, "app")
        sent = await service.update_confirmation(
            confirmed.id, {"status": "sent", "result": {"receipt": 7}}
        )
        assert (sent.status, sent.result, sent.recipient) == ("sent", {"receipt": 7}, "agent-a")
        with pytest.raises(ActionConflictError, match="sent"):
            await service.update_confirmation(confirmed.id, {"status": "failed"})
        with pytest.raises(ActionConflictError, match="sent"):
            await service.update_confirmation(confirmed.id, {"key_epoch": 4})
        with pytest.raises(ActionConflictError, match="sent"):
            await service.update_confirmation(confirmed.id, {"result": None})
        with pytest.raises(ActionError, match="recipient"):
            await service.update_confirmation(confirmed.id, {"recipient": "agent-z"})
        reconciled = await service.update_confirmation(confirmed.id, {"reconciled": "matched"})
        assert reconciled.reconciled == "matched"
        with pytest.raises(ActionConflictError, match="exists"):
            await service.confirm(action.id, **values)

    async def test_a_confirmation_is_checked_before_it_is_written(self):
        service = await _service()
        action = await service.create(kind="message", by="local")
        for bad, message in (
            ({"confirmed_at": datetime(2026, 9, 27, 8, 0)}, "offset"),
            ({"text_sha256": "short"}, "text_sha256"),
            ({"confirmation_id": "not-a-uuid"}, "UUID"),
            ({"source": "email"}, "source"),
        ):
            with pytest.raises(ActionError, match=message):
                await service.confirm(action.id, **_confirmation(**bad))
        with pytest.raises(ActionNotFoundError):
            await service.confirm(42, **_confirmation())
        other = await service.confirm(action.id, **_confirmation())
        with pytest.raises(ActionNotFoundError):  # a confirmation belongs to its own action
            await service.update_confirmation(other.id, {"status": "sent"}, action_id=action.id + 1)

    async def test_confirmations_read_in_insertion_order_for_reconciliation(self):
        service = await _service()
        first = await service.create(kind="message", by="local")
        second = await service.create(kind="message", by="local")
        a = await service.confirm(second.id, **_confirmation(recipient="agent-b"))
        b = await service.confirm(first.id, **_confirmation(key_epoch=1))
        c = await service.confirm(second.id, **_confirmation(recipient="agent-c", key_epoch=1))
        await service.update_confirmation(c.id, {"reconciled": "matched"})
        assert [r.id for r in await service.confirmations()] == [a.id, b.id, c.id]
        assert [r.id for r in await service.confirmations(after=a.seq)] == [b.id, c.id]
        assert [r.id for r in await service.confirmations(signed=True, reconciled=False)] == [b.id]
        _, listed = await service.get(second.id)
        assert [r.id for r in listed] == [a.id, c.id]
