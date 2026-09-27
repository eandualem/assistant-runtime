"""A session created for a subject keeps its profile and subject for every later turn."""

from __future__ import annotations

import pytest

from assistant_runtime.app.assistant._session_store import SessionStore
from assistant_runtime.app.assistant.exceptions import SessionError
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.app.streaming._turn import TurnPlanner


def _request(message_id: str, **fields) -> AssistantRequest:
    return AssistantRequest(id=message_id, session_id="s-1", content="hi", **fields)


async def test_the_first_message_binds_the_session_and_later_turns_inherit_it():
    sessions = SessionStore()
    planner = TurnPlanner(sessions)
    first = await planner.plan(_request("m1", profile="tracker", subject="agent-a"))
    assert (first.request.profile, first.request.subject) == ("tracker", "agent-a")
    context = sessions.get_context("s-1")
    assert (context["profile"], context["subject"]) == ("tracker", "agent-a")
    later = await planner.plan(_request("m2"))
    assert (later.request.profile, later.request.subject) == ("tracker", "agent-a")


@pytest.mark.parametrize(
    "fields",
    [{"subject": "agent-b"}, {"profile": "other"}, {"profile": "other", "subject": "agent-a"}],
)
async def test_another_profile_or_subject_is_rejected(fields):
    sessions = SessionStore()
    planner = TurnPlanner(sessions)
    await planner.plan(_request("m1", profile="tracker", subject="agent-a"))
    with pytest.raises(SessionError, match="not"):
        await planner.plan(_request("m2", **fields))


async def test_a_session_without_a_subject_still_selects_its_profile_per_turn():
    sessions = SessionStore()
    planner = TurnPlanner(sessions)
    await planner.plan(_request("m1", profile="tracker"))
    later = await planner.plan(_request("m2", profile="other"))
    assert later.request.profile == "other"
    assert sessions.get_context("s-1")["subject"] is None


async def test_an_omitted_profile_binds_the_default_so_naming_it_later_is_accepted():
    sessions = SessionStore()
    planner = TurnPlanner(sessions, default_profile="neutral")
    first = await planner.plan(_request("m1", subject="agent-a"))
    assert first.request.profile == "neutral"
    assert sessions.get_context("s-1")["profile"] == "neutral"
    later = await planner.plan(_request("m2", profile="neutral"))
    assert later.request.subject == "agent-a"


async def test_steering_admitted_with_the_binding_is_promoted_in_a_bound_session():
    """A delivered message names no profile; it inherits the binding at admission."""
    from assistant_runtime.app.streaming._turn import apply_binding

    sessions = SessionStore()
    planner = TurnPlanner(sessions)
    await planner.plan(_request("m1", profile="tracker", subject="agent-a"))
    steering = _request("st-1", message_type="steering")
    admitted = apply_binding(steering, sessions.get_context("s-1"))
    await sessions.queue_steering("s-1", admitted)
    plan = await planner.plan(steering)
    assert plan.kind == "steering"
    assert (plan.request.profile, plan.request.subject) == ("tracker", "agent-a")


async def test_steering_about_another_subject_is_refused_at_admission():
    from assistant_runtime.app.streaming._turn import apply_binding

    sessions = SessionStore()
    await TurnPlanner(sessions).plan(_request("m1", profile="tracker", subject="agent-a"))
    with pytest.raises(SessionError, match="agent-a"):
        apply_binding(
            _request("st-1", message_type="steering", subject="agent-b"),
            sessions.get_context("s-1"),
        )


async def test_a_stranger_is_refused_access_before_the_binding_is_checked():
    from assistant_runtime.app.access.exceptions import AccessDeniedError
    from assistant_runtime.principal import Principal

    sessions = SessionStore()
    planner = TurnPlanner(sessions)
    await planner.plan(_request("m1", profile="tracker", subject="agent-a"), Principal(id="alice"))
    with pytest.raises(AccessDeniedError):
        await planner.plan(_request("m2", subject="agent-b"), Principal(id="mallory"))


async def test_the_binding_reaches_a_row_created_before_the_first_message():
    from unittest.mock import AsyncMock

    store = SessionStore()
    db = AsyncMock()
    db.load = AsyncMock(return_value=None)
    store._db = db
    await store.register_user_message(
        AssistantRequest(
            id="m1", session_id="s-1", content="hi", profile="tracker", subject="agent-a"
        ),
        owner_id="alice",
    )
    db.set_binding.assert_awaited_once_with("s-1", "tracker", "agent-a")
