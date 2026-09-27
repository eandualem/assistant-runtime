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
