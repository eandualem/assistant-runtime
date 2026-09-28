"""Persistent agent endpoints — start, move, stop and message agents, and read their records.

A persistent agent keeps one continuing session; each message runs as its
next turn (see ``app/tasks``). Starting, moving, stopping and messaging
agents is administration; callers read their own records, and an
administrator reads all.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from assistant_runtime.app.access.deps import AdminDep, PrincipalDep
from assistant_runtime.app.tasks.deps import TaskServiceDep
from assistant_runtime.app.tasks.exceptions import (
    AgentConflictError,
    AgentNotFoundError,
    TaskError,
    TaskLimitError,
    TasksDisabledError,
)
from assistant_runtime.services.artifacts.exceptions import UnknownProfileError

router = APIRouter(prefix="/agents", tags=["agents"])

_STATUS = {
    TasksDisabledError: 503,
    AgentNotFoundError: 404,
    AgentConflictError: 409,
    TaskLimitError: 429,
    UnknownProfileError: 404,
}


class StartAgentRequest(BaseModel):
    profile: str | None = None
    subject: str | None = None
    session_id: str | None = Field(default=None, max_length=64)


class MoveAgentRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=64)


class AgentMessageRequest(BaseModel):
    content: str = Field(..., min_length=1)
    parent_session_id: str | None = Field(default=None, max_length=64)


def _http_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=_STATUS.get(type(exc), 422), detail=str(exc))


@router.post("", status_code=201)
async def start_agent(body: StartAgentRequest, tasks: TaskServiceDep, admin: AdminDep) -> dict:
    """Start a persistent agent for a profile and subject; one active agent per pair."""
    try:
        record = await tasks.start_agent(
            profile=body.profile, subject=body.subject, session_id=body.session_id, principal=admin
        )
    except (TaskError, UnknownProfileError) as e:
        raise _http_error(e) from e
    return record.to_dict()


@router.get("")
async def list_agents(
    tasks: TaskServiceDep,
    principal: PrincipalDep,
    status: str | None = Query(None, pattern="^(active|stopped)$"),
) -> list[dict]:
    """Newest first: the caller's agents (every agent for an administrator)."""
    return [record.to_dict() for record in await tasks.list_agents(principal, status=status)]


@router.get("/{agent_id}")
async def get_agent(agent_id: str, tasks: TaskServiceDep, principal: PrincipalDep) -> dict:
    try:
        return (await tasks.get_agent(agent_id, principal)).to_dict()
    except TaskError as e:
        raise _http_error(e) from e


@router.post("/{agent_id}/session")
async def move_agent(
    agent_id: str, body: MoveAgentRequest, tasks: TaskServiceDep, admin: AdminDep
) -> dict:
    """Move the agent to a fresh session (a new id unless one is given)."""
    try:
        return (
            await tasks.move_agent(agent_id, session_id=body.session_id, principal=admin)
        ).to_dict()
    except TaskError as e:
        raise _http_error(e) from e


@router.post("/{agent_id}/stop")
async def stop_agent(agent_id: str, tasks: TaskServiceDep, admin: AdminDep) -> dict:
    """Stop the agent; its queued and running messages end ``cancelled``."""
    try:
        return (await tasks.stop_agent(agent_id, admin)).to_dict()
    except TaskError as e:
        raise _http_error(e) from e


@router.post("/{agent_id}/messages", status_code=202)
async def message_agent(
    agent_id: str, body: AgentMessageRequest, tasks: TaskServiceDep, admin: AdminDep
) -> dict:
    """Queue a message; it runs as the agent's next turn and ends with ``agent_message_finished``."""
    try:
        record = await tasks.message_agent(
            body.content,
            agent_id=agent_id,
            parent_session_id=body.parent_session_id,
            principal=admin,
        )
    except TaskError as e:
        raise _http_error(e) from e
    return record.to_dict()


@router.get("/{agent_id}/messages")
async def list_agent_messages(
    agent_id: str,
    tasks: TaskServiceDep,
    principal: PrincipalDep,
    parent_session_id: str | None = None,
    status: str | None = Query(
        None, pattern="^(queued|running|done|failed|cancelled|interrupted)$"
    ),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    """Newest first: the agent's messages that the caller may read."""
    try:
        await tasks.get_agent(agent_id, principal)
    except TaskError as e:
        raise _http_error(e) from e
    records = await tasks.list_messages(
        principal,
        agent_id=agent_id,
        parent_session_id=parent_session_id,
        status=status,
        limit=limit,
    )
    return [record.to_dict() for record in records]


@router.get("/{agent_id}/messages/{message_id}")
async def get_agent_message(
    agent_id: str, message_id: str, tasks: TaskServiceDep, principal: PrincipalDep
) -> dict:
    try:
        record = await tasks.get_message(message_id, principal)
    except TaskError as e:
        raise _http_error(e) from e
    if record.agent_id != agent_id:
        raise HTTPException(status_code=404, detail=f"No agent message '{message_id}'")
    return record.to_dict()
