"""Data types of the tasks module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

TaskStatus = Literal["queued", "running", "done", "failed", "cancelled", "interrupted"]
FINISHED: frozenset[str] = frozenset({"done", "failed", "cancelled", "interrupted"})


@dataclass(frozen=True)
class TaskRecord:
    """One task: what it was asked, where it runs, and how it ended.

    ``session_id`` is the task's own fresh session (``task-<id>``); a task
    with a ``subject`` runs bound to that profile and subject, one at a time
    per (profile, subject). ``parent_session_id`` is the session that asked
    for it, if any.
    """

    id: str
    session_id: str
    task: str
    status: TaskStatus
    created_by: str
    profile: str | None = None
    subject: str | None = None
    context: str | None = None
    parent_session_id: str | None = None
    result: str | None = None
    error: str | None = None
    usage: dict[str, Any] | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def finished(self) -> bool:
        return self.status in FINISHED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "profile": self.profile,
            "subject": self.subject,
            "task": self.task,
            "context": self.context,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "usage": self.usage,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


AgentStatus = Literal["active", "stopped"]


@dataclass(frozen=True)
class AgentRecord:
    """A persistent agent: one continuing session its messages run in, in order.

    The host starts it for a profile and an optional subject (one active
    agent per pair), moves it to a fresh session when its conversation
    should start over, and stops it. ``session_id`` is where the next
    message runs; ``config`` holds the tunables its turns send as a
    request's ``config`` (only those set).
    """

    id: str
    session_id: str
    status: AgentStatus
    created_by: str
    profile: str | None = None
    subject: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    stopped_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "profile": self.profile,
            "subject": self.subject,
            "config": dict(self.config),
            "status": self.status,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
        }


@dataclass(frozen=True)
class AgentMessageRecord:
    """One message to a persistent agent, and how the turn that answered it ended.

    ``session_id`` is the agent's session the turn ran in, set when it
    starts; ``parent_session_id`` is the session that sent the message, if any.
    """

    id: str
    agent_id: str
    content: str
    status: TaskStatus
    created_by: str
    session_id: str | None = None
    parent_session_id: str | None = None
    result: str | None = None
    error: str | None = None
    usage: dict[str, Any] | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def finished(self) -> bool:
        return self.status in FINISHED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "content": self.content,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "usage": self.usage,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
