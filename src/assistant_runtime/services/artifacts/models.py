"""Data types of the artifact service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

ActorKind = Literal["assistant", "host"]
VersionStatus = Literal["active", "pending", "superseded", "rejected"]


@dataclass(frozen=True)
class Actor:
    """Who is asking for a mutation. Permissions depend on ``kind`` only.

    ``label`` is recorded as ``proposed_by`` (or ``decided_by``) on the
    version; it is informational and never widens what the actor may do.
    ``session_id`` is the session the actor works in, when there is one; it
    travels on the published events so a host can place them.
    """

    kind: ActorKind
    label: str = ""
    session_id: str | None = None

    @property
    def proposed_by(self) -> str:
        return self.label or self.kind


@dataclass(frozen=True)
class ArtifactVersion:
    """One stored version of an artifact.

    ``status`` is its place in the lifecycle: a proposal is ``pending``
    until it is approved (``active``) or ``rejected``; an active version
    that another replaces becomes ``superseded``. ``actor_kind`` is who
    wrote it: ``assistant``, ``host`` or ``seed`` (the definition's default,
    stored at startup).
    """

    name: str
    content: str
    version: int
    is_active: bool
    proposed_by: str
    created_at: datetime | None = None
    id: int | None = None
    status: VersionStatus = "superseded"
    actor_kind: str = "host"
    rationale: str | None = None
    decided_by: str | None = None
    decided_at: datetime | None = None
    decision_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "version": self.version,
            "is_active": self.is_active,
            "status": self.status,
            "proposed_by": self.proposed_by,
            "actor_kind": self.actor_kind,
            "rationale": self.rationale,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "decision_reason": self.decision_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass(frozen=True)
class MutationResult:
    """What a write did: the version it concerns and whether it is live."""

    version: ArtifactVersion
    activated: bool
    """The version named here is the active one after the call."""
    live_version: int | None
    """The active version after the call, if any."""
    unchanged: bool = False
    """The content already matched the active version; nothing was written."""
    durable: bool = True
    """Whether the store survives a process restart."""
