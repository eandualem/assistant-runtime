"""Data types of the artifact service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

ActorKind = Literal["assistant", "host"]


@dataclass(frozen=True)
class Actor:
    """Who is asking for a mutation. Permissions depend on ``kind`` only.

    ``label`` is recorded as ``proposed_by`` on the written version; it is
    informational and never widens what the actor may do.
    """

    kind: ActorKind
    label: str = ""

    @property
    def proposed_by(self) -> str:
        return self.label or self.kind


@dataclass(frozen=True)
class ArtifactVersion:
    """One stored version of an artifact."""

    name: str
    content: str
    version: int
    is_active: bool
    proposed_by: str
    created_at: datetime | None = None
    id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "version": self.version,
            "is_active": self.is_active,
            "proposed_by": self.proposed_by,
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
