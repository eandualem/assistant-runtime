"""Data types of the actions module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

ActionStatus = Literal["proposed", "scheduled", "sending", "sent", "failed", "discarded", "undone"]
ACTION_STATUSES: tuple[str, ...] = (
    "proposed",
    "scheduled",
    "sending",
    "sent",
    "failed",
    "discarded",
    "undone",
)
RECONCILED: tuple[str, ...] = ("matched", "altered", "missing", "undelivered")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass(frozen=True)
class ActionRecord:
    """An action a host proposed, and what became of it.

    ``history`` lists each status with when and by whom it was set.
    ``revision`` counts the text's versions: an edit raises it and keeps the
    first text as ``proposed_text``. ``results`` holds one result per
    recipient.
    """

    id: int
    kind: str
    status: str
    created_by: str
    text: str | None = None
    arguments: dict[str, Any] | None = None
    profile: str | None = None
    subject: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    confirmed_by: str | None = None
    results: dict[str, Any] = field(default_factory=dict)
    decided_at: datetime | None = None
    revision: int = 1
    proposed_text: str | None = None
    revised_at: datetime | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "arguments": self.arguments,
            "profile": self.profile,
            "subject": self.subject,
            "status": self.status,
            "history": self.history,
            "confirmed_by": self.confirmed_by,
            "results": self.results,
            "decided_at": _iso(self.decided_at),
            "revision": self.revision,
            "proposed_text": self.proposed_text,
            "revised_at": _iso(self.revised_at),
            "created_by": self.created_by,
            "created_at": _iso(self.created_at),
        }


@dataclass(frozen=True)
class ConfirmationRecord:
    """The owner's confirmation of one action's text for one recipient.

    Written before the send. What was confirmed never changes; the signing
    fields (``key_epoch``, ``sender``, ``audience``) are set while it is
    still ``confirmed``, ``status`` moves once to ``sent`` or ``failed``
    with its ``result``, and ``reconciled`` records a later check. ``seq``
    is its place in insertion order.
    """

    seq: int
    id: str
    action_id: int
    recipient: str
    kind: str
    revision: int
    text_sha256: str
    source: str
    confirmed_at: datetime
    status: str = "confirmed"
    key_epoch: int | None = None
    sender: str | None = None
    audience: str | None = None
    result: Any = None
    reconciled: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "id": self.id,
            "action_id": self.action_id,
            "recipient": self.recipient,
            "kind": self.kind,
            "revision": self.revision,
            "text_sha256": self.text_sha256,
            "source": self.source,
            "confirmed_at": self.confirmed_at.isoformat(),
            "key_epoch": self.key_epoch,
            "sender": self.sender,
            "audience": self.audience,
            "status": self.status,
            "result": self.result,
            "reconciled": self.reconciled,
        }
