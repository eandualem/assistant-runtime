"""Data types of the event log module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

DIRECTIONS: tuple[str, ...] = ("inbound", "outbound")
SEVERITIES: tuple[str, ...] = ("info", "warning", "critical")
# An outbound notice only moves forward: pending, delivered (shown), heard.
NOTICE_ORDER: dict[str, int] = {"pending": 0, "delivered": 1, "heard": 2}


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass(frozen=True)
class EventRecord:
    """One event, in arrival order (``id``).

    ``(source, event_id)`` is the idempotency key. An inbound event is
    ``received``, and ``delivered`` once steered into its target session
    (``delivery`` says how); an outbound notice is ``pending``, then
    ``delivered`` and ``heard``. ``history`` marks events imported from
    before, which are never steered.
    """

    id: int
    event_id: str
    direction: str
    source: str
    kind: str
    severity: str
    status: str
    agent: str | None = None
    summary: str = ""
    payload: dict[str, Any] | None = None
    target_session_id: str | None = None
    history: bool = False
    delivery: dict[str, Any] | None = None
    occurred_at: datetime | None = None
    created_at: datetime | None = None
    delivered_at: datetime | None = None
    heard_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "direction": self.direction,
            "source": self.source,
            "agent": self.agent,
            "kind": self.kind,
            "severity": self.severity,
            "summary": self.summary,
            "payload": self.payload,
            "target_session_id": self.target_session_id,
            "history": self.history,
            "status": self.status,
            "delivery": self.delivery,
            "occurred_at": _iso(self.occurred_at),
            "created_at": _iso(self.created_at),
            "delivered_at": _iso(self.delivered_at),
            "heard_at": _iso(self.heard_at),
        }
