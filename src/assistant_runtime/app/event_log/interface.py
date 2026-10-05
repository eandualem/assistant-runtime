"""EventLogService — events from other systems and notices for the owner, in order.

A host that watches other systems posts what happened here instead of keeping
its own event table. Every event is stored once per ``(source, event_id)`` and
numbered in arrival order, so a reader continues from the last id it saw. An
inbound event is only stored, unless it names a target session: then it is
steered into that session through ingress, queued behind a running turn, or
waiting for the session's next turn while a voice call holds it. Outbound
events are the notices a host shows the owner; the host moves each forward
from ``pending`` to ``delivered`` to ``heard``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from assistant_runtime.app.event_log._store import (
    DatabaseEventStore,
    EventStore,
    InMemoryEventStore,
)
from assistant_runtime.app.event_log.config import EventLogConfig
from assistant_runtime.app.event_log.exceptions import (
    EventConflictError,
    EventLogError,
    EventNotFoundError,
)
from assistant_runtime.app.event_log.models import (
    DIRECTIONS,
    NOTICE_ORDER,
    SEVERITIES,
    EventRecord,
)

if TYPE_CHECKING:
    from assistant_runtime.app.ingress.interface import IngressService
    from assistant_runtime.services.database.interface import DatabaseService

# The inbox orders waiting messages by its own severity names.
_INBOX_SEVERITY = {"info": "info", "warning": "action_needed", "critical": "urgent"}


class EventLogService:
    """Record, deliver and list events. Implements LifecycleAware."""

    def __init__(
        self,
        config: EventLogConfig,
        ingress_service: IngressService | None = None,
        database_service: DatabaseService | None = None,
    ) -> None:
        self._config = config
        self._ingress = ingress_service
        self._database = database_service
        self._store: EventStore | None = None

    async def start(self) -> None:
        database = self._database
        if database is not None and getattr(database, "healthy", False):
            self._store = DatabaseEventStore(database)
        else:
            self._store = InMemoryEventStore()
        logger.info("Event log started", durable=self._store.durable)

    async def stop(self) -> None:
        self._store = None
        logger.info("Event log stopped")

    async def health_check(self) -> dict[str, Any]:
        return {
            "healthy": self._store is not None,
            "durable": self._store.durable if self._store is not None else None,
        }

    async def record(
        self,
        *,
        event_id: str,
        direction: str,
        source: str,
        kind: str,
        severity: str = "info",
        agent: str | None = None,
        summary: str = "",
        payload: dict[str, Any] | None = None,
        target_session_id: str | None = None,
        history: bool = False,
        occurred_at: datetime | None = None,
    ) -> tuple[EventRecord, bool]:
        """Store an event once; returns it and whether it is new.

        A new inbound event with a target, not imported history, is steered
        into the target session. A repeated ``(source, event_id)`` returns the
        stored event; it is delivered then only if no request ever started
        delivering it (the process stopped after storing it), never again.
        """
        if direction not in DIRECTIONS:
            raise EventLogError("direction is inbound or outbound")
        if severity not in SEVERITIES:
            raise EventLogError(f"severity is one of {', '.join(SEVERITIES)}")
        if direction == "outbound" and target_session_id is not None:
            raise EventLogError("An outbound notice has no target session")
        async with self._require_store().transaction() as tx:
            record, created = await tx.create_if_new(
                event_id=event_id,
                direction=direction,
                source=source,
                kind=kind,
                severity=severity,
                agent=agent,
                summary=summary,
                payload=payload,
                target_session_id=target_session_id,
                history=history,
                occurred_at=occurred_at,
                status="received" if direction == "inbound" else "pending",
            )
        if record.target_session_id is not None and not record.history and record.delivery is None:
            record = await self._deliver(record)
        return record, created

    async def get(self, event_id: int) -> EventRecord:
        async with self._require_store().transaction() as tx:
            record = await tx.get(event_id)
        if record is None:
            raise EventNotFoundError(f"No event {event_id}")
        return record

    async def list(
        self,
        *,
        after: int = 0,
        limit: int = 100,
        direction: str | None = None,
        source: str | None = None,
        agent: str | None = None,
        kind: str | None = None,
        news_only: bool = False,
    ) -> list[EventRecord]:
        """Events after ``after`` in arrival order; ``news_only`` leaves out imported history."""
        async with self._require_store().transaction() as tx:
            return await tx.list(
                after=after,
                limit=min(limit, self._config.max_page),
                direction=direction,
                source=source,
                agent=agent,
                kind=kind,
                history=False if news_only else None,
            )

    async def count(self, start: datetime, end: datetime) -> int:
        """Events stored in ``[start, end)``; 0 while the service is not started."""
        if self._store is None:
            return 0
        async with self._store.transaction() as tx:
            return await tx.count_created(start, end)

    async def update_notice(self, event_id: int, status: str) -> EventRecord:
        """Move an outbound notice forward: ``pending`` → ``delivered`` → ``heard``."""
        if status not in ("delivered", "heard"):
            raise EventLogError("A notice moves to delivered or heard")
        now = datetime.now(UTC)
        async with self._require_store().transaction() as tx:
            current = await tx.get(event_id, lock=True)
            if current is None:
                raise EventNotFoundError(f"No event {event_id}")
            if current.direction != "outbound":
                raise EventConflictError(f"Event {event_id} is inbound, not a notice")
            if NOTICE_ORDER[status] < NOTICE_ORDER[current.status]:
                raise EventConflictError(f"Notice {event_id} is {current.status} already")
            if status == current.status:
                return current
            values: dict[str, Any] = {"status": status}
            if current.delivered_at is None:
                values["delivered_at"] = now  # heard implies shown
            if status == "heard":
                values["heard_at"] = now
            return await tx.update(event_id, **values)

    async def _deliver(self, record: EventRecord) -> EventRecord:
        """Steer an inbound event into its target session and note how it went.

        The start is recorded first (``delivery`` holds only the session), so
        concurrent requests never deliver an event twice; one left like that
        by a stopped process has an unknown outcome and is not retried.
        """
        target = record.target_session_id
        async with self._require_store().transaction() as tx:
            current = await tx.get(record.id, lock=True)
            if current is None or current.delivery is not None:
                return current or record  # another request has started it
            await tx.update(record.id, delivery={"session_id": target})
        if self._ingress is None:
            delivery: dict[str, Any] = {"session_id": target, "error": "No ingress service"}
        else:
            text = f"{record.kind} ({record.severity})"
            if record.summary:
                text += f": {record.summary}"
            try:
                outcome = await self._ingress.deliver(
                    from_agent=record.agent or record.source,
                    via=record.source,
                    message=text,
                    session_id=target,
                    severity=_INBOX_SEVERITY[record.severity],
                )
            except Exception as e:
                logger.warning("Event delivery failed", event=record.id, error=str(e))
                delivery = {"session_id": target, "error": str(e) or type(e).__name__}
            else:
                how = outcome.get("delivery") if outcome.get("status") == "delivered" else "inbox"
                delivery = {"session_id": target, "how": how}
        values: dict[str, Any] = {"delivery": delivery}
        if "how" in delivery:
            values.update(status="delivered", delivered_at=datetime.now(UTC))
        async with self._require_store().transaction() as tx:
            return await tx.update(record.id, **values)

    def _require_store(self) -> EventStore:
        if self._store is None:
            raise EventLogError("Event log not started")
        return self._store
