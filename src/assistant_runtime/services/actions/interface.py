"""ActionService — the record of actions a host proposed and carried out.

A host with its own confirmation flow (drafts the owner edits, scheduled
sends, Undo, results per recipient) keeps that flow and writes its state
here instead of in a database of its own. The runtime stores and orders the
records; it neither decides nor sends. Status values are checked, the
transitions are the host's, and a change can be made conditional on the
status and text revision the host last saw. Confirmations are the owner's
per-recipient approvals, written before a send: what was confirmed never
changes afterwards.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from assistant_runtime.services.actions._store import (
    ActionStore,
    DatabaseActionStore,
    InMemoryActionStore,
)
from assistant_runtime.services.actions.config import ActionsConfig
from assistant_runtime.services.actions.exceptions import (
    ActionConflictError,
    ActionError,
    ActionNotFoundError,
)
from assistant_runtime.services.actions.models import (
    ACTION_STATUSES,
    RECONCILED,
    ActionRecord,
    ConfirmationRecord,
)

if TYPE_CHECKING:
    from assistant_runtime.services.database.interface import DatabaseService

_ACTION_CHANGES = frozenset(
    {"status", "text", "arguments", "confirmed_by", "decided_at", "results"}
)
_CONFIRMATION_CHANGES = frozenset(
    {"status", "result", "key_epoch", "sender", "audience", "reconciled"}
)
_SIGNING = ("key_epoch", "sender", "audience")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ActionService:
    """Create, change and list action records and confirmations. Implements LifecycleAware."""

    def __init__(
        self, config: ActionsConfig, database_service: DatabaseService | None = None
    ) -> None:
        self._config = config
        self._database = database_service
        self._store: ActionStore | None = None

    async def start(self) -> None:
        database = self._database
        if database is not None and getattr(database, "healthy", False):
            self._store = DatabaseActionStore(database)
        else:
            self._store = InMemoryActionStore()
        logger.info("Action service started", durable=self._store.durable)

    async def stop(self) -> None:
        self._store = None
        logger.info("Action service stopped")

    async def health_check(self) -> dict[str, Any]:
        return {
            "healthy": self._store is not None,
            "durable": self._store.durable if self._store is not None else None,
        }

    # --- actions ---

    async def create(
        self,
        *,
        kind: str,
        by: str,
        text: str | None = None,
        arguments: dict[str, Any] | None = None,
        profile: str | None = None,
        subject: str | None = None,
        status: str = "proposed",
    ) -> ActionRecord:
        _check_status(status)
        now = datetime.now(UTC)
        async with self._require_store().transaction() as tx:
            return await tx.create_action(
                kind=kind,
                text=text,
                arguments=arguments,
                profile=profile,
                subject=subject,
                status=status,
                history=[{"status": status, "at": now.isoformat(), "by": by}],
                results={},
                revision=1,
                revised_at=now,
                created_by=by,
            )

    async def get(self, action_id: int) -> tuple[ActionRecord, list[ConfirmationRecord]]:
        """One action and its confirmations, in insertion order."""
        async with self._require_store().transaction() as tx:
            record = await tx.get_action(action_id)
            if record is None:
                raise ActionNotFoundError(f"No action {action_id}")
            confirmations = await tx.list_confirmations(
                after=0, limit=self._config.max_page, action_id=action_id
            )
        return record, confirmations

    async def list(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        profile: str | None = None,
        subject: str | None = None,
        limit: int = 100,
        before: int | None = None,
    ) -> list[ActionRecord]:
        """Newest first; with ``before`` (an action's id), only the actions older than it."""
        async with self._require_store().transaction() as tx:
            if before is not None and await tx.get_action(before) is None:
                raise ActionNotFoundError(f"No action {before}")
            return await tx.list_actions(
                limit=min(limit, self._config.max_page),
                before=before,
                status=status,
                kind=kind,
                profile=profile,
                subject=subject,
            )

    async def count(self, start: datetime, end: datetime) -> int:
        """Actions created in ``[start, end)``; 0 while the service is not started."""
        if self._store is None:
            return 0
        async with self._store.transaction() as tx:
            return await tx.count_created(start, end)

    async def update(
        self,
        action_id: int,
        changes: dict[str, Any],
        *,
        by: str,
        expected_status: Sequence[str] | None = None,
        expected_revision: int | None = None,
    ) -> ActionRecord:
        """Apply ``changes`` if the action is in one of ``expected_status`` (and at the revision).

        A new status is added to the history; new text raises the revision
        and keeps the first text as ``proposed_text``; results merge per
        recipient.
        """
        unknown = set(changes) - _ACTION_CHANGES
        if unknown:
            raise ActionError(f"Cannot change {', '.join(sorted(unknown))} of an action")
        if "status" in changes:
            _check_status(changes["status"])
        now = datetime.now(UTC)
        async with self._require_store().transaction() as tx:
            current = await tx.get_action(action_id, lock=True)
            if current is None:
                raise ActionNotFoundError(f"No action {action_id}")
            if expected_status is not None and current.status not in expected_status:
                raise ActionConflictError(
                    f"Action {action_id} is {current.status}, not {' or '.join(expected_status)}"
                )
            if expected_revision is not None and current.revision != expected_revision:
                raise ActionConflictError(
                    f"Action {action_id} text is at revision {current.revision}, "
                    f"not {expected_revision}"
                )
            values: dict[str, Any] = {}
            if "status" in changes and changes["status"] != current.status:
                entry = {"status": changes["status"], "at": now.isoformat(), "by": by}
                values.update(status=changes["status"], history=[*current.history, entry])
            if "text" in changes and changes["text"] != current.text:
                values.update(
                    text=changes["text"],
                    revision=current.revision + 1,
                    proposed_text=(
                        current.proposed_text if current.proposed_text is not None else current.text
                    ),
                    revised_at=now,
                )
            if "results" in changes:
                values["results"] = {**current.results, **(changes["results"] or {})}
            for key in ("arguments", "confirmed_by", "decided_at"):
                if key in changes:
                    values[key] = changes[key]
            return await tx.update_action(action_id, **values) if values else current

    # --- confirmations ---

    async def confirm(
        self,
        action_id: int,
        *,
        confirmation_id: str,
        recipient: str,
        kind: str,
        revision: int,
        text_sha256: str,
        source: str,
        confirmed_at: datetime,
        key_epoch: int | None = None,
    ) -> ConfirmationRecord:
        """Record the owner's confirmation of an action for one recipient, before the send."""
        try:
            uuid.UUID(confirmation_id)
        except ValueError:
            raise ActionError("A confirmation id must be a UUID") from None
        if kind not in ("message", "steer"):
            raise ActionError("A confirmation kind is message or steer")
        if source not in ("button", "typed", "voice"):
            raise ActionError("A confirmation source is button, typed or voice")
        if not _SHA256.fullmatch(text_sha256):
            raise ActionError("text_sha256 must be 64 lowercase hexadecimal characters")
        if confirmed_at.tzinfo is None:
            raise ActionError("confirmed_at needs a UTC offset")
        async with self._require_store().transaction() as tx:
            if await tx.get_action(action_id) is None:
                raise ActionNotFoundError(f"No action {action_id}")
            record = await tx.add_confirmation(
                id=confirmation_id,
                action_id=action_id,
                recipient=recipient,
                kind=kind,
                revision=revision,
                text_sha256=text_sha256,
                source=source,
                confirmed_at=confirmed_at,
                key_epoch=key_epoch,
                status="confirmed",
            )
        if record is None:
            raise ActionConflictError(f"Confirmation {confirmation_id} exists already")
        return record

    async def update_confirmation(
        self, confirmation_id: str, changes: dict[str, Any], *, action_id: int | None = None
    ) -> ConfirmationRecord:
        """Sign a confirmation while it is ``confirmed``, settle it once, or note reconciliation.

        What was confirmed (recipient, kind, revision, text hash, source,
        time) never changes, and neither does the status or result once it
        is settled.
        """
        unknown = set(changes) - _CONFIRMATION_CHANGES
        if unknown:
            raise ActionError(f"Cannot change {', '.join(sorted(unknown))} of a confirmation")
        if "status" in changes and changes["status"] not in ("sent", "failed"):
            raise ActionError("A confirmation is settled as sent or failed")
        if changes.get("reconciled") not in (None, *RECONCILED):
            raise ActionError(f"reconciled is one of {', '.join(RECONCILED)}")
        async with self._require_store().transaction() as tx:
            current = await tx.get_confirmation(confirmation_id, lock=True)
            if current is None or action_id not in (None, current.action_id):
                raise ActionNotFoundError(f"No confirmation {confirmation_id}")
            if current.status != "confirmed" and (
                "status" in changes or "result" in changes or any(k in changes for k in _SIGNING)
            ):  # settled once: afterwards only its reconciliation is noted
                raise ActionConflictError(f"Confirmation {confirmation_id} is {current.status}")
            values = {key: changes[key] for key in changes}
            return await tx.update_confirmation(confirmation_id, **values) if values else current

    async def confirmations(
        self,
        *,
        after: int = 0,
        limit: int = 100,
        action_id: int | None = None,
        status: str | None = None,
        sender: str | None = None,
        audience: str | None = None,
        signed: bool | None = None,
        reconciled: bool | None = None,
    ) -> list[ConfirmationRecord]:
        """Confirmations after ``after`` in insertion order, for reconciliation."""
        async with self._require_store().transaction() as tx:
            return await tx.list_confirmations(
                after=after,
                limit=min(limit, self._config.max_page),
                action_id=action_id,
                status=status,
                sender=sender,
                audience=audience,
                signed=signed,
                reconciled=reconciled,
            )

    def _require_store(self) -> ActionStore:
        if self._store is None:
            raise ActionError("Action service not started")
        return self._store


def _check_status(status: str) -> None:
    if status not in ACTION_STATUSES:
        raise ActionError(f"An action status is one of {', '.join(ACTION_STATUSES)}")
