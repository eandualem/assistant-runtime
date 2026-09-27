"""Action records: in process memory, or the ``actions`` and ``action_confirmations`` tables.

Internal module — only accessed through ActionService (interface.py). Every
read and write happens inside ``transaction()``: one lock in memory, one
database transaction (with row locks where asked) in Postgres.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import fields, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from assistant_runtime.services.actions.models import ActionRecord, ConfirmationRecord

_ACTION_FIELDS = tuple(f.name for f in fields(ActionRecord))
_CONFIRMATION_FIELDS = tuple(f.name for f in fields(ConfirmationRecord))


class ActionTransaction(Protocol):
    async def create_action(self, **values: Any) -> ActionRecord: ...

    async def get_action(self, action_id: int, *, lock: bool = False) -> ActionRecord | None: ...

    async def update_action(self, action_id: int, **values: Any) -> ActionRecord: ...

    async def list_actions(self, *, limit: int, **filters: Any) -> list[ActionRecord]: ...

    async def add_confirmation(self, **values: Any) -> ConfirmationRecord | None: ...

    async def get_confirmation(
        self, confirmation_id: str, *, lock: bool = False
    ) -> ConfirmationRecord | None: ...

    async def update_confirmation(
        self, confirmation_id: str, **values: Any
    ) -> ConfirmationRecord: ...

    async def list_confirmations(
        self,
        *,
        after: int,
        limit: int,
        signed: bool | None = None,
        reconciled: bool | None = None,
        **filters: Any,
    ) -> list[ConfirmationRecord]: ...


class ActionStore(Protocol):
    durable: bool

    def transaction(self) -> contextlib.AbstractAsyncContextManager[ActionTransaction]: ...


def _matches(record: Any, filters: dict[str, Any]) -> bool:
    return all(value is None or getattr(record, name) == value for name, value in filters.items())


class InMemoryActionStore:
    """Actions kept in the process; lost on restart."""

    durable = False

    def __init__(self) -> None:
        self._actions: dict[int, ActionRecord] = {}
        self._confirmations: dict[str, ConfirmationRecord] = {}
        self._next_action = 1
        self._next_seq = 1
        self._lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[InMemoryActionStore]:
        async with self._lock:
            yield self

    async def create_action(self, **values: Any) -> ActionRecord:
        record = ActionRecord(id=self._next_action, created_at=datetime.now(UTC), **values)
        self._actions[record.id] = record
        self._next_action += 1
        return record

    async def get_action(self, action_id: int, *, lock: bool = False) -> ActionRecord | None:
        return self._actions.get(action_id)

    async def update_action(self, action_id: int, **values: Any) -> ActionRecord:
        record = self._actions[action_id] = replace(self._actions[action_id], **values)
        return record

    async def list_actions(self, *, limit: int, **filters: Any) -> list[ActionRecord]:
        records = [r for r in self._actions.values() if _matches(r, filters)]
        return sorted(records, key=lambda r: r.id, reverse=True)[:limit]

    async def add_confirmation(self, **values: Any) -> ConfirmationRecord | None:
        if values["id"] in self._confirmations:
            return None
        record = ConfirmationRecord(seq=self._next_seq, **values)
        self._confirmations[record.id] = record
        self._next_seq += 1
        return record

    async def get_confirmation(
        self, confirmation_id: str, *, lock: bool = False
    ) -> ConfirmationRecord | None:
        return self._confirmations.get(confirmation_id)

    async def update_confirmation(self, confirmation_id: str, **values: Any) -> ConfirmationRecord:
        record = replace(self._confirmations[confirmation_id], **values)
        self._confirmations[confirmation_id] = record
        return record

    async def list_confirmations(
        self,
        *,
        after: int,
        limit: int,
        signed: bool | None = None,
        reconciled: bool | None = None,
        **filters: Any,
    ) -> list[ConfirmationRecord]:
        records = [
            r
            for r in self._confirmations.values()
            if r.seq > after
            and _matches(r, filters)
            and (signed is None or (r.key_epoch is not None) == signed)
            and (reconciled is None or (r.reconciled is not None) == reconciled)
        ]
        return sorted(records, key=lambda r: r.seq)[:limit]


class DatabaseActionStore:
    """Actions in the ``actions`` and ``action_confirmations`` tables."""

    durable = True

    def __init__(self, database_service: Any) -> None:
        self._database = database_service

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[_DatabaseTransaction]:
        from assistant_runtime.services.database.repositories import ActionRepository

        async with self._database.session_context() as session:
            yield _DatabaseTransaction(ActionRepository(session))


class _DatabaseTransaction:
    def __init__(self, repository: Any) -> None:
        self._repository = repository

    @staticmethod
    def _action(row: Any) -> ActionRecord:
        return ActionRecord(**{name: getattr(row, name) for name in _ACTION_FIELDS})

    @staticmethod
    def _confirmation(row: Any) -> ConfirmationRecord:
        return ConfirmationRecord(**{name: getattr(row, name) for name in _CONFIRMATION_FIELDS})

    async def create_action(self, **values: Any) -> ActionRecord:
        return self._action(await self._repository.create(**values))

    async def get_action(self, action_id: int, *, lock: bool = False) -> ActionRecord | None:
        row = await self._repository.get(action_id, lock=lock)
        return self._action(row) if row is not None else None

    async def update_action(self, action_id: int, **values: Any) -> ActionRecord:
        return self._action(await self._repository.update(action_id, **values))

    async def list_actions(self, *, limit: int, **filters: Any) -> list[ActionRecord]:
        return [self._action(row) for row in await self._repository.list(limit=limit, **filters)]

    async def add_confirmation(self, **values: Any) -> ConfirmationRecord | None:
        row = await self._repository.add_confirmation(**values)
        return self._confirmation(row) if row is not None else None

    async def get_confirmation(
        self, confirmation_id: str, *, lock: bool = False
    ) -> ConfirmationRecord | None:
        row = await self._repository.get_confirmation(confirmation_id, lock=lock)
        return self._confirmation(row) if row is not None else None

    async def update_confirmation(self, confirmation_id: str, **values: Any) -> ConfirmationRecord:
        return self._confirmation(
            await self._repository.update_confirmation(confirmation_id, **values)
        )

    async def list_confirmations(
        self,
        *,
        after: int,
        limit: int,
        signed: bool | None = None,
        reconciled: bool | None = None,
        **filters: Any,
    ) -> list[ConfirmationRecord]:
        rows = await self._repository.list_confirmations(
            after=after, limit=limit, signed=signed, reconciled=reconciled, **filters
        )
        return [self._confirmation(row) for row in rows]
