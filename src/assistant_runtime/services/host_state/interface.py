"""HostStateService — small versioned values a host keeps by namespace and key.

A host that needs a little state of its own (the cursor it last read an
event source at, application settings, a daily budget) keeps it here
instead of in a database of its own. Each value has a version that rises
with every write; a write or delete can require the version the host last
read (``0`` for a key that must not exist yet), so two writers never
silently overwrite each other. A deleted key keeps counting, so a version
is never reused. The runtime never reads the values.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from assistant_runtime.services.host_state._store import (
    DatabaseHostStateStore,
    HostStateStore,
    InMemoryHostStateStore,
)
from assistant_runtime.services.host_state.config import HostStateConfig
from assistant_runtime.services.host_state.exceptions import (
    HostStateConflictError,
    HostStateError,
    HostStateNotFoundError,
)
from assistant_runtime.services.host_state.models import HostStateEntry

if TYPE_CHECKING:
    from assistant_runtime.services.database.interface import DatabaseService

NAMESPACE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
KEY = re.compile(r"^[A-Za-z0-9_.:@-]{1,200}$")


class HostStateService:
    """Read, write and delete host values. Implements LifecycleAware."""

    def __init__(
        self, config: HostStateConfig, database_service: DatabaseService | None = None
    ) -> None:
        self._config = config
        self._database = database_service
        self._store: HostStateStore | None = None

    async def start(self) -> None:
        database = self._database
        if database is not None and getattr(database, "healthy", False):
            self._store = DatabaseHostStateStore(database)
        else:
            self._store = InMemoryHostStateStore()
        logger.info("Host state started", durable=self._store.durable)

    async def stop(self) -> None:
        self._store = None
        logger.info("Host state stopped")

    async def health_check(self) -> dict[str, Any]:
        return {
            "healthy": self._store is not None,
            "durable": self._store.durable if self._store is not None else None,
        }

    async def get(self, namespace: str, key: str) -> HostStateEntry:
        _check(namespace, key)
        entry = await self._require_store().get(namespace, key)
        if entry is None:
            raise HostStateNotFoundError(f"Nothing is stored at {namespace}/{key}")
        return entry

    async def list(
        self, namespace: str, *, after: str | None = None, limit: int | None = None
    ) -> tuple[list[HostStateEntry], str | None]:
        """Values in a namespace by key, after ``after``; and the key to continue from, if any."""
        _check(namespace)
        limit = min(limit or self._config.max_page, self._config.max_page)
        entries = await self._require_store().list(namespace, after, limit + 1)
        more = len(entries) > limit
        entries = entries[:limit]
        return entries, entries[-1].key if more else None

    async def put(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        by: str,
        expected_version: int | None = None,
    ) -> HostStateEntry:
        """Store ``value``; with ``expected_version``, only over that version (0: none yet)."""
        _check(namespace, key)
        size = len(json.dumps(value, separators=(",", ":")).encode())
        if size > self._config.max_value_bytes:
            raise HostStateError(
                f"The value is {size} bytes; at most {self._config.max_value_bytes} are kept"
            )
        store = self._require_store()
        entry = await store.put(namespace, key, value, by, expected_version)
        if entry is None:
            raise await self._conflict(namespace, key, expected_version)
        return entry

    async def delete(
        self, namespace: str, key: str, *, by: str, expected_version: int | None = None
    ) -> None:
        _check(namespace, key)
        if not await self._require_store().delete(namespace, key, by, expected_version):
            if expected_version is None:
                raise HostStateNotFoundError(f"Nothing is stored at {namespace}/{key}")
            raise await self._conflict(namespace, key, expected_version)

    async def _conflict(
        self, namespace: str, key: str, expected: int | None
    ) -> HostStateConflictError:
        current = await self._require_store().get(namespace, key)
        version = current.version if current is not None else 0
        return HostStateConflictError(
            f"{namespace}/{key} is at version {version}, not {expected}", current_version=version
        )

    def _require_store(self) -> HostStateStore:
        if self._store is None:
            raise HostStateError("Host state not started")
        return self._store


def _check(namespace: str, key: str | None = None) -> None:
    if not NAMESPACE.match(namespace):
        raise HostStateError(
            "A namespace is lowercase letters, digits, '_', '.' or '-', up to 64 characters"
        )
    if key is not None and not KEY.match(key):
        raise HostStateError("A key is 1 to 200 letters, digits or _ . : @ -")
