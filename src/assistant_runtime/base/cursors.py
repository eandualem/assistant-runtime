"""Change cursors: positions in a ``(updated_at, id)`` ascending order of records.

A cursor is ``<ISO 8601 updated_at>|<id>``, or a bare ISO time meaning
everything strictly after that time. A time without a timezone is UTC. A
reader asks for the records after a cursor and continues from the last one
it received, so a record that changed again reappears.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# ``+`` in an unencoded query string arrives as a space: ``...T12:00:00 00:00``.
_SPACED_OFFSET = re.compile(r" (\d{2}(?::?\d{2})?)$")


@dataclass(frozen=True)
class ChangeCursor:
    """``updated_at`` (timezone-aware) and, unless the cursor was a bare time, an ``id``."""

    updated_at: datetime
    id: Any = None

    def admits(self, updated_at: datetime | None, record_id: Any) -> bool:
        """Whether a record with this ``(updated_at, id)`` comes after the cursor."""
        if updated_at is None:
            return False
        updated_at = aware(updated_at)
        if self.id is None:
            return updated_at > self.updated_at
        return (updated_at, record_id) > (self.updated_at, self.id)


def aware(value: datetime) -> datetime:
    """A timezone-less value is taken as UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def parse_cursor(value: str, *, id_type: Callable[[str], Any] = str) -> ChangeCursor:
    """Read a cursor; ``ValueError`` when it is malformed (or its id is not ``id_type``)."""
    time_part, separator, id_part = value.strip().partition("|")
    if separator and not id_part:
        raise ValueError("A cursor's id is empty")
    try:
        updated_at = datetime.fromisoformat(_SPACED_OFFSET.sub(r"+\1", time_part))
    except ValueError:
        raise ValueError(f"Not an ISO 8601 time: {time_part!r}") from None
    record_id = None
    if separator:
        try:
            record_id = id_type(id_part)
        except (TypeError, ValueError):
            raise ValueError(f"Not a record id: {id_part!r}") from None
    return ChangeCursor(aware(updated_at), record_id)


def format_cursor(updated_at: datetime, record_id: Any) -> str:
    """The cursor that continues after the record with this ``(updated_at, id)``."""
    return f"{aware(updated_at).isoformat()}|{record_id}"


def next_cursor(records: list[Any], after: str | None) -> str | None:
    """The cursor after the last record (objects with ``updated_at`` and ``id``), else ``after``."""
    if not records:
        return after
    last = records[-1]
    return format_cursor(last.updated_at, last.id)
