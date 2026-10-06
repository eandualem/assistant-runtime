"""Change-cursor query parameters (see ``base/cursors.py``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from assistant_runtime.base.cursors import ChangeCursor, parse_cursor


def cursor_param(
    value: str | None, name: str = "updated_after", *, id_type: Callable[[str], Any] = str
) -> ChangeCursor | None:
    """The parsed cursor, None when absent; ``422`` when it is malformed."""
    if value is None:
        return None
    try:
        return parse_cursor(value, id_type=id_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid '{name}' cursor: {e}") from e


def exclusive(updated_after: str | None, **others: Any) -> None:
    """``422`` when ``updated_after`` is combined with another way of paging."""
    given = [name for name, value in others.items() if value is not None]
    if updated_after is not None and given:
        raise HTTPException(
            status_code=422, detail=f"'updated_after' cannot be combined with '{given[0]}'"
        )
