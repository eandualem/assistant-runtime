"""Change cursors: ``<ISO time>|<id>`` or a bare time, ordered by ``(updated_at, id)``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from assistant_runtime.base.cursors import ChangeCursor, format_cursor, parse_cursor

NOON = datetime(2026, 10, 1, 12, tzinfo=UTC)


def test_a_bare_time_admits_strictly_later_changes():
    cursor = parse_cursor("2026-10-01T12:00:00+00:00")
    assert cursor == ChangeCursor(NOON)
    assert not cursor.admits(NOON, "zzz")
    assert cursor.admits(NOON + timedelta(microseconds=1), "a")


def test_a_time_with_an_id_continues_after_that_record():
    cursor = parse_cursor(format_cursor(NOON, "m2"))
    assert cursor == ChangeCursor(NOON, "m2")
    assert cursor.admits(NOON, "m3")
    assert not cursor.admits(NOON, "m2")
    assert not cursor.admits(NOON, "m1")
    assert parse_cursor("2026-10-01T12:00:00Z|7", id_type=int) == ChangeCursor(NOON, 7)


def test_a_naive_time_is_utc_and_an_unencoded_plus_is_read():
    assert parse_cursor("2026-10-01T12:00").updated_at == NOON
    # ``+02:00`` sent unencoded in a query string arrives as `` 02:00``.
    assert parse_cursor("2026-10-01T14:00:00 02:00|m1") == ChangeCursor(NOON, "m1")
    assert ChangeCursor(NOON).admits(datetime(2026, 10, 1, 13), "a")  # naive record time: UTC


@pytest.mark.parametrize("value", ["yesterday", "2026-10-01T12:00|", "|m1", "2026-13-01T00:00"])
def test_a_malformed_cursor_is_refused(value):
    with pytest.raises(ValueError, match="ISO 8601|id is empty"):
        parse_cursor(value)


def test_an_id_of_the_wrong_type_is_refused():
    with pytest.raises(ValueError, match="record id"):
        parse_cursor("2026-10-01T12:00|abc", id_type=int)
