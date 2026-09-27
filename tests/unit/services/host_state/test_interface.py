"""HostStateService: versioned values, conditional writes, bounds."""

from __future__ import annotations

import pytest

from assistant_runtime.services.host_state.config import HostStateConfig
from assistant_runtime.services.host_state.exceptions import (
    HostStateConflictError,
    HostStateError,
    HostStateNotFoundError,
)
from assistant_runtime.services.host_state.interface import HostStateService


async def _service(**config) -> HostStateService:
    service = HostStateService(HostStateConfig(**config))
    await service.start()
    return service


async def test_each_write_raises_the_version_and_a_stale_one_is_refused():
    state = await _service()
    first = await state.put("cursors", "hub", {"after": 10}, by="local", expected_version=0)
    second = await state.put("cursors", "hub", {"after": 12}, by="local", expected_version=1)
    assert (first.version, second.version, second.value) == (1, 2, {"after": 12})
    with pytest.raises(HostStateConflictError, match="version 2, not 1") as stale:
        await state.put("cursors", "hub", {"after": 11}, by="local", expected_version=1)
    assert stale.value.current_version == 2
    with pytest.raises(HostStateConflictError, match="version 2, not 0"):
        await state.put("cursors", "hub", {}, by="local", expected_version=0)
    unconditional = await state.put("cursors", "hub", {"after": 20}, by="local")
    assert unconditional.version == 3


async def test_values_are_listed_by_key_and_deleted_on_the_expected_version():
    state = await _service()
    await state.put("settings", "voice", "cove", by="local")
    await state.put("settings", "budget", 5, by="local")
    await state.put("other", "x", True, by="local")
    assert [e.key for e in await state.list("settings")] == ["budget", "voice"]
    with pytest.raises(HostStateConflictError):
        await state.delete("settings", "voice", expected_version=7)
    await state.delete("settings", "voice", expected_version=1)
    with pytest.raises(HostStateNotFoundError):
        await state.get("settings", "voice")
    with pytest.raises(HostStateNotFoundError):
        await state.delete("settings", "voice")


async def test_names_and_sizes_are_bounded():
    state = await _service(max_value_bytes=1024)
    with pytest.raises(HostStateError, match="namespace"):
        await state.put("Bad Namespace", "k", 1, by="local")
    with pytest.raises(HostStateError, match="key"):
        await state.put("ns", "a/b", 1, by="local")
    with pytest.raises(HostStateError, match="at most 1024"):
        await state.put("ns", "big", "x" * 2000, by="local")
