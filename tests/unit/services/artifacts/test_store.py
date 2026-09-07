"""The in-memory store's version semantics and the database store's mapping."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from assistant_runtime.services.artifacts._store import (
    DatabaseArtifactStore,
    InMemoryArtifactStore,
)


class TestInMemoryArtifactStore:
    async def test_versions_increment_per_scope_and_name(self):
        store = InMemoryArtifactStore()
        a1 = await store.propose("s", "a", "one", "host")
        a2 = await store.propose("s", "a", "two", "host")
        b1 = await store.propose("s", "b", "one", "host")
        other = await store.propose("t", "a", "one", "host")
        assert (a1.version, a2.version, b1.version, other.version) == (1, 2, 1, 1)
        assert len({a1.id, a2.id, b1.id, other.id}) == 4
        assert a1.is_active is False
        assert a1.created_at is not None
        assert a1.created_at.tzinfo is not None

    async def test_activate_switches_the_single_active_version(self):
        store = InMemoryArtifactStore()
        await store.propose("s", "a", "one", "host")
        await store.propose("s", "a", "two", "host")
        assert (await store.activate("s", "a", 1)).is_active
        assert (await store.activate("s", "a", 2)).version == 2
        assert (await store.get_active("s", "a")).content == "two"
        assert [v.is_active for v in await store.get_history("s", "a", 10)] == [True, False]
        assert await store.activate("s", "a", 3) is None

    async def test_get_all_active_is_scoped_and_sorted(self):
        store = InMemoryArtifactStore()
        for name in ("b", "a"):
            await store.propose("s", name, "x", "host")
            await store.activate("s", name, 1)
        await store.propose("t", "c", "x", "host")
        await store.activate("t", "c", 1)
        assert [v.name for v in await store.get_all_active("s")] == ["a", "b"]

    async def test_delete_removes_every_version(self):
        store = InMemoryArtifactStore()
        await store.propose("s", "a", "one", "host")
        await store.propose("s", "a", "two", "host")
        assert await store.delete("s", "a") == 2
        assert await store.delete("s", "a") == 0
        assert await store.get_history("s", "a", 10) == []


def _row(**overrides):
    base = {
        "id": 7,
        "name": "persona",
        "content": "text",
        "version": 2,
        "is_active": True,
        "proposed_by": "host",
        "created_at": datetime(2026, 9, 5, tzinfo=UTC),
    }
    return SimpleNamespace(**{**base, **overrides})


class TestDatabaseArtifactStore:
    def _store(self, repo):
        @asynccontextmanager
        async def session_context():
            yield object()

        store = DatabaseArtifactStore(SimpleNamespace(session_context=session_context))
        patcher = patch(
            "assistant_runtime.services.database.repositories.ArtifactRepository",
            return_value=repo,
        )
        return store, patcher

    async def test_maps_rows_and_passes_the_scope(self):
        repo = SimpleNamespace(
            get_active=AsyncMock(return_value=_row()),
            get_all_active=AsyncMock(return_value=[_row(name="a"), _row(name="b")]),
            get_history=AsyncMock(return_value=[_row(version=2), _row(version=1)]),
            propose=AsyncMock(return_value=_row(is_active=False, version=3)),
            approve=AsyncMock(return_value=None),
            delete_by_name=AsyncMock(return_value=3),
        )
        store, patcher = self._store(repo)
        with patcher:
            assert store.durable is True
            active = await store.get_active("shop", "persona")
            assert active.id == 7
            assert active.is_active is True
            assert active.version == 2
            repo.get_active.assert_awaited_once_with("shop", "persona")
            assert [v.name for v in await store.get_all_active("shop")] == ["a", "b"]
            assert [v.version for v in await store.get_history("shop", "persona", 5)] == [2, 1]
            repo.get_history.assert_awaited_once_with("shop", "persona", limit=5)
            proposed = await store.propose("shop", "persona", "new", "host")
            assert proposed.version == 3
            assert proposed.is_active is False
            repo.propose.assert_awaited_once_with("shop", "persona", "new", "host")
            assert await store.activate("shop", "persona", 9) is None
            assert await store.delete("shop", "persona") == 3
