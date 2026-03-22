"""Integration-style failure-path tests for persistence/runtime wiring."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.models import AssistantRequest
from lovely_assistant.services.history.interface import HistoryService


class _FailingDatabaseService:
    """Database service test double that fails on any session context usage."""

    @asynccontextmanager
    async def session_context(self):
        raise RuntimeError("database unavailable")
        yield


@pytest.mark.asyncio
async def test_session_store_persist_failure_keeps_in_memory_state():
    """When DB persistence fails during tree-message writes, the error propagates."""
    store = SessionStore(database_service=_FailingDatabaseService())

    with pytest.raises(RuntimeError, match="database unavailable"):
        await store.register_user_message(
            AssistantRequest(
                id="user-1",
                session_id="session-1",
                parent_id=None,
                content="hello",
            )
        )

    assert store.has_session("session-1") is False


@pytest.mark.asyncio
async def test_session_store_load_failure_propagates_error():
    """When DB read fails, error propagates instead of silently creating blank context."""
    store = SessionStore(database_service=_FailingDatabaseService())

    with pytest.raises(RuntimeError, match="database unavailable"):
        await store.get_context_if_exists_async("session-2")

    with pytest.raises(RuntimeError, match="database unavailable"):
        await store.get_context_async("session-2")


@pytest.mark.asyncio
async def test_history_service_runtime_settings_propagates_to_summarizer():
    """Runtime settings updates propagate to the active summarizer instance."""
    from lovely_assistant.services.history.config import HistoryConfig

    service = HistoryService(config=HistoryConfig(), llm_service=MagicMock())
    await service.start()

    runtime = MagicMock()
    service.set_runtime_settings(runtime)

    assert service._manager is not None
    assert service._manager._summarizer._runtime_settings is runtime
