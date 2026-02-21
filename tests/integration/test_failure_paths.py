"""Integration-style failure-path tests for persistence/runtime wiring."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.services.history.interface import HistoryService


class _FailingDatabaseService:
    """Database service test double that fails on any session context usage."""

    @asynccontextmanager
    async def session_context(self):
        raise RuntimeError("database unavailable")
        yield


@pytest.mark.asyncio
async def test_session_store_persist_failure_keeps_in_memory_state():
    """When DB persistence fails, session data remains available in memory."""
    store = SessionStore(database_service=_FailingDatabaseService())
    messages = [ModelRequest(parts=[UserPromptPart(content="hello")])]

    # Should not raise even though persistence path fails.
    await store.save_history_async("session-1", messages)

    history = store.get_history("session-1")
    assert len(history) == 1
    assert isinstance(history[0], ModelRequest)


@pytest.mark.asyncio
async def test_session_store_load_failure_falls_back_to_new_context():
    """When DB read fails, store returns a fresh in-memory context."""
    store = SessionStore(database_service=_FailingDatabaseService())

    missing = await store.get_context_if_exists_async("session-2")
    assert missing is None

    context = await store.get_context_async("session-2")
    assert context["turn_number"] == 0
    assert context["message_history"] == []


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
