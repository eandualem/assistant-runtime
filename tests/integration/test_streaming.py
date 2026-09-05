"""Integration tests for the streaming pipeline — real wiring, mocked Agent.run_stream_events()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.run import AgentRunResultEvent

from assistant_runtime.app.assistant.models import AssistantRequest

from .conftest import _make_mock_agent_result, _make_mock_agent_run


def _request(
    *, message_id: str, session_id: str, parent_id: str | None, content: str
) -> AssistantRequest:
    return AssistantRequest(
        id=message_id,
        session_id=session_id,
        parent_id=parent_id,
        content=content,
    )


class TestStreamingPipeline:
    @pytest.mark.asyncio
    async def test_stream_yields_correct_event_order(self, wired_services):
        """Streaming yields started → final_response → completed in correct order."""
        streaming = wired_services["streaming_service"]
        mock_run = _make_mock_agent_run("Streamed result")

        mock_agent = MagicMock()
        mock_agent.run_stream_events = MagicMock(return_value=mock_run)

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent):
            events = []
            async for event in streaming.stream_message(
                _request(
                    message_id="user-1",
                    session_id="stream-1",
                    parent_id=None,
                    content="Hi",
                )
            ):
                events.append(event)

        # Filter to protocol events (not debug_* events)
        protocol_events = [e for e in events if not e["type"].startswith("debug_")]
        types = [e["type"] for e in protocol_events]
        assert types[0] == "agent_status"
        assert protocol_events[0]["status"] == "started"
        assert "final_response" in types
        assert types[-1] == "agent_status"
        assert protocol_events[-1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_stream_final_response_content(self, wired_services):
        """Final response event contains the agent output."""
        streaming = wired_services["streaming_service"]
        mock_run = _make_mock_agent_run("The answer is 42")

        mock_agent = MagicMock()
        mock_agent.run_stream_events = MagicMock(return_value=mock_run)

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent):
            events = []
            async for event in streaming.stream_message(
                _request(
                    message_id="user-1",
                    session_id="stream-2",
                    parent_id=None,
                    content="What?",
                )
            ):
                events.append(event)

        final = [e for e in events if e["type"] == "final_response"]
        assert len(final) == 1
        assert final[0]["content"] == "The answer is 42"
        assert final[0]["streamed"] is False
        assert final[0]["message_id"]

    @pytest.mark.asyncio
    async def test_stream_saves_history(self, wired_services):
        """Streaming saves message history to session store."""
        streaming = wired_services["streaming_service"]
        assistant = wired_services["assistant_service"]

        mock_result = _make_mock_agent_result("Done")
        mock_result.all_messages.return_value = [MagicMock(), MagicMock()]

        class _MockRun:
            def __init__(self):
                self.result = mock_result
                self._done = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._done:
                    raise StopAsyncIteration
                self._done = True
                return AgentRunResultEvent(self.result)

        mock_agent = MagicMock()
        mock_agent.run_stream_events = MagicMock(return_value=_MockRun())

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent):
            async for _ in streaming.stream_message(
                _request(
                    message_id="user-1",
                    session_id="stream-hist",
                    parent_id=None,
                    content="Save me",
                )
            ):
                pass

        path = await assistant._sessions.get_message_path("stream-hist")
        assert len(path) == 2
        assert path[0]["id"] == "user-1"
        assert path[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_coordinator_dedup_in_real_flow(self, wired_services):
        """EventCoordinator prevents duplicate started/completed events."""
        streaming = wired_services["streaming_service"]
        mock_run = _make_mock_agent_run("OK")

        mock_agent = MagicMock()
        mock_agent.run_stream_events = MagicMock(return_value=mock_run)

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent):
            events = []
            async for event in streaming.stream_message(
                _request(
                    message_id="user-1",
                    session_id="stream-dedup",
                    parent_id=None,
                    content="test",
                )
            ):
                events.append(event)

        # Count protocol events — should be exactly one of each
        started = [e for e in events if e.get("status") == "started"]
        completed = [e for e in events if e.get("status") == "completed"]
        final = [e for e in events if e["type"] == "final_response"]

        assert len(started) == 1
        assert len(completed) == 1
        assert len(final) == 1

    @pytest.mark.asyncio
    async def test_stream_increments_turn(self, wired_services):
        """Streaming increments the session turn counter."""
        streaming = wired_services["streaming_service"]
        assistant = wired_services["assistant_service"]
        mock_run = _make_mock_agent_run("OK")

        mock_agent = MagicMock()
        mock_agent.run_stream_events = MagicMock(return_value=mock_run)

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent):
            async for _ in streaming.stream_message(
                _request(
                    message_id="user-1",
                    session_id="stream-turn",
                    parent_id=None,
                    content="test",
                )
            ):
                pass

        ctx = assistant._sessions.get_context("stream-turn")
        assert ctx["turn_number"] == 1
