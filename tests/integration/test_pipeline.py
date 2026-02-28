"""Integration tests for the assistant pipeline — request → response with real wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lovely_assistant.app.assistant.models import AssistantRequest

from .conftest import _make_mock_agent


class TestAssistantPipeline:
    @pytest.mark.asyncio
    async def test_request_flows_to_result(self, wired_services):
        """A request flows through the full pipeline and returns a result."""
        assistant = wired_services["assistant_service"]
        mock_agent = _make_mock_agent("Hello from the assistant!")

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent):
            result = await assistant.process_message(
                AssistantRequest(session_id="integ-1", message="Hi")
            )

        assert result.content == "Hello from the assistant!"
        assert result.session_id == "integ-1"
        assert result.turn_number == 1

    @pytest.mark.asyncio
    async def test_session_persists_across_turns(self, wired_services):
        """Session context persists — turn counter increments, history accumulates."""
        assistant = wired_services["assistant_service"]

        mock_agent1 = _make_mock_agent("Turn 1")
        mock_agent1.run.return_value.all_messages.return_value = [MagicMock()]
        mock_agent2 = _make_mock_agent("Turn 2")
        mock_agent2.run.return_value.all_messages.return_value = [
            MagicMock(),
            MagicMock(),
        ]

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent1):
            result1 = await assistant.process_message(
                AssistantRequest(session_id="integ-persist", message="First")
            )

        assert result1.turn_number == 1

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent2):
            result2 = await assistant.process_message(
                AssistantRequest(session_id="integ-persist", message="Second")
            )

        assert result2.turn_number == 2

        # History was saved and re-loaded for turn 2
        sessions = assistant._sessions
        history = sessions.get_history("integ-persist")
        assert len(history) == 2  # From turn 2's all_messages()

    @pytest.mark.asyncio
    async def test_tools_passed_to_agent(self, wired_services):
        """Real ToolService provides toolsets to build_agent."""
        assistant = wired_services["assistant_service"]
        mock_agent = _make_mock_agent("OK")

        with patch.object(
            wired_services["llm_service"], "build_agent", return_value=mock_agent
        ) as build_mock:
            await assistant.process_message(
                AssistantRequest(session_id="integ-tools", message="test")
            )

        # build_agent was called with toolsets from the real ToolService
        call_kwargs = build_mock.call_args[1]
        assert "toolsets" in call_kwargs
        assert "system_prompt" in call_kwargs
        assert "Jarvis" in call_kwargs["system_prompt"]

    @pytest.mark.asyncio
    async def test_machine_state_in_prompt(self, wired_services):
        """Machine state is included in the system prompt."""
        assistant = wired_services["assistant_service"]
        mock_agent = _make_mock_agent("OK")

        with patch.object(
            wired_services["llm_service"], "build_agent", return_value=mock_agent
        ) as build_mock:
            await assistant.process_message(
                AssistantRequest(
                    session_id="integ-state",
                    message="test",
                    machine_state={
                        "active_page": {
                            "name": "agents",
                            "data": {"sessions": [{"name": "leo", "state": "idle"}]},
                        },
                    },
                )
            )

        prompt = build_mock.call_args[1]["system_prompt"]
        assert "agents page" in prompt
