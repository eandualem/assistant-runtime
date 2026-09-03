"""Integration tests for the assistant pipeline — request → response with real wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lovely_assistant.app.assistant.models import AssistantRequest

from .conftest import _make_mock_agent


def _request(
    *, message_id: str, session_id: str, parent_id: str | None, content: str
) -> AssistantRequest:
    return AssistantRequest(
        id=message_id,
        session_id=session_id,
        parent_id=parent_id,
        content=content,
    )


class TestAssistantPipeline:
    @pytest.mark.asyncio
    async def test_request_flows_to_result(self, wired_services):
        """A request flows through the full pipeline and returns a result."""
        assistant = wired_services["assistant_service"]
        mock_agent = _make_mock_agent("Hello from the assistant!")

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent):
            result = await assistant.process_message(
                _request(
                    message_id="user-1",
                    session_id="integ-1",
                    parent_id=None,
                    content="Hi",
                )
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
                _request(
                    message_id="user-1",
                    session_id="integ-persist",
                    parent_id=None,
                    content="First",
                )
            )

        assert result1.turn_number == 1
        parent_id = (await assistant._sessions.get_message_path("integ-persist"))[-1]["id"]

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent2):
            result2 = await assistant.process_message(
                _request(
                    message_id="user-2",
                    session_id="integ-persist",
                    parent_id=parent_id,
                    content="Second",
                )
            )

        assert result2.turn_number == 2

        # History was saved and re-loaded for turn 2
        sessions = assistant._sessions
        path = await sessions.get_message_path("integ-persist")
        assert len(path) == 4

    @pytest.mark.asyncio
    async def test_tools_passed_to_agent(self, wired_services):
        """Real ToolService provides toolsets to build_agent."""
        assistant = wired_services["assistant_service"]
        mock_agent = _make_mock_agent("OK")

        with patch.object(
            wired_services["llm_service"], "build_agent", return_value=mock_agent
        ) as build_mock:
            await assistant.process_message(
                _request(
                    message_id="user-1",
                    session_id="integ-tools",
                    parent_id=None,
                    content="test",
                )
            )

        # prepare_agent_context wires toolsets into the assistant agent build call.
        assistant_call = next(
            kwargs for _, kwargs in build_mock.call_args_list if "toolsets" in kwargs
        )
        call_kwargs = assistant_call
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
                    id="user-1",
                    session_id="integ-state",
                    parent_id=None,
                    content="test",
                    machine_state={
                        "active_page": {
                            "name": "agents",
                            "data": {"sessions": [{"name": "leo", "state": "idle"}]},
                        },
                    },
                )
            )

        prompt = next(
            kwargs["system_prompt"]
            for _, kwargs in build_mock.call_args_list
            if "toolsets" in kwargs and "system_prompt" in kwargs
        )
        assert "agents page" in prompt
