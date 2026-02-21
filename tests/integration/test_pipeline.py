"""Integration tests for the assistant pipeline — request → response with real wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.result import DeferredToolRequests

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


class TestDeferredToolRoundTrip:
    @pytest.mark.asyncio
    async def test_deferred_then_continuation(self, wired_services):
        """First call returns deferred tool request, continuation completes."""
        assistant = wired_services["assistant_service"]

        # Step 1: Agent returns DeferredToolRequests
        mock_call = MagicMock(spec=ToolCallPart)
        mock_call.tool_call_id = "call-round-trip"
        mock_call.tool_name = "navigate"
        mock_call.args = {"message": "hello"}

        mock_deferred = MagicMock()
        mock_deferred.__class__ = DeferredToolRequests
        mock_deferred.calls = [mock_call]

        mock_agent1 = _make_mock_agent(mock_deferred)

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent1):
            # Need frontend tools for output_type to include DeferredToolRequests
            from lovely_assistant.services.tools.models import (
                ToolCategory,
                ToolDefinition,
                ToolSet,
            )

            with patch.object(
                wired_services["tool_service"],
                "get_available_tools",
                return_value=ToolSet(
                    frontend_tools=[
                        ToolDefinition(
                            name="navigate",
                            description="Notify",
                            parameters_schema={},
                            category=ToolCategory.FRONTEND,
                        )
                    ]
                ),
            ):
                result1 = await assistant.process_message(
                    AssistantRequest(session_id="integ-defer", message="Notify user")
                )

        assert result1.is_tool_call
        assert result1.deferred_tool_request.tool_name == "navigate"

        # Verify pending tool call was stored
        sessions = assistant._sessions
        ctx = sessions.get_context("integ-defer")
        assert ctx["pending_tool_call"]["tool_call_id"] == "call-round-trip"

        # Step 2: Continuation with tool result
        mock_agent2 = _make_mock_agent("Notification sent")

        with patch.object(wired_services["llm_service"], "build_agent", return_value=mock_agent2):
            result2 = await assistant.process_message(
                AssistantRequest(
                    session_id="integ-defer",
                    message="Continue",
                    tool_call_id="call-round-trip",
                    tool_result="User notified",
                )
            )

        assert result2.content == "Notification sent"
        assert result2.turn_number == 2

        # Pending tool call should be cleared
        ctx = sessions.get_context("integ-defer")
        assert ctx["pending_tool_call"] is None
