from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from assistant_runtime.app.assistant.config import AssistantConfig
from assistant_runtime.app.assistant.exceptions import AgentRunError, AssistantError
from assistant_runtime.app.assistant.interface import AssistantService
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.services.tools._request_context import record_current_telegram_chat_binding
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition, ToolSet


def _request(
    *,
    message_id: str,
    session_id: str = "sess-1",
    parent_id: str | None = None,
    content: str = "Hello",
) -> AssistantRequest:
    return AssistantRequest(
        id=message_id,
        session_id=session_id,
        parent_id=parent_id,
        content=content,
    )


def _tool_set() -> ToolSet:
    return ToolSet(
        backend_tools=[
            ToolDefinition(
                name="get_time",
                description="Get the current time",
                parameters_schema={},
                category=ToolCategory.BACKEND,
            )
        ]
    )


def _run_result(output: str = "Hello!") -> MagicMock:
    result = MagicMock()
    result.output = output
    messages = [
        ModelResponse(
            parts=[TextPart(content=output)],
            timestamp=datetime(2026, 3, 21, 12, 0, tzinfo=UTC),
        )
    ]
    result.all_messages.return_value = messages
    result.new_messages.return_value = messages
    usage = MagicMock(input_tokens=11, output_tokens=13, total_tokens=24)
    result.usage = usage
    return result


@pytest.fixture(autouse=True)
def _mock_artifacts() -> Any:
    with patch.object(
        AssistantService,
        "_load_active_artifacts",
        new=AsyncMock(
            return_value={
                "persona": "You are the assistant.",
                "communication_protocol": "Envelope tags may be present.",
                "ecosystem": "Agents are available.",
                "soul": "Increase the operator's leverage.",
            }
        ),
    ):
        yield


@pytest.fixture
def llm_service() -> MagicMock:
    service = MagicMock()
    service.resolve_model = MagicMock(return_value="openai:gpt-5.4")
    agent = MagicMock()
    agent.run = AsyncMock(return_value=_run_result())
    service.build_agent = MagicMock(return_value=agent)
    return service


@pytest.fixture
def history_service() -> AsyncMock:
    service = AsyncMock()
    service.prepare_history = AsyncMock(return_value=([], False))
    service.extract_memory_delta = AsyncMock(return_value={"goal": "ship tree model"})
    return service


@pytest.fixture
def tool_service() -> MagicMock:
    service = MagicMock()
    service.get_available_tools = MagicMock(return_value=_tool_set())
    service.build_toolset = MagicMock(return_value=[])
    service.get_mcp_summary = AsyncMock(return_value=None)
    return service


@pytest.fixture
def service(
    llm_service: MagicMock,
    history_service: AsyncMock,
    tool_service: MagicMock,
) -> AssistantService:
    return AssistantService(
        config=AssistantConfig(),
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
    )


class TestAssistantServiceLifecycle:
    async def test_process_message_requires_start(self, service: AssistantService) -> None:
        with pytest.raises(AssistantError, match="not started"):
            await service.process_message(_request(message_id="user-1"))


class TestProcessMessage:
    async def test_persists_tree_messages_for_a_turn(
        self,
        service: AssistantService,
    ) -> None:
        await service.start()

        result = await service.process_message(_request(message_id="user-1"))

        assert result.content == "Hello!"
        assert result.model == "openai:gpt-5.4"
        assert result.turn_number == 1
        path = await service.get_session_store().get_message_path("sess-1")
        assert [message["id"] for message in path] == ["user-1", path[-1]["id"]]
        assert path[-1]["role"] == "assistant"
        assert path[-1]["content"] == "Hello!"

    async def test_passes_resolved_path_history_into_history_service(
        self,
        service: AssistantService,
        history_service: AsyncMock,
    ) -> None:
        await service.start()
        sessions = service.get_session_store()
        assert sessions is not None
        await sessions.register_user_message(_request(message_id="user-1"))
        await sessions.register_assistant_message(
            "sess-1",
            message_id="assistant-1",
            parent_id="user-1",
            content="First answer",
            segments=[{"kind": "text", "text": "First answer"}],
            usage={"input_tokens": 1, "output_tokens": 2},
        )
        history_service.prepare_history = AsyncMock(
            side_effect=lambda history, ctx: (history, False)
        )

        await service.process_message(
            _request(
                message_id="user-2",
                parent_id="assistant-1",
                content="Follow up",
            )
        )

        prepared_history = history_service.prepare_history.await_args.args[0]
        assert len(prepared_history) == 2

    async def test_process_message_persists_telegram_binding(
        self,
        service: AssistantService,
        llm_service: MagicMock,
    ) -> None:
        await service.start()

        async def _run(*_args: Any, **_kwargs: Any) -> MagicMock:
            record_current_telegram_chat_binding("123456789")
            return _run_result("Telegram sent")

        llm_service.build_agent.return_value.run = AsyncMock(side_effect=_run)

        await service.process_message(_request(message_id="user-1"))

        ctx = service.get_session_store().get_context("sess-1")
        assert ctx["telegram_chat_id"] == "123456789"
        assert ctx["telegram_bound_at"] is not None

    async def test_agent_failures_raise_agent_run_error(
        self,
        service: AssistantService,
        llm_service: MagicMock,
    ) -> None:
        await service.start()
        llm_service.build_agent.return_value.run = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(AgentRunError, match="Agent execution failed"):
            await service.process_message(_request(message_id="user-1"))

    async def test_updates_working_memory_from_tree_path(
        self,
        service: AssistantService,
        history_service: AsyncMock,
    ) -> None:
        await service.start()

        await service.process_message(_request(message_id="user-1"))

        history_service.extract_memory_delta.assert_awaited_once()
        ctx = service.get_session_store().get_context("sess-1")
        assert ctx["working_memory"] == {"goal": "ship tree model"}
