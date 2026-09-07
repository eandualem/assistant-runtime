from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from assistant_runtime.app.assistant.config import AssistantConfig
from assistant_runtime.app.assistant.interface import AssistantService
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.artifacts import technical_operator_profile
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


@pytest.fixture
def artifact_service() -> MagicMock:
    service = MagicMock()
    service.profile = technical_operator_profile()
    service.active_texts = AsyncMock(
        return_value={
            "persona": "You are the assistant.",
            "communication_protocol": "Envelope tags may be present.",
            "ecosystem": "Agents are available.",
            "soul": "Increase the operator's leverage.",
        }
    )
    return service


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
    artifact_service: MagicMock,
) -> AssistantService:
    return AssistantService(
        config=AssistantConfig(),
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
        artifact_service=artifact_service,
    )


class TestWorkingMemory:
    async def test_update_working_memory_extracts_delta_from_tree_path(
        self,
        service: AssistantService,
        history_service: AsyncMock,
    ) -> None:
        await service.start()
        sessions = service.get_session_store()
        assert sessions is not None
        ctx, _record = await sessions.register_user_message(_request(message_id="user-1"))

        await service.update_working_memory("sess-1", ctx, turn_number=1)

        history_service.extract_memory_delta.assert_awaited_once()
        recent = history_service.extract_memory_delta.await_args.args[1]
        assert recent == [{"role": "user", "content": "Hello"}]
        assert ctx["working_memory"] == {"goal": "ship tree model"}

    async def test_update_working_memory_never_raises(
        self,
        service: AssistantService,
        history_service: AsyncMock,
    ) -> None:
        await service.start()
        history_service.extract_memory_delta = AsyncMock(side_effect=RuntimeError("llm down"))
        ctx, _record = await service.get_session_store().register_user_message(
            _request(message_id="user-1")
        )

        await service.update_working_memory("sess-1", ctx, turn_number=1)

        assert "working_memory" not in ctx or ctx["working_memory"] is None
