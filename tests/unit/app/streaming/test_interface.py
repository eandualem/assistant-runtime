from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.usage import UsageLimits
from pydantic_graph.nodes import End

from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.config import AssistantConfig
from lovely_assistant.app.assistant.models import AgentSetupContext, AssistantRequest, PromptResult
from lovely_assistant.app.settings import EffectiveConfig, RuntimeSettings
from lovely_assistant.app.streaming.config import StreamingConfig
from lovely_assistant.app.streaming.exceptions import StreamingError
from lovely_assistant.app.streaming.interface import StreamingService
from lovely_assistant.services.history.models import HistoryPreparationResult
from lovely_assistant.services.tools.models import ToolSet


def _request(
    *,
    message_id: str,
    session_id: str = "sess-1",
    parent_id: str | None = None,
    content: str = "Hello",
    message_type: str = "standard",
    tool_call_id: str | None = None,
    tool_result: Any | None = None,
) -> AssistantRequest:
    return AssistantRequest(
        id=message_id,
        session_id=session_id,
        parent_id=parent_id,
        content=content,
        message_type=message_type,
        tool_call_id=tool_call_id,
        tool_result=tool_result,
    )


@asynccontextmanager
async def _empty_stream() -> Any:
    async def _iter() -> Any:
        return
        yield  # pragma: no cover

    yield _iter()


class _MockRun:
    def __init__(self, *, output: Any, all_messages: list[Any]) -> None:
        self.result = MagicMock()
        self.result.output = output
        self.result.all_messages.return_value = all_messages
        self.result.usage.return_value = MagicMock(
            request_tokens=5,
            response_tokens=7,
            total_tokens=12,
        )
        self.next_node = End(data=output)
        self.ctx = MagicMock()
        self.ctx.state.message_history = all_messages

    async def __aenter__(self) -> "_MockRun":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None


def _agent_context(agent: Any) -> AgentSetupContext:
    effective_config = EffectiveConfig(
        default_model=None,
        thinking_budget=None,
        temperature=None,
        max_turns=10,
        enable_working_memory=True,
        summarization_model=None,
        working_memory_model=None,
        default_image_model=None,
        default_video_model=None,
        subagent_model=None,
        subagent_thinking_budget=None,
    )
    return AgentSetupContext(
        agent=agent,
        available_tools=ToolSet(),
        toolsets=[],
        prompt_result=PromptResult(content="You are Jarvis.", fragments=[]),
        resolved_model="openai:gpt-5.4",
        usage_limits=UsageLimits(request_limit=10),
        output_type=str,
        effective_config=effective_config,
        mcp_summary=None,
    )


def _make_service(*, sessions: SessionStore | None = None, run: _MockRun | None = None) -> StreamingService:
    sessions = sessions or SessionStore()
    run = run or _MockRun(
        output="Hello!",
        all_messages=[
            ModelResponse(
                parts=[TextPart(content="Hello!")],
                timestamp=datetime(2026, 3, 21, 12, 0, tzinfo=UTC),
            )
        ],
    )
    agent = MagicMock()
    agent.iter = MagicMock(return_value=run)

    history_service = AsyncMock()

    async def _prepare(history: list[Any], _ctx: dict[str, Any], is_continuation: bool = False) -> HistoryPreparationResult:
        return HistoryPreparationResult(
            history=history,
            was_compacted=False,
            message_count=len(history),
            estimated_tokens=0,
        )

    history_service.prepare_history_with_metadata = AsyncMock(side_effect=_prepare)

    assistant_service = MagicMock()
    assistant_service.get_session_store.return_value = sessions
    assistant_service.prepare_agent_context = AsyncMock(return_value=_agent_context(agent))
    assistant_service._update_working_memory = AsyncMock()

    service = StreamingService(
        config=StreamingConfig(emit_debug_events=False),
        llm_service=MagicMock(),
        history_service=history_service,
        tool_service=MagicMock(),
        assistant_service=assistant_service,
        runtime_settings=RuntimeSettings(frozen_config=AssistantConfig()),
        assistant_config=AssistantConfig(),
        database_service=None,
    )
    return service


class TestStreamingService:
    @pytest.mark.asyncio
    async def test_stream_message_requires_start(self) -> None:
        service = _make_service()

        with pytest.raises(StreamingError, match="not started"):
            async for _event in service.stream_message(_request(message_id="user-1")):
                pass

    @pytest.mark.asyncio
    async def test_new_message_persists_assistant_row_and_emits_message_id(self) -> None:
        service = _make_service()
        await service.start()

        events = [event async for event in service.stream_message(_request(message_id="user-1"))]

        final = next(event for event in events if event["type"] == "final_response")
        assert final["message_id"]
        path = await service._assistant_service.get_session_store().get_message_path("sess-1")
        assert [message["id"] for message in path] == ["user-1", final["message_id"]]
        assert path[-1]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_deferred_frontend_tool_uses_same_assistant_message_id(self) -> None:
        deferred = DeferredToolRequests(
            calls=[
                ToolCallPart(
                    tool_name="navigate",
                    args={"page": "agents"},
                    tool_call_id="call-nav-1",
                )
            ]
        )
        run = _MockRun(
            output=deferred,
            all_messages=[
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="navigate",
                            args={"page": "agents"},
                            tool_call_id="call-nav-1",
                        )
                    ],
                    timestamp=datetime(2026, 3, 21, 12, 0, tzinfo=UTC),
                )
            ],
        )
        service = _make_service(run=run)
        await service.start()

        events = [event async for event in service.stream_message(_request(message_id="user-1"))]

        final = next(event for event in events if event["type"] == "final_response")
        ctx = service._assistant_service.get_session_store().get_context("sess-1")
        assert final["message_id"] == ctx["pending_assistant_message_id"]
        assert final["pending_tool_call"]["call_id"] == "call-nav-1"

    @pytest.mark.asyncio
    async def test_continuation_updates_existing_assistant_message(self) -> None:
        sessions = SessionStore()
        await sessions.register_user_message(_request(message_id="user-1", parent_id=None))
        await sessions.register_assistant_message(
            "sess-1",
            message_id="assistant-1",
            parent_id="user-1",
            content="Opening page",
            segments=[
                {
                    "kind": "tool_group",
                    "tools": [{"id": "call-nav-1", "name": "navigate", "input": {"page": "agents"}}],
                }
            ],
            usage={"input_tokens": 1, "output_tokens": 2},
        )
        ctx = sessions.get_context("sess-1")
        ctx["pending_tool_call_id"] = "call-nav-1"
        ctx["pending_tool_name"] = "navigate"
        ctx["pending_assistant_message_id"] = "assistant-1"
        history = sessions.get_history("sess-1")
        run = _MockRun(
            output="Done",
            all_messages=[
                *history,
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="navigate",
                            content={"ok": True},
                            tool_call_id="call-nav-1",
                            timestamp=datetime(2026, 3, 21, 12, 1, tzinfo=UTC),
                        )
                    ],
                    timestamp=datetime(2026, 3, 21, 12, 1, tzinfo=UTC),
                ),
                ModelResponse(
                    parts=[TextPart(content="Done")],
                    timestamp=datetime(2026, 3, 21, 12, 1, tzinfo=UTC),
                ),
            ],
        )
        service = _make_service(sessions=sessions, run=run)
        await service.start()

        events = [
            event
            async for event in service.stream_message(
                _request(
                    message_id="cont-1",
                    parent_id="assistant-1",
                    content="",
                    tool_call_id="call-nav-1",
                    tool_result={"ok": True},
                )
            )
        ]

        final = next(event for event in events if event["type"] == "final_response")
        path = await sessions.get_message_path("sess-1")
        assert final["message_id"] == "assistant-1"
        assert path[-1]["id"] == "assistant-1"
        assert path[-1]["content"] == "Done"
        assert sessions.get_context("sess-1").get("pending_tool_call_id") is None

    @pytest.mark.asyncio
    async def test_guidance_is_queued_without_starting_a_new_run(self) -> None:
        sessions = SessionStore()
        ctx = sessions.get_context("sess-1")
        ctx["current_assistant_message_id"] = "assistant-1"
        service = _make_service(sessions=sessions)
        await service.start()

        events = [
            event
            async for event in service.stream_message(
                _request(
                    message_id="guidance-1",
                    parent_id="assistant-1",
                    content="Focus on Leo only",
                    message_type="guidance",
                )
            )
        ]

        assert events == []
        assert ctx["pending_guidance"][0]["id"] == "guidance-1"
