from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.exceptions import ModelHTTPError
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


class _MockRun:
    def __init__(self, *, output: Any, all_messages: list[Any], new_messages: list[Any] | None = None) -> None:
        self.result = MagicMock()
        self.result.output = output
        self.result.all_messages.return_value = all_messages
        self.result.new_messages.return_value = new_messages if new_messages is not None else all_messages
        self.result.usage.return_value = MagicMock(
            input_tokens=5,
            output_tokens=7,
            total_tokens=12,
            requests=1,
        )
        self.next_node = End(data=output)
        self.ctx = MagicMock()
        self.ctx.state.message_history = all_messages

    async def __aenter__(self) -> _MockRun:
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

    async def _prepare(
        history: list[Any],
        _ctx: dict[str, Any],
        is_continuation: bool = False,
        exclude_tool_call_ids: set[str] | None = None,
    ) -> HistoryPreparationResult:
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

    return StreamingService(
        config=StreamingConfig(emit_debug_events=False),
        llm_service=MagicMock(),
        history_service=history_service,
        tool_service=MagicMock(),
        assistant_service=assistant_service,
        runtime_settings=RuntimeSettings(frozen_config=AssistantConfig()),
        assistant_config=AssistantConfig(),
        database_service=None,
    )


async def _seed_basic_turn(store: SessionStore) -> None:
    await store.register_user_message(_request(message_id="user-1", parent_id=None))
    await store.register_assistant_message(
        "sess-1",
        message_id="assistant-1",
        parent_id="user-1",
        content="Opening page",
        segments=[{"kind": "text", "text": "Opening page"}],
        usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )


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
    async def test_deferred_frontend_tool_prefers_history_call_id_when_output_drifts(self) -> None:
        deferred = DeferredToolRequests(
            calls=[
                ToolCallPart(
                    tool_name="navigate",
                    args={"page": "agents"},
                    tool_call_id="call-output-id",
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
                            tool_call_id="call-history-id",
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

        assert final["pending_tool_call"]["call_id"] == "call-history-id"
        assert ctx["pending_tool_call_id"] == "call-history-id"
        assert ctx["pending_tool_name"] == "navigate"

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
            usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
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
            new_messages=[
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
    async def test_new_message_resolves_stale_pending_frontend_tool_before_new_turn(self) -> None:
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
                    "tools": [{"id": "call-old", "name": "navigate", "input": {"page": "agents"}}],
                }
            ],
            usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        )
        stale_ctx = sessions.get_context("sess-1")
        stale_ctx["pending_tool_call_id"] = "call-old"
        stale_ctx["pending_tool_name"] = "navigate"
        stale_ctx["pending_assistant_message_id"] = "assistant-1"

        deferred = DeferredToolRequests(
            calls=[
                ToolCallPart(
                    tool_name="navigate",
                    args={"page": "dashboard"},
                    tool_call_id="call-new",
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
                            args={"page": "dashboard"},
                            tool_call_id="call-new",
                        )
                    ],
                    timestamp=datetime(2026, 3, 21, 12, 2, tzinfo=UTC),
                )
            ],
        )
        service = _make_service(sessions=sessions, run=run)
        await service.start()

        events = [
            event
            async for event in service.stream_message(
                _request(
                    message_id="user-2",
                    parent_id="assistant-1",
                    content="Try again",
                )
            )
        ]

        final = next(event for event in events if event["type"] == "final_response")
        ctx = sessions.get_context("sess-1")
        stale_record = ctx["message_index"]["assistant-1"]
        stale_tool = stale_record["segments"][0]["tools"][0]

        assert stale_tool["output"] == service._STALE_FRONTEND_TOOL_OUTPUT
        assert final["pending_tool_call"]["call_id"] == "call-new"
        assert ctx["pending_tool_call_id"] == "call-new"
        assert ctx["pending_assistant_message_id"] == final["message_id"]

    @pytest.mark.asyncio
    async def test_accept_steering_queues_when_stream_is_live(self) -> None:
        sessions = SessionStore()
        await _seed_basic_turn(sessions)
        service = _make_service(sessions=sessions)
        await service.start()

        action = await service.accept_steering(
            _request(
                message_id="steering-1",
                content="Focus on Leo only",
                message_type="steering",
            ),
            has_live_stream=True,
        )

        ctx = sessions.get_context("sess-1")
        assert action == "queued"
        assert ctx["pending_steering_ids"] == ["steering-1"]

    @pytest.mark.asyncio
    async def test_accept_steering_promotes_when_idle(self) -> None:
        sessions = SessionStore()
        await _seed_basic_turn(sessions)
        service = _make_service(sessions=sessions)
        await service.start()

        action = await service.accept_steering(
            _request(
                message_id="steering-1",
                content="Focus on Leo only",
                message_type="steering",
            ),
            has_live_stream=False,
        )

        record = sessions.get_context("sess-1")["steering_index"]["steering-1"]
        assert action == "promoted"
        assert record["status"] == "promoted"

    @pytest.mark.asyncio
    async def test_deliver_steering_into_request_marks_pending_records_delivered(self) -> None:
        sessions = SessionStore()
        await _seed_basic_turn(sessions)
        await sessions.queue_steering(
            "sess-1",
            _request(
                message_id="steering-1",
                content="Focus on Leo only",
                message_type="steering",
            ),
        )
        await sessions.queue_steering(
            "sess-1",
            _request(
                message_id="steering-2",
                content="Skip Ada",
                message_type="steering",
            ),
        )
        service = _make_service(sessions=sessions)
        await service.start()

        node = MagicMock()
        node.request = ModelRequest(parts=[])

        delivered = await service._deliver_steering_into_request(
            session_id="sess-1",
            session_context=sessions.get_context("sess-1"),
            next_node=node,
        )

        assert [record["id"] for record in delivered] == ["steering-1", "steering-2"]
        assert len(node.request.parts) == 2
        assert "Additional user steering" in node.request.parts[0].content
        assert sessions.get_context("sess-1")["pending_steering_ids"] == []

    @pytest.mark.asyncio
    async def test_promoted_steering_updates_existing_assistant_message(self) -> None:
        sessions = SessionStore()
        await _seed_basic_turn(sessions)
        service = _make_service(sessions=sessions)
        await service.start()
        request = _request(
            message_id="steering-1",
            content="Focus on Leo only",
            message_type="steering",
        )

        action = await service.accept_steering(request, has_live_stream=False)
        events = [event async for event in service.stream_message(request)]

        final = next(event for event in events if event["type"] == "final_response")
        path = await sessions.get_message_path("sess-1")

        assert action == "promoted"
        assert final["message_id"] == "assistant-1"
        assert path[-1]["id"] == "assistant-1"
        assert path[-1]["content"].endswith("Hello!")

    @pytest.mark.asyncio
    async def test_provider_rate_limit_emits_retryable_rate_limit_error(self) -> None:
        class _FailingRun:
            async def __aenter__(self) -> _FailingRun:
                raise ModelHTTPError(
                    status_code=429,
                    model_name="claude-haiku-4-5",
                    body={
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": "too many tokens",
                        },
                    },
                )

            async def __aexit__(self, *_args: Any) -> None:
                return None

        service = _make_service()
        service._save_trace = AsyncMock()
        failing_agent = MagicMock()
        failing_agent.iter = MagicMock(return_value=_FailingRun())
        service._assistant_service.prepare_agent_context = AsyncMock(
            return_value=_agent_context(failing_agent)
        )
        await service.start()

        events = [event async for event in service.stream_message(_request(message_id="user-1"))]

        final = next(event for event in events if event["type"] == "final_response")
        error = next(event for event in events if event["type"] == "error")

        assert final["error"] is True
        assert final["error_type"] == "rate_limit"
        assert isinstance(final.get("trace_id"), str)
        assert error["error_type"] == "rate_limit"
        assert error["retry_allowed"] is True
        assert "RATE_LIMIT" in error["message"]
        assert error["trace_id"] == final["trace_id"]
        assert final["trace_id"] in error["message"]

        trace_args = service._save_trace.await_args
        assert trace_args is not None
        assert trace_args.kwargs["trace_id"] == final["trace_id"]
        assert any(event["type"] == "debug_error" for event in trace_args.args[1])

    @pytest.mark.asyncio
    async def test_provider_client_error_persists_debug_error_with_trace_id(self) -> None:
        class _FailingRun:
            async def __aenter__(self) -> _FailingRun:
                raise ModelHTTPError(
                    status_code=400,
                    model_name="gpt-5.4",
                    body={
                        "message": "No tool output found for function call call_hIufoBmnNsKcS4GlaEckDtRn.",
                        "type": "invalid_request_error",
                        "param": "input",
                        "code": None,
                    },
                )

            async def __aexit__(self, *_args: Any) -> None:
                return None

        service = _make_service()
        service._save_trace = AsyncMock()
        failing_agent = MagicMock()
        failing_agent.iter = MagicMock(return_value=_FailingRun())
        service._assistant_service.prepare_agent_context = AsyncMock(
            return_value=_agent_context(failing_agent)
        )
        await service.start()

        events = [event async for event in service.stream_message(_request(message_id="user-1"))]

        final = next(event for event in events if event["type"] == "final_response")
        error = next(event for event in events if event["type"] == "error")

        assert final["error"] is True
        assert final["error_type"] == "provider_client_error"
        assert isinstance(final.get("trace_id"), str)
        assert error["error_type"] == "provider_client_error"
        assert error["trace_id"] == final["trace_id"]
        assert "No tool output found" in error["message"]
        assert final["trace_id"] in error["message"]

        trace_args = service._save_trace.await_args
        assert trace_args is not None
        assert trace_args.kwargs["trace_id"] == final["trace_id"]
        assert any(
            event["type"] == "debug_error"
            and event["trace_id"] == final["trace_id"]
            and event["error_type"] == "provider_client_error"
            for event in trace_args.args[1]
        )


class TestMultiToolContinuation:
    """Integration tests for continuations when the assistant calls both
    backend and frontend tools in the same turn.

    These tests verify that the flat message reconstruction
    (assistant_record_to_flat_messages) produces a single ModelResponse
    with all tool calls — matching Pydantic AI's _handle_deferred_tool_results
    contract. The old split reconstruction would fail with:
    "Tool call results need to be provided for all deferred tool calls."
    """

    @pytest.mark.asyncio
    async def test_backend_plus_frontend_continuation_succeeds(self) -> None:
        """1 backend tool (resolved) + 1 frontend tool (pending) → continuation succeeds."""
        sessions = SessionStore()
        await sessions.register_user_message(_request(message_id="user-1", parent_id=None))

        # Assistant message with TWO tool groups:
        # 1. Backend tool (get_agent_status) — resolved, has output
        # 2. Frontend tool (ui_send_event) — pending, no output
        await sessions.register_assistant_message(
            "sess-1",
            message_id="assistant-1",
            parent_id="user-1",
            content="Checking status then sending event",
            segments=[
                {"kind": "text", "text": "Let me check status first."},
                {
                    "kind": "tool_group",
                    "tools": [
                        {
                            "id": "call-backend-1",
                            "name": "get_agent_status",
                            "input": {"agent": "ike"},
                            "output": {"status": "idle", "entity": "ike"},
                        }
                    ],
                },
                {"kind": "text", "text": "Now sending the event."},
                {
                    "kind": "tool_group",
                    "tools": [
                        {
                            "id": "call-frontend-1",
                            "name": "ui_send_event",
                            "input": {"event": "REFRESH", "data": {}},
                            # no "output" — pending frontend tool
                        }
                    ],
                },
            ],
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )

        ctx = sessions.get_context("sess-1")
        ctx["pending_tool_call_id"] = "call-frontend-1"
        ctx["pending_tool_name"] = "ui_send_event"
        ctx["pending_assistant_message_id"] = "assistant-1"

        # Build what agent.iter would return after a successful continuation:
        # The flat history + tool return for the frontend tool + LLM's final response
        from lovely_assistant.app.assistant._serialization import assistant_record_to_flat_messages

        flat_msgs = assistant_record_to_flat_messages(ctx["message_index"]["assistant-1"])
        user_msg = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="ui_send_event",
                    content={"ok": True},
                    tool_call_id="call-frontend-1",
                    timestamp=datetime(2026, 3, 22, 1, 0, tzinfo=UTC),
                )
            ],
            timestamp=datetime(2026, 3, 22, 1, 0, tzinfo=UTC),
        )
        final_response = ModelResponse(
            parts=[TextPart(content="Event sent successfully.")],
            timestamp=datetime(2026, 3, 22, 1, 0, tzinfo=UTC),
        )

        history_without_leaf = sessions.get_history("sess-1", exclude_leaf=True)
        all_messages = [*history_without_leaf, *flat_msgs, user_msg, final_response]

        run = _MockRun(
            output="Event sent successfully.",
            all_messages=all_messages,
            new_messages=[user_msg, final_response],
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
                    tool_call_id="call-frontend-1",
                    tool_result={"ok": True},
                )
            )
        ]

        # Continuation should succeed — no set-equality mismatch
        final = next(event for event in events if event["type"] == "final_response")
        assert final["message_id"] == "assistant-1"
        assert not final.get("error")

        # Verify agent.iter was called with the flat history structure
        agent = service._assistant_service.prepare_agent_context.return_value.agent
        iter_call = agent.iter.call_args
        message_history = iter_call.kwargs.get("message_history") or iter_call.args[1]

        # Find the last ModelResponse — it should contain BOTH tool calls
        last_model_response = None
        for msg in reversed(message_history):
            if isinstance(msg, ModelResponse):
                last_model_response = msg
                break

        assert last_model_response is not None
        tool_call_ids = {
            part.tool_call_id
            for part in last_model_response.parts
            if isinstance(part, ToolCallPart)
        }
        assert tool_call_ids == {"call-backend-1", "call-frontend-1"}, (
            f"Expected both tool calls in a single ModelResponse, got: {tool_call_ids}"
        )

        # Verify deferred_tool_results was passed correctly
        deferred = iter_call.kwargs.get("deferred_tool_results")
        assert deferred is not None
        assert "call-frontend-1" in deferred.calls

        # Verify pending state was cleared
        assert sessions.get_context("sess-1").get("pending_tool_call_id") is None

    @pytest.mark.asyncio
    async def test_two_backend_plus_frontend_continuation_succeeds(self) -> None:
        """2 backend tools (resolved) + 1 frontend tool (pending) → continuation succeeds."""
        sessions = SessionStore()
        await sessions.register_user_message(_request(message_id="user-1", parent_id=None))

        # Three tool calls: 2 backend (resolved) + 1 frontend (pending)
        await sessions.register_assistant_message(
            "sess-1",
            message_id="assistant-1",
            parent_id="user-1",
            content="Checking multiple things then navigating",
            segments=[
                {
                    "kind": "tool_group",
                    "tools": [
                        {
                            "id": "call-backend-1",
                            "name": "get_agent_status",
                            "input": {"agent": "ike"},
                            "output": {"status": "idle"},
                        }
                    ],
                },
                {
                    "kind": "tool_group",
                    "tools": [
                        {
                            "id": "call-backend-2",
                            "name": "get_agent_status",
                            "input": {"agent": "bell"},
                            "output": {"status": "processing"},
                        }
                    ],
                },
                {
                    "kind": "tool_group",
                    "tools": [
                        {
                            "id": "call-frontend-1",
                            "name": "navigate",
                            "input": {"page": "agents"},
                        }
                    ],
                },
            ],
            usage={"input_tokens": 15, "output_tokens": 25, "total_tokens": 40},
        )

        ctx = sessions.get_context("sess-1")
        ctx["pending_tool_call_id"] = "call-frontend-1"
        ctx["pending_tool_name"] = "navigate"
        ctx["pending_assistant_message_id"] = "assistant-1"

        from lovely_assistant.app.assistant._serialization import assistant_record_to_flat_messages

        flat_msgs = assistant_record_to_flat_messages(ctx["message_index"]["assistant-1"])
        user_msg = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="navigate",
                    content={"page": "agents", "ok": True},
                    tool_call_id="call-frontend-1",
                    timestamp=datetime(2026, 3, 22, 1, 0, tzinfo=UTC),
                )
            ],
            timestamp=datetime(2026, 3, 22, 1, 0, tzinfo=UTC),
        )
        final_response = ModelResponse(
            parts=[TextPart(content="Navigated to agents page.")],
            timestamp=datetime(2026, 3, 22, 1, 0, tzinfo=UTC),
        )

        history_without_leaf = sessions.get_history("sess-1", exclude_leaf=True)
        all_messages = [*history_without_leaf, *flat_msgs, user_msg, final_response]

        run = _MockRun(
            output="Navigated to agents page.",
            all_messages=all_messages,
            new_messages=[user_msg, final_response],
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
                    tool_call_id="call-frontend-1",
                    tool_result={"page": "agents", "ok": True},
                )
            )
        ]

        final = next(event for event in events if event["type"] == "final_response")
        assert final["message_id"] == "assistant-1"
        assert not final.get("error")

        # Verify ALL 3 tool calls are in a single ModelResponse
        agent = service._assistant_service.prepare_agent_context.return_value.agent
        iter_call = agent.iter.call_args
        message_history = iter_call.kwargs.get("message_history") or iter_call.args[1]

        last_model_response = None
        for msg in reversed(message_history):
            if isinstance(msg, ModelResponse):
                last_model_response = msg
                break

        assert last_model_response is not None
        tool_call_ids = {
            part.tool_call_id
            for part in last_model_response.parts
            if isinstance(part, ToolCallPart)
        }
        assert tool_call_ids == {"call-backend-1", "call-backend-2", "call-frontend-1"}, (
            f"Expected all 3 tool calls in a single ModelResponse, got: {tool_call_ids}"
        )

        # Verify the ModelRequest before it has ToolReturnParts for ONLY the resolved tools
        last_model_request = None
        for msg in reversed(message_history):
            if isinstance(msg, ModelRequest):
                last_model_request = msg
                break
            if isinstance(msg, ModelResponse):
                break

        # last_model_request should be the one right before last_model_response
        # It should contain ToolReturnParts for backend-1 and backend-2 only
        if last_model_request is not None:
            return_ids = {
                part.tool_call_id
                for part in last_model_request.parts
                if isinstance(part, ToolReturnPart)
            }
            assert "call-frontend-1" not in return_ids, (
                "Frontend tool should NOT have a ToolReturnPart in the history"
            )

    @pytest.mark.asyncio
    async def test_flat_reconstruction_differs_from_split(self) -> None:
        """Verify the flat form produces a different (correct) structure than the split form.

        This test demonstrates that the old split reconstruction would have
        produced multiple ModelResponses, while the flat form produces one.
        """
        from lovely_assistant.app.assistant._serialization import (
            assistant_record_to_flat_messages,
            path_records_to_model_history,
        )

        record = {
            "id": "assistant-1",
            "role": "assistant",
            "content": "text",
            "segments": [
                {
                    "kind": "tool_group",
                    "tools": [
                        {"id": "call-A", "name": "backend_tool", "input": {}, "output": "done"},
                    ],
                },
                {
                    "kind": "tool_group",
                    "tools": [
                        {"id": "call-B", "name": "frontend_tool", "input": {}},
                    ],
                },
            ],
            "usage": None,
            "created_at": datetime(2026, 3, 22, tzinfo=UTC),
        }

        # Split form: produces MULTIPLE ModelResponses (one per tool group)
        split_messages = path_records_to_model_history([record])
        split_responses = [m for m in split_messages if isinstance(m, ModelResponse)]
        assert len(split_responses) == 2, (
            f"Split form should produce 2 ModelResponses, got {len(split_responses)}"
        )

        # Flat form: produces ONE ModelResponse with ALL tool calls
        flat_messages = assistant_record_to_flat_messages(record)
        flat_responses = [m for m in flat_messages if isinstance(m, ModelResponse)]
        assert len(flat_responses) == 1, (
            f"Flat form should produce 1 ModelResponse, got {len(flat_responses)}"
        )

        # Flat response has BOTH tool calls
        flat_tool_ids = {
            part.tool_call_id
            for part in flat_responses[0].parts
            if isinstance(part, ToolCallPart)
        }
        assert flat_tool_ids == {"call-A", "call-B"}

        # Flat form's ModelRequest has only the resolved tool's ToolReturnPart
        flat_requests = [m for m in flat_messages if isinstance(m, ModelRequest)]
        assert len(flat_requests) == 1
        return_ids = {
            part.tool_call_id
            for part in flat_requests[0].parts
            if isinstance(part, ToolReturnPart)
        }
        assert return_ids == {"call-A"}, "Only the resolved tool should have a ToolReturnPart"
