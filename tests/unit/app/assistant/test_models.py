"""Tests for assistant module request/response models."""

from lovely_assistant.app.assistant.models import AssistantRequest, AssistantResult
from lovely_assistant.services.tools.models import DeferredToolRequest


class TestAssistantRequest:
    def test_minimal(self):
        req = AssistantRequest(session_id="s1", message="hello")
        assert req.session_id == "s1"
        assert req.message == "hello"
        assert req.machine_state is None
        assert req.tool_call_id is None
        assert req.tool_result is None

    def test_with_machine_state(self):
        req = AssistantRequest(
            session_id="s1",
            message="hello",
            machine_state={"current_state": "dashboard"},
        )
        assert req.machine_state == {"current_state": "dashboard"}

    def test_continuation_request(self):
        req = AssistantRequest(
            session_id="s1",
            message="",
            tool_call_id="tc-123",
            tool_result={"status": "ok"},
        )
        assert req.tool_call_id == "tc-123"
        assert req.tool_result == {"status": "ok"}


class TestAssistantResult:
    def test_text_result(self):
        result = AssistantResult(
            content="Hello!",
            model="anthropic:claude-sonnet-4-6",
            session_id="s1",
            turn_number=1,
        )
        assert result.content == "Hello!"
        assert result.deferred_tool_request is None
        assert result.is_tool_call is False

    def test_tool_call_result(self):
        result = AssistantResult(
            content=None,
            model="anthropic:claude-sonnet-4-6",
            deferred_tool_request=DeferredToolRequest(
                request_id="tc-123",
                tool_name="ui_notify",
                arguments={"message": "Done"},
            ),
            session_id="s1",
            turn_number=2,
        )
        assert result.content is None
        assert result.is_tool_call is True
        assert result.deferred_tool_request.tool_name == "ui_notify"

    def test_serialization(self):
        result = AssistantResult(
            content="test",
            model="anthropic:claude-sonnet-4-6",
            session_id="s1",
            turn_number=1,
        )
        data = result.model_dump()
        assert data["content"] == "test"
        assert data["model"] == "anthropic:claude-sonnet-4-6"
        assert data["deferred_tool_request"] is None
