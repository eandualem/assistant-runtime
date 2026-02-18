"""Tests for SSE event builder pure functions."""

from lovely_assistant.app.streaming._event_builder import (
    make_agent_status_event,
    make_error_event,
    make_final_response_event,
    make_text_delta_event,
    make_thinking_delta_event,
    make_tool_call_event,
    make_tool_status_event,
)


class TestAgentStatusEvent:
    def test_started(self):
        event = make_agent_status_event("started")
        assert event == {"type": "agent_status", "status": "started"}

    def test_completed(self):
        event = make_agent_status_event("completed")
        assert event == {"type": "agent_status", "status": "completed"}


class TestThinkingDeltaEvent:
    def test_basic(self):
        event = make_thinking_delta_event("Let me think...")
        assert event == {"type": "thinking_delta", "content": "Let me think..."}

    def test_empty_content(self):
        event = make_thinking_delta_event("")
        assert event["content"] == ""


class TestTextDeltaEvent:
    def test_basic(self):
        event = make_text_delta_event("Hello")
        assert event == {"type": "text_delta", "content": "Hello"}

    def test_unicode_content(self):
        event = make_text_delta_event("Hello 世界")
        assert event["content"] == "Hello 世界"


class TestToolCallEvent:
    def test_basic(self):
        event = make_tool_call_event(
            tool_name="ui_notify",
            arguments={"message": "hi", "level": "info"},
            call_id="call_123",
        )
        assert event == {
            "type": "tool_call",
            "tool_name": "ui_notify",
            "arguments": {"message": "hi", "level": "info"},
            "call_id": "call_123",
        }

    def test_empty_arguments(self):
        event = make_tool_call_event(
            tool_name="get_time",
            arguments={},
            call_id="call_456",
        )
        assert event["arguments"] == {}


class TestToolStatusEvent:
    def test_started(self):
        event = make_tool_status_event("get_time", "started")
        assert event == {"type": "tool_status", "tool_name": "get_time", "status": "started"}

    def test_completed(self):
        event = make_tool_status_event("get_time", "completed")
        assert event["status"] == "completed"

    def test_error(self):
        event = make_tool_status_event("get_time", "error")
        assert event["status"] == "error"


class TestFinalResponseEvent:
    def test_with_content(self):
        event = make_final_response_event("Done!", "claude-3-5-sonnet")
        assert event == {
            "type": "final_response",
            "content": "Done!",
            "model": "claude-3-5-sonnet",
            "streamed": True,
        }

    def test_null_content(self):
        event = make_final_response_event(None, "claude-3-5-sonnet")
        assert event["content"] is None
        assert event["streamed"] is True


class TestErrorEvent:
    def test_basic(self):
        event = make_error_event("Something went wrong")
        assert event == {"type": "error", "message": "Something went wrong"}
