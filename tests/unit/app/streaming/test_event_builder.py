"""Tests for SSE event builder pure functions."""

from lovely_assistant.app.streaming._event_builder import (
    _truncate,
    make_agent_status_event,
    make_debug_agent_config_event,
    make_debug_completed_event,
    make_debug_error_event,
    make_debug_final_response_event,
    make_debug_history_event,
    make_debug_request_event,
    make_debug_system_prompt_event,
    make_debug_thinking_event,
    make_debug_tool_selection_event,
    make_debug_usage_event,
    make_error_event,
    make_final_response_event,
    make_text_delta_event,
    make_thinking_delta_event,
    make_tool_call_event,
    make_tool_error_event,
    make_tool_result_event,
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
            tool_name="navigate",
            arguments={"message": "hi", "level": "info"},
            call_id="call_123",
        )
        assert event == {
            "type": "tool_call",
            "tool_name": "navigate",
            "arguments": {"message": "hi", "level": "info"},
            "call_id": "call_123",
            "category": "backend",
        }

    def test_empty_arguments(self):
        event = make_tool_call_event(
            tool_name="get_time",
            arguments={},
            call_id="call_456",
        )
        assert event["arguments"] == {}

    def test_frontend_category(self):
        event = make_tool_call_event(
            tool_name="navigate",
            arguments={"page": "agents"},
            call_id="call_789",
            category="frontend",
        )
        assert event["category"] == "frontend"

    def test_default_category_is_backend(self):
        event = make_tool_call_event("tool", {}, "c1")
        assert event["category"] == "backend"


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
            "streamed": False,
        }
        assert "thinking_streamed" not in event
        assert "error" not in event
        assert "error_type" not in event

    def test_null_content(self):
        event = make_final_response_event(None, "claude-3-5-sonnet")
        assert event["content"] is None
        assert event["streamed"] is False

    def test_streamed_true(self):
        event = make_final_response_event("Done!", "model", streamed=True)
        assert event["streamed"] is True

    def test_thinking_streamed(self):
        event = make_final_response_event("Done!", "model", thinking_streamed=True)
        assert event["thinking_streamed"] is True

    def test_error_fields(self):
        event = make_final_response_event("Oops", "model", error=True, error_type="timeout")
        assert event["error"] is True
        assert event["error_type"] == "timeout"

    def test_error_fields_absent(self):
        event = make_final_response_event("OK", "model")
        assert "error" not in event
        assert "error_type" not in event

    def test_session_id_included(self):
        event = make_final_response_event("Done!", "model", session_id="sess-42")
        assert event["session_id"] == "sess-42"

    def test_session_id_absent_when_none(self):
        event = make_final_response_event("Done!", "model")
        assert "session_id" not in event

    def test_trace_id_included(self):
        event = make_final_response_event("Done!", "model", trace_id="trace-1")
        assert event["trace_id"] == "trace-1"


class TestErrorEvent:
    def test_basic(self):
        event = make_error_event("Something went wrong")
        assert event["type"] == "error"
        assert event["message"] == "Something went wrong"
        assert event["terminal"] is True
        assert event["retry_allowed"] is False

    def test_error_type_timeout(self):
        event = make_error_event("Timed out", error_type="timeout", retry_allowed=True)
        assert event["error_type"] == "timeout"
        assert event["retry_allowed"] is True

    def test_error_type_absent(self):
        event = make_error_event("fail")
        assert "error_type" not in event

    def test_trace_id_included(self):
        event = make_error_event("fail", trace_id="trace-1")
        assert event["trace_id"] == "trace-1"


# --- Debug event tests ---


class TestTruncateHelper:
    def test_short_text_unchanged(self):
        assert _truncate("hello", max_len=2000) == "hello"

    def test_long_text_truncated(self):
        text = "x" * 3000
        result = _truncate(text, max_len=2000)
        assert len(result) < len(text)
        assert result.startswith("x" * 2000)
        assert "1000 chars truncated" in result

    def test_exact_limit(self):
        text = "a" * 2000
        assert _truncate(text, max_len=2000) == text

    def test_custom_max_len(self):
        text = "abcdefghij"
        result = _truncate(text, max_len=5)
        assert result.startswith("abcde")
        assert "5 chars truncated" in result

    def test_empty_string(self):
        assert _truncate("") == ""


class TestDebugRequestEvent:
    def test_basic(self):
        event = make_debug_request_event("sess-1", "Hello", False, True)
        assert event["type"] == "debug_request"
        assert event["session_id"] == "sess-1"
        assert event["message"] == "Hello"
        assert event["is_continuation"] is False
        assert event["has_machine_state"] is True
        assert event["machine_state"] is None

    def test_long_message_not_truncated(self):
        long_msg = "x" * 3000
        event = make_debug_request_event("sess-1", long_msg, False, False)
        assert event["message"] == long_msg

    def test_with_machine_state(self):
        state = {"active_page": {"name": "agents"}}
        event = make_debug_request_event("sess-1", "Hi", False, True, machine_state=state)
        assert event["machine_state"] == state

    def test_continuation_flag(self):
        event = make_debug_request_event("sess-2", "cont", True, False)
        assert event["is_continuation"] is True
        assert event["has_machine_state"] is False

    def test_image_count_defaults_to_zero(self):
        event = make_debug_request_event("sess-1", "Hello", False, True)
        assert event["image_count"] == 0

    def test_image_count_included(self):
        event = make_debug_request_event("sess-1", "Hello", False, True, image_count=3)
        assert event["image_count"] == 3


class TestDebugErrorEvent:
    def test_basic(self):
        event = make_debug_error_event(
            "boom",
            error_type="provider_client_error",
            retry_allowed=False,
            trace_id="trace-1",
            model="openai:gpt-5.4",
            phase="stream",
        )
        assert event == {
            "type": "debug_error",
            "message": "boom",
            "error_type": "provider_client_error",
            "retry_allowed": False,
            "trace_id": "trace-1",
            "model": "openai:gpt-5.4",
            "phase": "stream",
        }


class TestDebugSystemPromptEvent:
    def test_basic(self):
        fragments = [{"name": "persona", "char_count": 500, "content": "You are..."}]
        event = make_debug_system_prompt_event(
            total_length=500, fragment_count=1, fragments=fragments, content="You are..."
        )
        assert event["type"] == "debug_system_prompt"
        assert event["total_length"] == 500
        assert event["fragment_count"] == 1
        assert event["fragments"] == fragments
        assert event["content"] == "You are..."

    def test_multiple_fragments(self):
        fragments = [
            {"name": "persona", "char_count": 500, "content": "persona text"},
            {"name": "datetime", "char_count": 50, "content": "datetime text"},
            {"name": "tools", "char_count": 200, "content": "tools text"},
        ]
        event = make_debug_system_prompt_event(
            total_length=750,
            fragment_count=3,
            fragments=fragments,
            content="persona text\n\ndatetime text\n\ntools text",
        )
        assert event["fragment_count"] == 3
        assert len(event["fragments"]) == 3
        assert event["content"] == "persona text\n\ndatetime text\n\ntools text"


class TestDebugHistoryEvent:
    def test_basic(self):
        event = make_debug_history_event(
            message_count=5, estimated_tokens=1000, was_compacted=False, compacted_from=0
        )
        assert event["type"] == "debug_history"
        assert event["message_count"] == 5
        assert event["estimated_tokens"] == 1000
        assert event["was_compacted"] is False
        assert event["compacted_from"] == 0
        assert event["messages"] is None

    def test_compacted(self):
        event = make_debug_history_event(
            message_count=3, estimated_tokens=500, was_compacted=True, compacted_from=10
        )
        assert event["was_compacted"] is True
        assert event["compacted_from"] == 10
        assert event["message_count"] == 3

    def test_with_message_summaries(self):
        summaries = [
            {"role": "user", "content_preview": "Hello", "char_count": 5, "part_count": 1},
            {"role": "assistant", "content_preview": "Hi there", "char_count": 8, "part_count": 1},
        ]
        event = make_debug_history_event(
            message_count=2,
            estimated_tokens=50,
            was_compacted=False,
            compacted_from=0,
            messages=summaries,
        )
        assert event["messages"] == summaries
        assert len(event["messages"]) == 2


class TestDebugToolSelectionEvent:
    def test_basic(self):
        event = make_debug_tool_selection_event(
            page=None,
            backend_count=2,
            filtered_out=0,
            tool_names=["get_time", "list_agents"],
        )
        assert event["type"] == "debug_tool_selection"
        assert event["page"] is None
        assert event["backend_count"] == 2
        assert event["filtered_out"] == 0
        assert len(event["tool_names"]) == 2

    def test_with_filtering(self):
        event = make_debug_tool_selection_event(
            page="agents",
            backend_count=3,
            filtered_out=5,
            tool_names=["get_agent_status", "list_sessions", "check_health"],
        )
        assert event["page"] == "agents"
        assert event["filtered_out"] == 5
        assert event["tools"] is None

    def test_with_tool_details(self):
        tools = [
            {"name": "get_time", "description": "Get current time"},
            {"name": "list_agents", "description": "List all agents"},
        ]
        event = make_debug_tool_selection_event(
            page=None,
            backend_count=2,
            filtered_out=0,
            tool_names=["get_time", "list_agents"],
            tools=tools,
        )
        assert event["tools"] == tools
        assert len(event["tools"]) == 2

    def test_with_parameters_schema(self):
        tools = [
            {
                "name": "get_time",
                "description": "Get current time",
                "parameters_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "read_skill",
                "description": "Read a skill",
                "parameters_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        ]
        event = make_debug_tool_selection_event(
            page=None,
            backend_count=2,
            filtered_out=0,
            tool_names=["get_time", "read_skill"],
            tools=tools,
        )
        assert event["tools"][0]["parameters_schema"] == {"type": "object", "properties": {}}
        assert "required" in event["tools"][1]["parameters_schema"]


class TestDebugAgentConfigEvent:
    def test_basic(self):
        event = make_debug_agent_config_event(
            model="claude-3-5-sonnet",
            output_type="str",
            thinking_budget=None,
        )
        assert event["type"] == "debug_agent_config"
        assert event["model"] == "claude-3-5-sonnet"
        assert event["output_type"] == "str"
        assert event["thinking_budget"] is None

    def test_with_thinking(self):
        event = make_debug_agent_config_event(
            model="claude-3-opus",
            output_type="str",
            thinking_budget=10000,
        )
        assert event["thinking_budget"] == 10000

    def test_session_id_included(self):
        event = make_debug_agent_config_event(
            model="claude-3-5-sonnet",
            output_type="str",
            thinking_budget=None,
            session_id="sess-99",
        )
        assert event["session_id"] == "sess-99"

    def test_session_id_absent_when_none(self):
        event = make_debug_agent_config_event(
            model="claude-3-5-sonnet",
            output_type="str",
            thinking_budget=None,
        )
        assert "session_id" not in event


class TestDebugThinkingEvent:
    def test_basic(self):
        event = make_debug_thinking_event("Let me reason about this...")
        assert event["type"] == "debug_thinking"
        assert event["content"] == "Let me reason about this..."

    def test_empty_content(self):
        event = make_debug_thinking_event("")
        assert event["type"] == "debug_thinking"
        assert event["content"] == ""


class TestDebugFinalResponseEvent:
    def test_basic(self):
        event = make_debug_final_response_event("Hello world", "claude-3-5-sonnet")
        assert event["type"] == "debug_final_response"
        assert event["content"] == "Hello world"
        assert event["model"] == "claude-3-5-sonnet"
        assert "usage" not in event

    def test_with_usage(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        event = make_debug_final_response_event("response", "model-1", usage=usage)
        assert event["usage"] == usage

    def test_without_usage(self):
        event = make_debug_final_response_event("response", "model-1")
        assert "usage" not in event


class TestDebugUsageEvent:
    def test_basic(self):
        event = make_debug_usage_event(
            input_tokens=100,
            output_tokens=50,
            cache_read=20,
            cache_write=10,
            requests=1,
            total=150,
        )
        assert event["type"] == "debug_usage"
        assert event["input_tokens"] == 100
        assert event["output_tokens"] == 50
        assert event["cache_read"] == 20
        assert event["cache_write"] == 10
        assert event["requests"] == 1
        assert event["total_tokens"] == 150


class TestDebugCompletedEvent:
    def test_basic(self):
        event = make_debug_completed_event(duration_ms=123.456)
        assert event["type"] == "debug_completed"
        assert event["duration_ms"] == 123.5

    def test_duration(self):
        event = make_debug_completed_event(duration_ms=500.0)
        assert event["duration_ms"] == 500.0

    def test_duration_rounded(self):
        event = make_debug_completed_event(duration_ms=99.999)
        assert event["duration_ms"] == 100.0


# --- Tool error event tests ---


class TestToolErrorEvent:
    def test_basic(self):
        event = make_tool_error_event(
            tool_name="get_time",
            error="Something went wrong",
            call_id="call_789",
        )
        assert event == {
            "type": "tool_error",
            "tool_name": "get_time",
            "error": "Something went wrong",
            "call_id": "call_789",
        }

    def test_empty_error(self):
        event = make_tool_error_event(
            tool_name="navigate",
            error="",
            call_id="call_empty",
        )
        assert event["error"] == ""
        assert event["type"] == "tool_error"


# --- Enriched tool result event tests ---


class TestToolResultEventEnriched:
    def test_basic_output_only(self):
        event = make_tool_result_event("tool", "data", "call_1")
        assert event["type"] == "tool_result"
        assert event["tool_name"] == "tool"
        assert event["output"] == "data"
        assert event["call_id"] == "call_1"
        assert "result" not in event

    def test_with_duration_ms(self):
        event = make_tool_result_event("tool", "data", "call_1", duration_ms=123.456)
        assert event["duration_ms"] == 123.5

    def test_with_invalidates(self):
        event = make_tool_result_event("tool", "data", "call_1", invalidates=["agents"])
        assert event["invalidates"] == ["agents"]

    def test_without_optional_fields(self):
        event = make_tool_result_event("tool", "data", "call_1")
        assert "duration_ms" not in event
        assert "invalidates" not in event

    def test_with_all_fields(self):
        event = make_tool_result_event(
            "tool",
            "data",
            "call_1",
            duration_ms=50.0,
            invalidates=["agents", "sessions"],
        )
        assert event["duration_ms"] == 50.0
        assert event["invalidates"] == ["agents", "sessions"]
        assert event["output"] == "data"
        assert event["call_id"] == "call_1"


# --- Final response event with usage tests ---


class TestFinalResponseEventUsage:
    def test_with_usage(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        event = make_final_response_event("done", "model", usage=usage)
        assert event["usage"] == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

    def test_without_usage(self):
        event = make_final_response_event("done", "model")
        assert "usage" not in event

    def test_usage_with_other_fields(self):
        usage = {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280}
        event = make_final_response_event(
            "done",
            "claude-3-5-sonnet",
            session_id="sess-7",
            streamed=True,
            usage=usage,
        )
        assert event["session_id"] == "sess-7"
        assert event["streamed"] is True
        assert event["usage"] == usage
        assert event["model"] == "claude-3-5-sonnet"
