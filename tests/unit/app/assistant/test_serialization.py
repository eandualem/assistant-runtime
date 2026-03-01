"""Tests for message history serialization helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from lovely_assistant.app.assistant._serialization import (
    _sanitize_history,
    _sanitize_tool_return_content,
    deserialize_messages,
    messages_to_display_format,
    sanitize_image_tool_returns,
    serialize_messages,
)

# --- serialize_messages ---


class TestSerializeMessages:
    def test_empty_list_returns_empty(self):
        result = serialize_messages([])
        assert result == []

    def test_simple_model_request_round_trips(self):
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        serialized = serialize_messages([msg])
        assert isinstance(serialized, list)
        assert len(serialized) == 1
        # Verify the serialized dict is JSON-compatible (no pydantic objects)
        assert isinstance(serialized[0], dict)


# --- deserialize_messages ---


class TestDeserializeMessages:
    def test_empty_list_returns_empty(self):
        result = deserialize_messages([])
        assert result == []

    def test_empty_data_returns_empty(self):
        result = deserialize_messages([])
        assert result == []


# --- Round-trip tests ---


class TestRoundTrip:
    def test_user_prompt_content_preserved(self):
        original_msg = ModelRequest(parts=[UserPromptPart(content="What is 2+2?")])
        serialized = serialize_messages([original_msg])
        deserialized = deserialize_messages(serialized)

        assert len(deserialized) == 1
        restored = deserialized[0]
        assert isinstance(restored, ModelRequest)
        assert len(restored.parts) == 1
        assert isinstance(restored.parts[0], UserPromptPart)
        assert restored.parts[0].content == "What is 2+2?"

    def test_text_part_model_response_preserved(self):
        original_resp = ModelResponse(parts=[TextPart(content="The answer is 4.")])
        serialized = serialize_messages([original_resp])
        deserialized = deserialize_messages(serialized)

        assert len(deserialized) == 1
        restored = deserialized[0]
        assert isinstance(restored, ModelResponse)
        assert len(restored.parts) == 1
        assert isinstance(restored.parts[0], TextPart)
        assert restored.parts[0].content == "The answer is 4."


# --- _sanitize_tool_return_content ---


class TestSanitizeToolReturnContent:
    def test_plain_string_passes_through(self):
        result = _sanitize_tool_return_content("hello world")
        assert result == "hello world"

    def test_plain_dict_passes_through(self):
        data: dict[str, Any] = {"key": "value", "count": 42}
        result = _sanitize_tool_return_content(data)
        assert result == {"key": "value", "count": 42}

    def test_list_of_strings_passes_through(self):
        data = ["a", "b", "c"]
        result = _sanitize_tool_return_content(data)
        assert result == ["a", "b", "c"]

    def test_nested_dict_passes_through(self):
        data: dict[str, Any] = {
            "outer": {
                "inner": "value",
                "nums": [1, 2, 3],
            },
            "flag": True,
        }
        result = _sanitize_tool_return_content(data)
        assert result == {
            "outer": {
                "inner": "value",
                "nums": [1, 2, 3],
            },
            "flag": True,
        }


# --- _sanitize_history ---


class TestSanitizeHistory:
    def test_empty_list_returns_empty(self):
        result = _sanitize_history([])
        assert result == []

    def test_messages_without_tool_return_pass_through(self):
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        resp = ModelResponse(parts=[TextPart(content="world")])
        history = [msg, resp]

        result = _sanitize_history(history)
        assert len(result) == 2
        # Objects should be the same identity (no modification needed)
        assert result[0] is msg
        assert result[1] is resp


# --- messages_to_display_format ---


class TestMessagesToDisplayFormat:
    def test_empty_messages(self):
        result = messages_to_display_format([])
        assert result == []

    def test_user_message(self):
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        result = messages_to_display_format([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["text"] == "hello"

    def test_assistant_text_only(self):
        resp = ModelResponse(
            parts=[TextPart(content="The answer is 42.")],
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        result = messages_to_display_format([resp])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["text"] == "The answer is 42."
        assert result[0]["segments"] == [{"kind": "text", "text": "The answer is 42."}]
        assert "thinking" not in result[0]
        assert "tool_calls" not in result[0]

    def test_assistant_with_thinking(self):
        resp = ModelResponse(
            parts=[
                ThinkingPart(content="Let me consider..."),
                TextPart(content="Here is the answer."),
            ],
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        result = messages_to_display_format([resp])

        assert len(result) == 1
        entry = result[0]
        assert entry["role"] == "assistant"
        assert entry["text"] == "Here is the answer."
        assert entry["thinking"] == "Let me consider..."
        assert entry["segments"] == [
            {"kind": "thinking", "text": "Let me consider..."},
            {"kind": "text", "text": "Here is the answer."},
        ]

    def test_tool_call_paired_with_result(self):
        call_id = "tc_001"
        response_msg = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="get_status",
                    args={"agent": "leo"},
                    tool_call_id=call_id,
                ),
            ],
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        return_msg = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="get_status",
                    content="idle",
                    tool_call_id=call_id,
                    timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
                ),
            ]
        )
        result = messages_to_display_format([response_msg, return_msg])

        # Only the assistant message should appear (tool return is not a standalone message)
        assert len(result) == 1
        entry = result[0]
        assert entry["role"] == "assistant"
        assert len(entry["tool_calls"]) == 1
        tc = entry["tool_calls"][0]
        assert tc["name"] == "get_status"
        assert tc["input"] == {"agent": "leo"}
        assert tc["id"] == call_id
        assert tc["output"] == "idle"
        assert entry["segments"] == [
            {
                "kind": "tool_group",
                "tools": [
                    {
                        "id": call_id,
                        "name": "get_status",
                        "input": {"agent": "leo"},
                        "output": "idle",
                    }
                ],
            }
        ]

    def test_tool_call_without_result(self):
        call_id = "tc_orphan"
        resp = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="restart_agent",
                    args={"name": "ada"},
                    tool_call_id=call_id,
                ),
            ],
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        result = messages_to_display_format([resp])

        assert len(result) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["name"] == "restart_agent"
        assert "output" not in tc
        assert result[0]["segments"] == [
            {
                "kind": "tool_group",
                "tools": [
                    {
                        "id": call_id,
                        "name": "restart_agent",
                        "input": {"name": "ada"},
                    }
                ],
            }
        ]

    def test_system_prompts_skipped(self):
        msg = ModelRequest(parts=[SystemPromptPart(content="You are an assistant.")])
        result = messages_to_display_format([msg])

        assert result == []

    def test_tool_returns_not_separate_messages(self):
        msg = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="check",
                    content="ok",
                    tool_call_id="tc_100",
                    timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            ]
        )
        result = messages_to_display_format([msg])

        assert result == []

    def test_multi_turn_ordering(self):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Question 1")]),
            ModelResponse(parts=[TextPart(content="Answer 1")], timestamp=ts),
            ModelRequest(parts=[UserPromptPart(content="Question 2")]),
            ModelResponse(parts=[TextPart(content="Answer 2")], timestamp=ts),
        ]
        result = messages_to_display_format(messages)

        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[0]["text"] == "Question 1"
        assert result[1]["role"] == "assistant"
        assert result[1]["text"] == "Answer 1"
        assert result[2]["role"] == "user"
        assert result[2]["text"] == "Question 2"
        assert result[3]["role"] == "assistant"
        assert result[3]["text"] == "Answer 2"
        assert result[1]["segments"] == [{"kind": "text", "text": "Answer 1"}]
        assert result[3]["segments"] == [{"kind": "text", "text": "Answer 2"}]

    def test_timestamps_propagated(self):
        ts_user = datetime(2026, 2, 15, 10, 30, 0, tzinfo=UTC)
        ts_assistant = datetime(2026, 2, 15, 10, 30, 5, tzinfo=UTC)
        messages = [
            ModelRequest(
                parts=[UserPromptPart(content="hi", timestamp=ts_user)],
                timestamp=ts_user,
            ),
            ModelResponse(
                parts=[TextPart(content="hello")],
                timestamp=ts_assistant,
            ),
        ]
        result = messages_to_display_format(messages)

        assert len(result) == 2
        assert result[0]["timestamp"] == ts_user.isoformat()
        assert result[1]["timestamp"] == ts_assistant.isoformat()

    def test_segments_interleaved_multi_tool_turn(self):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Check agents")]),
            ModelResponse(
                parts=[
                    TextPart(content="Let me check."),
                    ToolCallPart(
                        tool_name="get_status",
                        args={"agent": "leo"},
                        tool_call_id="tc_1",
                    ),
                ],
                timestamp=ts,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="get_status",
                        content="idle",
                        tool_call_id="tc_1",
                        timestamp=ts,
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content="Leo is idle. Checking Ada."),
                    ToolCallPart(
                        tool_name="get_status",
                        args={"agent": "ada"},
                        tool_call_id="tc_2",
                    ),
                ],
                timestamp=ts,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="get_status",
                        content="busy",
                        tool_call_id="tc_2",
                        timestamp=ts,
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content="Ada is busy.")],
                timestamp=ts,
            ),
        ]
        result = messages_to_display_format(messages)

        # 1 user entry + 1 collapsed assistant entry
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["text"] == "Check agents"

        entry = result[1]
        assert entry["role"] == "assistant"
        assert entry["segments"] == [
            {"kind": "text", "text": "Let me check."},
            {
                "kind": "tool_group",
                "tools": [
                    {
                        "id": "tc_1",
                        "name": "get_status",
                        "input": {"agent": "leo"},
                        "output": "idle",
                    }
                ],
            },
            {"kind": "text", "text": "Leo is idle. Checking Ada."},
            {
                "kind": "tool_group",
                "tools": [
                    {
                        "id": "tc_2",
                        "name": "get_status",
                        "input": {"agent": "ada"},
                        "output": "busy",
                    }
                ],
            },
            {"kind": "text", "text": "Ada is busy."},
        ]

    def test_segments_consecutive_tool_calls_grouped(self):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        messages = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_status",
                        args={"agent": "leo"},
                        tool_call_id="tc_a",
                    ),
                    ToolCallPart(
                        tool_name="get_status",
                        args={"agent": "ada"},
                        tool_call_id="tc_b",
                    ),
                ],
                timestamp=ts,
            ),
        ]
        result = messages_to_display_format(messages)

        assert len(result) == 1
        entry = result[0]
        assert entry["role"] == "assistant"
        assert entry["segments"] == [
            {
                "kind": "tool_group",
                "tools": [
                    {
                        "id": "tc_a",
                        "name": "get_status",
                        "input": {"agent": "leo"},
                    },
                    {
                        "id": "tc_b",
                        "name": "get_status",
                        "input": {"agent": "ada"},
                    },
                ],
            }
        ]

    def test_segments_collapsed_assistant_turn(self):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        messages = [
            ModelResponse(
                parts=[TextPart(content="First part.")],
                timestamp=ts,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="x",
                        content="r",
                        tool_call_id="tc_x",
                        timestamp=ts,
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content="Second part.")],
                timestamp=ts,
            ),
        ]
        result = messages_to_display_format(messages)

        assert len(result) == 1
        entry = result[0]
        assert entry["role"] == "assistant"
        assert entry["segments"] == [
            {"kind": "text", "text": "First part."},
            {"kind": "text", "text": "Second part."},
        ]

    def test_segments_text_only_no_tool_group(self):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        messages = [
            ModelResponse(
                parts=[TextPart(content="Just text.")],
                timestamp=ts,
            ),
        ]
        result = messages_to_display_format(messages)

        assert len(result) == 1
        entry = result[0]
        assert entry["segments"] == [{"kind": "text", "text": "Just text."}]
        assert "tool_calls" not in entry
        # No tool_group segments should be present
        assert all(seg["kind"] != "tool_group" for seg in entry["segments"])

    def test_segments_backward_compat_flat_fields(self):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        messages = [
            ModelResponse(
                parts=[
                    ThinkingPart(content="Hmm..."),
                    TextPart(content="Here's what I found."),
                    ToolCallPart(
                        tool_name="check",
                        args={},
                        tool_call_id="tc_z",
                    ),
                ],
                timestamp=ts,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="check",
                        content="all good",
                        tool_call_id="tc_z",
                        timestamp=ts,
                    ),
                ]
            ),
        ]
        result = messages_to_display_format(messages)

        assert len(result) == 1
        entry = result[0]

        # Backward-compatible flat fields
        assert entry["text"] == "Here's what I found."
        assert entry["thinking"] == "Hmm..."
        assert len(entry["tool_calls"]) == 1
        assert entry["tool_calls"][0]["name"] == "check"
        assert entry["tool_calls"][0]["output"] == "all good"

        # Ordered segments field
        assert entry["segments"] == [
            {"kind": "thinking", "text": "Hmm..."},
            {"kind": "text", "text": "Here's what I found."},
            {
                "kind": "tool_group",
                "tools": [
                    {
                        "id": "tc_z",
                        "name": "check",
                        "input": {},
                        "output": "all good",
                    }
                ],
            },
        ]

    def test_interleaved_segments_across_multi_response_turn(self):
        """Five-segment interleaving: text -> tool -> text -> tool -> text across collapsed turns."""
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Check everything")]),
            # Turn 1: text + tool call
            ModelResponse(
                parts=[
                    TextPart(content="Starting checks."),
                    ToolCallPart(
                        tool_name="check_a", args={"target": "alpha"}, tool_call_id="tc_i1"
                    ),
                ],
                timestamp=ts,
            ),
            # Tool return for check_a
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="check_a",
                        content="alpha ok",
                        tool_call_id="tc_i1",
                        timestamp=ts,
                    ),
                ]
            ),
            # Turn 2: text + tool call
            ModelResponse(
                parts=[
                    TextPart(content="Alpha passed. Now beta."),
                    ToolCallPart(
                        tool_name="check_b", args={"target": "beta"}, tool_call_id="tc_i2"
                    ),
                ],
                timestamp=ts,
            ),
            # Tool return for check_b
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="check_b",
                        content="beta ok",
                        tool_call_id="tc_i2",
                        timestamp=ts,
                    ),
                ]
            ),
            # Turn 3: final text
            ModelResponse(
                parts=[TextPart(content="All checks passed.")],
                timestamp=ts,
            ),
        ]
        result = messages_to_display_format(messages)

        # 1 user + 1 collapsed assistant
        assert len(result) == 2
        assert result[0]["role"] == "user"

        entry = result[1]
        assert entry["role"] == "assistant"
        assert entry["segments"] == [
            {"kind": "text", "text": "Starting checks."},
            {
                "kind": "tool_group",
                "tools": [
                    {
                        "id": "tc_i1",
                        "name": "check_a",
                        "input": {"target": "alpha"},
                        "output": "alpha ok",
                    }
                ],
            },
            {"kind": "text", "text": "Alpha passed. Now beta."},
            {
                "kind": "tool_group",
                "tools": [
                    {
                        "id": "tc_i2",
                        "name": "check_b",
                        "input": {"target": "beta"},
                        "output": "beta ok",
                    }
                ],
            },
            {"kind": "text", "text": "All checks passed."},
        ]
        # Backward compat: text concatenates all text parts, tool_calls has both
        assert entry["text"] == "Starting checks.Alpha passed. Now beta.All checks passed."
        assert len(entry["tool_calls"]) == 2

    def test_user_message_with_multimodal_content(self):
        """User message with text + image (list content) should extract text."""
        screenshot = BinaryContent(data=b"\xff\xd8\xff\xe0", media_type="image/jpeg")
        msg = ModelRequest(
            parts=[UserPromptPart(content=["What is this?", screenshot])],
        )
        result = messages_to_display_format([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["text"] == "What is this?"

    def test_user_message_image_only_gets_placeholder(self):
        """User message with only image content (no text) should get placeholder."""
        screenshot = BinaryContent(data=b"\xff\xd8\xff\xe0", media_type="image/jpeg")
        msg = ModelRequest(
            parts=[UserPromptPart(content=[screenshot])],
        )
        result = messages_to_display_format([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["text"] == "[Image attachment]"

    def test_user_message_multimodal_multi_text_segments(self):
        """User message with multiple text segments in list content."""
        screenshot = BinaryContent(data=b"\xff\xd8\xff\xe0", media_type="image/jpeg")
        msg = ModelRequest(
            parts=[UserPromptPart(content=["Hello", screenshot, "World"])],
        )
        result = messages_to_display_format([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["text"] == "Hello\nWorld"

    def test_multi_turn_with_multimodal_user_messages(self):
        """Full conversation with multimodal user messages should preserve all turns."""
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        screenshot = BinaryContent(data=b"\xff\xd8\xff\xe0", media_type="image/jpeg")
        messages = [
            ModelRequest(
                parts=[UserPromptPart(content=["Question 1", screenshot])],
            ),
            ModelResponse(parts=[TextPart(content="Answer 1")], timestamp=ts),
            ModelRequest(
                parts=[UserPromptPart(content=["Question 2", screenshot])],
            ),
            ModelResponse(parts=[TextPart(content="Answer 2")], timestamp=ts),
        ]
        result = messages_to_display_format(messages)

        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[0]["text"] == "Question 1"
        assert result[1]["role"] == "assistant"
        assert result[1]["text"] == "Answer 1"
        assert result[2]["role"] == "user"
        assert result[2]["text"] == "Question 2"
        assert result[3]["role"] == "assistant"
        assert result[3]["text"] == "Answer 2"


# --- sanitize_image_tool_returns ---


class TestSanitizeImageToolReturns:
    def test_messages_without_look_at_screen_pass_through_unchanged(self):
        """Regular messages without look_at_screen should pass through unmodified."""
        msg = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="get_time",
                    content="2026-01-01",
                    tool_call_id="tc_1",
                ),
                UserPromptPart(content="hello"),
            ]
        )
        result = sanitize_image_tool_returns([msg])

        assert len(result) == 1
        req = result[0]
        assert isinstance(req, ModelRequest)
        assert isinstance(req.parts[0], ToolReturnPart)
        assert req.parts[0].content == "2026-01-01"
        assert isinstance(req.parts[1], UserPromptPart)
        assert req.parts[1].content == "hello"

    def test_look_at_screen_tool_return_content_replaced(self):
        """ToolReturnPart with tool_name='look_at_screen' gets content replaced."""
        msg = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="look_at_screen",
                    content="some binary data",
                    tool_call_id="tc_1",
                ),
            ]
        )
        result = sanitize_image_tool_returns([msg])

        assert len(result) == 1
        req = result[0]
        assert isinstance(req, ModelRequest)
        part = req.parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.content == "[Inspected current screen]"

    def test_synthetic_user_prompt_with_binary_content_removed(self):
        """Synthetic UserPromptPart with BinaryContent is removed entirely (not replaced)."""
        msg = ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "This is file xyz:",
                        BinaryContent(data=b"\xff\xd8", media_type="image/jpeg"),
                    ]
                ),
            ]
        )
        result = sanitize_image_tool_returns([msg])

        assert len(result) == 1
        req = result[0]
        assert isinstance(req, ModelRequest)
        # The synthetic UserPromptPart should be completely removed
        assert len(req.parts) == 0

    def test_other_tool_returns_and_user_prompts_not_affected(self):
        """Non-look_at_screen ToolReturnParts and plain string UserPromptParts are unchanged."""
        msg = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="get_time",
                    content="2026-01-01",
                    tool_call_id="tc_2",
                ),
                UserPromptPart(content="hello"),
            ]
        )
        result = sanitize_image_tool_returns([msg])

        assert len(result) == 1
        req = result[0]
        assert isinstance(req, ModelRequest)
        tool_part = req.parts[0]
        assert isinstance(tool_part, ToolReturnPart)
        assert tool_part.tool_name == "get_time"
        assert tool_part.content == "2026-01-01"
        user_part = req.parts[1]
        assert isinstance(user_part, UserPromptPart)
        assert user_part.content == "hello"

    def test_both_tool_return_and_user_prompt_sanitized_in_same_message(self):
        """ToolReturnPart content replaced, synthetic UserPromptPart removed entirely."""
        msg = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="look_at_screen",
                    content="raw screenshot bytes",
                    tool_call_id="tc_3",
                ),
                UserPromptPart(
                    content=[
                        "Screenshot:",
                        BinaryContent(data=b"\xff\xd8\xff\xe0", media_type="image/jpeg"),
                    ]
                ),
            ]
        )
        result = sanitize_image_tool_returns([msg])

        assert len(result) == 1
        req = result[0]
        assert isinstance(req, ModelRequest)
        # Only the ToolReturnPart should remain — synthetic UserPromptPart removed
        assert len(req.parts) == 1
        tool_part = req.parts[0]
        assert isinstance(tool_part, ToolReturnPart)
        assert tool_part.content == "[Inspected current screen]"
