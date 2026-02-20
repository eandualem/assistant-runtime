"""Tests for message history serialization helpers."""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from lovely_assistant.app.assistant._serialization import (
    _sanitize_history,
    _sanitize_tool_return_content,
    deserialize_messages,
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
