from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from assistant_runtime.app.assistant._serialization import (
    assistant_segments_to_text,
    build_assistant_message_content,
    build_steering_request,
    canonicalize_assistant_segments,
    merge_display_messages,
    normalize_tool_output_for_storage,
    path_records_to_model_history,
    sanitize_image_tool_returns,
    steering_records_to_display,
    tree_messages_to_display,
    tree_messages_to_tree,
)


def _strip_segment_metadata(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in segment.items() if key not in {"segment_id", "segment_index"}}
        for segment in segments
    ]


def test_canonicalize_assistant_segments_merges_adjacent_content() -> None:
    segments = [
        {"kind": "thinking", "text": "A"},
        {"kind": "thinking", "text": "B"},
        {"kind": "text", "text": "C"},
        {"kind": "text", "text": "D"},
        {"kind": "tool_group", "tools": [{"id": "tc-1", "name": "list_agents", "input": {}}]},
        {"kind": "text", "text": "E"},
    ]

    assert canonicalize_assistant_segments(segments) == [
        {"kind": "thinking", "text": "AB"},
        {"kind": "text", "text": "CD"},
        {"kind": "tool_group", "tools": [{"id": "tc-1", "name": "list_agents", "input": {}}]},
        {"kind": "text", "text": "E"},
    ]


def test_path_records_to_model_history_expands_assistant_turns() -> None:
    created_at = datetime(2026, 3, 21, 12, 0, tzinfo=UTC)
    history = path_records_to_model_history(
        [
            {
                "id": "user-1",
                "session_id": "sess-1",
                "parent_id": None,
                "role": "user",
                "message_type": "standard",
                "content": "Check the agents page",
                "created_at": created_at,
            },
            {
                "id": "assistant-1",
                "session_id": "sess-1",
                "parent_id": "user-1",
                "role": "assistant",
                "message_type": "standard",
                "content": "Leo is idle.",
                "segments": [
                    {"kind": "thinking", "text": "Inspecting."},
                    {
                        "kind": "tool_group",
                        "tools": [
                            {
                                "id": "call-1",
                                "name": "get_active_agents",
                                "input": {"scope": "all"},
                                "output": [{"name": "leo", "state": "idle"}],
                            }
                        ],
                    },
                    {"kind": "text", "text": "Leo is idle."},
                ],
                "usage": {"input_tokens": 5, "output_tokens": 7},
                "created_at": created_at,
            },
        ]
    )

    assert len(history) == 4
    assert isinstance(history[0], ModelRequest)
    assert isinstance(history[1], ModelResponse)
    assert isinstance(history[2], ModelRequest)
    assert isinstance(history[3], ModelResponse)
    assert isinstance(history[1].parts[0], ThinkingPart)
    assert isinstance(history[1].parts[1], ToolCallPart)
    assert isinstance(history[2].parts[0], ToolReturnPart)
    assert isinstance(history[3].parts[0], TextPart)
    assert history[3].usage == RequestUsage(input_tokens=5, output_tokens=7)


def test_tree_messages_to_display_returns_root_to_leaf_view() -> None:
    created_at = datetime(2026, 3, 21, 12, 0, tzinfo=UTC)
    display = tree_messages_to_display(
        [
            {
                "id": "user-1",
                "parent_id": None,
                "role": "user",
                "message_type": "standard",
                "content": "Hello",
                "created_at": created_at,
            },
            {
                "id": "assistant-1",
                "parent_id": "user-1",
                "role": "assistant",
                "content": "Hi there",
                "segments": [
                    {"kind": "thinking", "text": "Thinking..."},
                    {"kind": "text", "text": "Hi there"},
                ],
                "usage": {"input_tokens": 1, "output_tokens": 2},
                "created_at": created_at,
            },
        ]
    )

    assert display[0] == {
        "id": "user-1",
        "parent_id": None,
        "role": "user",
        "text": "Hello",
        "message_type": "standard",
        "timestamp": created_at.isoformat(),
    }
    assert display[1]["id"] == "assistant-1"
    assert display[1]["text"] == "Hi there"
    assert display[1]["usage"] == {"input_tokens": 1, "output_tokens": 2}
    assert _strip_segment_metadata(display[1]["segments"]) == [
        {"kind": "thinking", "text": "Thinking..."},
        {"kind": "text", "text": "Hi there"},
    ]


def test_tree_messages_to_tree_preserves_parent_links() -> None:
    created_at = datetime(2026, 3, 21, 12, 0, tzinfo=UTC)
    assert tree_messages_to_tree(
        [
            {
                "id": "root",
                "parent_id": None,
                "role": "user",
                "message_type": "standard",
                "content": "Root",
                "created_at": created_at,
            },
            {
                "id": "child",
                "parent_id": "root",
                "role": "assistant",
                "message_type": "standard",
                "content": "Child",
                "created_at": created_at,
            },
        ]
    ) == [
        {
            "id": "root",
            "parent_id": None,
            "role": "user",
            "message_type": "standard",
            "content": "Root",
            "created_at": created_at.isoformat(),
        },
        {
            "id": "child",
            "parent_id": "root",
            "role": "assistant",
            "message_type": "standard",
            "content": "Child",
            "created_at": created_at.isoformat(),
        },
    ]


def test_steering_records_to_display_filters_pending_records() -> None:
    created_at = datetime(2026, 3, 21, 12, 0, tzinfo=UTC)
    delivered_at = datetime(2026, 3, 21, 12, 1, tzinfo=UTC)

    display = steering_records_to_display(
        [
            {
                "id": "steering-1",
                "content": "Focus on Leo",
                "status": "pending",
                "created_at": created_at,
                "delivered_at": None,
            },
            {
                "id": "steering-2",
                "content": "Skip Ada",
                "status": "delivered",
                "created_at": created_at,
                "delivered_at": delivered_at,
            },
        ]
    )

    assert display == [
        {
            "id": "steering-2",
            "role": "steering",
            "text": "Skip Ada",
            "message_type": "steering",
            "status": "delivered",
            "timestamp": delivered_at.isoformat(),
        }
    ]


def test_merge_display_messages_inserts_steering_by_delivery_time() -> None:
    user_time = datetime(2026, 3, 21, 12, 0, tzinfo=UTC)
    assistant_time = datetime(2026, 3, 21, 12, 1, tzinfo=UTC)
    steering_time = datetime(2026, 3, 21, 12, 2, tzinfo=UTC)

    merged = merge_display_messages(
        [
            {
                "id": "user-1",
                "parent_id": None,
                "role": "user",
                "message_type": "standard",
                "content": "Hello",
                "created_at": user_time,
            },
            {
                "id": "assistant-1",
                "parent_id": "user-1",
                "role": "assistant",
                "message_type": "standard",
                "content": "Hi there",
                "segments": [{"kind": "text", "text": "Hi there"}],
                "usage": None,
                "created_at": assistant_time,
            },
        ],
        [
            {
                "id": "steering-1",
                "content": "Focus on Leo",
                "status": "delivered",
                "created_at": user_time,
                "delivered_at": steering_time,
            }
        ],
    )

    assert [message["id"] for message in merged] == ["user-1", "assistant-1", "steering-1"]


def test_build_steering_request_frames_each_steering_item() -> None:
    request = build_steering_request(
        [
            {
                "id": "steering-1",
                "content": "Focus on Leo",
                "status": "delivered",
                "created_at": datetime(2026, 3, 21, 12, 0, tzinfo=UTC),
                "delivered_at": None,
            },
            {
                "id": "steering-2",
                "content": "Skip Ada",
                "status": "delivered",
                "created_at": datetime(2026, 3, 21, 12, 1, tzinfo=UTC),
                "delivered_at": None,
            },
        ]
    )

    assert len(request.parts) == 2
    assert all(isinstance(part, UserPromptPart) for part in request.parts)
    assert "Additional user steering" in request.parts[0].content
    assert request.parts[1].content.endswith("Skip Ada")


def test_build_assistant_message_content_collects_segments_and_usage() -> None:
    timestamp = datetime(2026, 3, 21, 12, 0, tzinfo=UTC)
    text, segments, created_at = build_assistant_message_content(
        [
            ModelResponse(
                parts=[
                    ThinkingPart(content="Let me check."),
                    ToolCallPart(
                        tool_name="look_at_screen",
                        args={},
                        tool_call_id="call-1",
                    ),
                ],
                timestamp=timestamp,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="look_at_screen",
                        content=BinaryContent(data=b"\xff\xd8", media_type="image/jpeg"),
                        tool_call_id="call-1",
                        timestamp=timestamp,
                    )
                ],
                timestamp=timestamp,
            ),
            ModelResponse(
                parts=[TextPart(content="I found the agents page.")],
                timestamp=timestamp,
            ),
        ]
    )

    assert text == "I found the agents page."
    assert assistant_segments_to_text(segments) == "I found the agents page."
    assert segments == [
        {"kind": "thinking", "text": "Let me check."},
        {
            "kind": "tool_group",
            "tools": [
                {
                    "id": "call-1",
                    "name": "look_at_screen",
                    "input": {},
                    "output": "[Inspected current screen]",
                }
            ],
        },
        {"kind": "text", "text": "I found the agents page."},
    ]
    assert created_at == timestamp


def test_normalize_tool_output_for_storage_handles_nested_binary_values() -> None:
    result = normalize_tool_output_for_storage(
        "get_payload",
        {
            "items": [
                BinaryContent(data=b"\x00", media_type="application/octet-stream"),
                {"nested": BinaryContent(data=b"\x01", media_type="application/octet-stream")},
            ]
        },
    )

    assert result == {
        "items": [
            "[Binary content omitted]",
            {"nested": "[Binary content omitted]"},
        ]
    }


def test_sanitize_image_tool_returns_rewrites_screen_payloads() -> None:
    image = BinaryContent(data=b"\xff\xd8", media_type="image/jpeg")
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="look_at_screen",
                    content=image,
                    tool_call_id="call-1",
                ),
                UserPromptPart(content=["Reference: look_at_screen", image]),
            ]
        )
    ]

    sanitized = sanitize_image_tool_returns(messages)

    request = sanitized[0]
    assert isinstance(request, ModelRequest)
    assert len(request.parts) == 1
    assert isinstance(request.parts[0], ToolReturnPart)
    assert request.parts[0].content == "[Inspected current screen]"
