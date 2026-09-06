from __future__ import annotations

import pytest
from pydantic import ValidationError

from assistant_runtime.app.assistant.config import TunableOverrides
from assistant_runtime.app.assistant.models import (
    AssistantRequest,
    AssistantResult,
    normalize_host_context,
)
from assistant_runtime.host_context import camel_to_snake, host_context_from_payload


class TestHelpers:
    def test_camel_to_snake(self) -> None:
        assert camel_to_snake("eventType") == "event_type"
        assert camel_to_snake("HTMLParser") == "html_parser"

    def test_contract_keys_normalize_but_host_payloads_do_not(self) -> None:
        context = normalize_host_context(
            {"view": {"name": "agents", "state": {"listMode": 1}}, "capturedAt": None}
        )
        assert context["view"]["state"] == {"listMode": 1}
        assert "captured_at" not in context  # None values are dropped from the canonical form

    def test_host_context_attachments_join_the_request(self) -> None:
        request = AssistantRequest.model_validate(
            {
                "id": "user-1",
                "session_id": "sess-1",
                "content": "Look",
                "attachments": [{"kind": "text", "text": "same"}],
                "host_context": {
                    "attachments": [
                        {
                            "kind": "image",
                            "purpose": "screenshot",
                            "dataUri": "data:image/png;base64,s",
                        },
                        {"kind": "text", "text": "same"},
                        {"kind": "text", "text": "other"},
                    ]
                },
            }
        )
        assert request.screenshot == "data:image/png;base64,s"
        assert [a.text for a in request.reference_attachments] == ["same", "other"]
        assert len(request.host_context["attachments"]) == 3


class TestAssistantRequest:
    def test_requires_unified_message_contract(self) -> None:
        request = AssistantRequest(
            id="user-1",
            session_id="sess-1",
            parent_id=None,
            content="Hello",
        )

        assert request.id == "user-1"
        assert request.content == "Hello"
        assert request.is_continuation is False
        assert request.is_steering is False

    def test_normalizes_camel_case_host_context(self) -> None:
        request = AssistantRequest.model_validate(
            {
                "id": "user-1",
                "sessionId": "sess-1",
                "parentId": None,
                "content": "Check agents",
                "hostContext": {
                    "page": {"name": "agents", "data": {"entities": [{"name": "leo"}]}}
                },
                "config": {"defaultModel": "openai/gpt-5.4"},
            }
        )

        assert request.session_id == "sess-1"
        # Canonical form: version 1, ``page`` becomes ``view``, empty fields dropped.
        assert request.host_context == {
            "version": 1,
            "view": {
                "name": "agents",
                "description": "",
                "data": {"entities": [{"name": "leo"}]},
                "state": {},
            },
            "navigation": [],
            "background": {},
            "actions": [],
            "attachments": [],
            "extensions": {},
        }
        assert request.config == TunableOverrides(default_model="openai/gpt-5.4")

    def test_invalid_host_context_is_a_validation_error(self) -> None:
        with pytest.raises(ValueError, match="surprise"):
            AssistantRequest.model_validate(
                {
                    "id": "user-1",
                    "session_id": "sess-1",
                    "content": "x",
                    "host_context": {"surprise": True},
                }
            )

    def test_normalize_host_context_rejects_non_mappings(self) -> None:
        assert normalize_host_context(None) is None
        assert normalize_host_context("x") is None
        assert normalize_host_context([1]) is None

    def test_host_context_from_payload_accepts_both_spellings(self) -> None:
        payload = {"hostContext": {"page": {"name": "a"}}}
        assert host_context_from_payload(payload)["view"]["name"] == "a"
        assert (
            host_context_from_payload({"host_context": {"view": {"name": "b"}}})["view"]["name"]
            == "b"
        )
        assert host_context_from_payload({"session_id": "s"}) is None
        assert host_context_from_payload("nope") is None

    def test_host_context_none_by_default(self) -> None:
        request = AssistantRequest(id="u", session_id="s", content="hi")
        assert request.host_context is None

    def test_steering_has_distinct_shape(self) -> None:
        request = AssistantRequest(
            id="steering-1",
            session_id="sess-1",
            content="Focus on Leo",
            message_type="steering",
        )

        assert request.is_steering is True
        assert request.is_continuation is False
        assert request.parent_id is None

    def test_steering_rejects_parent_id(self) -> None:
        with pytest.raises(ValueError, match="must not include parent_id"):
            AssistantRequest(
                id="steering-1",
                session_id="sess-1",
                parent_id="assistant-1",
                content="Focus on Leo",
                message_type="steering",
            )

    def test_steering_rejects_tool_continuation_fields(self) -> None:
        with pytest.raises(ValueError, match="must not include tool continuation fields"):
            AssistantRequest(
                id="steering-1",
                session_id="sess-1",
                content="Focus on Leo",
                message_type="steering",
                tool_call_id="call-1",
                tool_result={"ok": True},
            )

    def test_tool_outcome_requires_a_continuation(self) -> None:
        with pytest.raises(ValueError, match="tool_outcome requires tool_call_id"):
            AssistantRequest(id="user-1", session_id="sess-1", content="", tool_outcome="failed")

    def test_failed_tool_outcome_on_a_continuation(self) -> None:
        request = AssistantRequest.model_validate(
            {
                "id": "continuation-1",
                "sessionId": "sess-1",
                "content": "",
                "toolCallId": "call-1",
                "toolResult": {"error": "not found"},
                "toolOutcome": "failed",
            }
        )
        assert request.is_continuation
        assert request.tool_outcome == "failed"
        with pytest.raises(ValueError, match="tool_outcome"):
            AssistantRequest(
                id="c", session_id="s", content="", tool_call_id="call-1", tool_outcome="unknown"
            )

    def test_screenshot_fields_fold_into_images(self) -> None:
        request = AssistantRequest.model_validate(
            {
                "id": "user-1",
                "session_id": "sess-1",
                "parent_id": None,
                "content": "Look",
                "imageDataUri": "data:image/png;base64,abc123",
            }
        )

        assert request.images == ["data:image/png;base64,abc123"]
        assert request.screenshot == "data:image/png;base64,abc123"
        assert [a.purpose for a in request.attachments] == ["screenshot"]
        assert request.reference_attachments == []

    def test_reference_attachments_are_separate_from_the_screenshot(self) -> None:
        request = AssistantRequest.model_validate(
            {
                "id": "user-1",
                "session_id": "sess-1",
                "content": "Compare",
                "images": ["data:image/png;base64,shot"],
                "attachments": [
                    {"kind": "image", "dataUri": "data:image/jpeg;base64,ref"},
                    {"kind": "text", "text": "notes", "name": "notes.txt"},
                ],
            }
        )
        assert request.screenshot == "data:image/png;base64,shot"
        assert [a.kind for a in request.reference_attachments] == ["image", "text"]

    def test_missing_id_is_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            AssistantRequest.model_validate(
                {"session_id": "sess-1", "parent_id": None, "content": "Hello"}
            )


class TestAssistantResult:
    def test_serializes(self) -> None:
        result = AssistantResult(
            content="Done",
            model="openai:gpt-5.4",
            session_id="sess-1",
            turn_number=1,
        )

        assert result.model_dump() == {
            "content": "Done",
            "model": "openai:gpt-5.4",
            "session_id": "sess-1",
            "turn_number": 1,
            "message_id": None,
            "pending_tool_call": None,
        }
