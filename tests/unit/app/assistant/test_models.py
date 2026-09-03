from __future__ import annotations

import pytest
from pydantic import ValidationError

from assistant_runtime.app.assistant.models import (
    AssistantRequest,
    AssistantResult,
    RequestConfigOverride,
    _build_user_prompt,
    _camel_to_snake,
    _normalize_keys,
)


class TestHelpers:
    def test_camel_to_snake(self) -> None:
        assert _camel_to_snake("eventType") == "event_type"
        assert _camel_to_snake("HTMLParser") == "html_parser"

    def test_normalize_keys(self) -> None:
        assert _normalize_keys({"activePage": {"pageName": "agents"}}) == {
            "active_page": {"page_name": "agents"}
        }

    def test_build_user_prompt_is_passthrough(self) -> None:
        assert _build_user_prompt("hello") == "hello"


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
        assert request.message == "Hello"
        assert request.is_continuation is False
        assert request.is_steering is False

    def test_normalizes_camel_case_and_machine_state_aliases(self) -> None:
        request = AssistantRequest.model_validate(
            {
                "id": "user-1",
                "sessionId": "sess-1",
                "parentId": None,
                "content": "Check agents",
                "machineState": {
                    "activePage": {"name": "agents", "data": {"entities": [{"name": "leo"}]}}
                },
                "config": {"defaultModel": "openai/gpt-5.4"},
            }
        )

        assert request.session_id == "sess-1"
        assert request.machine_state == {
            "active_page": {"name": "agents", "data": {"sessions": [{"name": "leo"}]}}
        }
        assert request.config == RequestConfigOverride(default_model="openai/gpt-5.4")

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

    def test_guidance_alias_normalizes_to_steering(self) -> None:
        request = AssistantRequest.model_validate(
            {
                "id": "steering-1",
                "session_id": "sess-1",
                "content": "Focus on Leo",
                "message_type": "guidance",
            }
        )

        assert request.message_type == "steering"
        assert request.is_steering is True

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
        }
