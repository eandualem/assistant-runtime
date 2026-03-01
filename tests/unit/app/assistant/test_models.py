"""Tests for assistant module request/response models."""

import pytest

from lovely_assistant.app.assistant.models import (
    AssistantRequest,
    AssistantResult,
    RequestConfigOverride,
    _build_user_prompt,
    _camel_to_snake,
    _normalize_keys,
)


class TestCamelToSnake:
    def test_simple_camel(self):
        assert _camel_to_snake("eventType") == "event_type"

    def test_already_snake(self):
        assert _camel_to_snake("event_type") == "event_type"

    def test_consecutive_caps(self):
        assert _camel_to_snake("HTMLParser") == "html_parser"

    def test_single_word(self):
        assert _camel_to_snake("name") == "name"

    def test_pascal_case(self):
        assert _camel_to_snake("ActivePage") == "active_page"


class TestNormalizeKeys:
    def test_flat_dict(self):
        result = _normalize_keys({"activePage": "tasks", "sessionId": "s1"})
        assert result == {"active_page": "tasks", "session_id": "s1"}

    def test_nested_dict(self):
        result = _normalize_keys({"activePage": {"pageName": "tasks"}})
        assert result == {"active_page": {"page_name": "tasks"}}

    def test_list_of_dicts(self):
        result = _normalize_keys([{"eventType": "NAV"}, {"eventType": "REFRESH"}])
        assert result == [{"event_type": "NAV"}, {"event_type": "REFRESH"}]

    def test_dict_with_list(self):
        result = _normalize_keys({"availableActions": [{"eventType": "NAV"}]})
        assert result == {"available_actions": [{"event_type": "NAV"}]}

    def test_already_snake(self):
        data = {"active_page": {"name": "tasks"}}
        assert _normalize_keys(data) == data

    def test_non_dict_passthrough(self):
        assert _normalize_keys("hello") == "hello"
        assert _normalize_keys(42) == 42
        assert _normalize_keys(None) is None

    def test_empty_dict(self):
        assert _normalize_keys({}) == {}


class TestMachineStateNormalization:
    """Tests that AssistantRequest.model_validator normalizes machine_state."""

    def test_top_level_keys_normalized(self):
        req = AssistantRequest(
            session_id="s1",
            message="hi",
            machine_state={"activePage": {"name": "tasks", "data": {}}, "availableActions": []},
        )
        assert "active_page" in req.machine_state
        assert "available_actions" in req.machine_state
        assert "activePage" not in req.machine_state

    def test_nested_keys_normalized(self):
        req = AssistantRequest(
            session_id="s1",
            message="hi",
            machine_state={
                "active_page": {
                    "name": "tasks",
                    "data": {"selectedIssue": {"commentCount": 3}},
                }
            },
        )
        data = req.machine_state["active_page"]["data"]
        assert "selected_issue" in data
        assert "comment_count" in data["selected_issue"]

    def test_snake_case_passthrough(self):
        original = {
            "active_page": {"name": "tasks", "data": {"issues": []}},
            "available_actions": [],
        }
        req = AssistantRequest(session_id="s1", message="hi", machine_state=original)
        assert req.machine_state["active_page"]["name"] == "tasks"
        assert req.machine_state["available_actions"] == []

    def test_none_passthrough(self):
        req = AssistantRequest(session_id="s1", message="hi", machine_state=None)
        assert req.machine_state is None

    def test_entities_aliased_to_sessions_on_agents_page(self):
        req = AssistantRequest(
            session_id="s1",
            message="hi",
            machine_state={
                "active_page": {
                    "name": "agents",
                    "data": {"entities": [{"name": "leo", "state": "idle"}]},
                }
            },
        )
        data = req.machine_state["active_page"]["data"]
        assert "sessions" in data
        assert "entities" not in data
        assert data["sessions"][0]["name"] == "leo"

    def test_coding_agents_aliased_to_sessions_on_agents_page(self):
        req = AssistantRequest(
            session_id="s1",
            message="hi",
            machine_state={
                "activePage": {
                    "name": "agents",
                    "data": {"codingAgents": [{"name": "platform-api", "state": "idle"}]},
                }
            },
        )
        data = req.machine_state["active_page"]["data"]
        assert "sessions" in data
        assert "coding_agents" not in data
        assert data["sessions"][0]["name"] == "platform-api"

    def test_filters_aliased_to_active_filters_on_tasks_page(self):
        req = AssistantRequest(
            session_id="s1",
            message="hi",
            machine_state={
                "active_page": {
                    "name": "tasks",
                    "data": {"filters": {"for": "ike"}},
                }
            },
        )
        data = req.machine_state["active_page"]["data"]
        assert "active_filters" in data
        assert "filters" not in data

    def test_list_items_normalized(self):
        req = AssistantRequest(
            session_id="s1",
            message="hi",
            machine_state={
                "available_actions": [
                    {"eventType": "NAVIGATE", "label": "Go"},
                    {"eventType": "REFRESH"},
                ]
            },
        )
        actions = req.machine_state["available_actions"]
        assert actions[0]["event_type"] == "NAVIGATE"
        assert actions[1]["event_type"] == "REFRESH"

    def test_event_type_normalized_in_actions(self):
        """Full round-trip: camelCase actions from frontend → snake_case for prompt builder."""
        req = AssistantRequest(
            session_id="s1",
            message="hi",
            machine_state={
                "activePage": {"name": "tasks", "data": {}},
                "availableActions": [{"eventType": "NAVIGATE", "label": "Go"}],
            },
        )
        action = req.machine_state["available_actions"][0]
        assert action["event_type"] == "NAVIGATE"
        assert action["label"] == "Go"


class TestAssistantRequest:
    def test_minimal(self):
        req = AssistantRequest(session_id="s1", message="hello")
        assert req.session_id == "s1"
        assert req.message == "hello"
        assert req.machine_state is None

    def test_with_machine_state(self):
        req = AssistantRequest(
            session_id="s1",
            message="hello",
            machine_state={"current_state": "dashboard"},
        )
        assert req.machine_state == {"current_state": "dashboard"}


class TestAssistantRequestContinuation:
    def test_is_continuation_when_tool_call_id_present(self):
        req = AssistantRequest(session_id="s", message="", tool_call_id="call_1")
        assert req.is_continuation is True

    def test_is_not_continuation_without_tool_call_id(self):
        req = AssistantRequest(session_id="s", message="hi")
        assert req.is_continuation is False

    def test_tool_result_field_accepted(self):
        req = AssistantRequest(
            session_id="s",
            message="",
            tool_call_id="call_1",
            tool_result={"success": True},
        )
        assert req.tool_result == {"success": True}

    def test_camel_case_normalization_for_continuation(self):
        req = AssistantRequest(
            **{"sessionId": "s", "message": "", "toolCallId": "c1", "toolResult": 42}
        )
        assert req.tool_call_id == "c1"
        assert req.tool_result == 42


class TestTopLevelCamelCaseNormalization:
    """Tests that the model_validator normalizes camelCase top-level keys."""

    def test_camel_case_top_level_keys(self):
        """Frontend sends sessionId/machineState — should be normalized."""
        req = AssistantRequest.model_validate(
            {"sessionId": "s1", "message": "hi", "machineState": {"activePage": {"name": "home"}}}
        )
        assert req.session_id == "s1"
        assert req.machine_state is not None
        assert "active_page" in req.machine_state

    def test_snake_case_passthrough(self):
        """Snake_case keys still work (idempotent normalization)."""
        req = AssistantRequest(session_id="s1", message="hi")
        assert req.session_id == "s1"


class TestConfigCamelCaseNormalization:
    """Tests that camelCase config keys are normalized before RequestConfigOverride validates."""

    def test_camel_case_config_keys(self):
        req = AssistantRequest.model_validate(
            {
                "sessionId": "s1",
                "message": "hi",
                "config": {"defaultModel": "anthropic:claude-haiku-4-5", "thinkingBudget": 5000},
            }
        )
        assert req.config is not None
        assert req.config.default_model == "anthropic:claude-haiku-4-5"
        assert req.config.thinking_budget == 5000

    def test_mixed_case_config_keys(self):
        req = AssistantRequest.model_validate(
            {
                "session_id": "s1",
                "message": "hi",
                "config": {"default_model": "openai:gpt-4o", "enableWorkingMemory": True},
            }
        )
        assert req.config.default_model == "openai:gpt-4o"
        assert req.config.enable_working_memory is True

    def test_config_none_passthrough(self):
        req = AssistantRequest(session_id="s1", message="hi", config=None)
        assert req.config is None


class TestAssistantResult:
    def test_text_result(self):
        result = AssistantResult(
            content="Hello!",
            model="anthropic:claude-sonnet-4-6",
            session_id="s1",
            turn_number=1,
        )
        assert result.content == "Hello!"

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


class TestRequestConfigOverride:
    def test_all_none_default(self):
        override = RequestConfigOverride()
        assert override.default_model is None
        assert override.thinking_budget is None
        assert override.temperature is None
        assert override.max_turns is None
        assert override.enable_working_memory is None

    def test_model_only(self):
        override = RequestConfigOverride(default_model="anthropic:claude-haiku-4-5")
        assert override.default_model == "anthropic:claude-haiku-4-5"
        assert override.temperature is None

    def test_temperature_bounds_valid(self):
        override = RequestConfigOverride(temperature=0.0)
        assert override.temperature == 0.0
        override = RequestConfigOverride(temperature=2.0)
        assert override.temperature == 2.0

    def test_temperature_bounds_invalid(self):
        with pytest.raises(ValueError, match="less than or equal to 2"):
            RequestConfigOverride(temperature=2.5)
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            RequestConfigOverride(temperature=-0.1)

    def test_assistant_request_with_config(self):
        req = AssistantRequest(
            session_id="s1",
            message="hi",
            config={"temperature": 0.5, "max_turns": 5},
        )
        assert req.config is not None
        assert req.config.temperature == 0.5
        assert req.config.max_turns == 5

    def test_config_none_default(self):
        req = AssistantRequest(session_id="s1", message="hi")
        assert req.config is None


class TestBuildUserPrompt:
    """Tests for _build_user_prompt — returns message string (images handled by look_at_screen tool)."""

    def test_returns_message_string(self):
        result = _build_user_prompt("hello")
        assert result == "hello"
        assert isinstance(result, str)

    def test_empty_message(self):
        result = _build_user_prompt("")
        assert result == ""

    def test_images_field_defaults_empty(self):
        req = AssistantRequest(session_id="s1", message="hi")
        assert req.images == []
