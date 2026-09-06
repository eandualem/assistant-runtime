"""Mapping of AG-UI run input onto runtime requests (no execution)."""

from __future__ import annotations

import pytest

core = pytest.importorskip("ag_ui.core")

from assistant_runtime.app.routes._agui import (  # noqa: E402
    AGUIRequestError,
    build_assistant_request,
)


def run_input(**overrides):
    return core.RunAgentInput.model_validate(
        {
            "threadId": "thread-1",
            "runId": "run-1",
            "state": {},
            "messages": [{"id": "user-1", "role": "user", "content": "Hi"}],
            "tools": [],
            "context": [],
            "forwardedProps": {},
            **overrides,
        }
    )


class TestBuildAssistantRequest:
    def test_user_message_appends_to_the_active_leaf(self):
        request = build_assistant_request(run_input(), {"active_leaf_id": "assistant-9"})
        assert request.id == "user-1"
        assert request.session_id == "thread-1"
        assert request.parent_id == "assistant-9"
        assert request.content == "Hi"
        # Per-request: no tools this run means no host actions this run.
        assert request.host_context["version"] == 1
        assert request.host_context["actions"] == []
        assert request.config is None

    def test_first_message_of_a_new_session_is_the_root(self):
        assert build_assistant_request(run_input(), None).parent_id is None

    def test_typed_user_content_becomes_reference_attachments(self):
        body = run_input(
            messages=[
                {
                    "id": "user-1",
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look at this"},
                        {"type": "text", "text": "and this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "data",
                                "value": "iVBORw0KGgo=",
                                "mimeType": "image/png",
                            },
                        },
                        {
                            "type": "document",
                            "source": {"type": "url", "value": "https://example.test/spec.pdf"},
                        },
                        {
                            "type": "audio",
                            "source": {"type": "url", "value": "https://example.test/a.mp3"},
                        },
                    ],
                }
            ]
        )
        request = build_assistant_request(body, None)
        assert request.content == "Look at this\nand this"
        assert [(a.kind, a.purpose, a.media_type) for a in request.attachments] == [
            ("image", "reference", "image/png"),
            ("document", "reference", None),
        ]
        assert request.attachments[0].data_uri == "data:image/png;base64,iVBORw0KGgo="
        assert request.attachments[1].url == "https://example.test/spec.pdf"

    @pytest.mark.filterwarnings("ignore:BinaryInputContent is deprecated:DeprecationWarning")
    def test_legacy_binary_user_content_is_still_mapped(self):
        body = run_input(
            messages=[
                {
                    "id": "user-1",
                    "role": "user",
                    "content": [
                        {
                            "type": "binary",
                            "mimeType": "application/pdf",
                            "url": "https://example.test/spec.pdf",
                            "filename": "spec.pdf",
                        },
                        {"type": "binary", "mimeType": "image/jpeg", "data": "/9j/4AAQ"},
                    ],
                }
            ]
        )
        request = build_assistant_request(body, None)
        assert [(a.kind, a.media_type, a.name) for a in request.attachments] == [
            ("document", "application/pdf", "spec.pdf"),
            ("image", "image/jpeg", None),
        ]
        assert request.attachments[0].url == "https://example.test/spec.pdf"
        assert request.attachments[1].data_uri == "data:image/jpeg;base64,/9j/4AAQ"

    def test_tool_message_is_a_continuation(self):
        body = run_input(
            messages=[
                {"id": "user-1", "role": "user", "content": "Pick"},
                {"id": "tool-1", "role": "tool", "toolCallId": "host-1", "content": '{"ok": 1}'},
            ]
        )
        request = build_assistant_request(body, {"pending_tool_call_id": "host-1"})
        assert request.is_continuation
        assert request.id == "tool-1"
        assert request.tool_call_id == "host-1"
        assert request.tool_result == {"ok": 1}
        assert request.tool_outcome == "success"

    def test_tool_message_error_is_a_failed_outcome_with_text_result(self):
        body = run_input(
            messages=[
                {
                    "id": "tool-1",
                    "role": "tool",
                    "toolCallId": "host-1",
                    "content": "not json",
                    "error": "boom",
                }
            ]
        )
        request = build_assistant_request(body, None)
        assert request.tool_result == "not json"
        assert request.tool_outcome == "failed"

    def test_tools_context_and_state_form_the_host_context(self):
        body = run_input(
            tools=[{"name": "go", "description": "", "parameters": {"type": "object"}}],
            context=[{"description": "page", "value": "inventory"}],
            state={"cart": 2},
        )
        request = build_assistant_request(body, None)
        assert request.host_context["version"] == 1
        assert request.host_context["actions"][0]["name"] == "go"
        assert request.host_context["actions"][0]["description"] == "Host action go"
        assert request.host_context["background"] == {"page": "inventory"}
        assert request.host_context["extensions"] == {"state": {"cart": 2}}

    def test_versioned_state_is_the_host_context(self):
        body = run_input(
            state={"version": 1, "view": {"name": "orders"}},
            tools=[{"name": "go", "description": "Go", "parameters": {"type": "object"}}],
        )
        request = build_assistant_request(body, None)
        assert request.host_context["view"]["name"] == "orders"
        assert [a["name"] for a in request.host_context["actions"]] == ["go"]

    def test_forwarded_config_is_the_request_override(self):
        request = build_assistant_request(
            run_input(forwardedProps={"config": {"max_turns": 2}, "other": 1}), None
        )
        assert request.config.max_turns == 2

    def test_invalid_host_action_is_a_validation_error(self):
        body = run_input(tools=[{"name": "bad name!", "description": "x", "parameters": {}}])
        with pytest.raises(ValueError, match="(?i)name|action"):
            build_assistant_request(body, None)

    @pytest.mark.parametrize(
        "messages",
        [[], [{"id": "a", "role": "assistant", "content": "hi"}]],
        ids=["empty", "assistant-last"],
    )
    def test_unmappable_transcripts_are_request_errors(self, messages):
        with pytest.raises(AGUIRequestError):
            build_assistant_request(run_input(messages=messages), None)
