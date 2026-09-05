"""The host contract (version 1): parsing, validation, adapters and limits."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from assistant_runtime.host_context import (
    MAX_ATTACHMENT_CHARS,
    MAX_CONTEXT_CHARS,
    Attachment,
    HostAction,
    HostContext,
    actions_of,
    camel_to_snake,
    host_context_from_payload,
    view_name_of,
)

BROWSER = {
    "version": 1,
    "host": {"name": "web-dashboard", "kind": "browser", "version": "2.1"},
    "view": {
        "name": "tasks",
        "description": "Open bugs.",
        "data": {"issues": [{"number": 42, "title": "Fix login"}]},
        "state": {"filters": {"label": "bug"}},
    },
    "navigation": [{"name": "agents", "description": "Running agents"}],
    "actions": [
        {
            "name": "select_issue",
            "description": "Select an issue in the list.",
            "parameters": {
                "type": "object",
                "properties": {"number": {"type": "integer"}},
                "required": ["number"],
            },
        }
    ],
    "attachments": [
        {"kind": "image", "purpose": "screenshot", "data_uri": "data:image/png;base64,x"}
    ],
    "background": {"agents": {"state": "idle", "summary": {"count": 3}}},
    "captured_at": "2026-09-05T10:00:00Z",
}

SERVICE = {
    "host": {"name": "billing-worker", "kind": "service"},
    "actions": [{"name": "issue_refund", "description": "Refund an order."}],
    "extensions": {"tenant": "acme", "queue_depth": 12},
}


class TestExamples:
    def test_browser_and_service_hosts_validate_against_the_same_schema(self):
        browser = HostContext.from_payload(BROWSER)
        service = HostContext.from_payload(SERVICE)
        assert browser.view_name == "tasks"
        assert browser.host.kind == "browser"
        assert [a.name for a in browser.actions] == ["select_issue"]
        assert browser.attachments[0].purpose == "screenshot"
        assert service.view is None
        assert service.version == 1
        assert service.actions[0].parameters == {"type": "object", "properties": {}}
        assert service.extensions == {"tenant": "acme", "queue_depth": 12}

    def test_canonical_form_round_trips(self):
        context = HostContext.from_payload(BROWSER)
        stored = context.to_dict()
        assert stored["view"]["name"] == "tasks"
        assert "page" not in stored
        assert HostContext.from_payload(stored) == context
        assert view_name_of(stored) == "tasks"
        assert [a.name for a in actions_of(stored)] == ["select_issue"]

    def test_absent_context(self):
        assert HostContext.from_payload(None) is None
        assert HostContext.from_payload("text") is None
        assert host_context_from_payload({"session_id": "s"}) is None
        assert view_name_of(None) is None
        assert actions_of({}) == []


class TestAdapters:
    def test_camel_case_and_page_alias(self):
        context = HostContext.from_payload(
            {"page": {"name": "agents"}, "capturedAt": "2026-09-05T10:00:00Z"}
        )
        assert context.view_name == "agents"
        assert context.captured_at == datetime(2026, 9, 5, 10, tzinfo=UTC)
        assert camel_to_snake("HTMLParser") == "html_parser"

    def test_page_and_view_together_is_an_error(self):
        with pytest.raises(ValueError, match="both 'page' and 'view'"):
            HostContext.from_payload({"page": {"name": "a"}, "view": {"name": "b"}})

    def test_legacy_page_actions_explain_the_migration(self):
        with pytest.raises(ValueError, match="host_context.actions"):
            HostContext.from_payload({"page": {"name": "a", "actions": [{"event_type": "x"}]}})

    def test_payload_helper_returns_the_canonical_dict(self):
        stored = host_context_from_payload({"hostContext": {"page": {"name": "a"}}})
        assert stored["version"] == 1
        assert stored["view"]["name"] == "a"


class TestValidation:
    def test_unknown_fields_are_rejected_with_their_path(self):
        with pytest.raises(ValidationError, match="surprise"):
            HostContext.from_payload({"surprise": 1})
        with pytest.raises(ValidationError, match="view.color"):
            HostContext.from_payload({"view": {"name": "a", "color": "red"}})

    def test_unsupported_version(self):
        with pytest.raises(ValidationError, match="version 2 is not supported"):
            HostContext.from_payload({"version": 2})

    def test_size_limit_on_curated_data(self):
        big = {"view": {"name": "a", "data": {"blob": "x" * MAX_CONTEXT_CHARS}}}
        with pytest.raises(ValidationError, match="Curate what the model needs"):
            HostContext.from_payload(big)

    def test_opaque_payload_keys_are_not_renamed(self):
        context = HostContext.from_payload(
            {
                "view": {"name": "a", "data": {"activeFilters": [1]}, "state": {"listMode": 1}},
                "background": {"agentSessions": {"state": "idle"}},
                "extensions": {"tenantId": "acme"},
                "actions": [
                    {
                        "name": "go",
                        "description": "Go.",
                        "parameters": {"type": "object", "properties": {"orderId": {}}},
                    }
                ],
            }
        )
        assert context.view.data == {"activeFilters": [1]}
        assert context.view.state == {"listMode": 1}
        assert context.background == {"agentSessions": {"state": "idle"}}
        assert context.extensions == {"tenantId": "acme"}
        assert context.actions[0].parameters["properties"] == {"orderId": {}}

    def test_cardinality_limits(self):
        with pytest.raises(ValidationError, match="at most 32"):
            HostContext.from_payload(
                {"actions": [{"name": f"a{i}", "description": "x"} for i in range(33)]}
            )
        with pytest.raises(ValidationError, match="at most 16"):
            HostContext.from_payload(
                {"attachments": [{"kind": "text", "text": str(i)} for i in range(17)]}
            )
        with pytest.raises(ValidationError, match="at most 50"):
            HostContext.from_payload({"navigation": [{"name": f"n{i}"} for i in range(51)]})

    def test_duplicate_action_names(self):
        with pytest.raises(ValidationError, match="duplicate names"):
            HostContext.from_payload(
                {"actions": [{"name": "a", "description": "x"}, {"name": "a", "description": "y"}]}
            )

    @pytest.mark.parametrize("name", ["with space", "", "a" * 65, "émoji"])
    def test_action_names_follow_tool_rules(self, name):
        with pytest.raises(ValidationError, match="action name"):
            HostAction(name=name, description="x")

    def test_action_parameters_must_be_an_object_schema(self):
        with pytest.raises(ValidationError, match="type object"):
            HostAction(name="a", description="x", parameters={"type": "array"})

    def test_freshness(self):
        now = datetime(2026, 9, 5, 10, 5, tzinfo=UTC)
        context = HostContext(captured_at=now - timedelta(minutes=5))
        assert context.age_seconds(now) == 300
        assert HostContext().age_seconds(now) is None
        naive = HostContext(captured_at=datetime(2026, 9, 5, 10, 4))
        assert naive.age_seconds(now) == 60


class TestAttachments:
    def test_exactly_one_source(self):
        with pytest.raises(ValidationError, match="exactly one of"):
            Attachment()
        with pytest.raises(ValidationError, match="exactly one of"):
            Attachment(data_uri="data:image/png;base64,x", url="https://x/y.png")

    def test_data_uri_shape_and_size(self):
        with pytest.raises(ValidationError, match="data: URI"):
            Attachment(data_uri="https://not-a-data-uri")
        with pytest.raises(ValidationError, match="exceeds"):
            Attachment(data_uri="data:image/png;base64," + "x" * MAX_ATTACHMENT_CHARS)

    def test_text_attachments(self):
        assert Attachment(kind="text", text="notes").text == "notes"
        with pytest.raises(ValidationError, match="kind 'text'"):
            Attachment(kind="image", text="notes")
        with pytest.raises(ValidationError, match="needs 'text'"):
            Attachment(kind="text", url="https://x/notes.txt")

    def test_screenshot_must_be_an_inline_image(self):
        with pytest.raises(ValidationError, match="screenshot must be an image"):
            Attachment(purpose="screenshot", url="https://x/shot.png")
        shot = Attachment.screenshot("data:image/png;base64,x")
        assert shot.purpose == "screenshot"
        assert shot.kind == "image"

    def test_defaults_are_reference_images(self):
        attachment = Attachment(data_uri="data:image/png;base64,x")
        assert attachment.purpose == "reference"
        assert attachment.kind == "image"
