"""Tests for the system prompt builder."""

import pytest

from assistant_runtime.app.assistant._prompt_builder import (
    REQUIRED_ARTIFACT_NAMES,
    _datetime_fragment,
    _host_context_fragment,
    _mcp_connections_fragment,
    _render_state,
    _working_memory_fragment,
    build_system_prompt,
)
from assistant_runtime.app.assistant.models import PromptResult
from assistant_runtime.services.history.models import WorkingMemory
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition, ToolSet

REQUIRED_ARTIFACTS = {
    "soul": "The assistant exists to increase the operator's leverage in a live AI workbench.",
    "persona": "You are the assistant, the operational assistant.",
    "communication_protocol": "Messages may arrive with envelope tags.",
    "ecosystem": "Agents: Leo, Ike, Feynman.",
}


class TestDatetimeFragment:
    def test_contains_utc(self):
        frag = _datetime_fragment()
        assert "UTC" in frag

    def test_contains_current_time(self):
        frag = _datetime_fragment()
        assert "Current time:" in frag


class TestWorkingMemoryFragment:
    def test_no_memory(self):
        assert _working_memory_fragment({}) == ""

    def test_none_memory(self):
        assert _working_memory_fragment({"working_memory": None}) == ""

    def test_empty_memory(self):
        wm = WorkingMemory()
        assert _working_memory_fragment({"working_memory": wm}) == ""

    def test_with_memory(self):
        wm = WorkingMemory(active_goal="Fix the bug")
        frag = _working_memory_fragment({"working_memory": wm})
        assert "Fix the bug" in frag


class TestMcpConnectionsFragment:
    def test_none_returns_empty(self):
        assert _mcp_connections_fragment(None) == ""

    def test_empty_list_returns_empty(self):
        assert _mcp_connections_fragment([]) == ""

    def test_simple_server_no_tools(self):
        frag = _mcp_connections_fragment([{"name": "memory"}])
        assert "**memory**" in frag

    def test_server_with_tools_shows_names_and_count(self):
        summary = [
            {
                "name": "memory",
                "tools": ["create_entities", "add_observations", "search_nodes"],
                "tool_count": 3,
            }
        ]
        frag = _mcp_connections_fragment(summary)
        assert "**memory** (3 tools)" in frag
        assert "add_observations" in frag
        assert "create_entities" in frag
        assert "search_nodes" in frag

    def test_more_than_3_tools_shows_overflow(self):
        summary = [
            {
                "name": "memory",
                "tools": ["tool_a", "tool_b", "tool_c", "tool_d", "tool_e"],
                "tool_count": 5,
            }
        ]
        frag = _mcp_connections_fragment(summary)
        assert "(5 tools)" in frag
        assert "+2 more" in frag

    def test_multiple_servers(self):
        summary = [
            {"name": "memory", "tools": ["create_entities"], "tool_count": 1},
            {"name": "brave-search", "tools": ["brave_web_search"], "tool_count": 1},
        ]
        frag = _mcp_connections_fragment(summary)
        assert "**memory**" in frag
        assert "**brave-search**" in frag
        assert "brave_web_search" in frag

    def test_includes_calling_guidance(self):
        summary = [{"name": "memory", "tools": ["tool_a"], "tool_count": 1}]
        frag = _mcp_connections_fragment(summary)
        assert "MCP tools are called directly by name" in frag


class TestHostContextFragment:
    def test_none(self):
        assert _host_context_fragment(None) == ""

    def test_empty_dict(self):
        assert _host_context_fragment({}) == ""

    def test_missing_page(self):
        assert _host_context_fragment({"something": "else"}) == ""

    def test_page_without_name(self):
        assert _host_context_fragment({"page": {"data": {"x": 1}}}) == ""

    def test_missing_page_logs_warning(self):
        from io import StringIO

        from loguru import logger

        sink = StringIO()
        handler_id = logger.add(sink, format="{message}", level="WARNING")
        try:
            assert _host_context_fragment({"something": "else"}) == ""
            assert "no page name" in sink.getvalue()
        finally:
            logger.remove(handler_id)

    def test_header_names_the_page(self):
        frag = _host_context_fragment({"page": {"name": "settings"}})
        assert frag == "The host application is showing: settings."

    def test_header_includes_description(self):
        frag = _host_context_fragment(
            {"page": {"name": "editor", "description": "A file is open for editing."}}
        )
        assert frag.startswith("The host application is showing: editor. A file is open")

    def test_data_renders_structured_items(self):
        ctx = {
            "page": {
                "name": "tasks",
                "data": {
                    "issues": [
                        {"number": 42, "title": "Fix bug", "state": "open", "labels": ["bug"]},
                        {"number": 43, "title": "Add feature", "state": "open", "labels": []},
                    ]
                },
            }
        }
        frag = _host_context_fragment(ctx)
        assert "issues:" in frag
        assert "number: 42" in frag
        assert "title: Fix bug" in frag
        assert "number: 43" in frag

    def test_data_no_truncation(self):
        long_title = "x" * 500
        ctx = {"page": {"name": "tasks", "data": {"issues": [{"title": long_title}]}}}
        assert long_title in _host_context_fragment(ctx)

    def test_data_lists_as_bullets(self):
        ctx = {"page": {"name": "home", "data": {"recent": ["a", "b"]}}}
        frag = _host_context_fragment(ctx)
        assert "recent:" in frag
        assert "- a" in frag
        assert "- b" in frag

    def test_scalar_data_inline(self):
        ctx = {"page": {"name": "home", "data": {"theme": "dark"}}}
        assert "theme: dark" in _host_context_fragment(ctx)

    def test_no_data_key(self):
        assert "showing: agents" in _host_context_fragment({"page": {"name": "agents"}})

    def test_state_rendered_as_json(self):
        ctx = {
            "page": {
                "name": "agents",
                "data": {"sessions": [{"name": "leo", "state": "idle"}]},
                "state": {
                    "list": {
                        "current": "loaded",
                        "context": {"count": 5},
                        "transitions": [{"event_type": "SELECT", "description": "Select"}],
                    }
                },
            }
        }
        frag = _host_context_fragment(ctx)
        assert "Page state:" in frag
        assert '"current": "loaded"' in frag
        assert "SELECT" in frag

    def test_no_state_key_no_section(self):
        frag = _host_context_fragment({"page": {"name": "agents", "data": {"sessions": []}}})
        assert "Page state:" not in frag

    def test_empty_state_no_section(self):
        frag = _host_context_fragment({"page": {"name": "agents", "state": {}}})
        assert "Page state:" not in frag

    def test_background_summary(self):
        ctx = {
            "page": {"name": "tasks", "data": {}},
            "background": {
                "agents": {"state": "idle", "summary": {"entityCount": 5}},
                "sessions": {"state": "loaded", "summary": {"sessionCount": 3}},
            },
        }
        frag = _host_context_fragment(ctx)
        assert "Background:" in frag
        assert "agents (idle)" in frag
        assert "sessions (loaded)" in frag
        assert "5 entityCount" in frag
        assert "3 sessionCount" in frag

    def test_background_flat_dict_fallback(self):
        ctx = {"page": {"name": "tasks"}, "background": {"agents": {"online": 5, "offline": 2}}}
        frag = _host_context_fragment(ctx)
        assert "Background:" in frag
        assert "5 online" in frag

    def test_actions(self):
        ctx = {
            "page": {
                "name": "tasks",
                "actions": [
                    {"event_type": "NAVIGATE", "label": "Go to agents"},
                    {"event_type": "REFRESH", "label": ""},
                ],
            },
        }
        frag = _host_context_fragment(ctx)
        assert "Available host actions:" in frag
        assert "- NAVIGATE (Go to agents)" in frag
        assert "- REFRESH" in frag

    def test_actions_with_params(self):
        ctx = {
            "page": {
                "name": "meetings",
                "actions": [
                    {
                        "event_type": "user.selectRoom",
                        "label": "Select Room",
                        "params": [
                            {"name": "id", "type": "string", "required": True},
                            {"name": "focus", "type": "boolean", "required": False},
                            {"name": "note"},
                        ],
                    },
                ],
            },
        }
        frag = _host_context_fragment(ctx)
        assert (
            "- user.selectRoom (Select Room) — params: id (string, required), focus (boolean), note"
            in frag
        )

    def test_empty_actions_omitted(self):
        frag = _host_context_fragment({"page": {"name": "tasks", "actions": []}})
        assert "Available host actions:" not in frag

    def test_actions_not_read_from_top_level(self):
        ctx = {"page": {"name": "tasks"}, "actions": [{"event_type": "X"}]}
        assert "Available host actions:" not in _host_context_fragment(ctx)

    def test_navigation_targets(self):
        ctx = {
            "page": {"name": "home"},
            "navigation": [
                {"route": "/agents", "name": "Agents", "description": "Monitor agent sessions"},
                {"route": "/tasks", "name": "Tasks", "description": "View GitHub issues"},
            ],
        }
        frag = _host_context_fragment(ctx)
        assert "Navigation:" in frag
        assert "- Agents — Monitor agent sessions" in frag
        assert "- Tasks — View GitHub issues" in frag

    def test_navigation_targets_no_description(self):
        ctx = {"page": {"name": "home"}, "navigation": [{"name": "Settings"}, "About"]}
        frag = _host_context_fragment(ctx)
        assert "- Settings" in frag
        assert "- About" in frag
        assert "— " not in frag.split("Navigation:")[1]

    def test_empty_navigation_omitted(self):
        assert "Navigation:" not in _host_context_fragment(
            {"page": {"name": "home"}, "navigation": []}
        )

    def test_section_order(self):
        ctx = {
            "page": {
                "name": "p",
                "data": {"k": "v"},
                "state": {"s": 1},
                "actions": [{"event_type": "A"}],
            },
            "navigation": ["n"],
            "background": {"b": {"x": 1}},
        }
        frag = _host_context_fragment(ctx)
        positions = [
            frag.index("showing: p"),
            frag.index("k: v"),
            frag.index("Page state:"),
            frag.index("Navigation:"),
            frag.index("Available host actions:"),
            frag.index("Background:"),
        ]
        assert positions == sorted(positions)


class TestRenderState:
    def test_empty_returns_empty(self):
        assert _render_state({}) == ""

    def test_none_returns_empty(self):
        assert _render_state(None) == ""

    def test_json_serialized_faithfully(self):
        state = {"wizard": {"current": "step2", "context": {"draft": True}}}
        result = _render_state(state)
        assert result.startswith("Page state:\n```json\n")
        assert '"current": "step2"' in result
        assert '"draft": true' in result

    def test_non_serializable_values_stringified(self):
        from datetime import date

        result = _render_state({"since": date(2026, 1, 2)})
        assert "2026-01-02" in result


class TestBuildSystemPrompt:
    def test_contains_soul(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "live AI workbench" in result.content

    def test_contains_persona(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "The assistant" in result.content

    def test_contains_communication_protocol(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "envelope tags" in result.content

    def test_contains_datetime(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "Current time:" in result.content

    def test_tools_not_in_prompt_text(self):
        """Tools are registered natively with the agent, not duplicated in the system prompt."""
        ts = ToolSet(
            backend_tools=[
                ToolDefinition(
                    name="test_tool",
                    description="A test",
                    parameters_schema={},
                    category=ToolCategory.BACKEND,
                )
            ]
        )
        result = build_system_prompt(
            available_tools=ts,
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "test_tool" not in result.content
        fragment_names = [f["name"] for f in result.fragments]
        assert "tools" not in fragment_names

    def test_includes_host_context(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            host_context={
                "page": {
                    "name": "agents",
                    "data": {"sessions": [{"name": "leo", "state": "idle"}]},
                },
            },
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "showing: agents" in result.content

    def test_includes_host_context_state(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            host_context={
                "page": {
                    "name": "agents",
                    "state": {"list": {"current": "loaded", "events": ["REFRESH"]}},
                },
            },
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "Page state:" in result.content
        assert "REFRESH" in result.content

    def test_includes_working_memory(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={"working_memory": WorkingMemory(active_goal="Deploy v2")},
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "Deploy v2" in result.content

    def test_fragments_separated(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        # Fragments joined by double newline
        assert "\n\n" in result.content

    def test_returns_prompt_result(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert isinstance(result, PromptResult)
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert isinstance(result.fragments, list)
        assert len(result.fragments) > 0

    def test_fragments_metadata(self):
        ts = ToolSet(
            backend_tools=[
                ToolDefinition(
                    name="test_tool",
                    description="A test",
                    parameters_schema={},
                    category=ToolCategory.BACKEND,
                )
            ]
        )
        result = build_system_prompt(
            available_tools=ts,
            session_context={"working_memory": WorkingMemory(active_goal="Deploy v2")},
            host_context={
                "page": {
                    "name": "agents",
                    "data": {"sessions": [{"name": "leo", "state": "idle"}]},
                },
            },
            artifacts=REQUIRED_ARTIFACTS,
        )
        fragment_names = [f["name"] for f in result.fragments]
        assert "soul" in fragment_names
        assert "persona" in fragment_names
        assert "ecosystem" in fragment_names
        assert "datetime" in fragment_names
        assert "host_context" in fragment_names
        assert "working_memory" in fragment_names
        assert "tools" not in fragment_names
        # Each fragment has name and char_count
        for frag in result.fragments:
            assert "name" in frag
            assert "char_count" in frag
            assert isinstance(frag["char_count"], int)
            assert frag["char_count"] > 0

    def test_working_memory_in_fragments(self):
        """Working memory content appears as a named fragment in the system prompt."""
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={"working_memory": WorkingMemory(active_goal="Ship v3")},
            artifacts=REQUIRED_ARTIFACTS,
        )
        fragment_names = [f["name"] for f in result.fragments]
        assert "working_memory" in fragment_names
        wm_frag = next(f for f in result.fragments if f["name"] == "working_memory")
        assert "Ship v3" in wm_frag["content"]

    def test_minimal_fragments(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        fragment_names = [f["name"] for f in result.fragments]
        # Minimum: soul + persona + communication_protocol + ecosystem + datetime (always present)
        assert "soul" in fragment_names
        assert "persona" in fragment_names
        assert "communication_protocol" in fragment_names
        assert "ecosystem" in fragment_names
        assert "datetime" in fragment_names
        # No tools, host_context, etc. when not provided
        assert "tools" not in fragment_names
        assert "host_context" not in fragment_names
        assert "working_memory" not in fragment_names

    def test_ecosystem_included_from_artifact(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        fragment_names = [f["name"] for f in result.fragments]
        assert "ecosystem" in fragment_names
        assert "Leo" in result.content

    def test_required_artifacts_follow_catalog_order(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        fragment_names = [f["name"] for f in result.fragments[: len(REQUIRED_ARTIFACT_NAMES)]]
        assert fragment_names == list(REQUIRED_ARTIFACT_NAMES)


class TestArtifactIntegration:
    """Tests for DB artifact validation in prompt builder."""

    def test_all_artifacts_appear_in_prompt(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts={
                "soul": "Deeper identity guidance",
                "persona": "I am TestBot",
                "communication_protocol": "Custom protocol rules",
                "ecosystem": "Test agents here",
                "scratchpad": "Test scratchpad",
            },
        )
        assert "Deeper identity guidance" in result.content
        assert "I am TestBot" in result.content
        assert "Custom protocol rules" in result.content
        assert "Test agents here" in result.content
        assert "Test scratchpad" in result.content
        fragment_names = [f["name"] for f in result.fragments]
        assert "soul" in fragment_names
        assert "persona" in fragment_names
        assert "communication_protocol" in fragment_names
        assert "ecosystem" in fragment_names
        assert "scratchpad" in fragment_names

    def test_scratchpad_appears_when_present(self):
        artifacts = {**REQUIRED_ARTIFACTS, "scratchpad": "Remember: the user prefers dark mode"}
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=artifacts,
        )
        assert "Remember: the user prefers dark mode" in result.content
        fragment_names = [f["name"] for f in result.fragments]
        assert "scratchpad" in fragment_names

    def test_empty_scratchpad_excluded(self):
        artifacts = {**REQUIRED_ARTIFACTS, "scratchpad": ""}
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=artifacts,
        )
        fragment_names = [f["name"] for f in result.fragments]
        assert "scratchpad" not in fragment_names

    def test_communication_protocol_content_appears(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts={
                **REQUIRED_ARTIFACTS,
                "communication_protocol": "Custom protocol rules here",
            },
        )
        assert "Custom protocol rules here" in result.content

    def test_missing_persona_raises_error(self):
        with pytest.raises(ValueError, match="Missing required artifact: persona"):
            build_system_prompt(
                available_tools=ToolSet(),
                session_context={},
                artifacts={
                    "soul": "soul",
                    "communication_protocol": "protocol",
                    "ecosystem": "ecosystem",
                },
            )

    def test_missing_communication_protocol_raises_error(self):
        with pytest.raises(ValueError, match="Missing required artifact: communication_protocol"):
            build_system_prompt(
                available_tools=ToolSet(),
                session_context={},
                artifacts={
                    "soul": "soul",
                    "persona": "persona",
                    "ecosystem": "ecosystem",
                },
            )

    def test_missing_soul_raises_error(self):
        with pytest.raises(ValueError, match="Missing required artifact: soul"):
            build_system_prompt(
                available_tools=ToolSet(),
                session_context={},
                artifacts={
                    "persona": "persona",
                    "communication_protocol": "protocol",
                    "ecosystem": "ecosystem",
                },
            )

    def test_missing_ecosystem_raises_error(self):
        with pytest.raises(ValueError, match="Missing required artifact: ecosystem"):
            build_system_prompt(
                available_tools=ToolSet(),
                session_context={},
                artifacts={
                    "soul": "soul",
                    "persona": "persona",
                    "communication_protocol": "protocol",
                },
            )

    def test_empty_persona_raises(self):
        with pytest.raises(ValueError, match="Missing required artifact: persona"):
            build_system_prompt(
                available_tools=ToolSet(),
                session_context={},
                artifacts={
                    "soul": "soul",
                    "persona": "",
                    "communication_protocol": "protocol",
                    "ecosystem": "ecosystem",
                },
            )

    def test_empty_communication_protocol_raises(self):
        with pytest.raises(ValueError, match="Missing required artifact: communication_protocol"):
            build_system_prompt(
                available_tools=ToolSet(),
                session_context={},
                artifacts={
                    "soul": "soul",
                    "persona": "persona",
                    "communication_protocol": "",
                    "ecosystem": "ecosystem",
                },
            )

    def test_empty_soul_raises(self):
        with pytest.raises(ValueError, match="Missing required artifact: soul"):
            build_system_prompt(
                available_tools=ToolSet(),
                session_context={},
                artifacts={
                    "soul": "",
                    "persona": "persona",
                    "communication_protocol": "protocol",
                    "ecosystem": "ecosystem",
                },
            )

    def test_empty_ecosystem_raises(self):
        with pytest.raises(ValueError, match="Missing required artifact: ecosystem"):
            build_system_prompt(
                available_tools=ToolSet(),
                session_context={},
                artifacts={
                    "soul": "soul",
                    "persona": "persona",
                    "communication_protocol": "protocol",
                    "ecosystem": "",
                },
            )
