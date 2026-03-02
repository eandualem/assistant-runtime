"""Tests for the system prompt builder."""

import pytest

from lovely_assistant.app.assistant._prompt_builder import (
    _dashboard_context_fragment,
    _datetime_fragment,
    _mcp_connections_fragment,
    _smart_hints,
    _tools_fragment,
    _working_memory_fragment,
    build_system_prompt,
)
from lovely_assistant.app.assistant.models import PromptResult
from lovely_assistant.services.history.models import WorkingMemory
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet

REQUIRED_ARTIFACTS = {
    "persona": "You are Jarvis, the operational assistant.",
    "communication_protocol": "Messages may arrive with envelope tags.",
    "ecosystem": "The Lovely Universe agents: Leo, Ike, Feynman.",
}


class TestDatetimeFragment:
    def test_contains_utc(self):
        frag = _datetime_fragment()
        assert "UTC" in frag

    def test_contains_current_time(self):
        frag = _datetime_fragment()
        assert "Current time:" in frag


class TestToolsFragment:
    def test_empty_tools(self):
        ts = ToolSet()
        assert _tools_fragment(ts) == ""

    def test_backend_tools(self):
        ts = ToolSet(
            backend_tools=[
                ToolDefinition(
                    name="get_time",
                    description="Get current time",
                    parameters_schema={},
                    category=ToolCategory.BACKEND,
                )
            ]
        )
        frag = _tools_fragment(ts)
        assert "get_time" in frag
        assert "Available tools:" in frag


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


class TestDashboardContextFragment:
    def test_none_state(self):
        assert _dashboard_context_fragment(None) == ""

    def test_empty_dict(self):
        assert _dashboard_context_fragment({}) == ""

    def test_missing_active_page(self):
        assert _dashboard_context_fragment({"something": "else"}) == ""

    def test_tasks_page_renders_issues(self):
        """Tasks page uses generic renderer — issues rendered as structured key-value pairs."""
        state = {
            "active_page": {
                "name": "tasks",
                "data": {
                    "issues": [
                        {"number": 42, "title": "Fix bug", "state": "open", "labels": ["bug"]},
                        {"number": 43, "title": "Add feature", "state": "open", "labels": []},
                    ]
                },
            }
        }
        frag = _dashboard_context_fragment(state)
        assert "tasks page" in frag
        assert "number: 42" in frag
        assert "title: Fix bug" in frag
        assert "number: 43" in frag
        assert "title: Add feature" in frag

    def test_tasks_page_renders_selected_issue(self):
        """Selected issue rendered generically — no custom format."""
        state = {
            "active_page": {
                "name": "tasks",
                "data": {
                    "selected_issue": {
                        "title": "Important bug",
                        "body": "This needs fixing",
                        "comment_count": 3,
                    }
                },
            }
        }
        frag = _dashboard_context_fragment(state)
        assert "title: Important bug" in frag
        assert "This needs fixing" in frag
        assert "comment_count: 3" in frag

    def test_agents_page_renders_sessions(self):
        """Agents page uses generic renderer — sessions as structured items."""
        state = {
            "active_page": {
                "name": "agents",
                "data": {
                    "sessions": [
                        {"name": "leo", "state": "idle"},
                        {"name": "ike", "state": "processing", "context": "Working on #50"},
                    ]
                },
            }
        }
        frag = _dashboard_context_fragment(state)
        assert "agents page" in frag
        assert "name: leo" in frag
        assert "state: idle" in frag
        assert "name: ike" in frag
        assert "state: processing" in frag
        assert "Working on #50" in frag

    def test_sessions_page_renders(self):
        """Sessions page uses generic renderer."""
        state = {
            "active_page": {
                "name": "sessions",
                "data": {
                    "sessions": [
                        {"name": "feynman", "state": "idle"},
                    ]
                },
            }
        }
        frag = _dashboard_context_fragment(state)
        assert "sessions page" in frag
        assert "name: feynman" in frag

    def test_home_page_renders(self):
        state = {
            "active_page": {
                "name": "home",
                "data": {"dashboard_version": "1.0"},
            }
        }
        frag = _dashboard_context_fragment(state)
        assert "home page" in frag
        assert len(frag) > 0

    def test_unknown_page_renders_generically(self):
        state = {
            "active_page": {
                "name": "settings",
                "data": {"theme": "dark", "language": "en"},
            }
        }
        frag = _dashboard_context_fragment(state)
        assert "settings page" in frag
        assert "theme" in frag

    def test_generic_renderer_no_truncation(self):
        """Long values are NOT truncated — the dashboard curates what it sends."""
        long_transcript = "Speaker A said something very important. " * 20  # ~820 chars
        state = {
            "active_page": {
                "name": "meetings",
                "data": {"transcript": long_transcript},
            }
        }
        frag = _dashboard_context_fragment(state)
        # The full transcript should be present, not truncated
        assert long_transcript in frag
        assert "..." not in frag

    def test_generic_renderer_lists_as_bullets(self):
        """Lists are rendered as bullet points, not Python repr."""
        state = {
            "active_page": {
                "name": "meetings",
                "data": {
                    "rooms": [
                        {"name": "standup", "participants": 3},
                        {"name": "planning", "participants": 5},
                    ]
                },
            }
        }
        frag = _dashboard_context_fragment(state)
        assert "standup" in frag
        assert "planning" in frag
        assert "- " in frag  # bullet point formatting

    def test_background_summary(self):
        state = {
            "active_page": {"name": "tasks", "data": {}},
            "background": {
                "agents": {"state": "idle", "summary": {"entityCount": 5}},
                "sessions": {"state": "loaded", "summary": {"sessionCount": 3}},
            },
        }
        frag = _dashboard_context_fragment(state)
        assert "Background:" in frag
        assert "agents (idle)" in frag
        assert "sessions (loaded)" in frag
        assert "5 entityCount" in frag
        assert "3 sessionCount" in frag

    def test_background_flat_dict_fallback(self):
        """Background entries without state/summary sub-structure still render."""
        state = {
            "active_page": {"name": "tasks", "data": {}},
            "background": {
                "agents": {"online": 5, "offline": 2},
            },
        }
        frag = _dashboard_context_fragment(state)
        assert "Background:" in frag
        assert "agents" in frag
        assert "5 online" in frag

    def test_available_actions(self):
        state = {
            "active_page": {
                "name": "tasks",
                "data": {},
                "available_actions": [
                    {"event_type": "NAVIGATE", "label": "Go to agents"},
                    {"event_type": "REFRESH", "label": ""},
                ],
            },
        }
        frag = _dashboard_context_fragment(state)
        assert "Available UI actions:" in frag
        assert "- NAVIGATE (Go to agents)" in frag
        assert "- REFRESH" in frag

    def test_available_actions_with_params(self):
        """Actions with params render parameter names and types for the agent."""
        state = {
            "active_page": {
                "name": "meetings",
                "data": {},
                "available_actions": [
                    {
                        "event_type": "user.selectRoom",
                        "label": "Select Room",
                        "params": [
                            {"name": "id", "type": "string", "required": True},
                        ],
                    },
                    {
                        "event_type": "user.createRoom",
                        "label": "Create Room",
                        "params": [
                            {"name": "title", "type": "string", "required": True},
                        ],
                    },
                ],
            },
        }
        frag = _dashboard_context_fragment(state)
        assert "user.selectRoom (Select Room) — params: id (string, required)" in frag
        assert "user.createRoom (Create Room) — params: title (string, required)" in frag

    def test_empty_actions(self):
        state = {
            "active_page": {"name": "tasks", "data": {}, "available_actions": []},
        }
        frag = _dashboard_context_fragment(state)
        assert "Available UI actions" not in frag

    def test_available_actions_not_read_from_top_level(self):
        """available_actions at top level of machine_state should be ignored."""
        state = {
            "active_page": {"name": "tasks", "data": {}},
            "available_actions": [
                {"event_type": "NAVIGATE", "label": "Go to agents"},
            ],
        }
        frag = _dashboard_context_fragment(state)
        assert "Available UI actions" not in frag

    def test_graceful_partial_data(self):
        state = {
            "active_page": {
                "name": "tasks",
                "data": {"issues": [{"title": "No number"}]},
            }
        }
        frag = _dashboard_context_fragment(state)
        assert "No number" in frag

    def test_active_page_no_data_key(self):
        state = {"active_page": {"name": "agents"}}
        frag = _dashboard_context_fragment(state)
        assert "agents page" in frag

    def test_missing_active_page_logs_warning(self):
        """machine_state present but no active_page triggers a warning log."""
        from io import StringIO

        from loguru import logger

        sink = StringIO()
        handler_id = logger.add(sink, format="{message}", level="WARNING")
        try:
            frag = _dashboard_context_fragment({"something": "else"})
            assert frag == ""
            assert "missing active_page" in sink.getvalue()
        finally:
            logger.remove(handler_id)

    def test_navigation_targets(self):
        """Navigation entries with name and description render as bullets."""
        state = {
            "active_page": {"name": "home", "data": {}},
            "navigation": [
                {"route": "/agents", "name": "Agents", "description": "Monitor agent sessions"},
                {"route": "/tasks", "name": "Tasks", "description": "View GitHub issues"},
            ],
        }
        frag = _dashboard_context_fragment(state)
        assert "Navigation:" in frag
        assert "- Agents — Monitor agent sessions" in frag
        assert "- Tasks — View GitHub issues" in frag

    def test_navigation_targets_no_description(self):
        """Navigation entries without description render name only."""
        state = {
            "active_page": {"name": "home", "data": {}},
            "navigation": [
                {"route": "/settings", "name": "Settings"},
            ],
        }
        frag = _dashboard_context_fragment(state)
        assert "Navigation:" in frag
        assert "- Settings" in frag
        assert "— " not in frag.split("Navigation:")[1]

    def test_empty_navigation_omitted(self):
        """Empty navigation list produces no Navigation section."""
        state = {
            "active_page": {"name": "home", "data": {}},
            "navigation": [],
        }
        frag = _dashboard_context_fragment(state)
        assert "Navigation:" not in frag

    def test_no_navigation_key_omitted(self):
        """Missing navigation key produces no Navigation section."""
        state = {
            "active_page": {"name": "home", "data": {}},
        }
        frag = _dashboard_context_fragment(state)
        assert "Navigation:" not in frag


class TestSmartHints:
    def test_none_state(self):
        assert _smart_hints(None) == ""

    def test_empty_state(self):
        assert _smart_hints({}) == ""

    def test_no_active_page(self):
        assert _smart_hints({"something": "else"}) == ""

    def test_idle_agents_hint(self):
        state = {
            "active_page": {
                "name": "agents",
                "data": {
                    "sessions": [
                        {"name": "leo", "state": "idle"},
                        {"name": "ike", "state": "processing"},
                        {"name": "ada", "state": "idle"},
                    ]
                },
            }
        }
        hints = _smart_hints(state)
        assert "Idle agents" in hints
        assert "leo" in hints
        assert "ada" in hints
        assert "ike" not in hints

    def test_plan_waiting_hint(self):
        state = {
            "active_page": {
                "name": "agents",
                "data": {
                    "sessions": [
                        {"name": "leo", "state": "plan_waiting"},
                        {"name": "ike", "state": "idle"},
                    ]
                },
            }
        }
        hints = _smart_hints(state)
        assert "plan approval" in hints
        assert "leo" in hints

    def test_no_hints_when_all_busy(self):
        state = {
            "active_page": {
                "name": "agents",
                "data": {
                    "sessions": [
                        {"name": "leo", "state": "processing"},
                        {"name": "ike", "state": "processing"},
                    ]
                },
            }
        }
        assert _smart_hints(state) == ""

    def test_unfiltered_tasks_hint(self):
        state = {
            "active_page": {
                "name": "tasks",
                "data": {
                    "issues": [{"number": i} for i in range(10)],
                },
            }
        }
        hints = _smart_hints(state)
        assert "10 issues" in hints
        assert "no filters" in hints

    def test_no_hint_when_filtered(self):
        state = {
            "active_page": {
                "name": "tasks",
                "data": {
                    "issues": [{"number": i} for i in range(10)],
                    "active_filters": {"for": "ike"},
                },
            }
        }
        assert _smart_hints(state) == ""

    def test_no_hint_few_issues(self):
        state = {
            "active_page": {
                "name": "tasks",
                "data": {
                    "issues": [{"number": 1}, {"number": 2}],
                },
            }
        }
        assert _smart_hints(state) == ""

    def test_unrelated_page_no_hints(self):
        state = {
            "active_page": {
                "name": "flows",
                "data": {"something": "here"},
            }
        }
        assert _smart_hints(state) == ""


class TestBuildSystemPrompt:
    def test_contains_persona(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "Jarvis" in result.content

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

    def test_includes_tools_when_present(self):
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
        assert "test_tool" in result.content

    def test_includes_machine_state(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            machine_state={
                "active_page": {
                    "name": "agents",
                    "data": {"sessions": [{"name": "leo", "state": "idle"}]},
                },
            },
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "agents page" in result.content

    def test_includes_working_memory(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={"working_memory": WorkingMemory(active_goal="Deploy v2")},
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "Deploy v2" in result.content

    def test_includes_smart_hints(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            machine_state={
                "active_page": {
                    "name": "agents",
                    "data": {
                        "sessions": [
                            {"name": "leo", "state": "idle"},
                            {"name": "ike", "state": "plan_waiting"},
                        ]
                    },
                },
            },
            artifacts=REQUIRED_ARTIFACTS,
        )
        assert "Hints:" in result.content
        assert "Idle agents" in result.content
        assert "plan approval" in result.content

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
            machine_state={
                "active_page": {
                    "name": "agents",
                    "data": {"sessions": [{"name": "leo", "state": "idle"}]},
                },
            },
            artifacts=REQUIRED_ARTIFACTS,
        )
        fragment_names = [f["name"] for f in result.fragments]
        assert "persona" in fragment_names
        assert "ecosystem" in fragment_names
        assert "datetime" in fragment_names
        assert "tools" in fragment_names
        assert "dashboard_context" in fragment_names
        assert "working_memory" in fragment_names
        # Each fragment has name and char_count
        for frag in result.fragments:
            assert "name" in frag
            assert "char_count" in frag
            assert isinstance(frag["char_count"], int)
            assert frag["char_count"] > 0

    def test_minimal_fragments(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=REQUIRED_ARTIFACTS,
        )
        fragment_names = [f["name"] for f in result.fragments]
        # Minimum: persona + communication_protocol + ecosystem + datetime (always present)
        assert "persona" in fragment_names
        assert "communication_protocol" in fragment_names
        assert "ecosystem" in fragment_names
        assert "datetime" in fragment_names
        # No tools, dashboard_context, etc. when not provided
        assert "tools" not in fragment_names
        assert "dashboard_context" not in fragment_names
        assert "smart_hints" not in fragment_names
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


class TestArtifactIntegration:
    """Tests for DB artifact validation in prompt builder."""

    def test_all_artifacts_appear_in_prompt(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts={
                "persona": "I am TestBot",
                "communication_protocol": "Custom protocol rules",
                "ecosystem": "Test agents here",
                "scratchpad": "Test scratchpad",
            },
        )
        assert "I am TestBot" in result.content
        assert "Custom protocol rules" in result.content
        assert "Test agents here" in result.content
        assert "Test scratchpad" in result.content
        fragment_names = [f["name"] for f in result.fragments]
        assert "persona" in fragment_names
        assert "communication_protocol" in fragment_names
        assert "ecosystem" in fragment_names
        assert "scratchpad" in fragment_names

    def test_scratchpad_appears_when_present(self):
        artifacts = {**REQUIRED_ARTIFACTS, "scratchpad": "Remember: Elias prefers dark mode"}
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            artifacts=artifacts,
        )
        assert "Remember: Elias prefers dark mode" in result.content
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
                    "persona": "persona",
                    "ecosystem": "ecosystem",
                },
            )

    def test_missing_ecosystem_raises_error(self):
        with pytest.raises(ValueError, match="Missing required artifact: ecosystem"):
            build_system_prompt(
                available_tools=ToolSet(),
                session_context={},
                artifacts={
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
                    "persona": "persona",
                    "communication_protocol": "",
                    "ecosystem": "ecosystem",
                },
            )

    def test_empty_ecosystem_raises(self):
        with pytest.raises(ValueError, match="Missing required artifact: ecosystem"):
            build_system_prompt(
                available_tools=ToolSet(),
                session_context={},
                artifacts={
                    "persona": "persona",
                    "communication_protocol": "protocol",
                    "ecosystem": "",
                },
            )
