"""Tests for the system prompt builder."""

from lovely_assistant.app.assistant._prompt_builder import (
    _dashboard_context_fragment,
    _datetime_fragment,
    _ecosystem_fragment,
    _persona_fragment,
    _smart_hints,
    _tools_fragment,
    _working_memory_fragment,
    build_system_prompt,
)
from lovely_assistant.app.assistant.models import PromptResult
from lovely_assistant.services.history.models import WorkingMemory
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet


class TestPersonaFragment:
    def test_contains_identity(self):
        frag = _persona_fragment()
        assert "Jarvis" in frag
        assert "operational nervous system" in frag

    def test_contains_behavioral_instruction(self):
        frag = _persona_fragment()
        assert "direct" in frag.lower()
        assert "anticipat" in frag.lower()

    def test_action_oriented(self):
        frag = _persona_fragment()
        assert "force multiplier" in frag.lower()
        assert "agency" in frag.lower()

    def test_assistant_first_model(self):
        frag = _persona_fragment()
        assert "everything flows through you" in frag.lower()


class TestEcosystemFragment:
    def test_contains_agents(self):
        frag = _ecosystem_fragment()
        assert "Leo" in frag
        assert "Ike" in frag
        assert "Feynman" in frag

    def test_contains_routing(self):
        frag = _ecosystem_fragment()
        assert "Route:" in frag

    def test_format(self):
        frag = _ecosystem_fragment()
        assert frag.startswith("The Lovely Universe")
        assert len(frag) > 0

    def test_all_agents_present(self):
        frag = _ecosystem_fragment()
        expected = [
            "Leo",
            "Ike",
            "Hamilton",
            "Curie",
            "Bell",
            "Feynman",
            "Ada",
            "Brunel",
            "Coding Agents",
        ]
        for name in expected:
            assert name in frag, f"{name} not found in ecosystem fragment"


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

    def test_frontend_tools_marked(self):
        ts = ToolSet(
            frontend_tools=[
                ToolDefinition(
                    name="navigate",
                    description="Send notification",
                    parameters_schema={},
                    category=ToolCategory.FRONTEND,
                )
            ]
        )
        frag = _tools_fragment(ts)
        assert "(UI)" in frag


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


class TestDashboardContextFragment:
    def test_none_state(self):
        assert _dashboard_context_fragment(None) == ""

    def test_empty_dict(self):
        assert _dashboard_context_fragment({}) == ""

    def test_missing_active_page(self):
        assert _dashboard_context_fragment({"something": "else"}) == ""

    def test_tasks_page_renders_issues(self):
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
        assert "#42 Fix bug" in frag
        assert "#43 Add feature" in frag
        assert "2 issues loaded" in frag

    def test_tasks_page_renders_selected_issue(self):
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
        assert "Important bug" in frag
        assert "This needs fixing" in frag
        assert "Comments: 3" in frag

    def test_agents_page_renders_sessions(self):
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
        assert "leo: idle" in frag
        assert "ike: processing" in frag
        assert "Working on #50" in frag

    def test_sessions_page_renders(self):
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
        assert "feynman: idle" in frag

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

    def test_background_summary(self):
        state = {
            "active_page": {"name": "tasks", "data": {}},
            "background": {
                "agents": {"online": 5, "offline": 2},
                "sessions": {"active": 3},
            },
        }
        frag = _dashboard_context_fragment(state)
        assert "Background:" in frag
        assert "agents" in frag
        assert "sessions" in frag

    def test_available_actions(self):
        state = {
            "active_page": {"name": "tasks", "data": {}},
            "available_actions": [
                {"event_type": "NAVIGATE", "label": "Go to agents"},
                {"event_type": "REFRESH", "label": ""},
            ],
        }
        frag = _dashboard_context_fragment(state)
        assert "Available UI actions:" in frag
        assert "NAVIGATE" in frag
        assert "REFRESH" in frag

    def test_empty_actions(self):
        state = {
            "active_page": {"name": "tasks", "data": {}},
            "available_actions": [],
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
        )
        assert "Jarvis" in result.content

    def test_contains_datetime(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
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
        )
        assert "agents page" in result.content

    def test_includes_working_memory(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={"working_memory": WorkingMemory(active_goal="Deploy v2")},
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
        )
        assert "Hints:" in result.content
        assert "Idle agents" in result.content
        assert "plan approval" in result.content

    def test_fragments_separated(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
        )
        # Fragments joined by double newline
        assert "\n\n" in result.content

    def test_returns_prompt_result(self):
        result = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
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
        )
        fragment_names = [f["name"] for f in result.fragments]
        # Minimum: persona + ecosystem + datetime (always present)
        assert "persona" in fragment_names
        assert "ecosystem" in fragment_names
        assert "datetime" in fragment_names
        # No tools, dashboard_context, smart_hints, or working_memory
        assert "tools" not in fragment_names
        assert "dashboard_context" not in fragment_names
        assert "smart_hints" not in fragment_names
        assert "working_memory" not in fragment_names
