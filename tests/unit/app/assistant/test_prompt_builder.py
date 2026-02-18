"""Tests for the system prompt builder."""

from lovely_assistant.app.assistant._prompt_builder import (
    _datetime_fragment,
    _machine_state_fragment,
    _persona_fragment,
    _tools_fragment,
    _working_memory_fragment,
    build_system_prompt,
)
from lovely_assistant.services.history.models import WorkingMemory
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet


class TestPersonaFragment:
    def test_contains_identity(self):
        frag = _persona_fragment()
        assert "Lovely Assistant" in frag

    def test_contains_behavioral_instruction(self):
        frag = _persona_fragment()
        assert "direct" in frag.lower() or "concise" in frag.lower()


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
                    name="ui_notify",
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


class TestMachineStateFragment:
    def test_none_state(self):
        assert _machine_state_fragment(None) == ""

    def test_empty_state(self):
        assert _machine_state_fragment({}) == ""

    def test_with_current_state(self):
        frag = _machine_state_fragment({"current_state": "dashboard.agents"})
        assert "dashboard.agents" in frag

    def test_with_available_events(self):
        frag = _machine_state_fragment({"available_events": ["NAVIGATE", "REFRESH"]})
        assert "NAVIGATE" in frag
        assert "REFRESH" in frag


class TestBuildSystemPrompt:
    def test_contains_persona(self):
        prompt = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
        )
        assert "Lovely Assistant" in prompt

    def test_contains_datetime(self):
        prompt = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
        )
        assert "Current time:" in prompt

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
        prompt = build_system_prompt(
            available_tools=ts,
            session_context={},
        )
        assert "test_tool" in prompt

    def test_includes_machine_state(self):
        prompt = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
            machine_state={"current_state": "overview"},
        )
        assert "overview" in prompt

    def test_includes_working_memory(self):
        prompt = build_system_prompt(
            available_tools=ToolSet(),
            session_context={"working_memory": WorkingMemory(active_goal="Deploy v2")},
        )
        assert "Deploy v2" in prompt

    def test_fragments_separated(self):
        prompt = build_system_prompt(
            available_tools=ToolSet(),
            session_context={},
        )
        # Fragments joined by double newline
        assert "\n\n" in prompt
