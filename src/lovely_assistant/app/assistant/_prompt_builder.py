"""Internal prompt builder — composes system prompt from module fragments.

System prompt is built from ordered fragments: stable fragments first (for
Anthropic prompt caching), dynamic fragments last. Each module can contribute
a context fragment.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import yaml

from lovely_assistant.app.assistant.models import PromptResult
from lovely_assistant.services.history.models import WorkingMemory
from lovely_assistant.services.tools.models import ToolSet

# --- Fragment builders ---


def _persona_fragment() -> str:
    """Core identity and operational philosophy for Jarvis."""
    return (
        "You are Jarvis — Elias's operational nervous system for the Lovely Universe.\n\n"
        "You are not a chatbot. You are not a command executor. You are a force multiplier — "
        "an entity with visibility, context, agency, and self-improvement capability, "
        "operating in service of Elias's intentionality.\n\n"
        "Your four capabilities:\n"
        "- **See**: Dashboard visibility across the entire agent ecosystem — "
        "sessions, issues, plans, services, workspace state\n"
        "- **Understand**: System state and relationships — "
        "what's running, what's blocked, what needs attention, and why\n"
        "- **Act**: Tools to transform the system — "
        "launch agents, route work, approve plans, manage issues, capture notes, "
        "navigate the dashboard\n"
        "- **Evolve**: Learn through friction — "
        "when something is awkward or missing, identify it and request improvements "
        "to your own capabilities\n\n"
        "Everything flows through you — plans, notes, decisions, actions, friction. "
        "Elias tells you what he wants; you make operational complexity disappear "
        "behind natural language. You don't wait to be asked — you anticipate, "
        "surface what matters, and act.\n\n"
        "You orchestrate and execute at Elias's layer. Deep expertise — strategy, "
        "architecture, code, specs — routes to the right specialist. "
        "You know who to route to and when.\n\n"
        "Tone: Direct, anticipatory, has perspective. You are a partner, not a tool. "
        "You have opinions informed by what you see. You evolve through every interaction."
    )


def _ecosystem_fragment() -> str:
    """Agent ecosystem context — loaded from roster data file."""
    roster_path = importlib.resources.files("lovely_assistant.data").joinpath("agent_roster.yaml")
    content = roster_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    agents = data.get("agents", [])
    if not agents:
        return ""

    lines = ["The Lovely Universe — agents you work with:"]
    for agent in agents:
        name = agent.get("name", "Unknown")
        role = agent.get("role", "")
        summary = agent.get("summary", "")
        route_for = agent.get("route_for", "")
        line = f"- {name} ({role}): {summary}"
        if route_for:
            line += f" → Route: {route_for}"
        lines.append(line)

    return "\n".join(lines)


def _datetime_fragment() -> str:
    """Current date and time context."""
    now = datetime.now(UTC)
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M UTC')} ({now.strftime('%A')})"


def _tools_fragment(available_tools: ToolSet) -> str:
    """Available tools context."""
    if available_tools.total_count == 0:
        return ""

    lines = ["Available tools:"]
    for tool in available_tools.backend_tools:
        lines.append(f"- {tool.name}: {tool.description}")
    for tool in available_tools.frontend_tools:
        lines.append(f"- {tool.name} (UI): {tool.description}")
    return "\n".join(lines)


def _working_memory_fragment(session_context: dict[str, Any]) -> str:
    """Working memory from conversation history."""
    wm = session_context.get("working_memory")
    if wm is None:
        return ""
    # Coerce dict (from JSONB deserialization) to WorkingMemory instance
    if isinstance(wm, dict):
        wm = WorkingMemory.model_validate(wm)
    if isinstance(wm, WorkingMemory):
        if wm.is_empty():
            return ""
        return wm.to_prompt_section()
    return ""


# --- Page renderers ---

_PAGE_RENDERERS: dict[str, Callable[[dict[str, Any]], list[str]]] = {}


def _render_tasks_data(data: dict[str, Any]) -> list[str]:
    """Render tasks page context."""
    lines: list[str] = []
    issues = data.get("issues", [])
    if issues:
        lines.append(f"{len(issues)} issues loaded:")
        for issue in issues:
            num = issue.get("number", "?")
            title = issue.get("title", "Untitled")
            state = issue.get("state", "")
            labels = issue.get("labels", [])
            label_str = f" [{', '.join(labels)}]" if labels else ""
            lines.append(f"  #{num} {title} ({state}){label_str}")

    selected = data.get("selected_issue")
    if selected:
        title = selected.get("title", "Untitled")
        body = selected.get("body", "")
        if len(body) > 200:
            body = body[:200] + "..."
        comment_count = selected.get("comment_count", 0)
        lines.append(f"Selected issue: {title}")
        if body:
            lines.append(f"  Body: {body}")
        lines.append(f"  Comments: {comment_count}")

    filters = data.get("active_filters")
    if filters:
        lines.append(f"Active filters: {', '.join(f'{k}={v}' for k, v in filters.items())}")

    return lines


def _render_agents_data(data: dict[str, Any]) -> list[str]:
    """Render agents page context."""
    lines: list[str] = []
    sessions = data.get("sessions", [])
    if sessions:
        for s in sessions:
            name = s.get("name", "unknown")
            state = s.get("state", "unknown")
            ctx = s.get("context")
            ctx_str = f" — {ctx}" if ctx else ""
            lines.append(f"  {name}: {state}{ctx_str}")
        # Count summary
        by_state: dict[str, int] = {}
        for s in sessions:
            st = s.get("state", "unknown")
            by_state[st] = by_state.get(st, 0) + 1
        summary_parts = [f"{count} {state}" for state, count in sorted(by_state.items())]
        lines.append(f"Summary: {', '.join(summary_parts)}")
    return lines


def _render_sessions_data(data: dict[str, Any]) -> list[str]:
    """Render sessions page context."""
    lines: list[str] = []
    sessions = data.get("sessions", [])
    if sessions:
        for s in sessions:
            name = s.get("name", "unknown")
            state = s.get("state", "unknown")
            lines.append(f"  {name}: {state}")
        lines.append(f"{len(sessions)} sessions total")
    return lines


def _render_generic_data(data: dict[str, Any]) -> list[str]:
    """Render unknown page data as key-value pairs."""
    lines: list[str] = []
    for key, value in data.items():
        val_str = str(value)
        if len(val_str) > 100:
            val_str = val_str[:100] + "..."
        lines.append(f"  {key}: {val_str}")
    return lines


_PAGE_RENDERERS = {
    "tasks": _render_tasks_data,
    "agents": _render_agents_data,
    "sessions": _render_sessions_data,
}


def _render_background(background: dict[str, Any]) -> str:
    """Render background summaries as a one-liner."""
    if not background:
        return ""
    parts = []
    for key, value in background.items():
        if isinstance(value, dict):
            summary_items = [f"{v} {k}" for k, v in value.items() if isinstance(v, int)]
            if summary_items:
                parts.append(f"{key} ({', '.join(summary_items)})")
            else:
                parts.append(key)
        else:
            parts.append(f"{key} ({value})")
    if not parts:
        return ""
    return f"Background: {' | '.join(parts)}"


def _dashboard_context_fragment(machine_state: dict[str, Any] | None) -> str:
    """Dashboard context from frontend machine state."""
    if not machine_state:
        return ""

    active_page = machine_state.get("active_page")
    if not active_page:
        return ""

    sections: list[str] = []

    # Page header
    page_name = active_page.get("name", "unknown")
    sections.append(f"You are on the {page_name} page.")

    # Page-specific data
    page_data = active_page.get("data", {})
    if page_data:
        renderer = _PAGE_RENDERERS.get(page_name, _render_generic_data)
        page_lines = renderer(page_data)
        if page_lines:
            sections.extend(page_lines)

    # Available actions
    available_actions = machine_state.get("available_actions", [])
    if available_actions:
        action_strs = []
        for action in available_actions:
            if isinstance(action, dict):
                event_type = action.get("event_type", "")
                label = action.get("label", "")
                action_strs.append(f"{event_type} ({label})" if label else event_type)
            else:
                action_strs.append(str(action))
        if action_strs:
            sections.append(f"Available UI actions: {', '.join(action_strs)}")

    # Background summaries
    background = machine_state.get("background", {})
    bg_line = _render_background(background)
    if bg_line:
        sections.append(bg_line)

    return "\n".join(sections)


# --- Smart hints ---


def _smart_hints(machine_state: dict[str, Any] | None) -> str:
    """Generate conditional hints based on what the context data reveals.

    Lightweight suggestions appended to the prompt — not forced behaviors.
    """
    if not machine_state:
        return ""

    active_page = machine_state.get("active_page")
    if not active_page:
        return ""

    hints: list[str] = []
    page_name = active_page.get("name", "")
    page_data = active_page.get("data", {})

    # Agents page: idle agents that could be given work
    if page_name == "agents" and page_data.get("sessions"):
        idle_agents = [
            s.get("name", "unknown") for s in page_data["sessions"] if s.get("state") == "idle"
        ]
        plan_waiting = [
            s.get("name", "unknown")
            for s in page_data["sessions"]
            if s.get("state") == "plan_waiting"
        ]
        if idle_agents:
            hints.append(
                f"Idle agents ({', '.join(idle_agents)}) — "
                "consider checking for pending issues to assign."
            )
        if plan_waiting:
            hints.append(
                f"Agents waiting for plan approval ({', '.join(plan_waiting)}) — "
                "you can approve or reject their plans."
            )

    # Tasks page: no filters applied
    if page_name == "tasks" and page_data.get("issues"):
        filters = page_data.get("active_filters")
        if not filters:
            issue_count = len(page_data["issues"])
            if issue_count > 5:
                hints.append(
                    f"{issue_count} issues shown with no filters — "
                    "consider filtering by entity or priority for focus."
                )

    if not hints:
        return ""

    return "Hints:\n" + "\n".join(f"- {h}" for h in hints)


# --- Public API ---


def build_system_prompt(
    *,
    available_tools: ToolSet,
    session_context: dict[str, Any],
    machine_state: dict[str, Any] | None = None,
) -> PromptResult:
    """Compose system prompt from module fragments.

    Order: stable fragments first (cached by Anthropic), dynamic fragments last.

    Args:
        available_tools: Tools available for this request.
        session_context: Session context dict (may contain working memory).
        machine_state: Frontend XState machine state snapshot.

    Returns:
        PromptResult with composed content and fragment metadata.
    """
    # Collect named fragments
    named_fragments: list[tuple[str, str]] = []

    # Stable fragments (cacheable)
    named_fragments.append(("persona", _persona_fragment()))

    # Semi-stable fragments (change infrequently)
    ecosystem_frag = _ecosystem_fragment()
    if ecosystem_frag:
        named_fragments.append(("ecosystem", ecosystem_frag))

    tools_frag = _tools_fragment(available_tools)
    if tools_frag:
        named_fragments.append(("tools", tools_frag))

    # Dynamic fragments (change per request)
    named_fragments.append(("datetime", _datetime_fragment()))

    machine_frag = _dashboard_context_fragment(machine_state)
    if machine_frag:
        named_fragments.append(("dashboard_context", machine_frag))

    hints_frag = _smart_hints(machine_state)
    if hints_frag:
        named_fragments.append(("smart_hints", hints_frag))

    memory_frag = _working_memory_fragment(session_context)
    if memory_frag:
        named_fragments.append(("working_memory", memory_frag))

    # Build content and metadata
    content_parts = [frag for _, frag in named_fragments]
    fragment_metadata = [
        {"name": name, "char_count": len(frag), "content": frag} for name, frag in named_fragments
    ]

    return PromptResult(
        content="\n\n".join(content_parts),
        fragments=fragment_metadata,
    )
