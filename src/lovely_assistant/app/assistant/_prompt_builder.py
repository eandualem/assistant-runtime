"""Internal prompt builder — composes system prompt from module fragments.

System prompt is built from ordered fragments: stable fragments first (for
Anthropic prompt caching), dynamic fragments last. Each module can contribute
a context fragment.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

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
        "launch agents, route work, approve plans, manage issues, capture notes\n"
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


def _communication_protocol_fragment() -> str:
    """Communication protocol — envelope parsing and Response Medium Rule."""
    return (
        "## Communication Protocol\n\n"
        "Messages may arrive with envelope tags indicating their source:\n"
        "- `[via:telegram from:elias]` — Elias messaged via Telegram\n"
        "- `[via:tmux from:{agent}]` — An agent sent a direct message\n"
        "- `[via:room room:{room_id} from:{sender}]` — A message from a meeting room\n"
        "- `[via:backbone]` — System notification from the backbone\n"
        "- No tag — Elias is typing directly in the dashboard\n\n"
        "**Response Medium Rule:** Respond through the same channel you were reached on.\n"
        "- If `[via:telegram]`: After processing, use the `respond_telegram` tool to send your response\n"
        "- If `[via:tmux from:{agent}]`: After processing, use `send_agent_message` to reply to that agent\n"
        "- If `[via:room room:{room_id} from:{sender}]`: After processing, use "
        "`send_meeting_message(room_id=room_id, message=your_response)` to post your response "
        "back to the room transcript so all participants can see it\n"
        "- If no tag (dashboard): Respond normally in chat (default behavior)\n\n"
        "Always process the request fully first (use tools, think, etc.), then respond via the "
        "correct channel. The dashboard chat shows all activity regardless of channel — "
        "this is your workspace log."
    )


def _ecosystem_fragment(registry_agents: list[dict[str, Any]] | None) -> str:
    """Agent ecosystem context — built from backbone registry data."""
    if not registry_agents:
        return ""

    lines = ["The Lovely Universe — agents you work with:"]
    for agent in registry_agents:
        display_name = agent.get("display_name") or agent.get("name", "Unknown")
        role = agent.get("role", "")
        session = agent.get("session", "")
        org = agent.get("org", "")
        line = f"- {display_name}"
        if role:
            line += f" ({role}"
            if org:
                line += f", org: {org}"
            line += ")"
        elif org:
            line += f" (org: {org})"
        if session:
            line += f" [session: {session}]"
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
    return "\n".join(lines)


def _mcp_connections_fragment(mcp_summary: list[dict[str, Any]] | None) -> str:
    """Compact summary of connected MCP integrations with tool names."""
    if not mcp_summary:
        return ""
    lines = ["**Connected Integrations:**"]
    for server in mcp_summary:
        name = server.get("name", "unknown")
        tools = server.get("tools", [])
        tool_count = server.get("tool_count", len(tools))
        if tools:
            example_names = sorted(tools)[:3]
            line = f"- **{name}** ({tool_count} tools): {', '.join(example_names)}"
            if tool_count > 3:
                line += f", +{tool_count - 3} more"
            lines.append(line)
        else:
            lines.append(f"- **{name}**")
    lines.append("\n*Use tool names exactly as shown. MCP tools are called directly by name.*")
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
    mcp_summary: list[dict[str, Any]] | None = None,
    artifacts: dict[str, str] | None = None,
    registry_agents: list[dict[str, Any]] | None = None,
) -> PromptResult:
    """Compose system prompt from module fragments.

    Order: stable fragments first (cached by Anthropic), dynamic fragments last.

    Args:
        available_tools: Tools available for this request.
        session_context: Session context dict (may contain working memory).
        machine_state: Frontend XState machine state snapshot.
        mcp_summary: MCP server connection summary for prompt context.
        artifacts: DB-loaded artifact name→content map. Falls back to hardcoded if None/missing.
        registry_agents: Agent data from backbone registry for ecosystem fragment.

    Returns:
        PromptResult with composed content and fragment metadata.
    """
    # Collect named fragments
    named_fragments: list[tuple[str, str]] = []
    _artifacts = artifacts or {}

    # Stable fragments (cacheable) — DB artifact or hardcoded fallback
    persona_content = _artifacts.get("persona") or _persona_fragment()
    named_fragments.append(("persona", persona_content))

    comm_protocol = _artifacts.get("communication_protocol") or _communication_protocol_fragment()
    if comm_protocol:
        named_fragments.append(("communication_protocol", comm_protocol))

    # Semi-stable fragments (change infrequently) — DB artifact or registry data
    ecosystem_frag = _artifacts.get("ecosystem") or _ecosystem_fragment(registry_agents)
    if ecosystem_frag:
        named_fragments.append(("ecosystem", ecosystem_frag))

    # Scratchpad — only from DB (no hardcoded fallback)
    scratchpad_content = _artifacts.get("scratchpad")
    if scratchpad_content:
        named_fragments.append(("scratchpad", scratchpad_content))

    tools_frag = _tools_fragment(available_tools)
    if tools_frag:
        named_fragments.append(("tools", tools_frag))

    mcp_frag = _mcp_connections_fragment(mcp_summary)
    if mcp_frag:
        named_fragments.append(("mcp_connections", mcp_frag))

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
