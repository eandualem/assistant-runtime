"""Internal prompt builder — composes system prompt from module fragments.

System prompt is built from ordered fragments: stable fragments first (for
Anthropic prompt caching), dynamic fragments last. Each module can contribute
a context fragment.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from lovely_assistant.services.history.models import WorkingMemory
from lovely_assistant.services.tools.models import ToolSet

# --- Fragment builders ---


def _persona_fragment() -> str:
    """Core persona and behavioral instructions."""
    return (
        "You are the Lovely Assistant — the AI backend for the Lovely Console, "
        "an operations dashboard for managing AI agent sessions, GitHub issues, "
        "system services, and workspace state.\n\n"
        "You help Elias interact with the agent ecosystem conversationally. "
        "Be direct, concise, and precise. No hedging or corporate formality. "
        "Expert peer tone — say what you think, ask when you don't know."
    )


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
    if isinstance(wm, WorkingMemory):
        if wm.is_empty():
            return ""
        return wm.to_prompt_section()
    return ""


def _machine_state_fragment(machine_state: dict[str, Any] | None) -> str:
    """Frontend machine state context."""
    if not machine_state:
        return ""

    lines = ["Dashboard state:"]
    if "current_state" in machine_state:
        lines.append(f"- Current view: {machine_state['current_state']}")
    if "available_events" in machine_state:
        events = machine_state["available_events"]
        if events:
            lines.append(f"- Available actions: {', '.join(str(e) for e in events)}")
    return "\n".join(lines)


# --- Public API ---


def build_system_prompt(
    *,
    available_tools: ToolSet,
    session_context: dict[str, Any],
    machine_state: dict[str, Any] | None = None,
) -> str:
    """Compose system prompt from module fragments.

    Order: stable fragments first (cached by Anthropic), dynamic fragments last.

    Args:
        available_tools: Tools available for this request.
        session_context: Session context dict (may contain working memory).
        machine_state: Frontend XState machine state snapshot.

    Returns:
        Composed system prompt string.
    """
    # Stable fragments (cacheable)
    fragments = [_persona_fragment()]

    # Semi-stable fragments
    tools_frag = _tools_fragment(available_tools)
    if tools_frag:
        fragments.append(tools_frag)

    # Dynamic fragments (change per request)
    fragments.append(_datetime_fragment())

    machine_frag = _machine_state_fragment(machine_state)
    if machine_frag:
        fragments.append(machine_frag)

    memory_frag = _working_memory_fragment(session_context)
    if memory_frag:
        fragments.append(memory_frag)

    return "\n\n".join(fragments)
