"""Internal prompt builder — composes system prompt from module fragments.

System prompt is built from ordered fragments: stable fragments first (for
Anthropic prompt caching), dynamic fragments last. Each module can contribute
a context fragment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from assistant_runtime.app.assistant.models import PromptResult
from assistant_runtime.services.history.models import WorkingMemory
from assistant_runtime.services.tools.models import ToolSet

# --- Artifact catalog ---


@dataclass(frozen=True)
class ArtifactDefinition:
    """Definition of a first-class prompt artifact."""

    name: str
    role_boundary: str
    required: bool
    scratchpad_special_case: bool = False


ARTIFACT_CATALOG: tuple[ArtifactDefinition, ...] = (
    ArtifactDefinition(
        name="soul",
        role_boundary="enduring purpose, values, non-negotiables, deepest identity guidance",
        required=True,
    ),
    ArtifactDefinition(
        name="persona",
        role_boundary="style, stance, behavioral voice",
        required=True,
    ),
    ArtifactDefinition(
        name="communication_protocol",
        role_boundary="interaction and routing rules",
        required=True,
    ),
    ArtifactDefinition(
        name="ecosystem",
        role_boundary="world model, roles, org structure, system topology",
        required=True,
    ),
    ArtifactDefinition(
        name="scratchpad",
        role_boundary="short-lived operational memory",
        required=False,
        scratchpad_special_case=True,
    ),
)
_ARTIFACTS_BY_NAME: dict[str, ArtifactDefinition] = {item.name: item for item in ARTIFACT_CATALOG}
_REQUIRED_ARTIFACTS = tuple(item.name for item in ARTIFACT_CATALOG if item.required)
REQUIRED_ARTIFACT_NAMES = _REQUIRED_ARTIFACTS
KNOWN_ARTIFACT_NAMES = tuple(item.name for item in ARTIFACT_CATALOG)
SCRATCHPAD_ARTIFACT_NAME = next(
    item.name for item in ARTIFACT_CATALOG if item.scratchpad_special_case
)
_ARTIFACT_ORDER = {item.name: idx for idx, item in enumerate(ARTIFACT_CATALOG)}


def artifact_sort_key(name: str) -> tuple[int, str]:
    """Sort first-class artifacts in canonical prompt/dashboard order."""
    return (_ARTIFACT_ORDER.get(name, len(ARTIFACT_CATALOG)), name)


def is_known_artifact_name(name: str) -> bool:
    """Return whether the name is a first-class artifact."""
    return name in _ARTIFACTS_BY_NAME


def known_artifact_names_text() -> str:
    """Human-readable list of known first-class artifact names."""
    return ", ".join(KNOWN_ARTIFACT_NAMES)


def artifact_role_boundaries_text() -> str:
    """Human-readable summary of the artifact role split."""
    return "; ".join(f"{item.name} = {item.role_boundary}" for item in ARTIFACT_CATALOG)


# --- Fragment builders ---


def _datetime_fragment() -> str:
    """Current date and time context."""
    now = datetime.now(UTC)
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M UTC')} ({now.strftime('%A')})"


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


def _format_value(value: Any, indent: int = 2) -> str:
    """Format a value readably for the LLM — lists as bullets, dicts as key-value pairs."""
    prefix = " " * indent
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return "(none)"
        lines: list[str] = []
        for item in value:
            if isinstance(item, dict):
                # Compact dict items on one line: "key1: val1, key2: val2"
                parts = [f"{k}: {_format_value(v, 0)}" for k, v in item.items()]
                lines.append(f"{prefix}- {', '.join(parts)}")
            else:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines)
    if isinstance(value, dict):
        if not value:
            return "(empty)"
        parts = [f"{k}: {_format_value(v, 0)}" for k, v in value.items()]
        return ", ".join(parts)
    return str(value)


def _render_generic_data(data: dict[str, Any]) -> list[str]:
    """Render page data as readable key-value pairs. No truncation — trust the dashboard."""
    lines: list[str] = []
    for key, value in data.items():
        formatted = _format_value(value)
        if "\n" in formatted:
            # Multi-line value (list of items) — header then indented items
            lines.append(f"  {key}:")
            lines.append(formatted)
        else:
            lines.append(f"  {key}: {formatted}")
    return lines


def _render_background(background: dict[str, Any]) -> str:
    """Render background summaries as a one-liner.

    Expected structure per entry: {"state": "idle", "summary": {"entityCount": 5}}.
    Falls back gracefully for flat dicts.
    """
    if not background:
        return ""
    parts = []
    for key, value in background.items():
        if isinstance(value, dict):
            state = value.get("state", "")
            summary = value.get("summary", {})
            # If no "summary" sub-dict, treat the whole dict as summary (flat structure)
            if not isinstance(summary, dict):
                summary = {}
            if not summary and "state" not in value:
                summary = value
            summary_items = [f"{v} {k}" for k, v in summary.items() if isinstance(v, (int, float))]
            label = f"{key} ({state})" if state else key
            if summary_items:
                label += f" — {', '.join(summary_items)}"
            parts.append(label)
        else:
            parts.append(f"{key} ({value})")
    if not parts:
        return ""
    return f"Background: {' | '.join(parts)}"


def _render_machines(machines: dict[str, Any]) -> str:
    """Serialize page machines as-is for the LLM prompt.

    The dashboard already curates what it sends — event types, guards,
    transition targets, definitions, and context are all needed by the
    assistant to call ``ui_send_event`` correctly. No transformation,
    no field extraction — just faithful JSON serialization.
    """
    if not machines:
        return ""

    return "Page machines:\n```json\n" + json.dumps(machines, indent=2, default=str) + "\n```"


def _dashboard_context_fragment(machine_state: dict[str, Any] | None) -> str:
    """Dashboard context from frontend machine state."""
    if not machine_state:
        return ""

    active_page = machine_state.get("active_page")
    if not active_page:
        logger.warning("machine_state present but missing active_page — possible dashboard bug")
        return ""

    sections: list[str] = []

    # Page header
    page_name = active_page.get("name", "unknown")
    sections.append(f"You are on the {page_name} page.")

    # Page data (generic structure-aware rendering for all pages)
    page_data = active_page.get("data", {})
    if page_data:
        page_lines = _render_generic_data(page_data)
        if page_lines:
            sections.extend(page_lines)

    # Page machines (XState machine states, context, transitions)
    machines = active_page.get("machines", {})
    if machines:
        machines_text = _render_machines(machines)
        if machines_text:
            sections.append(machines_text)

    # Navigation targets
    navigation = machine_state.get("navigation", [])
    if navigation:
        nav_lines = ["Navigation:"]
        for target in navigation:
            if isinstance(target, dict):
                name = target.get("name", "unknown")
                desc = target.get("description", "")
                nav_lines.append(f"- {name} — {desc}" if desc else f"- {name}")
            else:
                nav_lines.append(f"- {target}")
        if len(nav_lines) > 1:
            sections.append("\n".join(nav_lines))

    # Available actions (nested inside active_page per PageAwareContext)
    available_actions = active_page.get("available_actions", [])
    if available_actions:
        action_lines = ["Available UI actions:"]
        for action in available_actions:
            if isinstance(action, dict):
                event_type = action.get("event_type", "")
                label = action.get("label", "")
                line = f"- {event_type} ({label})" if label else f"- {event_type}"
                # Include parameter definitions so the agent knows the payload shape
                params = action.get("params", [])
                if params:
                    param_strs = []
                    for p in params:
                        if isinstance(p, dict):
                            p_name = p.get("name", "?")
                            p_type = p.get("type", "")
                            p_req = p.get("required", False)
                            desc = f"{p_name} ({p_type}" if p_type else p_name
                            if p_type:
                                desc += ", required)" if p_req else ")"
                            elif p_req:
                                desc += " (required)"
                            param_strs.append(desc)
                    if param_strs:
                        line += f" — params: {', '.join(param_strs)}"
                action_lines.append(line)
            else:
                action_lines.append(f"- {action}")
        if len(action_lines) > 1:
            sections.append("\n".join(action_lines))

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

    # Any page: machines in error states
    machines = active_page.get("machines", {})
    if machines:
        error_machines = [
            name
            for name, m in machines.items()
            if isinstance(m, dict) and "error" in str(m.get("current_state", "")).lower()
        ]
        if error_machines:
            hints.append(
                f"Machines in error state ({', '.join(error_machines)}) — "
                "investigate or suggest recovery actions."
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
    artifacts: dict[str, str],
) -> PromptResult:
    """Compose system prompt from module fragments.

    Order: stable fragments first (cached by Anthropic), dynamic fragments last.
    Artifact role boundary:
    - soul: enduring purpose, values, non-negotiables, deepest identity guidance
    - persona: style, stance, behavioral voice
    - communication_protocol: interaction and routing rules
    - ecosystem: world model, roles, org structure, system topology
    - scratchpad: short-lived operational memory

    Args:
        available_tools: Tools available for this request.
        session_context: Session context dict (may contain working memory).
        machine_state: Frontend XState machine state snapshot.
        mcp_summary: MCP server connection summary for prompt context.
        artifacts: DB-loaded artifact name→content map. Must contain soul,
            persona, communication_protocol, and ecosystem.

    Returns:
        PromptResult with composed content and fragment metadata.

    Raises:
        ValueError: If a required artifact is missing or empty.
    """
    # Validate required artifacts
    for name in _REQUIRED_ARTIFACTS:
        if not artifacts.get(name):
            raise ValueError(f"Missing required artifact: {name}")

    # Collect named fragments
    named_fragments: list[tuple[str, str]] = []

    # Stable fragments (cacheable) — from DB artifacts
    for artifact in ARTIFACT_CATALOG:
        if artifact.required:
            named_fragments.append((artifact.name, artifacts[artifact.name]))

    # Scratchpad — optional
    scratchpad_content = artifacts.get(SCRATCHPAD_ARTIFACT_NAME)
    if scratchpad_content:
        named_fragments.append((SCRATCHPAD_ARTIFACT_NAME, scratchpad_content))

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
