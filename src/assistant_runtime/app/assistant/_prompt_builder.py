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
    """Sort first-class artifacts in canonical prompt order."""
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
    """Render page data as readable key-value pairs. No truncation; the host curates it."""
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


def _render_state(state: dict[str, Any]) -> str:
    """Serialize the host's page state as JSON, untouched.

    The host curates what it sends (selection, filters, workflow state, the
    events its actions accept). No transformation, no field extraction.
    """
    if not state:
        return ""
    return "Page state:\n```json\n" + json.dumps(state, indent=2, default=str) + "\n```"


def _render_actions(actions: list[Any]) -> str:
    """List the host actions the model may trigger, with their parameter shapes."""
    lines = ["Available host actions:"]
    for action in actions:
        if not isinstance(action, dict):
            lines.append(f"- {action}")
            continue
        event_type = action.get("event_type", "")
        label = action.get("label", "")
        line = f"- {event_type} ({label})" if label else f"- {event_type}"
        param_strs = []
        for p in action.get("params", []) or []:
            if not isinstance(p, dict):
                continue
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
        lines.append(line)
    return "\n".join(lines) if len(lines) > 1 else ""


def _render_navigation(navigation: list[Any]) -> str:
    """List the places the host can navigate to."""
    lines = ["Navigation:"]
    for target in navigation:
        if isinstance(target, dict):
            name = target.get("name", "unknown")
            desc = target.get("description", "")
            lines.append(f"- {name} — {desc}" if desc else f"- {name}")
        else:
            lines.append(f"- {target}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _host_context_fragment(host_context: dict[str, Any] | None) -> str:
    """What the host application is showing right now.

    Shape (every key optional except ``page.name``):
    ``page``: ``{"name", "description", "data", "state", "actions"}``;
    ``navigation``: places the host can navigate to;
    ``background``: summaries of things not on screen.
    """
    if not host_context:
        return ""

    page = host_context.get("page")
    if not isinstance(page, dict) or not page.get("name"):
        logger.warning("host_context present but has no page name")
        return ""

    sections: list[str] = []
    header = f"The host application is showing: {page['name']}."
    description = page.get("description")
    if description:
        header += f" {description}"
    sections.append(header)

    page_data = page.get("data", {})
    if isinstance(page_data, dict) and page_data:
        sections.extend(_render_generic_data(page_data))

    state = page.get("state", {})
    if isinstance(state, dict) and state:
        sections.append(_render_state(state))

    navigation = host_context.get("navigation", [])
    if isinstance(navigation, list) and navigation:
        nav_text = _render_navigation(navigation)
        if nav_text:
            sections.append(nav_text)

    actions = page.get("actions", [])
    if isinstance(actions, list) and actions:
        actions_text = _render_actions(actions)
        if actions_text:
            sections.append(actions_text)

    background = host_context.get("background", {})
    if isinstance(background, dict):
        bg_line = _render_background(background)
        if bg_line:
            sections.append(bg_line)

    return "\n".join(sections)


# --- Public API ---


def build_system_prompt(
    *,
    available_tools: ToolSet,
    session_context: dict[str, Any],
    host_context: dict[str, Any] | None = None,
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
        host_context: What the host application is showing (see _host_context_fragment).
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

    host_frag = _host_context_fragment(host_context)
    if host_frag:
        named_fragments.append(("host_context", host_frag))

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
