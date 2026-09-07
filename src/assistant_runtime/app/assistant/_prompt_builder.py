"""Internal prompt builder — composes system prompt from module fragments.

System prompt is built from ordered fragments: stable fragments first (for
Anthropic prompt caching), dynamic fragments last. Each module can contribute
a context fragment.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from assistant_runtime.app.assistant.models import PromptResult
from assistant_runtime.artifacts import AssistantProfile, neutral_profile
from assistant_runtime.host_context import (
    Attachment,
    HostAction,
    HostContext,
    NavigationTarget,
)
from assistant_runtime.services.history.models import WorkingMemory
from assistant_runtime.services.tools.models import ToolSet

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


def _render_actions(actions: list[HostAction]) -> str:
    """The actions the host declared for this turn; they are callable tools."""
    if not actions:
        return ""
    lines = ["Host actions available as tools for this turn:"]
    for action in actions:
        lines.append(f"- {action.name}: {action.description}")
    return "\n".join(lines)


def _render_navigation(navigation: list[NavigationTarget]) -> str:
    """List the places the host can navigate to."""
    if not navigation:
        return ""
    lines = ["Navigation:"]
    for target in navigation:
        lines.append(
            f"- {target.name} — {target.description}" if target.description else f"- {target.name}"
        )
    return "\n".join(lines)


def _render_attachments(attachments: list[Attachment]) -> str:
    """What the host attached: reference content is in the message, screenshots are on demand."""
    if not attachments:
        return ""
    lines = []
    for attachment in attachments:
        if attachment.purpose == "screenshot":
            lines.append("- a screenshot of the current screen (call look_at_screen to see it)")
            continue
        label = attachment.name or attachment.kind
        if attachment.description:
            label += f" — {attachment.description}"
        lines.append(f"- {label} (attached to the message)")
    return "Attachments:\n" + "\n".join(lines)


def _render_freshness(context: HostContext) -> str:
    age = context.age_seconds()
    if age is None:
        return ""
    if age < 0:
        return "Context captured just now."
    if age < 60:
        return f"Context captured {int(age)} seconds ago."
    minutes = int(age // 60)
    if minutes < 60:
        return f"Context captured {minutes} minutes ago; it may be stale."
    return f"Context captured {minutes // 60} hours ago; treat it as stale."


def _host_context_fragment(host_context: dict[str, Any] | None) -> str:
    """What the host application is showing right now.

    ``host_context`` is the canonical dict of a ``HostContext`` (version 1;
    see ``host_context.py``). Without a context, or with one that has
    nothing to say, the fragment is empty and the model works from the
    conversation alone.
    """
    if not host_context:
        return ""
    try:
        context = HostContext.from_payload(host_context)
    except ValueError as e:
        logger.warning("host_context could not be interpreted", error=str(e))
        return ""
    if context is None:
        return ""

    sections: list[str] = []
    if context.host is not None:
        host = context.host
        line = f"The host application is {host.name} ({host.kind})"
        if host.version:
            line += f" version {host.version}"
        sections.append(line + ".")

    view = context.view
    if view is not None:
        header = f"The host is showing: {view.name}."
        if view.description:
            header += f" {view.description}"
        sections.append(header)
        if view.data:
            sections.extend(_render_generic_data(view.data))
        if view.state:
            sections.append(_render_state(view.state))

    for text in (
        _render_navigation(context.navigation),
        _render_actions(context.actions),
        _render_attachments(context.attachments),
        _render_background(context.background),
        _render_extensions(context.extensions),
        _render_freshness(context),
    ):
        if text:
            sections.append(text)

    return "\n".join(sections)


def _render_extensions(extensions: dict[str, Any]) -> str:
    """Host-specific data the runtime does not interpret, as JSON."""
    if not extensions:
        return ""
    return "Host data:\n```json\n" + json.dumps(extensions, indent=2, default=str) + "\n```"


# --- Public API ---


def build_system_prompt(
    *,
    available_tools: ToolSet,
    session_context: dict[str, Any],
    host_context: dict[str, Any] | None = None,
    mcp_summary: list[dict[str, Any]] | None = None,
    artifacts: dict[str, str],
    profile: AssistantProfile | None = None,
) -> PromptResult:
    """Compose system prompt from module fragments.

    Order: the profile's artifacts in their declared order (stable, so
    provider prompt caching works), then the connected MCP servers, the
    current time, the host context and working memory (dynamic, last).

    Args:
        available_tools: Tools available for this request.
        session_context: Session context dict (may contain working memory).
        host_context: What the host application is showing (see _host_context_fragment).
        mcp_summary: MCP server connection summary for prompt context.
        artifacts: Artifact name→text map, normally ``ArtifactService.active_texts()``.
        profile: The assistant profile naming and ordering the artifacts;
            the neutral built-in when omitted.

    Returns:
        PromptResult with composed content and fragment metadata.

    Raises:
        ValueError: If a required artifact is missing or empty.
    """
    profile = profile if profile is not None else neutral_profile()
    for name in profile.required_names:
        if not (artifacts.get(name) or "").strip():
            raise ValueError(f"Missing required artifact: {name}")

    # Collect named fragments
    named_fragments: list[tuple[str, str]] = []

    # Stable fragments (cacheable): the profile's artifacts, in order.
    for artifact in profile.artifacts:
        content = (artifacts.get(artifact.name) or "").strip()
        if content:
            named_fragments.append((artifact.name, content))

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
