"""Capabilities: what the assistant can do, independent of who does it.

Each module declares the tools for one capability and the Protocol a
provider implements to serve it. ``register_capabilities`` registers the
tools of every capability that has a provider.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from assistant_runtime.services.tools.capabilities.activity import register_activity_tools
from assistant_runtime.services.tools.capabilities.approvals import register_approvals_tools
from assistant_runtime.services.tools.capabilities.issues import register_issues_tools
from assistant_runtime.services.tools.capabilities.library import register_library_tools
from assistant_runtime.services.tools.capabilities.messaging import register_messaging_tools
from assistant_runtime.services.tools.capabilities.notes import register_notes_tools
from assistant_runtime.services.tools.capabilities.peers import register_peers_tools
from assistant_runtime.services.tools.capabilities.reminders import register_reminders_tools
from assistant_runtime.services.tools.capabilities.repositories import (
    register_repositories_tools,
)
from assistant_runtime.services.tools.capabilities.rooms import register_rooms_tools
from assistant_runtime.services.tools.capabilities.workgroups import register_workgroups_tools

if TYPE_CHECKING:
    from assistant_runtime.services.tools._registry import ToolRegistry

CAPABILITIES: dict[str, Callable[[ToolRegistry, Any], None]] = {
    "notes": register_notes_tools,
    "library": register_library_tools,
    "peers": register_peers_tools,
    "rooms": register_rooms_tools,
    "reminders": register_reminders_tools,
    "activity": register_activity_tools,
    "workgroups": register_workgroups_tools,
    "repositories": register_repositories_tools,
    "approvals": register_approvals_tools,
    "issues": register_issues_tools,
    "messaging": register_messaging_tools,
}


def register_capabilities(registry: ToolRegistry, providers: dict[str, Any]) -> list[str]:
    """Register the tools of each capability that has a provider; returns their names."""
    registered = []
    for name, register in CAPABILITIES.items():
        provider = providers.get(name)
        if provider is None:
            continue
        register(registry, provider)
        registered.append(name)
    return registered


__all__ = ["CAPABILITIES", "register_capabilities"]
