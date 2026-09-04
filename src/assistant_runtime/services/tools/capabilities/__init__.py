"""Capabilities: what the assistant can do, independent of who does it.

Each module declares the tools for one capability and the Protocol a
provider implements to serve it. ``register_capabilities`` registers the
tools of every capability that has a provider.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from assistant_runtime.services.tools.capabilities.library import register_library_tools
from assistant_runtime.services.tools.capabilities.notes import register_notes_tools

if TYPE_CHECKING:
    from assistant_runtime.services.tools._registry import ToolRegistry

CAPABILITIES: dict[str, Callable[[ToolRegistry, Any], None]] = {
    "notes": register_notes_tools,
    "library": register_library_tools,
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
