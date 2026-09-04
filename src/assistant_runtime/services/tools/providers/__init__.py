"""Providers: the integrations that serve capabilities.

``build_providers`` turns the configuration into provider objects keyed by
the capability they serve; ``ToolService`` registers a capability's tools
only for keys that are present.
"""

from __future__ import annotations

from typing import Any

from assistant_runtime.services.tools.providers.config import ProvidersConfig
from assistant_runtime.services.tools.providers.filesystem import (
    FilesystemLibrary,
    MarkdownNotes,
)


def build_providers(config: ProvidersConfig) -> dict[str, Any]:
    """Provider objects by capability name, for the providers that are configured."""
    providers: dict[str, Any] = {}
    if config.notes_path is not None:
        providers["notes"] = MarkdownNotes(config.notes_path)
    if config.library_paths:
        providers["library"] = FilesystemLibrary(config.library_paths)
    return providers


__all__ = ["ProvidersConfig", "build_providers"]
