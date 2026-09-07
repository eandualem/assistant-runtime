"""Providers: the integrations that serve capabilities.

``build_providers`` turns the configuration into provider objects keyed by
the capability they serve; ``ToolService`` registers a capability's tools
only for keys that are present.
"""

from __future__ import annotations

from typing import Any

from assistant_runtime.services.tools.providers.backbone import build_backbone_providers
from assistant_runtime.services.tools.providers.claude_code import StateFileApprovals
from assistant_runtime.services.tools.providers.config import ProvidersConfig
from assistant_runtime.services.tools.providers.filesystem import (
    FilesystemLibrary,
    MarkdownNotes,
)
from assistant_runtime.services.tools.providers.github import GitHubIssues
from assistant_runtime.services.tools.providers.telegram import TelegramMessaging


def build_providers(config: ProvidersConfig) -> dict[str, Any]:
    """Provider objects by capability name, for the providers that are configured."""
    providers: dict[str, Any] = {}
    if config.notes_path is not None:
        providers["notes"] = MarkdownNotes(config.notes_path)
    if config.library_paths:
        providers["library"] = FilesystemLibrary(config.library_paths)
    if config.backbone_url:
        providers.update(build_backbone_providers(config))
    if config.agent_state_dir is not None:
        providers["approvals"] = StateFileApprovals(config.agent_state_dir)
    if config.github_token and config.github_repo:
        providers["issues"] = GitHubIssues()
    if config.telegram_token and config.telegram_chat_id:
        providers["messaging"] = TelegramMessaging()
    return providers


__all__ = ["ProvidersConfig", "build_providers"]
