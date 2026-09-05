"""Runtime-owned tool groups, selected by configuration.

The current time, the host's screenshot, the prompt artifacts, subagents and
media generation. Everything else the assistant can do is a *capability*
(``services/tools/capabilities``) served by a configured *provider*
(``services/tools/providers``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from assistant_runtime.services.tools.builtin._video import register_video_tools
from assistant_runtime.services.tools.builtin.artifacts import register_artifact_tools
from assistant_runtime.services.tools.builtin.media import register_media_tools
from assistant_runtime.services.tools.builtin.screen import register_screen_tools
from assistant_runtime.services.tools.builtin.subagent import register_subagent_tools
from assistant_runtime.services.tools.builtin.time import register_time_tools

if TYPE_CHECKING:
    from assistant_runtime.services.tools._registry import ToolRegistry


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    database_service: Any | None,
    llm_service: Any | None,
    media_service: Any | None,
    backend_toolsets: Callable[[], list[Any]],
    runtime_settings: Callable[[], Any | None],
    enabled: frozenset[str] | None = None,
) -> None:
    """Register the built-in tools; the optional ones only when their service exists."""
    if enabled is None or "time" in enabled:
        register_time_tools(registry)
    if enabled is None or "screen" in enabled:
        register_screen_tools(registry)
    if enabled is None or "artifacts" in enabled:
        register_artifact_tools(registry, database_service)
    if llm_service is not None and (enabled is None or "subagent" in enabled):
        register_subagent_tools(
            registry,
            llm_service,
            backend_toolsets=backend_toolsets,
            runtime_settings=runtime_settings,
        )
    if media_service is not None:
        if enabled is None or "media" in enabled:
            register_media_tools(registry, media_service)
        if enabled is None or "video" in enabled:
            register_video_tools(registry, media_service)


__all__ = ["register_builtin_tools"]
