"""Runtime-owned tool groups, selected by configuration.

The current time, the host's screenshot, the prompt artifacts, subagents and
media generation. Everything else the assistant can do is a *capability*
(``services/tools/capabilities``) served by a configured *provider*
(``services/tools/providers``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

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
    artifact_service: Any | None,
    llm_service: Any | None,
    media_service: Any | None,
    backend_toolsets: Callable[[], list[Any]],
    runtime_settings: Callable[[], Any | None],
    subagent_usage_limits: Any | None = None,
    subagent_defaults: dict[str, Any] | None = None,
    enabled: frozenset[str],
) -> None:
    """Register the enabled groups that can work; none unless the application enables them."""
    if "time" in enabled:
        register_time_tools(registry)
    if "screen" in enabled:
        register_screen_tools(registry)
    if "artifacts" in enabled:
        register_artifact_tools(registry, artifact_service)
    if llm_service is not None and "subagent" in enabled:
        register_subagent_tools(
            registry,
            llm_service,
            backend_toolsets=backend_toolsets,
            runtime_settings=runtime_settings,
            usage_limits=subagent_usage_limits,
            defaults=subagent_defaults,
        )
    has_media = media_service is not None
    for group, register, can_work in (
        ("media", register_media_tools, has_media and media_service.can_generate_images()),
        ("video", register_video_tools, has_media and media_service.can_generate_videos()),
    ):
        if group not in enabled:
            continue
        if can_work:
            register(registry, media_service)
        else:
            # Never offer the model a tool that cannot work.
            logger.warning("Built-in tool group skipped: no provider key", group=group)


__all__ = ["register_builtin_tools"]
