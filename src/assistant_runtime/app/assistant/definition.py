"""Application-owned composition using native Pydantic AI extension types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.tools import Tool, ToolFuncEither
from pydantic_ai.toolsets import AbstractToolset

from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.artifacts import AssistantProfile


@dataclass(frozen=True, kw_only=True)
class AssistantDefinition[DepsT]:
    """Native agent extensions shared by the server and in-process runtime.

    ``deps_factory`` runs once for each accepted turn/continuation, with that
    request's payload. It may be async. Resolve trusted identity/resources in
    host code; payload fields are not an authentication mechanism. Resources
    returned by the factory remain owned by the host application.

    ``profile`` names the prompt artifacts of this assistant, their order,
    defaults and mutation policies; it takes precedence over the
    ``ASSISTANT__PROFILE`` setting.

    These tools and toolsets keep Pydantic AI's schemas, metadata and error
    semantics. Runtime provider-tool scoping does not filter native extensions;
    use a native tool ``prepare`` callback or ``PrepareTools`` for those.
    """

    tools: Sequence[Tool[DepsT] | ToolFuncEither[DepsT, ...]] = ()
    toolsets: Sequence[AbstractToolset[DepsT]] = ()
    capabilities: Sequence[AgentCapability[DepsT]] = ()
    deps_type: type[DepsT] = type(None)
    deps_factory: Callable[[AssistantRequest], DepsT | Awaitable[DepsT]] | None = None
    profile: AssistantProfile | None = None

    def __post_init__(self) -> None:
        # Snapshot containers, while keeping the native extension objects intact.
        for name in ("tools", "toolsets", "capabilities"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


__all__ = ["AssistantDefinition"]
