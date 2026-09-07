"""Configuration for the tool service module."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolConfig(BaseModel):
    """Tool module configuration. Nested into AppSettings as `tools`.

    The three host-integration fields are empty by default: without them the
    runtime exposes every backend tool on every request, defers no tool to the
    host and reports no invalidation domains. A host application (dashboard,
    IDE, chat client) sets them to describe itself; see the README section on
    host context.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tools_per_request: int = Field(default=64, ge=1, le=100)
    tool_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    builtin_tools: frozenset[
        Literal["time", "screen", "artifacts", "subagent", "media", "video"]
    ] = Field(
        default=frozenset({"time", "screen", "artifacts", "subagent", "media", "video"}),
        description="Built-in tool groups to register; an empty set disables all built-ins.",
    )
    provider_capabilities: frozenset[str] | None = Field(
        default=None,
        description="Runtime business capabilities to expose from configured providers. None enables every configured capability; an empty set disables all.",
    )

    host_tools: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Tools the host executes itself, keyed by name: "
            '{"name": {"description": str, "parameters": <JSON schema>}}. '
            "The model can call them; the runtime emits a tool_call event and waits for "
            "the host's continuation. Env: TOOLS__HOST_TOOLS as JSON."
        ),
    )
    host_tools_path: str | None = Field(
        default=None,
        description="Path to a JSON file with the same shape as host_tools (merged over it).",
    )
    page_scopes: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Backend tool names allowed while the host reports a given page name: "
            '{"tasks": ["create_issue", ...]}. Pages not listed get every tool.'
        ),
    )
    invalidations: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Host data domains a backend tool invalidates when it runs: "
            '{"create_issue": ["tasks"]}. Reported on tool_result events.'
        ),
    )
