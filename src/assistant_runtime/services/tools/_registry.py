"""Internal tool registry -- manages tool definitions and builds Pydantic AI toolsets."""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable
from typing import Any

from loguru import logger
from pydantic_ai.toolsets import FunctionToolset

from assistant_runtime.base.resilience import retry_with_backoff
from assistant_runtime.host_context import HostAction, actions_of, view_name_of
from assistant_runtime.services.tools._host_tools import (
    build_host_toolset,
    get_host_definitions,
    load_host_tool_schemas,
)
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.exceptions import ToolValidationError
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition, ToolSet


class ToolRegistry:
    """Manages tool definitions and handlers, builds Pydantic AI toolsets per request."""

    def __init__(self, config: ToolConfig) -> None:
        self._config = config
        self._backend_handlers: dict[str, Callable] = {}
        self._backend_definitions: dict[str, ToolDefinition] = {}
        self._host_definitions: list[ToolDefinition] = []
        self._host_toolset: Any | None = None
        self._host_tool_names: frozenset[str] = frozenset()
        self._available_tools_cache: dict[str | None, ToolSet] = {}
        self._toolset_cache: dict[str | None, list[Any]] = {}

    def register_backend_tool(self, definition: ToolDefinition, handler: Callable) -> None:
        """Register a backend tool with its async handler function."""
        if definition.category != ToolCategory.BACKEND:
            raise ToolValidationError(
                f"Expected backend tool, got category '{definition.category}'"
            )
        if definition.name in self._backend_definitions:
            raise ToolValidationError(f"Backend tool '{definition.name}' already registered")
        if definition.name in self._host_tool_names:
            raise ToolValidationError(
                f"Backend tool '{definition.name}' clashes with a configured host tool"
            )
        self._backend_definitions[definition.name] = definition
        self._backend_handlers[definition.name] = handler
        self._available_tools_cache.clear()
        self._toolset_cache.clear()
        logger.debug("Registered backend tool", tool=definition.name)

    def register_host_tools(self, schemas: dict[str, dict[str, Any]] | None = None) -> None:
        """Register the tools the host executes (from config unless given explicitly)."""
        if schemas is None:
            schemas = load_host_tool_schemas(self._config)
        for name in schemas:
            if name in self._backend_definitions:
                raise ToolValidationError(f"Host tool '{name}' clashes with a backend tool")
        self._host_definitions = get_host_definitions(schemas)
        self._host_toolset = build_host_toolset(schemas)
        self._host_tool_names = frozenset(schemas)
        self._available_tools_cache.clear()
        self._toolset_cache.clear()
        logger.debug("Registered host tools", count=len(schemas), names=sorted(schemas))

    def is_host_tool(self, tool_name: str) -> bool:
        """Whether the host, not the runtime, executes this tool."""
        return tool_name in self._host_tool_names

    def invalidates_for(self, tool_name: str) -> list[str] | None:
        """Host data domains a backend tool invalidates, or None."""
        domains = self._config.invalidations.get(tool_name)
        return list(domains) if domains else None

    @staticmethod
    def _wrap_handler(handler: Callable, tool_name: str) -> Callable:
        """Wrap a tool handler with retry on transient errors and a safety net.

        Retries once (2 total attempts) on ConnectionError/TimeoutError before
        falling through to the structured error response.
        """

        @retry_with_backoff(
            max_attempts=2,
            min_wait=0.5,
            max_wait=5.0,
            retry_on=(ConnectionError, TimeoutError),
            name=f"tool:{tool_name}",
        )
        async def _retryable(*args: Any, **kwargs: Any) -> Any:
            return await handler(*args, **kwargs)

        @functools.wraps(handler)
        async def _safe_handler(*args: Any, **kwargs: Any) -> Any:
            try:
                return await _retryable(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "[TOOLS] Unhandled tool exception",
                    tool=tool_name,
                    error_type=e.__class__.__name__,
                    error=str(e),
                )
                return {
                    "success": False,
                    "error": f"Internal error: {e}",
                    "error_code": "TOOL_EXECUTION_ERROR",
                }

        return _safe_handler

    def build_toolset(self, host_context: dict[str, Any] | None = None) -> list:
        """Build Pydantic AI toolsets for a request.

        Returns list of AbstractToolset instances:
        - FunctionToolset for backend tools (with real handlers, wrapped with safety net)
        - ExternalToolset for host tools (deferred execution via DeferredToolRequests)
        """
        cache_key = self._cache_key(host_context)
        cached = self._toolset_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        available = self._resolve_available_tools(host_context)
        toolsets: list = []

        if available.backend_tools:
            func_toolset = FunctionToolset()
            for defn in available.backend_tools:
                handler = self._backend_handlers[defn.name]
                safe_handler = self._wrap_handler(handler, defn.name)
                func_toolset.add_function(
                    safe_handler,
                    name=defn.name,
                    description=defn.description,
                )
            toolsets.append(func_toolset)

        # Host tools — always appended, bypass page scoping
        if self._host_toolset is not None:
            toolsets.append(self._host_toolset)
        request_actions = self._request_actions(host_context)
        if request_actions:
            toolsets.append(
                build_host_toolset(
                    {
                        a.name: {"description": a.description, "parameters": a.parameters}
                        for a in request_actions
                    }
                )
            )

        logger.debug(
            "[TOOLS] Built toolsets",
            backend=len(available.backend_tools),
            host=len(self._host_definitions),
            toolsets=len(toolsets),
        )
        self._toolset_cache[cache_key] = list(toolsets)
        return list(toolsets)

    def build_subagent_toolset(self) -> list:
        """Build toolsets for subagent execution — backend tools only, excluding run_subagent.

        Returns a list with a single FunctionToolset containing all backend tools
        except run_subagent (prevents recursion). No host tools — subagents
        don't interact with the UI.
        """
        toolsets: list = []
        backend_defs = [
            defn for defn in self._backend_definitions.values() if defn.name != "run_subagent"
        ]

        if backend_defs:
            func_toolset = FunctionToolset()
            for defn in backend_defs:
                handler = self._backend_handlers[defn.name]
                safe_handler = self._wrap_handler(handler, defn.name)
                func_toolset.add_function(
                    safe_handler,
                    name=defn.name,
                    description=defn.description,
                )
            toolsets.append(func_toolset)

        logger.debug(
            "[TOOLS] Built subagent toolsets",
            backend=len(backend_defs),
            excluded="run_subagent",
        )
        return toolsets

    def get_available_tools(self, host_context: dict[str, Any] | None = None) -> ToolSet:
        """List tools available for a given host context."""
        return self._resolve_available_tools(host_context)

    def warm_host_context(self, host_context: dict[str, Any] | None = None) -> None:
        """Precompute page-scoped availability and toolset caches."""
        self._resolve_available_tools(host_context)
        self.build_toolset(host_context)

    def get_tool_names(self) -> list[str]:
        """All registered tool names."""
        return list(self._backend_definitions.keys())

    def backend_tool_count(self) -> int:
        """Number of registered backend tools."""
        return len(self._backend_definitions)

    def host_tool_count(self) -> int:
        """Number of registered host tools."""
        return len(self._host_definitions)

    def _resolve_available_tools(self, host_context: dict[str, Any] | None = None) -> ToolSet:
        """Determine which backend tools are available for the host's current page.

        ``ToolConfig.page_scopes`` maps a page name to the backend tool names
        allowed while the host reports that page. No host context, no page, or
        a page that is not listed means every backend tool. Host tools are
        never scoped.
        """
        page_name = self._page_name(host_context)
        cache_key = self._cache_key(host_context)
        cached = self._available_tools_cache.get(cache_key)
        if cached is not None:
            return cached

        backend = list(self._backend_definitions.values())
        total_before = len(backend)

        scope = self._config.page_scopes.get(page_name) if page_name else None
        if scope is not None:
            allowed = set(scope)
            backend = [t for t in backend if t.name in allowed]

        total = len(backend)
        if total > self._config.max_tools_per_request:
            logger.warning(
                "Tool count exceeds max_tools_per_request",
                total=total,
                max=self._config.max_tools_per_request,
            )

        toolset = ToolSet(
            backend_tools=backend,
            host_tools=[
                *self._host_definitions,
                *(
                    ToolDefinition(
                        name=a.name,
                        description=a.description,
                        parameters_schema=a.parameters,
                        category=ToolCategory.HOST,
                    )
                    for a in self._request_actions(host_context)
                ),
            ],
            page=page_name,
            filtered_out_count=total_before - total,
        )
        self._available_tools_cache[cache_key] = toolset
        return toolset

    @staticmethod
    def _page_name(host_context: dict[str, Any] | None = None) -> str | None:
        """The view (page) name of a host context."""
        return view_name_of(host_context)

    def _request_actions(self, host_context: dict[str, Any] | None) -> list[HostAction]:
        """Actions the host declared for this turn, minus names already taken."""
        actions = []
        for action in actions_of(host_context):
            if action.name in self._backend_definitions or any(
                action.name == d.name for d in self._host_definitions
            ):
                logger.warning(
                    "Request-declared action shadows a registered tool; ignored",
                    action=action.name,
                )
                continue
            actions.append(action)
        return actions

    def _cache_key(self, host_context: dict[str, Any] | None = None) -> str | None:
        """Cache by page and by the declared actions, which change availability."""
        page = self._page_name(host_context)
        actions = self._request_actions(host_context)
        if not actions:
            return page or None
        signature = hashlib.blake2b(
            json.dumps([a.model_dump() for a in actions], sort_keys=True).encode(),
            digest_size=8,
        ).hexdigest()
        return f"{page or ''}#{signature}"
