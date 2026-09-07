"""Host tools: tools the host application executes, not the runtime.

They use Pydantic AI's ExternalToolset + DeferredToolRequests pattern:
1. The model decides to call a host tool.
2. The runtime emits a ``tool_call`` event (category ``host``) with a call id
   and ends the turn with ``final_response.pending_tool_call``.
3. The host executes the tool (navigate, select, refresh, ...).
4. The host sends a continuation request carrying ``tool_call_id`` and ``tool_result``.
5. The runtime resumes the agent with that result.

Which tools exist is configuration (``ToolConfig.host_tools`` and
``host_tools_path``); the runtime ships none.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic_ai.tools import ToolDefinition as PydanticToolDefinition
from pydantic_ai.toolsets import ExternalToolset

from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.exceptions import ToolValidationError
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition

HostToolSchemas = dict[str, dict[str, Any]]


# What providers accept as a tool name (Anthropic and OpenAI both enforce this shape).
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate(schemas: HostToolSchemas, source: str) -> None:
    for name, schema in schemas.items():
        if not isinstance(name, str) or not TOOL_NAME_RE.fullmatch(name):
            raise ToolValidationError(
                f"Host tool name {name!r} from {source} must match {TOOL_NAME_RE.pattern}"
            )
        if not isinstance(schema, dict) or not schema.get("description"):
            raise ToolValidationError(f"Host tool '{name}' from {source} needs a description")
        if not isinstance(schema.get("parameters"), dict):
            raise ToolValidationError(
                f"Host tool '{name}' from {source} needs a 'parameters' JSON schema object"
            )


def load_host_tool_schemas(config: ToolConfig) -> HostToolSchemas:
    """Host tool schemas from the config dict, overlaid with the optional JSON file."""
    schemas: HostToolSchemas = dict(config.host_tools)
    _validate(schemas, "TOOLS__HOST_TOOLS")
    if config.host_tools_path:
        path = Path(config.host_tools_path)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ToolValidationError(f"Cannot read host tools file {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ToolValidationError(f"Host tools file {path} must contain a JSON object")
        _validate(loaded, str(path))
        schemas.update(loaded)
        logger.info("Loaded host tools", path=str(path), count=len(loaded))
    return schemas


def build_host_toolset(schemas: HostToolSchemas) -> ExternalToolset | None:
    """A Pydantic AI ExternalToolset for the host tools, or None when there are none."""
    if not schemas:
        return None
    return ExternalToolset(
        [
            PydanticToolDefinition(
                name=name,
                description=schema["description"],
                parameters_json_schema=schema["parameters"],
            )
            for name, schema in schemas.items()
        ]
    )


def get_host_definitions(schemas: HostToolSchemas) -> list[ToolDefinition]:
    """ToolDefinition models for the host tools (for ToolSet metadata, not execution)."""
    return [
        ToolDefinition(
            name=name,
            description=schema["description"],
            parameters_schema=schema["parameters"],
            category=ToolCategory.HOST,
        )
        for name, schema in schemas.items()
    ]
