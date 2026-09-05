"""Artifact management tool — the assistant's access to its own prompt artifacts.

Every action goes through ``ArtifactService``, which enforces the profile's
policies for the ``assistant`` actor; the tool only translates results and
errors into the registry's dict contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger

from assistant_runtime.services.artifacts.exceptions import ArtifactError
from assistant_runtime.services.artifacts.interface import ArtifactService
from assistant_runtime.services.artifacts.models import Actor, ArtifactVersion, MutationResult
from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition

ASSISTANT = Actor(kind="assistant")
ACTIONS = ("list", "view", "history", "propose", "update", "activate")


def _version_dict(row: ArtifactVersion, *, content: bool = False) -> dict[str, Any]:
    data = {
        "name": row.name,
        "version": row.version,
        "is_active": row.is_active,
        "proposed_by": row.proposed_by,
        "char_count": len(row.content),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
    if content:
        data["content"] = row.content
    return data


def _mutation_dict(result: MutationResult, message: str) -> dict[str, Any]:
    return {
        **_version_dict(result.version),
        "activated": result.activated,
        "live_version": result.live_version,
        "unchanged": result.unchanged,
        "durable": result.durable,
        "message": message,
        "success": True,
    }


def build_manage_artifacts(artifacts: ArtifactService | None) -> Callable[..., Any]:
    """The ``manage_artifacts`` handler, bound to the artifact service (or None)."""

    async def manage_artifacts(
        action: str,
        name: str = "",
        content: str = "",
        version: int = 0,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        """Read and evolve the prompt artifacts the profile allows the assistant to change."""
        if action not in ACTIONS:
            return {
                "error": f"Unknown action '{action}'. Valid actions: {', '.join(ACTIONS)}",
                "error_code": "unknown_action",
                "success": False,
            }
        if artifacts is None:
            return {
                "error": "Artifact service not available",
                "error_code": "artifacts_unavailable",
                "success": False,
            }
        if action != "list" and not name:
            return {
                "error": f"Name is required for {action}",
                "error_code": "missing_name",
                "success": False,
            }
        try:
            if action == "list":
                return await _list(artifacts)
            if action == "view":
                return await _view(artifacts, name)
            if action == "history":
                return await _history(artifacts, name)
            if action == "propose":
                result = await artifacts.propose(
                    name, content, actor=ASSISTANT, expected_version=expected_version
                )
                message = (
                    "Content already active; nothing changed."
                    if result.unchanged
                    else f"Version {result.version.version} proposed; it stays inactive until activated."
                )
                return _mutation_dict(result, message)
            if action == "update":
                result = await artifacts.update(
                    name, content, actor=ASSISTANT, expected_version=expected_version
                )
                message = (
                    "Content already active; nothing changed."
                    if result.unchanged
                    else f"Version {result.version.version} is active; later prompts use it."
                )
                return _mutation_dict(result, message)
            if not version:
                return {
                    "error": "Version is required for activate",
                    "error_code": "missing_version",
                    "success": False,
                }
            result = await artifacts.activate(name, version, actor=ASSISTANT)
            return _mutation_dict(result, f"Version {version} of '{name}' is now active.")
        except ArtifactError as e:
            return {"error": str(e), "error_code": e.error_code, "success": False}

    return manage_artifacts


async def _list(artifacts: ArtifactService) -> dict[str, Any]:
    """Every artifact of the profile: its role, what the assistant may do, its live version."""
    active = {row.name: row for row in await artifacts.list_active()}
    items = []
    for definition in artifacts.profile.artifacts:
        row = active.get(definition.name)
        items.append(
            {
                "name": definition.name,
                "role": definition.role,
                "required": definition.required,
                "allowed_actions": artifacts.allowed_actions(definition.name, "assistant"),
                **(
                    _version_dict(row)
                    if row is not None
                    else {"version": None, "char_count": len(definition.default)}
                ),
            }
        )
    return {"artifacts": items, "count": len(items), "durable": artifacts.durable, "success": True}


async def _view(artifacts: ArtifactService, name: str) -> dict[str, Any]:
    row = await artifacts.get_active(name)
    if row is None:
        definition = artifacts.profile.get(name)
        assert definition is not None  # get_active validated the name
        return {
            "name": name,
            "version": None,
            "content": definition.default,
            "source": "default",
            "success": True,
        }
    return {**_version_dict(row, content=True), "source": "store", "success": True}


async def _history(artifacts: ArtifactService, name: str) -> dict[str, Any]:
    rows = await artifacts.history(name)
    return {
        "name": name,
        "versions": [_version_dict(row) for row in rows],
        "count": len(rows),
        "success": True,
    }


def register_artifact_tools(registry: ToolRegistry, artifacts: ArtifactService | None) -> None:
    """Register the artifact management tool, described from the profile."""
    if artifacts is not None:
        profile = artifacts.profile
        lines = [
            f"{a.name} ({a.role or 'no role given'}; you may "
            f"{', '.join(artifacts.allowed_actions(a.name, 'assistant')) or 'only read'})"
            for a in profile.artifacts
        ]
        catalog = "Artifacts: " + "; ".join(lines) + ". "
        names = profile.names_text()
    else:
        catalog = ""
        names = ""

    registry.register_backend_tool(
        ToolDefinition(
            name="manage_artifacts",
            description=(
                "Read and evolve the versioned prompt artifacts that shape your behavior. "
                + catalog
                + "Actions: list (every artifact with its live version and what you may do), "
                "view (active content), history (all versions), propose (new inactive version "
                "for review), update (write and activate at once, only where allowed), "
                "activate (make a version live, only where allowed). Pass expected_version to "
                "fail instead of overwriting a change you have not seen."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": list(ACTIONS),
                        "description": "The action to perform",
                    },
                    "name": {
                        "type": "string",
                        "description": "Artifact name (required for every action but list)"
                        + (f". Known artifacts: {names}" if names else ""),
                    },
                    "content": {
                        "type": "string",
                        "description": "New content (required for propose and update)",
                    },
                    "version": {
                        "type": "integer",
                        "description": "Version number (required for activate)",
                    },
                    "expected_version": {
                        "type": "integer",
                        "description": (
                            "The active version you based the change on; the write is "
                            "rejected if another version is active by then (0 = none)"
                        ),
                    },
                },
                "required": ["action"],
            },
            category=ToolCategory.BACKEND,
        ),
        build_manage_artifacts(artifacts),
    )

    logger.info("Registered artifact management tool")
