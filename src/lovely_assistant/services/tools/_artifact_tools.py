"""Artifact management tool — CRUD for versioned prompt artifacts."""

from __future__ import annotations

from typing import Any

from loguru import logger

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition


def _unknown_artifact_error(name: str) -> dict[str, Any]:
    """Build a consistent unknown-artifact error payload."""
    from lovely_assistant.app.assistant._prompt_builder import (
        artifact_role_boundaries_text,
        known_artifact_names_text,
    )

    return {
        "error": (
            f"Unknown artifact '{name}'. "
            f"Known artifacts: {known_artifact_names_text()}. "
            f"Role boundaries: {artifact_role_boundaries_text()}."
        ),
        "success": False,
    }


async def manage_artifacts(
    action: str,
    name: str = "",
    content: str = "",
    version: int = 0,
) -> dict[str, Any]:
    """Manage versioned prompt artifacts (soul, persona, communication_protocol, ecosystem, scratchpad).

    Supports list, view, propose_edit, update_scratchpad, approve, and history actions.
    """
    actions = {
        "list": _list_artifacts,
        "view": _view_artifact,
        "propose_edit": _propose_edit,
        "update_scratchpad": _update_scratchpad,
        "approve": _approve_artifact,
        "history": _artifact_history,
    }

    if action not in actions:
        return {
            "error": f"Unknown action '{action}'. Valid actions: {', '.join(actions)}",
            "success": False,
        }

    # Get database service from handler deps
    deps = getattr(manage_artifacts, "_handler_deps", None)
    if deps is None or deps.get("database_service") is None:
        return {
            "error": "Artifact store not available (no database connection)",
            "success": False,
        }

    return await actions[action](
        name=name,
        content=content,
        version=version,
        database_service=deps["database_service"],
    )


async def _list_artifacts(
    database_service: Any,
    **_kwargs: Any,
) -> dict[str, Any]:
    """List all active artifacts with versions and sizes."""
    from lovely_assistant.app.assistant._prompt_builder import artifact_sort_key
    from lovely_assistant.services.database.repositories import ArtifactRepository

    async with database_service.session_context() as session:
        repo = ArtifactRepository(session)
        rows = await repo.get_all_active()
    rows = sorted(rows, key=lambda row: artifact_sort_key(row.name))

    return {
        "artifacts": [
            {
                "name": row.name,
                "version": row.version,
                "char_count": len(row.content),
                "proposed_by": row.proposed_by,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
        "count": len(rows),
        "success": True,
    }


async def _view_artifact(
    name: str,
    database_service: Any,
    **_kwargs: Any,
) -> dict[str, Any]:
    """View the active content of an artifact."""
    if not name:
        return {"error": "Name is required for view", "success": False}
    from lovely_assistant.app.assistant._prompt_builder import is_known_artifact_name

    if not is_known_artifact_name(name):
        return _unknown_artifact_error(name)

    from lovely_assistant.services.database.repositories import ArtifactRepository

    async with database_service.session_context() as session:
        repo = ArtifactRepository(session)
        row = await repo.get_active(name)

    if row is None:
        return {"error": f"No active artifact found: {name}", "success": False}

    return {
        "name": row.name,
        "version": row.version,
        "content": row.content,
        "proposed_by": row.proposed_by,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "success": True,
    }


async def _propose_edit(
    name: str,
    content: str,
    database_service: Any,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Propose a new version of an artifact (requires approval to activate)."""
    if not name:
        return {"error": "Name is required for propose_edit", "success": False}
    from lovely_assistant.app.assistant._prompt_builder import is_known_artifact_name

    if not is_known_artifact_name(name):
        return _unknown_artifact_error(name)
    if not content:
        return {"error": "Content is required for propose_edit", "success": False}

    from lovely_assistant.services.database.repositories import ArtifactRepository

    async with database_service.session_context() as session:
        repo = ArtifactRepository(session)
        row = await repo.propose(name, content, proposed_by="jarvis")

    logger.info("Proposed artifact edit", name=name, version=row.version)
    return {
        "name": row.name,
        "version": row.version,
        "is_active": row.is_active,
        "message": f"Version {row.version} proposed. Requires approval to activate.",
        "success": True,
    }


async def _update_scratchpad(
    content: str,
    database_service: Any,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Update scratchpad content (auto-approved)."""
    if not content:
        return {"error": "Content is required for update_scratchpad", "success": False}

    from lovely_assistant.services.database.repositories import ArtifactRepository

    async with database_service.session_context() as session:
        repo = ArtifactRepository(session)
        row = await repo.update_scratchpad(content, proposed_by="jarvis")

    logger.info("Updated scratchpad", version=row.version)
    return {
        "name": row.name,
        "version": row.version,
        "is_active": True,
        "message": "Scratchpad updated and activated.",
        "success": True,
    }


async def _approve_artifact(
    name: str,
    version: int,
    database_service: Any,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Activate a specific version of an artifact."""
    if not name:
        return {"error": "Name is required for approve", "success": False}
    from lovely_assistant.app.assistant._prompt_builder import is_known_artifact_name

    if not is_known_artifact_name(name):
        return _unknown_artifact_error(name)
    if not version:
        return {"error": "Version is required for approve", "success": False}

    from lovely_assistant.services.database.repositories import ArtifactRepository

    async with database_service.session_context() as session:
        repo = ArtifactRepository(session)
        row = await repo.approve(name, version)

    if row is None:
        return {
            "error": f"No version {version} found for artifact '{name}'",
            "success": False,
        }

    logger.info("Approved artifact version", name=name, version=version)
    return {
        "name": row.name,
        "version": row.version,
        "is_active": row.is_active,
        "message": f"Version {version} of '{name}' is now active.",
        "success": True,
    }


async def _artifact_history(
    name: str,
    database_service: Any,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Return version history for an artifact."""
    if not name:
        return {"error": "Name is required for history", "success": False}
    from lovely_assistant.app.assistant._prompt_builder import is_known_artifact_name

    if not is_known_artifact_name(name):
        return _unknown_artifact_error(name)

    from lovely_assistant.services.database.repositories import ArtifactRepository

    async with database_service.session_context() as session:
        repo = ArtifactRepository(session)
        rows = await repo.get_history(name)

    return {
        "name": name,
        "versions": [
            {
                "version": row.version,
                "is_active": row.is_active,
                "proposed_by": row.proposed_by,
                "char_count": len(row.content),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
        "count": len(rows),
        "success": True,
    }


def register_artifact_tools(registry: ToolRegistry) -> None:
    """Register the artifact management tool."""
    from lovely_assistant.app.assistant._prompt_builder import (
        artifact_role_boundaries_text,
        known_artifact_names_text,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="manage_artifacts",
            description=(
                "Manage versioned prompt artifacts "
                f"({known_artifact_names_text()}). "
                f"Role boundaries: {artifact_role_boundaries_text()}. "
                "Actions: list (all active), view (active content), propose_edit (create pending "
                "version — requires approval), update_scratchpad (auto-approved), history (list all "
                "versions with active status), approve (activate a specific version). "
                "Use history then approve to surface and activate pending edits. "
                "Use update_scratchpad to persist observations, patterns, and operational notes "
                "that should influence your behavior across conversations."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "list",
                            "view",
                            "propose_edit",
                            "update_scratchpad",
                            "approve",
                            "history",
                        ],
                        "description": "The action to perform",
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Artifact name (required for view/propose_edit/approve/history). "
                            f"Known artifacts: {known_artifact_names_text()}"
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "New content (required for propose_edit and update_scratchpad)",
                    },
                    "version": {
                        "type": "integer",
                        "description": "Version number (required for approve)",
                    },
                },
                "required": ["action"],
            },
            category=ToolCategory.BACKEND,
        ),
        manage_artifacts,
    )

    logger.info("Registered artifact management tool")
