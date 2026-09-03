"""Artifact endpoints — versioned prompt content management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from lovely_assistant.services.database.deps import get_database_service
from lovely_assistant.services.database.repositories import ArtifactRepository

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ProposeRequest(BaseModel):
    content: str = Field(..., min_length=1)
    proposed_by: str = Field(default="dashboard")


class ScratchpadUpdateRequest(BaseModel):
    content: str = Field(..., min_length=1)
    proposed_by: str = Field(default="dashboard")


class ArtifactActionRequest(BaseModel):
    """Unified action request for the dashboard proxy endpoint."""

    action: str = Field(..., pattern="^(approve|rollback|propose)$")
    version: int | None = None
    content: str | None = None
    proposed_by: str = Field(default="dashboard")


def _ensure_known_artifact_name(name: str) -> None:
    """Reject unknown artifact names at the route boundary."""
    from lovely_assistant.app.assistant._prompt_builder import (
        artifact_role_boundaries_text,
        is_known_artifact_name,
        known_artifact_names_text,
    )

    if not is_known_artifact_name(name):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown artifact '{name}'. "
                f"Known artifacts: {known_artifact_names_text()}. "
                f"Role boundaries: {artifact_role_boundaries_text()}."
            ),
        )


def _row_to_response(row: Any) -> dict:
    """Convert an ArtifactORM row to a response dict."""
    return {
        "id": row.id,
        "name": row.name,
        "content": row.content,
        "version": row.version,
        "is_active": row.is_active,
        "proposed_by": row.proposed_by,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _build_mutation_response(
    row: Any,
    *,
    message: str,
    live_version: int | None,
    effective_on_next_request: bool,
) -> dict[str, Any]:
    """Build an explicit mutation response for dashboard artifact actions."""
    return {
        **_row_to_response(row),
        "success": True,
        "message": message,
        "live_version": live_version,
        "effective_on_next_request": effective_on_next_request,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_artifacts(request: Request) -> list[dict]:
    """List all active artifacts."""
    db = get_database_service(request)
    from lovely_assistant.app.assistant._prompt_builder import artifact_sort_key

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            rows = await repo.get_all_active()
            rows = sorted(rows, key=lambda row: artifact_sort_key(row.name))
            return [_row_to_response(row) for row in rows]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list artifacts", error=str(e))
        raise HTTPException(status_code=503, detail="Database unavailable") from e


@router.get("/{name}")
async def get_artifact(name: str, request: Request) -> dict:
    """Get the active version of an artifact by name."""
    _ensure_known_artifact_name(name)
    db = get_database_service(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            row = await repo.get_active(name)
            if row is None:
                raise HTTPException(status_code=404, detail=f"Artifact not found: {name}")
            return _row_to_response(row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get artifact", name=name, error=str(e))
        raise HTTPException(status_code=503, detail="Database unavailable") from e


@router.get("/{name}/history")
async def get_artifact_history(
    name: str,
    request: Request,
    limit: int = Query(20, ge=1, le=100),
) -> list[dict]:
    """Get version history for an artifact."""
    _ensure_known_artifact_name(name)
    db = get_database_service(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            rows = await repo.get_history(name, limit=limit)
            return [_row_to_response(row) for row in rows]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get artifact history", name=name, error=str(e))
        raise HTTPException(status_code=503, detail="Database unavailable") from e


@router.post("/{name}/propose", status_code=201)
async def propose_artifact(name: str, body: ProposeRequest, request: Request) -> dict:
    """Propose a new version of an artifact (inactive until approved)."""
    _ensure_known_artifact_name(name)
    db = get_database_service(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            row = await repo.propose(name, body.content, body.proposed_by)
            active = await repo.get_active(name)
            live_version = active.version if active is not None else None
            return _build_mutation_response(
                row,
                message=f"Version {row.version} proposed. Approval required before activation.",
                live_version=live_version,
                effective_on_next_request=False,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to propose artifact", name=name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to propose artifact") from e


@router.post("/{name}/approve/{version}")
async def approve_artifact(name: str, version: int, request: Request) -> dict:
    """Approve (activate) a specific version of an artifact."""
    _ensure_known_artifact_name(name)
    db = get_database_service(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            row = await repo.approve(name, version)
            if row is None:
                raise HTTPException(
                    status_code=404, detail=f"Version {version} not found for artifact: {name}"
                )
            return _build_mutation_response(
                row,
                message=f"Version {version} approved and now active.",
                live_version=row.version,
                effective_on_next_request=True,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to approve artifact", name=name, version=version, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to approve artifact") from e


@router.post("/{name}/rollback/{version}")
async def rollback_artifact(name: str, version: int, request: Request) -> dict:
    """Rollback to a previous version of an artifact."""
    _ensure_known_artifact_name(name)
    db = get_database_service(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            row = await repo.rollback(name, version)
            if row is None:
                raise HTTPException(
                    status_code=404, detail=f"Version {version} not found for artifact: {name}"
                )
            return _build_mutation_response(
                row,
                message=f"Rolled back {name} to version {version}.",
                live_version=row.version,
                effective_on_next_request=True,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to rollback artifact", name=name, version=version, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to rollback artifact") from e


@router.post("/{name}/actions")
async def artifact_action(name: str, body: ArtifactActionRequest, request: Request) -> dict:
    """Unified action endpoint for the dashboard proxy.

    Dispatches approve, rollback, and propose actions for a named artifact.
    """
    _ensure_known_artifact_name(name)
    db = get_database_service(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)

            if body.action == "approve":
                if body.version is None:
                    raise HTTPException(status_code=422, detail="version is required for approve")
                row = await repo.approve(name, body.version)
                if row is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Version {body.version} not found for artifact: {name}",
                    )
                return _build_mutation_response(
                    row,
                    message=f"Version {body.version} approved and now active.",
                    live_version=row.version,
                    effective_on_next_request=True,
                )

            if body.action == "rollback":
                if body.version is None:
                    raise HTTPException(status_code=422, detail="version is required for rollback")
                row = await repo.rollback(name, body.version)
                if row is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Version {body.version} not found for artifact: {name}",
                    )
                return _build_mutation_response(
                    row,
                    message=f"Rolled back {name} to version {body.version}.",
                    live_version=row.version,
                    effective_on_next_request=True,
                )

            # action == "propose"
            if not body.content:
                raise HTTPException(status_code=422, detail="content is required for propose")
            row = await repo.propose(name, body.content, body.proposed_by)
            active = await repo.get_active(name)
            live_version = active.version if active is not None else None
            return _build_mutation_response(
                row,
                message=f"Version {row.version} proposed. Approval required before activation.",
                live_version=live_version,
                effective_on_next_request=False,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Artifact action failed", name=name, action=body.action, error=str(e))
        raise HTTPException(status_code=500, detail=f"Artifact action failed: {body.action}") from e


@router.delete("/{name}")
async def delete_artifact(name: str, request: Request) -> dict:
    """Delete all versions of an artifact by name."""
    _ensure_known_artifact_name(name)
    db = get_database_service(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            count = await repo.delete_by_name(name)
            if count == 0:
                raise HTTPException(status_code=404, detail=f"Artifact not found: {name}")
            return {"success": True, "name": name, "deleted_versions": count}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete artifact", name=name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete artifact") from e


@router.patch("/scratchpad")
async def update_scratchpad(body: ScratchpadUpdateRequest, request: Request) -> dict:
    """Direct scratchpad update (auto-approved)."""
    db = get_database_service(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            row = await repo.update_scratchpad(body.content, body.proposed_by)
            return _build_mutation_response(
                row,
                message=f"Scratchpad updated to version {row.version} and activated immediately.",
                live_version=row.version,
                effective_on_next_request=True,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update scratchpad", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update scratchpad") from e
