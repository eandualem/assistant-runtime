"""Artifact endpoints — versioned prompt content management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from lovely_assistant.services.database.interface import DatabaseService
from lovely_assistant.services.database.repositories import ArtifactRepository

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


async def _get_db(request: Request) -> DatabaseService:
    """Retrieve DatabaseService from app state."""
    db: DatabaseService | None = getattr(request.app.state, "database_service", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ProposeRequest(BaseModel):
    content: str = Field(..., min_length=1)
    proposed_by: str = Field(default="dashboard")


class ScratchpadUpdateRequest(BaseModel):
    content: str = Field(..., min_length=1)
    proposed_by: str = Field(default="dashboard")


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_artifacts(request: Request) -> list[dict]:
    """List all active artifacts."""
    db = await _get_db(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            rows = await repo.get_all_active()
            return [_row_to_response(row) for row in rows]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list artifacts", error=str(e))
        raise HTTPException(status_code=503, detail="Database unavailable") from e


@router.get("/{name}")
async def get_artifact(name: str, request: Request) -> dict:
    """Get the active version of an artifact by name."""
    db = await _get_db(request)

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
    db = await _get_db(request)

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
    db = await _get_db(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            row = await repo.propose(name, body.content, body.proposed_by)
            return _row_to_response(row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to propose artifact", name=name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to propose artifact") from e


@router.post("/{name}/approve/{version}")
async def approve_artifact(name: str, version: int, request: Request) -> dict:
    """Approve (activate) a specific version of an artifact."""
    db = await _get_db(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            row = await repo.approve(name, version)
            if row is None:
                raise HTTPException(
                    status_code=404, detail=f"Version {version} not found for artifact: {name}"
                )
            return _row_to_response(row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to approve artifact", name=name, version=version, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to approve artifact") from e


@router.post("/{name}/rollback/{version}")
async def rollback_artifact(name: str, version: int, request: Request) -> dict:
    """Rollback to a previous version of an artifact."""
    db = await _get_db(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            row = await repo.rollback(name, version)
            if row is None:
                raise HTTPException(
                    status_code=404, detail=f"Version {version} not found for artifact: {name}"
                )
            return _row_to_response(row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to rollback artifact", name=name, version=version, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to rollback artifact") from e


@router.patch("/scratchpad")
async def update_scratchpad(body: ScratchpadUpdateRequest, request: Request) -> dict:
    """Direct scratchpad update (auto-approved)."""
    db = await _get_db(request)

    try:
        async with db.session_context() as session:
            repo = ArtifactRepository(session)
            row = await repo.update_scratchpad(body.content, body.proposed_by)
            return _row_to_response(row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update scratchpad", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update scratchpad") from e
