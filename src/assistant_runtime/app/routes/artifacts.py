"""Artifact endpoints — the host's access to versioned prompt content.

The routes act as the ``host`` actor of the profile's policies. They work
with or without Postgres; responses carry ``durable`` so a client knows
whether versions survive a restart.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from assistant_runtime.services.artifacts.deps import get_artifact_service
from assistant_runtime.services.artifacts.exceptions import (
    ArtifactConflictError,
    ArtifactError,
    ArtifactPermissionError,
    ArtifactVersionNotFoundError,
    UnknownArtifactError,
)
from assistant_runtime.services.artifacts.interface import ArtifactService
from assistant_runtime.services.artifacts.models import Actor, MutationResult

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ProposeRequest(BaseModel):
    content: str = Field(..., min_length=1)
    proposed_by: str = Field(default="host", max_length=32)
    expected_version: int | None = None


class UpdateRequest(BaseModel):
    content: str = Field(..., min_length=1)
    proposed_by: str = Field(default="host", max_length=32)
    expected_version: int | None = None


class ArtifactActionRequest(BaseModel):
    """Unified action request for a host proxy endpoint."""

    action: str = Field(..., pattern="^(approve|rollback|propose|update)$")
    version: int | None = None
    content: str | None = None
    proposed_by: str = Field(default="host", max_length=32)
    expected_version: int | None = None


_STATUS = {
    UnknownArtifactError: 422,
    ArtifactPermissionError: 403,
    ArtifactConflictError: 409,
    ArtifactVersionNotFoundError: 404,
}


def _http_error(exc: ArtifactError) -> HTTPException:
    return HTTPException(status_code=_STATUS.get(type(exc), 400), detail=str(exc))


def _mutation_response(result: MutationResult, *, message: str) -> dict[str, Any]:
    return {
        **result.version.to_dict(),
        "success": True,
        "message": message,
        "live_version": result.live_version,
        "effective_on_next_request": result.activated,
        "unchanged": result.unchanged,
        "durable": result.durable,
    }


async def _propose(
    artifacts: ArtifactService,
    name: str,
    content: str,
    proposed_by: str,
    expected_version: int | None,
) -> dict[str, Any]:
    result = await artifacts.propose(
        name, content, actor=Actor("host", proposed_by), expected_version=expected_version
    )
    message = (
        "Content already active; nothing changed."
        if result.unchanged
        else f"Version {result.version.version} proposed. Approval required before activation."
    )
    return _mutation_response(result, message=message)


async def _update(
    artifacts: ArtifactService,
    name: str,
    content: str,
    proposed_by: str,
    expected_version: int | None,
) -> dict[str, Any]:
    result = await artifacts.update(
        name, content, actor=Actor("host", proposed_by), expected_version=expected_version
    )
    message = (
        "Content already active; nothing changed."
        if result.unchanged
        else f"Version {result.version.version} of {name} is active immediately."
    )
    return _mutation_response(result, message=message)


async def _activate(
    artifacts: ArtifactService, name: str, version: int, *, rollback: bool
) -> dict[str, Any]:
    result = await artifacts.activate(name, version, actor=Actor("host"))
    message = (
        f"Rolled back {name} to version {version}."
        if rollback
        else f"Version {version} approved and now active."
    )
    return _mutation_response(result, message=message)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_artifacts(request: Request) -> list[dict]:
    """Active versions of the profile's artifacts, in prompt order."""
    artifacts = get_artifact_service(request)
    try:
        return [row.to_dict() for row in await artifacts.list_active()]
    except Exception as e:
        logger.error("Failed to list artifacts", error=str(e))
        raise HTTPException(status_code=503, detail="Artifact store unavailable") from e


@router.get("/profile")
async def get_profile(request: Request) -> dict:
    """The profile: every artifact, its role, policy and whether a version is active."""
    artifacts = get_artifact_service(request)
    try:
        active = {row.name: row.version for row in await artifacts.list_active()}
    except Exception as e:
        logger.error("Failed to read artifacts", error=str(e))
        raise HTTPException(status_code=503, detail="Artifact store unavailable") from e
    profile = artifacts.profile
    return {
        "name": profile.name,
        "durable": artifacts.durable,
        "artifacts": [
            {
                "name": a.name,
                "role": a.role,
                "required": a.required,
                "policy": {
                    "assistant_edit": a.policy.assistant_edit,
                    "assistant_activate": a.policy.assistant_activate,
                    "host_edit": a.policy.host_edit,
                },
                "live_version": active.get(a.name),
            }
            for a in profile.artifacts
        ],
    }


@router.get("/{name}")
async def get_artifact(name: str, request: Request) -> dict:
    """The active version of an artifact, or its default text when none is active."""
    artifacts = get_artifact_service(request)
    try:
        row = await artifacts.get_active(name)
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to get artifact", name=name, error=str(e))
        raise HTTPException(status_code=503, detail="Artifact store unavailable") from e
    if row is None:
        definition = artifacts.profile.get(name)
        return {
            "id": None,
            "name": name,
            "content": definition.default if definition else "",
            "version": None,
            "is_active": False,
            "proposed_by": "default",
            "created_at": None,
            "source": "default",
        }
    return {**row.to_dict(), "source": "store"}


@router.get("/{name}/history")
async def get_artifact_history(
    name: str,
    request: Request,
    limit: int = Query(20, ge=1, le=100),
) -> list[dict]:
    """Version history for an artifact, newest first."""
    artifacts = get_artifact_service(request)
    try:
        return [row.to_dict() for row in await artifacts.history(name, limit=limit)]
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to get artifact history", name=name, error=str(e))
        raise HTTPException(status_code=503, detail="Artifact store unavailable") from e


@router.post("/{name}/propose", status_code=201)
async def propose_artifact(name: str, body: ProposeRequest, request: Request) -> dict:
    """Propose a new version of an artifact (inactive until approved)."""
    artifacts = get_artifact_service(request)
    try:
        return await _propose(
            artifacts, name, body.content, body.proposed_by, body.expected_version
        )
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to propose artifact", name=name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to propose artifact") from e


@router.patch("/{name}")
async def update_artifact(name: str, body: UpdateRequest, request: Request) -> dict:
    """Write a new version and activate it at once."""
    artifacts = get_artifact_service(request)
    try:
        return await _update(artifacts, name, body.content, body.proposed_by, body.expected_version)
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to update artifact", name=name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update artifact") from e


@router.post("/{name}/approve/{version}")
async def approve_artifact(name: str, version: int, request: Request) -> dict:
    """Approve (activate) a specific version of an artifact."""
    artifacts = get_artifact_service(request)
    try:
        return await _activate(artifacts, name, version, rollback=False)
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to approve artifact", name=name, version=version, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to approve artifact") from e


@router.post("/{name}/rollback/{version}")
async def rollback_artifact(name: str, version: int, request: Request) -> dict:
    """Reactivate an earlier version of an artifact."""
    artifacts = get_artifact_service(request)
    try:
        return await _activate(artifacts, name, version, rollback=True)
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to rollback artifact", name=name, version=version, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to rollback artifact") from e


@router.post("/{name}/actions")
async def artifact_action(name: str, body: ArtifactActionRequest, request: Request) -> dict:
    """Unified action endpoint: approve, rollback, propose or update a named artifact."""
    artifacts = get_artifact_service(request)
    try:
        if body.action in ("approve", "rollback"):
            if body.version is None:
                raise HTTPException(
                    status_code=422, detail=f"version is required for {body.action}"
                )
            return await _activate(
                artifacts, name, body.version, rollback=body.action == "rollback"
            )
        if not body.content:
            raise HTTPException(status_code=422, detail=f"content is required for {body.action}")
        handler = _propose if body.action == "propose" else _update
        return await handler(artifacts, name, body.content, body.proposed_by, body.expected_version)
    except HTTPException:
        raise
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Artifact action failed", name=name, action=body.action, error=str(e))
        raise HTTPException(status_code=500, detail=f"Artifact action failed: {body.action}") from e


@router.delete("/{name}")
async def delete_artifact(name: str, request: Request) -> dict:
    """Delete every stored version; the profile's default text applies again."""
    artifacts = get_artifact_service(request)
    try:
        count = await artifacts.delete(name, actor=Actor("host"))
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to delete artifact", name=name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete artifact") from e
    if count == 0:
        raise HTTPException(status_code=404, detail=f"No stored versions for artifact: {name}")
    return {"success": True, "name": name, "deleted_versions": count, "durable": artifacts.durable}
