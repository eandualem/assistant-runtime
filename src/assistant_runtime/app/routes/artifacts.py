"""Artifact endpoints — the host's access to versioned prompt content.

The routes act as the ``host`` actor of the profile's policies. Reads need
an authenticated caller; every mutation is administration (the ``admin``
role) attributed to the authenticated principal, never to a name in the
body. They work with or without Postgres; responses carry ``durable`` so
a client knows whether versions survive a restart.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from assistant_runtime.app.access.deps import AdminDep, PrincipalDep
from assistant_runtime.services.artifacts.deps import get_artifact_service
from assistant_runtime.services.artifacts.exceptions import (
    ArtifactConflictError,
    ArtifactError,
    ArtifactPermissionError,
    ArtifactSubjectRequiredError,
    ArtifactVersionNotFoundError,
    UnknownArtifactError,
    UnknownProfileError,
)
from assistant_runtime.services.artifacts.interface import ArtifactService
from assistant_runtime.services.artifacts.models import Actor, MutationResult

router = APIRouter(prefix="/artifacts", tags=["artifacts"])
proposals_router = APIRouter(prefix="/artifact-proposals", tags=["artifacts"])
subjects_router = APIRouter(prefix="/artifact-subjects", tags=["artifacts"])
prompt_router = APIRouter(prefix="/artifact-prompt", tags=["artifacts"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


_LABEL = r"^[a-z][a-z0-9_-]{0,31}$"


class ProposeRequest(BaseModel):
    content: str = Field(..., min_length=1)
    expected_version: int | None = None
    rationale: str | None = None
    label: str | None = Field(default=None, pattern=_LABEL)
    """Who in the host made the change (``owner``, ``import``, ...), recorded after the principal."""


class RejectRequest(BaseModel):
    reason: str | None = None
    label: str | None = Field(default=None, pattern=_LABEL)
    """Who in the host made the change (``owner``, ``import``, ...), recorded after the principal."""


class UpdateRequest(BaseModel):
    content: str = Field(..., min_length=1)
    expected_version: int | None = None
    label: str | None = Field(default=None, pattern=_LABEL)
    """Who in the host made the change (``owner``, ``import``, ...), recorded after the principal."""


class LabelRequest(BaseModel):
    label: str | None = Field(default=None, pattern=_LABEL)
    """Who in the host made the change (``owner``, ``import``, ...), recorded after the principal."""


class ArtifactActionRequest(BaseModel):
    """Unified action request for a host proxy endpoint."""

    action: str = Field(..., pattern="^(approve|rollback|reject|propose|update)$")
    version: int | None = None
    content: str | None = None
    expected_version: int | None = None
    rationale: str | None = None
    reason: str | None = None
    label: str | None = Field(default=None, pattern=_LABEL)
    """Who in the host made the change (``owner``, ``import``, ...), recorded after the principal."""


_STATUS = {
    UnknownArtifactError: 422,
    UnknownProfileError: 404,
    ArtifactPermissionError: 403,
    ArtifactConflictError: 409,
    ArtifactVersionNotFoundError: 404,
    ArtifactSubjectRequiredError: 422,
}


def _who(admin: Any, body: Any) -> str:
    """The principal, and the host's own label for the change when it gave one."""
    label = getattr(body, "label", None)
    return (f"{admin.id}:{label}" if label else admin.id)[:128]


def _http_error(exc: ArtifactError) -> HTTPException:
    return HTTPException(status_code=_STATUS.get(type(exc), 400), detail=str(exc))


def _scoped_artifacts(
    request: Request,
    principal: PrincipalDep,
    profile: str | None = Query(None, description="Registered assistant profile name"),
    subject: str | None = Query(
        None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
        description="The subject, for the profile's subject-scoped artifacts",
    ),
) -> ArtifactService:
    try:
        return get_artifact_service(request).for_profile(profile).for_subject(subject)
    except ArtifactError as exc:
        raise _http_error(exc) from exc


ScopedArtifactDep = Annotated[ArtifactService, Depends(_scoped_artifacts)]


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
    rationale: str | None = None,
) -> dict[str, Any]:
    result = await artifacts.propose(
        name,
        content,
        actor=Actor("host", proposed_by),
        expected_version=expected_version,
        rationale=rationale,
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
    artifacts: ArtifactService, name: str, version: int, actor_id: str, *, rollback: bool
) -> dict[str, Any]:
    result = await artifacts.activate(name, version, actor=Actor("host", actor_id))
    message = (
        f"Rolled back {name} to version {version}."
        if rollback
        else f"Version {version} approved and now active."
    )
    return _mutation_response(result, message=message)


async def _reject(
    artifacts: ArtifactService, name: str, version: int, actor_id: str, reason: str | None
) -> dict[str, Any]:
    result = await artifacts.reject(name, version, actor=Actor("host", actor_id), reason=reason)
    return _mutation_response(
        result, message=f"Version {version} of {name} rejected; the active version is unchanged."
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_artifacts(artifacts: ScopedArtifactDep, principal: PrincipalDep) -> list[dict]:
    """Active versions of the profile's artifacts, in prompt order."""
    try:
        return [row.to_dict() for row in await artifacts.list_active()]
    except Exception as e:
        logger.error("Failed to list artifacts", error=str(e))
        raise HTTPException(status_code=503, detail="Artifact store unavailable") from e


@router.get("/profile")
async def get_profile(artifacts: ScopedArtifactDep, principal: PrincipalDep) -> dict:
    """The profile: every artifact, its role, policy and whether a version is active."""
    try:
        active = {row.name: row.version for row in await artifacts.list_active()}
    except Exception as e:
        logger.error("Failed to read artifacts", error=str(e))
        raise HTTPException(status_code=503, detail="Artifact store unavailable") from e
    profile = artifacts.profile
    return {
        "name": profile.name,
        "subject": artifacts.subject,
        "available_profiles": list(artifacts.available_profiles),
        "durable": artifacts.durable,
        "include": list(profile.include),
        "artifacts": [
            {
                "name": a.name,
                "role": a.role,
                "required": a.required,
                "scope": a.scope,
                "keep_versions": a.keep_versions,
                "policy": _policy(a.policy),
                "live_version": active.get(a.name),
            }
            for a in profile.artifacts
        ],
        "collections": [
            {
                "prefix": c.prefix,
                "role": c.role,
                "keep_versions": c.keep_versions,
                "policy": _policy(c.policy),
                "documents": sorted(name for name in active if c.matches(name)),
            }
            for c in profile.collections
        ],
    }


def _policy(policy: Any) -> dict[str, Any]:
    return {
        "assistant_edit": policy.assistant_edit,
        "assistant_activate": policy.assistant_activate,
        "host_edit": policy.host_edit,
    }


@router.get("/{name}")
async def get_artifact(name: str, artifacts: ScopedArtifactDep, principal: PrincipalDep) -> dict:
    """The active version of an artifact, or its default text when none is active."""
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
    artifacts: ScopedArtifactDep,
    principal: PrincipalDep,
    limit: int = Query(20, ge=1, le=100),
) -> list[dict]:
    """Version history for an artifact, newest first."""
    try:
        return [row.to_dict() for row in await artifacts.history(name, limit=limit)]
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to get artifact history", name=name, error=str(e))
        raise HTTPException(status_code=503, detail="Artifact store unavailable") from e


@router.get("/{name}/versions/{version}")
async def get_artifact_version(
    name: str, version: int, artifacts: ScopedArtifactDep, principal: PrincipalDep
) -> dict:
    """One version as a proposal record: its content, the active text and a unified diff."""
    try:
        return await artifacts.version_record(name, version)
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to get artifact version", name=name, version=version, error=str(e))
        raise HTTPException(status_code=503, detail="Artifact store unavailable") from e


@router.post("/{name}/propose", status_code=201)
async def propose_artifact(
    name: str, body: ProposeRequest, artifacts: ScopedArtifactDep, admin: AdminDep
) -> dict:
    """Propose a new version of an artifact (inactive until approved)."""
    try:
        return await _propose(
            artifacts, name, body.content, _who(admin, body), body.expected_version, body.rationale
        )
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to propose artifact", name=name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to propose artifact") from e


@router.patch("/{name}")
async def update_artifact(
    name: str, body: UpdateRequest, artifacts: ScopedArtifactDep, admin: AdminDep
) -> dict:
    """Write a new version and activate it at once."""
    try:
        return await _update(
            artifacts, name, body.content, _who(admin, body), body.expected_version
        )
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to update artifact", name=name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update artifact") from e


@router.post("/{name}/approve/{version}")
async def approve_artifact(
    name: str,
    version: int,
    artifacts: ScopedArtifactDep,
    admin: AdminDep,
    body: LabelRequest | None = None,
) -> dict:
    """Approve (activate) a specific version of an artifact."""
    try:
        return await _activate(artifacts, name, version, _who(admin, body), rollback=False)
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to approve artifact", name=name, version=version, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to approve artifact") from e


@router.post("/{name}/reject/{version}")
async def reject_artifact(
    name: str,
    version: int,
    artifacts: ScopedArtifactDep,
    admin: AdminDep,
    body: RejectRequest | None = None,
) -> dict:
    """Reject a pending version; the active version stays as it is."""
    try:
        return await _reject(
            artifacts, name, version, _who(admin, body), body.reason if body else None
        )
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to reject artifact", name=name, version=version, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to reject artifact") from e


@router.post("/{name}/rollback/{version}")
async def rollback_artifact(
    name: str,
    version: int,
    artifacts: ScopedArtifactDep,
    admin: AdminDep,
    body: LabelRequest | None = None,
) -> dict:
    """Reactivate an earlier version of an artifact."""
    try:
        return await _activate(artifacts, name, version, _who(admin, body), rollback=True)
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to rollback artifact", name=name, version=version, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to rollback artifact") from e


@router.post("/{name}/actions")
async def artifact_action(
    name: str, body: ArtifactActionRequest, artifacts: ScopedArtifactDep, admin: AdminDep
) -> dict:
    """Unified action endpoint: approve, rollback, reject, propose or update a named artifact."""
    try:
        if body.action in ("approve", "rollback", "reject"):
            if body.version is None:
                raise HTTPException(
                    status_code=422, detail=f"version is required for {body.action}"
                )
            if body.action == "reject":
                return await _reject(artifacts, name, body.version, _who(admin, body), body.reason)
            return await _activate(
                artifacts, name, body.version, _who(admin, body), rollback=body.action == "rollback"
            )
        if not body.content:
            raise HTTPException(status_code=422, detail=f"content is required for {body.action}")
        if body.action == "propose":
            return await _propose(
                artifacts,
                name,
                body.content,
                _who(admin, body),
                body.expected_version,
                body.rationale,
            )
        return await _update(
            artifacts, name, body.content, _who(admin, body), body.expected_version
        )
    except HTTPException:
        raise
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Artifact action failed", name=name, action=body.action, error=str(e))
        raise HTTPException(status_code=500, detail=f"Artifact action failed: {body.action}") from e


@router.delete("/{name}")
async def delete_artifact(name: str, artifacts: ScopedArtifactDep, admin: AdminDep) -> dict:
    """Delete every stored version; the profile's default text applies again."""
    try:
        count = await artifacts.delete(name, actor=Actor("host", admin.id))
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to delete artifact", name=name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete artifact") from e
    if count == 0:
        raise HTTPException(status_code=404, detail=f"No stored versions for artifact: {name}")
    return {"success": True, "name": name, "deleted_versions": count, "durable": artifacts.durable}


@proposals_router.get("")
async def list_proposals(
    request: Request,
    principal: PrincipalDep,
    profile: str | None = Query(None, description="One registered profile; omit for all of them"),
    status: str = Query("pending", pattern="^(pending|active|superseded|rejected)$"),
    limit: int = Query(100, ge=1, le=500, description="Page size, newest first"),
    before_id: int | None = Query(None, description="The previous page's next_before"),
) -> dict:
    """Versions in ``status`` (pending proposals by default) as records with their diff.

    An omitted ``profile`` lists every registered profile, so an owner reviews
    all waiting proposals in one place; the profiles' records are merged
    newest first before the page is cut. ``next_before`` continues to older
    records (null on the last page). The path sits outside ``/artifacts`` so
    it never shadows an artifact's own name.
    """
    service = get_artifact_service(request)
    try:
        names = [profile] if profile is not None else list(service.available_profiles)
        records: list[dict] = []
        for name in names:
            records.extend(await service.for_profile(name).proposals(status, limit + 1, before_id))
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to list proposals", error=str(e))
        raise HTTPException(status_code=503, detail="Artifact store unavailable") from e
    records.sort(key=lambda record: record["id"] or 0, reverse=True)
    page = records[:limit]
    more = len(records) > limit
    return {"proposals": page, "next_before": page[-1]["id"] if more and page else None}


@subjects_router.get("")
async def list_subjects(
    request: Request,
    principal: PrincipalDep,
    profile: str | None = Query(None, description="Registered assistant profile name"),
) -> dict:
    """The subjects a profile keeps versions for (outside ``/artifacts``, like the proposals)."""
    try:
        artifacts = get_artifact_service(request).for_profile(profile)
        return {"profile": artifacts.profile.name, "subjects": await artifacts.subjects()}
    except ArtifactError as e:
        raise _http_error(e) from e
    except Exception as e:
        logger.error("Failed to list subjects", error=str(e))
        raise HTTPException(status_code=503, detail="Artifact store unavailable") from e


@prompt_router.get("")
async def preview_prompt(
    request: Request,
    principal: PrincipalDep,
    profile: str | None = Query(None, description="Registered assistant profile name"),
    subject: str | None = Query(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"),
) -> dict:
    """The system prompt a turn of ``profile`` about ``subject`` would start from now.

    The same record as a message's prompt, with ``content``; there is no
    host context or session working memory, since no turn is running.
    """
    try:
        return await request.app.state.assistant_service.preview_prompt(profile, subject)
    except ArtifactError as e:
        raise _http_error(e) from e
    except ValueError as e:  # a required artifact without text
        raise HTTPException(status_code=422, detail=str(e)) from e
