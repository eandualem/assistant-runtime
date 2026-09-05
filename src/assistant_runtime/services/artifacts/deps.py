"""Dependency injection for the artifact service module."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from assistant_runtime.services.artifacts.interface import ArtifactService


def get_artifact_service(request: Request) -> ArtifactService:
    """Access ArtifactService from app.state (created during lifespan)."""
    service: ArtifactService | None = getattr(request.app.state, "artifact_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Artifact service not available")
    return service


ArtifactServiceDep = Annotated[ArtifactService, Depends(get_artifact_service)]
