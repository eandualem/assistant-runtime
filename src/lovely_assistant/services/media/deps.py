"""Dependency injection for the media service module."""

from typing import Annotated

from fastapi import Depends, Request

from lovely_assistant.services.media.interface import MediaService


def get_media_service(request: Request) -> MediaService:
    """Access MediaService from app.state (created during lifespan)."""
    return request.app.state.media_service


MediaServiceDep = Annotated[MediaService, Depends(get_media_service)]
