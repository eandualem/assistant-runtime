"""Dependency injection for the OAuth service module."""

from typing import Annotated

from fastapi import Depends, Request

from lovely_assistant.services.oauth.interface import OAuthService


def get_oauth_service(request: Request) -> OAuthService:
    """Access OAuthService from app.state (created during lifespan)."""
    return request.app.state.oauth_service


OAuthServiceDep = Annotated[OAuthService, Depends(get_oauth_service)]
