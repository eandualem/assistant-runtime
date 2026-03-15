"""OAuth service module — OpenAI subscription token management."""

from lovely_assistant.services.oauth.deps import OAuthServiceDep, get_oauth_service
from lovely_assistant.services.oauth.exceptions import (
    OAuthDeviceCodeError,
    OAuthError,
    OAuthNotConfiguredError,
    OAuthRefreshError,
    OAuthTokenExpiredError,
)
from lovely_assistant.services.oauth.interface import OAuthService

__all__ = [
    "OAuthService",
    "OAuthServiceDep",
    "get_oauth_service",
    "OAuthError",
    "OAuthTokenExpiredError",
    "OAuthRefreshError",
    "OAuthNotConfiguredError",
    "OAuthDeviceCodeError",
]
