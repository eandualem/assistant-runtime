"""OAuth service module — OpenAI subscription token management."""

from assistant_runtime.services.oauth.deps import OAuthServiceDep, get_oauth_service
from assistant_runtime.services.oauth.exceptions import (
    OAuthCodexSyncError,
    OAuthDeviceCodeError,
    OAuthError,
    OAuthNotConfiguredError,
    OAuthRefreshError,
    OAuthTokenExpiredError,
)
from assistant_runtime.services.oauth.interface import OAuthService

__all__ = [
    "OAuthService",
    "OAuthServiceDep",
    "get_oauth_service",
    "OAuthError",
    "OAuthTokenExpiredError",
    "OAuthRefreshError",
    "OAuthNotConfiguredError",
    "OAuthDeviceCodeError",
    "OAuthCodexSyncError",
]
