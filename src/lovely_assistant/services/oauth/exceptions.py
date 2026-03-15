"""Exception hierarchy for the OAuth service module."""

from __future__ import annotations

from lovely_assistant.base.exceptions import LovelyAssistantError


class OAuthError(LovelyAssistantError):
    """Base exception for all OAuth module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "oauth")
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)


class OAuthTokenExpiredError(OAuthError):
    """OAuth token has expired and refresh failed."""

    def __init__(self, message: str = "OAuth token expired", **kwargs) -> None:
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class OAuthRefreshError(OAuthError):
    """Failed to refresh OAuth token."""

    def __init__(self, message: str = "OAuth token refresh failed", **kwargs) -> None:
        super().__init__(message, **kwargs)


class OAuthNotConfiguredError(OAuthError):
    """OAuth is not configured (missing encryption key or client ID)."""

    def __init__(self, message: str = "OAuth not configured", **kwargs) -> None:
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class OAuthDeviceCodeError(OAuthError):
    """Device code flow error."""

    def __init__(self, message: str = "Device code flow error", **kwargs) -> None:
        super().__init__(message, **kwargs)


class OAuthCodexSyncError(OAuthError):
    """Failed to sync OAuth state from the local Codex CLI."""

    def __init__(self, message: str = "Codex CLI sync failed", **kwargs) -> None:
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)
