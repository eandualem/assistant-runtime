"""Exception hierarchy for the access module."""

from assistant_runtime.base.exceptions import AssistantRuntimeError


class AccessError(AssistantRuntimeError):
    """Base exception for all access module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "access")
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class AuthenticationError(AccessError):
    """The caller could not be identified (401)."""


class AccessDeniedError(AccessError):
    """The caller is identified but may not do this (403)."""
