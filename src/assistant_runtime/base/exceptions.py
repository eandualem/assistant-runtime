"""Exception hierarchy for the Assistant Runtime."""

from typing import Literal

Severity = Literal["low", "medium", "high", "critical"]


class AssistantRuntimeError(Exception):
    """Base exception for all Assistant Runtime errors."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "general",
        severity: Severity = "medium",
        retry_allowed: bool = True,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.retry_allowed = retry_allowed


class ConfigurationError(AssistantRuntimeError):
    """Invalid or missing configuration."""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(
            message,
            category="configuration",
            severity="critical",
            retry_allowed=False,
            **kwargs,
        )


class ExternalServiceError(AssistantRuntimeError):
    """Failed to communicate with an external service."""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(
            message,
            category="external_service",
            severity="high",
            **kwargs,
        )
