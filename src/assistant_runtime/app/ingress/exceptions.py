"""Exception hierarchy for the ingress module."""

from assistant_runtime.base.exceptions import AssistantRuntimeError


class IngressError(AssistantRuntimeError):
    """Base exception for message ingress errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "ingress")
        kwargs.setdefault("severity", "medium")
        super().__init__(message, **kwargs)
