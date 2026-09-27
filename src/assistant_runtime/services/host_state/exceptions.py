"""Exception hierarchy for the host state module."""

from assistant_runtime.base.exceptions import AssistantRuntimeError


class HostStateError(AssistantRuntimeError):
    """Base exception for all host state errors."""

    error_code = "host_state_error"

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "host_state")
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class HostStateNotFoundError(HostStateError):
    """Nothing is stored under that namespace and key."""

    error_code = "host_state_not_found"


class HostStateConflictError(HostStateError):
    """The stored version is not the one the write expected."""

    error_code = "host_state_conflict"

    def __init__(self, message: str, *, current_version: int | None = None, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.current_version = current_version
