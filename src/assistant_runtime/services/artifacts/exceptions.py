"""Exception hierarchy for the artifact service module."""

from assistant_runtime.base.exceptions import AssistantRuntimeError


class ArtifactError(AssistantRuntimeError):
    """Base exception for all artifact module errors."""

    error_code = "artifact_error"

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "artifacts")
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class UnknownArtifactError(ArtifactError):
    """The profile has no artifact of that name."""

    error_code = "unknown_artifact"


class ArtifactPermissionError(ArtifactError):
    """The artifact's policy does not allow this actor to perform the action."""

    error_code = "artifact_permission_denied"


class ArtifactConflictError(ArtifactError):
    """The caller's expected version is no longer the active one."""

    error_code = "artifact_version_conflict"


class ArtifactVersionNotFoundError(ArtifactError):
    """No stored version matches the request."""

    error_code = "artifact_version_not_found"
