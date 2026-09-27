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
    """The caller's expected version is no longer the active one.

    For a stale write, ``current_version`` and ``current_content`` carry the
    active text, so the writer can merge its change into it.
    """

    error_code = "artifact_version_conflict"

    def __init__(
        self,
        message: str,
        *,
        current_version: int | None = None,
        current_content: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, **kwargs)
        self.current_version = current_version
        self.current_content = current_content


class ArtifactVersionNotFoundError(ArtifactError):
    """No stored version matches the request."""

    error_code = "artifact_version_not_found"


class UnknownProfileError(ArtifactError):
    """No assistant profile with this name was registered at startup."""

    error_code = "unknown_profile"


class ArtifactSubjectRequiredError(ArtifactError):
    """The artifact is kept per subject, and the call named none."""

    error_code = "artifact_subject_required"
