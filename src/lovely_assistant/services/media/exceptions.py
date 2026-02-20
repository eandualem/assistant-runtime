"""Exception hierarchy for the media service module."""

from lovely_assistant.base.exceptions import LovelyAssistantError


class MediaError(LovelyAssistantError):
    """Base exception for all media module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "media")
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)


class ProviderError(MediaError):
    """Provider API call failed."""


class ProviderNotConfiguredError(MediaError):
    """Required API key is missing for the requested provider."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "critical")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class ImageNotFoundError(MediaError):
    """Requested image not found in cache (expired or never existed)."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "low")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class ContentPolicyError(MediaError):
    """Image generation rejected due to content policy violation."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class VideoJobError(MediaError):
    """Video generation job encountered an error."""


class VideoTimeoutError(VideoJobError):
    """Video generation job timed out."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", True)
        super().__init__(message, **kwargs)
