"""Exception hierarchy for the LLM service module."""

from __future__ import annotations

from lovely_assistant.base.exceptions import LovelyAssistantError


class LLMError(LovelyAssistantError):
    """Base exception for all LLM module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "llm")
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)


class ProviderConfigError(LLMError):
    """Invalid provider configuration or model ID."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "critical")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class LLMCallError(LLMError):
    """LLM call execution failed."""

    def __init__(
        self,
        message: str,
        *,
        error_category: str = "unknown",
        is_retryable: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("retry_allowed", is_retryable)
        super().__init__(message, **kwargs)
        self.error_category = error_category
        self.is_retryable = is_retryable


def classify_llm_error(exc: Exception) -> LLMCallError:
    """Classify a provider exception into a categorized LLMCallError.

    Lazily imports provider-specific exception types to avoid hard-coupling
    to optional SDKs (anthropic, openai, google-genai).
    """
    exc_type = type(exc)
    exc_name = exc_type.__name__

    # Build a mapping of exception class names to (category, is_retryable)
    # All three provider SDKs (anthropic, openai, google) share the same
    # exception naming convention.
    retryable_names: dict[str, tuple[str, bool]] = {
        "RateLimitError": ("RATE_LIMIT", True),
        "InternalServerError": ("SERVER_ERROR", True),
        "APIConnectionError": ("CONNECTION_ERROR", True),
        "APITimeoutError": ("TIMEOUT", True),
    }
    non_retryable_names: dict[str, tuple[str, bool]] = {
        "AuthenticationError": ("AUTH_ERROR", False),
        "BadRequestError": ("CLIENT_ERROR", False),
    }

    # Check by class name (works across all provider SDKs)
    if exc_name in retryable_names:
        cat, retryable = retryable_names[exc_name]
        return LLMCallError(
            f"LLM call failed ({cat}): {exc}",
            error_category=cat,
            is_retryable=retryable,
        )

    if exc_name in non_retryable_names:
        cat, retryable = non_retryable_names[exc_name]
        return LLMCallError(
            f"LLM call failed ({cat}): {exc}",
            error_category=cat,
            is_retryable=retryable,
        )

    # Also check for standard Python transient errors
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return LLMCallError(
            f"LLM call failed (CONNECTION_ERROR): {exc}",
            error_category="CONNECTION_ERROR",
            is_retryable=True,
        )

    # Fallback — unknown error
    return LLMCallError(
        f"LLM call failed: {exc}",
        error_category="UNKNOWN",
        is_retryable=False,
    )
