"""Exception hierarchy for the LLM service module."""

from __future__ import annotations

from assistant_runtime.base.exceptions import AssistantRuntimeError


class LLMError(AssistantRuntimeError):
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


class ProviderKeyStoreUnavailableError(ProviderConfigError):
    """Stored provider keys need the encryption key and a reachable database."""


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


def _classify_model_http_error(exc: Exception) -> tuple[str, bool] | None:
    """Map Pydantic AI ModelHTTPError instances to a provider error category."""
    if type(exc).__name__ != "ModelHTTPError":
        return None

    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    provider_error_type: str | None = None

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            raw_type = error.get("type")
            if isinstance(raw_type, str):
                provider_error_type = raw_type

    if status_code == 429 or provider_error_type == "rate_limit_error":
        return "RATE_LIMIT", True
    if provider_error_type == "overloaded_error":
        return "SERVER_ERROR", True
    if status_code == 408:
        return "TIMEOUT", True
    if status_code in {401, 403}:
        return "AUTH_ERROR", False
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return "CLIENT_ERROR", False
    if isinstance(status_code, int) and status_code >= 500:
        return "SERVER_ERROR", True

    return None


def _classify_model_api_error(exc: Exception) -> tuple[str, bool] | None:
    """Classify Pydantic AI's non-HTTP ``ModelAPIError`` (timeouts, interrupted requests).

    The provider SDK's own exception is usually chained as ``__cause__`` and
    is classified first by name; otherwise the message decides. A provider
    request that failed for any other reason is a retryable provider error,
    never an internal one.
    """
    try:
        from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
    except ImportError:  # pragma: no cover
        return None
    if not isinstance(exc, ModelAPIError) or isinstance(exc, ModelHTTPError):
        return None
    cause = exc.__cause__ or exc.__context__
    if isinstance(cause, Exception) and cause is not exc:
        nested = classify_llm_error(cause)
        if nested.error_category != "UNKNOWN":
            return nested.error_category, nested.is_retryable
    message = str(exc).lower()
    if "timed out" in message or "timeout" in message:
        return "TIMEOUT", True
    if "connection" in message or "interrupted" in message:
        return "CONNECTION_ERROR", True
    return "SERVER_ERROR", True


_PROVIDER_ERRORS = {
    "RateLimitError": ("RATE_LIMIT", True),
    "InternalServerError": ("SERVER_ERROR", True),
    "APIConnectionError": ("CONNECTION_ERROR", True),
    "APITimeoutError": ("TIMEOUT", True),
    "AuthenticationError": ("AUTH_ERROR", False),
    "BadRequestError": ("CLIENT_ERROR", False),
}


def classify_llm_error(exc: Exception) -> LLMCallError:
    """Classify native/provider failures without importing optional provider SDKs."""
    classification = (
        _PROVIDER_ERRORS.get(type(exc).__name__)
        or _classify_model_http_error(exc)
        or _classify_model_api_error(exc)
    )
    if classification is None:
        if isinstance(exc, TimeoutError):
            classification = ("TIMEOUT", True)
        elif isinstance(exc, ConnectionError):
            classification = ("CONNECTION_ERROR", True)
        else:
            classification = ("UNKNOWN", False)
    category, retryable = classification
    label = f" ({category})" if category != "UNKNOWN" else ""
    return LLMCallError(
        f"LLM call failed{label}: {exc}",
        error_category=category,
        is_retryable=retryable,
    )
