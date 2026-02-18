"""Exception hierarchy for the LLM service module."""

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
