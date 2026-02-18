"""Tests for the exception hierarchy."""

from lovely_assistant.base.exceptions import (
    ConfigurationError,
    ExternalServiceError,
    LovelyAssistantError,
)


def test_base_error_defaults():
    err = LovelyAssistantError("something failed")
    assert str(err) == "something failed"
    assert err.category == "general"
    assert err.severity == "medium"
    assert err.retry_allowed is True


def test_base_error_custom_fields():
    err = LovelyAssistantError(
        "custom",
        category="llm",
        severity="high",
        retry_allowed=False,
    )
    assert err.category == "llm"
    assert err.severity == "high"
    assert err.retry_allowed is False


def test_configuration_error():
    err = ConfigurationError("missing API key")
    assert isinstance(err, LovelyAssistantError)
    assert err.category == "configuration"
    assert err.severity == "critical"
    assert err.retry_allowed is False


def test_external_service_error():
    err = ExternalServiceError("timeout calling LLM")
    assert isinstance(err, LovelyAssistantError)
    assert err.category == "external_service"
    assert err.severity == "high"
    assert err.retry_allowed is True


def test_exception_inheritance():
    err = ConfigurationError("test")
    assert isinstance(err, LovelyAssistantError)
    assert isinstance(err, Exception)
