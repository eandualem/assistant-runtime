"""Tests for the exception hierarchy."""

from assistant_runtime.base.exceptions import (
    AssistantRuntimeError,
    ConfigurationError,
    ExternalServiceError,
)


def test_base_error_defaults():
    err = AssistantRuntimeError("something failed")
    assert str(err) == "something failed"
    assert err.category == "general"
    assert err.severity == "medium"
    assert err.retry_allowed is True


def test_base_error_custom_fields():
    err = AssistantRuntimeError(
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
    assert isinstance(err, AssistantRuntimeError)
    assert err.category == "configuration"
    assert err.severity == "critical"
    assert err.retry_allowed is False


def test_external_service_error():
    err = ExternalServiceError("timeout calling LLM")
    assert isinstance(err, AssistantRuntimeError)
    assert err.category == "external_service"
    assert err.severity == "high"
    assert err.retry_allowed is True


def test_exception_inheritance():
    err = ConfigurationError("test")
    assert isinstance(err, AssistantRuntimeError)
    assert isinstance(err, Exception)
