"""Tests for the LLM service exception hierarchy and error classification."""

from pydantic_ai.exceptions import ModelHTTPError

from lovely_assistant.services.llm.exceptions import LLMCallError, classify_llm_error


class TestLLMCallError:
    """LLMCallError construction and attribute defaults."""

    def test_default_error_category_is_unknown(self):
        err = LLMCallError("something broke")
        assert err.error_category == "unknown"

    def test_default_is_retryable_is_false(self):
        err = LLMCallError("something broke")
        assert err.is_retryable is False

    def test_custom_error_category_and_retryable(self):
        err = LLMCallError(
            "rate limited",
            error_category="RATE_LIMIT",
            is_retryable=True,
        )
        assert err.error_category == "RATE_LIMIT"
        assert err.is_retryable is True

    def test_retry_allowed_set_from_is_retryable(self):
        """is_retryable propagates to the base class retry_allowed attribute."""
        err_retryable = LLMCallError("transient", is_retryable=True)
        assert err_retryable.retry_allowed is True

        err_not_retryable = LLMCallError("permanent", is_retryable=False)
        assert err_not_retryable.retry_allowed is False

    def test_message_preserved(self):
        err = LLMCallError("specific failure message")
        assert str(err) == "specific failure message"


class TestClassifyLlmError:
    """classify_llm_error maps provider exceptions by class name."""

    # --- Stub exception classes to simulate provider SDK exceptions ---

    def test_rate_limit_error(self):
        class RateLimitError(Exception):
            pass

        result = classify_llm_error(RateLimitError("too many requests"))
        assert result.error_category == "RATE_LIMIT"
        assert result.is_retryable is True
        assert result.retry_allowed is True

    def test_internal_server_error(self):
        class InternalServerError(Exception):
            pass

        result = classify_llm_error(InternalServerError("500"))
        assert result.error_category == "SERVER_ERROR"
        assert result.is_retryable is True
        assert result.retry_allowed is True

    def test_api_connection_error(self):
        class APIConnectionError(Exception):
            pass

        result = classify_llm_error(APIConnectionError("connection refused"))
        assert result.error_category == "CONNECTION_ERROR"
        assert result.is_retryable is True
        assert result.retry_allowed is True

    def test_authentication_error(self):
        class AuthenticationError(Exception):
            pass

        result = classify_llm_error(AuthenticationError("invalid key"))
        assert result.error_category == "AUTH_ERROR"
        assert result.is_retryable is False
        assert result.retry_allowed is False

    def test_bad_request_error(self):
        class BadRequestError(Exception):
            pass

        result = classify_llm_error(BadRequestError("malformed input"))
        assert result.error_category == "CLIENT_ERROR"
        assert result.is_retryable is False
        assert result.retry_allowed is False

    def test_unknown_exception_fallback(self):
        result = classify_llm_error(ValueError("unexpected"))
        assert result.error_category == "UNKNOWN"
        assert result.is_retryable is False
        assert result.retry_allowed is False

    def test_python_connection_error(self):
        result = classify_llm_error(ConnectionError("network down"))
        assert result.error_category == "CONNECTION_ERROR"
        assert result.is_retryable is True
        assert result.retry_allowed is True

    def test_python_timeout_error(self):
        result = classify_llm_error(TimeoutError("timed out"))
        assert result.error_category == "CONNECTION_ERROR"
        assert result.is_retryable is True
        assert result.retry_allowed is True

    def test_model_http_error_429_maps_to_rate_limit(self):
        result = classify_llm_error(
            ModelHTTPError(
                status_code=429,
                model_name="claude-haiku-4-5",
                body={
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": "too many tokens",
                    },
                },
            )
        )
        assert result.error_category == "RATE_LIMIT"
        assert result.is_retryable is True
        assert result.retry_allowed is True
