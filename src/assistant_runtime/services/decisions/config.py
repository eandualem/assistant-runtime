"""Startup policy for the decision capability; independent of backend model routing."""

from __future__ import annotations

from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DecisionsConfig(BaseModel):
    """Decision module configuration. Nested into AppSettings as ``decisions``.

    The capability is configured when the variable named by ``api_key_env``
    is set in the runtime's environment. There is no fallback to a language
    model: without a key the capability is absent and a call fails clearly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key_env: str = Field(default="TYPESAFE_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")
    model: str = Field(default="jev-latest", min_length=1, max_length=128)
    base_url: str = Field(default="https://api.typesafe.ai", min_length=1, max_length=2048)
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_questions: int = Field(default=32, ge=1, le=256)
    max_state_bytes: int = Field(default=262144, ge=1024, le=8388608)

    @field_validator("base_url")
    @classmethod
    def _https_unless_loopback(cls, value: str) -> str:
        """The key travels as a bearer header, so only TLS or the local machine may carry it."""
        parts = urlsplit(value)
        if parts.scheme == "https" and parts.hostname:
            return value
        if parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1"):
            return value
        raise ValueError("DECISIONS__BASE_URL must be an https:// URL (http:// only for localhost)")
