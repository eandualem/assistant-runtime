"""Configuration for the OAuth service module."""

from pydantic import BaseModel, ConfigDict, Field


class OAuthConfig(BaseModel):
    """OAuth module configuration. Nested into AppSettings as `oauth`."""

    model_config = ConfigDict(frozen=True)

    encryption_key: str = Field(
        default="",
        description="Fernet key for token encryption. Empty = OAuth disabled.",
    )
    client_id: str = Field(
        default="",
        description="OpenAI OAuth client ID",
    )
    auth_url: str = Field(
        default="https://auth.openai.com/authorize",
        description="OpenAI authorization endpoint",
    )
    token_url: str = Field(
        default="https://auth.openai.com/oauth/token",
        description="OpenAI token endpoint",
    )
    device_code_url: str = Field(
        default="https://auth.openai.com/oauth/device/code",
        description="OpenAI device code endpoint",
    )
    api_key_exchange_url: str = Field(
        default="https://api.openai.com/v1/organization/api_keys/exchange",
        description="OpenAI API key exchange endpoint",
    )
    scopes: str = Field(
        default="openid profile email offline_access model.request.all",
        description="Space-separated OAuth scopes",
    )
    refresh_buffer_seconds: int = Field(
        default=3600,
        description="Refresh if expiry is within this window (seconds)",
    )
    device_code_poll_interval: int = Field(
        default=5,
        description="Seconds between polls during device code flow",
    )
    device_code_timeout: int = Field(
        default=900,
        description="Max seconds to wait for user authorization (15 min)",
    )
