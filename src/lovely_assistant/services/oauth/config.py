"""Configuration for the OAuth service module."""

from pydantic import BaseModel, ConfigDict, Field

# Public OAuth client ID from the open-source Codex CLI (Apache-2.0).
# PKCE public client — no secret, no registration required.
OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"


class OAuthConfig(BaseModel):
    """OAuth module configuration. Nested into AppSettings as `oauth`."""

    model_config = ConfigDict(frozen=True)

    encryption_key: str = Field(
        default="",
        description="Fernet key for token encryption. Empty = OAuth disabled.",
    )
    token_url: str = Field(
        default="https://auth.openai.com/oauth/token",
        description="OpenAI token endpoint",
    )
    device_code_url: str = Field(
        default="https://auth.openai.com/api/accounts/deviceauth/usercode",
        description="OpenAI device auth usercode endpoint",
    )
    device_auth_token_url: str = Field(
        default="https://auth.openai.com/api/accounts/deviceauth/token",
        description="OpenAI device auth polling endpoint",
    )
    device_callback_url: str = Field(
        default="https://auth.openai.com/deviceauth/callback",
        description="OpenAI device auth redirect URI used for authorization-code exchange",
    )
    device_verification_uri: str = Field(
        default="https://auth.openai.com/codex/device",
        description="Verification URL the user opens during device auth",
    )
    scopes: str = Field(
        default="openid profile email offline_access model.request api.responses.write",
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
    codex_auth_file: str = Field(
        default="~/.codex/auth.json",
        description="Path to Codex CLI auth.json for local ChatGPT/Codex auth sync.",
    )
    codex_auto_sync: bool = Field(
        default=False,
        description="Automatically sync local Codex CLI auth when no stored OAuth token exists.",
    )
