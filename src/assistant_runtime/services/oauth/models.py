"""OAuth flow state and subscription credential models."""

from enum import StrEnum

from pydantic import BaseModel


class DeviceCodeStatus(StrEnum):
    """State of the device code authorization flow."""

    IDLE = "idle"
    POLLING = "polling"
    AUTHORIZED = "authorized"
    EXPIRED = "expired"
    ERROR = "error"


class AuthSource(StrEnum):
    """How the current OpenAI auth session was obtained."""

    DEVICE_CODE = "device_code"
    CODEX_CLI = "codex_cli"
    DATABASE = "database"


class DeviceCodeResponse(BaseModel):
    """Response from initiating a device code flow."""

    user_code: str
    verification_uri: str
    expires_in: int


class AuthStatus(BaseModel):
    """Current OAuth connection status."""

    connected: bool = False
    status: DeviceCodeStatus = DeviceCodeStatus.IDLE
    source: AuthSource | None = None
    email: str | None = None
    api_key_preview: str | None = None
    expires_at: float | None = None
    error: str | None = None
    persisted: bool = False


class CodexSession(BaseModel):
    """Current ChatGPT/Codex auth context used for backend requests."""

    access_token: str
    account_id: str
    expires_at: float | None = None
    email: str | None = None
    source: AuthSource | None = None


class CodexCliAuth(BaseModel):
    """Relevant OAuth state imported from the local Codex CLI."""

    access_token: str
    refresh_token: str
    id_token: str | None = None
    account_id: str | None = None
    auth_mode: str | None = None
    email: str | None = None
