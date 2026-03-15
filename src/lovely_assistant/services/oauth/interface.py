"""OAuth service — manages OpenAI OAuth tokens via Device Code flow."""

from __future__ import annotations

import asyncio
import os
import time
from enum import Enum
from typing import Any

import httpx
from cryptography.fernet import Fernet
from loguru import logger
from pydantic import BaseModel

from lovely_assistant.services.oauth.config import OAuthConfig
from lovely_assistant.services.oauth.exceptions import (
    OAuthDeviceCodeError,
    OAuthNotConfiguredError,
    OAuthRefreshError,
    OAuthTokenExpiredError,
)


class DeviceCodeStatus(str, Enum):
    """State of the device code authorization flow."""

    IDLE = "idle"
    POLLING = "polling"
    AUTHORIZED = "authorized"
    EXPIRED = "expired"
    ERROR = "error"


class DeviceCodeResponse(BaseModel):
    """Response from initiating a device code flow."""

    user_code: str
    verification_uri: str
    expires_in: int


class AuthStatus(BaseModel):
    """Current OAuth connection status."""

    connected: bool = False
    status: DeviceCodeStatus = DeviceCodeStatus.IDLE
    email: str | None = None
    api_key_preview: str | None = None
    expires_at: float | None = None
    error: str | None = None


class OAuthService:
    """Manages OpenAI OAuth tokens — Device Code flow, encryption, refresh."""

    def __init__(self, config: OAuthConfig) -> None:
        self._config = config
        self._fernet: Fernet | None = None
        self._http: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task | None = None

        # In-memory state
        self._api_key: str | None = None
        self._refresh_token: str | None = None
        self._id_token: str | None = None
        self._expires_at: float = 0.0
        self._email: str | None = None

        # Device code flow state
        self._device_code_status = DeviceCodeStatus.IDLE
        self._device_code_error: str | None = None
        self._pending_device_code: str | None = None

        # Database service (set after startup via set_database_service)
        self._db_service: Any = None

    @property
    def configured(self) -> bool:
        """Whether OAuth is configured (has encryption key and client ID)."""
        return bool(self._config.encryption_key and self._config.client_id)

    def set_database_service(self, db_service: Any) -> None:
        """Inject database service after lifecycle registration."""
        self._db_service = db_service

    async def start(self) -> None:
        """Initialize Fernet cipher and load stored token from DB."""
        if not self.configured:
            logger.info("OAuth not configured — skipping initialization")
            return

        self._fernet = Fernet(self._config.encryption_key.encode())
        self._http = httpx.AsyncClient(timeout=30.0)

        # Load stored token from DB if available
        if self._db_service is not None:
            await self._load_stored_token()

    async def stop(self) -> None:
        """Cancel polling task and close HTTP client."""
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def health_check(self) -> dict[str, Any]:
        """Report OAuth connection health."""
        if not self.configured:
            return {"status": "disabled", "reason": "not_configured"}

        connected = self._api_key is not None and not self.needs_refresh()
        return {
            "status": "connected" if connected else "disconnected",
            "email": self._email,
            "expires_at": self._expires_at if self._expires_at > 0 else None,
        }

    async def initiate_device_code(self) -> DeviceCodeResponse:
        """Start the OAuth Device Code flow — returns user_code and verification_uri."""
        if not self.configured:
            raise OAuthNotConfiguredError()
        if self._http is None:
            raise OAuthNotConfiguredError("OAuth service not started")

        response = await self._http.post(
            self._config.device_code_url,
            data={
                "client_id": self._config.client_id,
                "scope": self._config.scopes,
            },
        )

        if response.status_code != 200:
            raise OAuthDeviceCodeError(
                f"Device code request failed: {response.status_code} {response.text}"
            )

        data = response.json()
        self._pending_device_code = data["device_code"]
        self._device_code_status = DeviceCodeStatus.POLLING
        self._device_code_error = None

        # Start background polling
        self._poll_task = asyncio.create_task(
            self._poll_for_token(
                device_code=data["device_code"],
                interval=data.get("interval", self._config.device_code_poll_interval),
                expires_in=data.get("expires_in", self._config.device_code_timeout),
            )
        )

        return DeviceCodeResponse(
            user_code=data["user_code"],
            verification_uri=data["verification_uri"],
            expires_in=data.get("expires_in", self._config.device_code_timeout),
        )

    def get_device_code_status(self) -> AuthStatus:
        """Get current authorization flow status."""
        connected = self._api_key is not None and not self.needs_refresh()
        return AuthStatus(
            connected=connected,
            status=self._device_code_status if not connected else DeviceCodeStatus.AUTHORIZED,
            email=self._email,
            api_key_preview=f"sk-...{self._api_key[-4:]}" if self._api_key else None,
            expires_at=self._expires_at if self._expires_at > 0 else None,
            error=self._device_code_error,
        )

    def get_openai_api_key(self) -> str | None:
        """Get the current OpenAI API key, or None if not connected."""
        return self._api_key

    def needs_refresh(self) -> bool:
        """Check if the API key expires within the refresh buffer."""
        if self._expires_at <= 0:
            return self._api_key is None
        return time.time() > (self._expires_at - self._config.refresh_buffer_seconds)

    async def refresh(self) -> None:
        """Refresh the OAuth token chain: refresh_token -> id_token -> API key."""
        if not self.configured:
            raise OAuthNotConfiguredError()
        if self._refresh_token is None:
            raise OAuthTokenExpiredError("No refresh token available — re-authenticate required")
        if self._http is None:
            raise OAuthNotConfiguredError("OAuth service not started")

        logger.info("Refreshing OpenAI OAuth token")

        # Step 1: Use refresh_token to get new id_token
        response = await self._http.post(
            self._config.token_url,
            data={
                "grant_type": "refresh_token",
                "client_id": self._config.client_id,
                "refresh_token": self._refresh_token,
            },
        )

        if response.status_code != 200:
            raise OAuthRefreshError(f"Token refresh failed: {response.status_code} {response.text}")

        token_data = response.json()
        self._id_token = token_data["id_token"]
        if "refresh_token" in token_data:
            self._refresh_token = token_data["refresh_token"]

        # Step 2: Exchange id_token for API key
        await self._exchange_for_api_key(self._id_token)

        # Step 3: Persist to DB
        await self._save_token()

        # Step 4: Export to environment
        self._export_api_key()

        logger.info("OpenAI OAuth token refreshed successfully")

    async def disconnect(self) -> None:
        """Disconnect — delete tokens from DB and clear env var."""
        self._api_key = None
        self._refresh_token = None
        self._id_token = None
        self._expires_at = 0.0
        self._email = None
        self._device_code_status = DeviceCodeStatus.IDLE

        # Clear env var
        os.environ.pop("OPENAI_API_KEY", None)

        # Delete from DB
        if self._db_service is not None:
            async with self._db_service.session_context() as session:
                from lovely_assistant.services.database.repositories import OAuthTokenRepository

                repo = OAuthTokenRepository(session)
                await repo.delete("openai")
                await session.commit()

        logger.info("OpenAI OAuth disconnected")

    # --- Private methods ---

    async def _poll_for_token(
        self, device_code: str, interval: int, expires_in: int
    ) -> None:
        """Poll OpenAI token endpoint during device code flow."""
        deadline = time.time() + expires_in

        while time.time() < deadline:
            await asyncio.sleep(interval)

            try:
                response = await self._http.post(
                    self._config.token_url,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "client_id": self._config.client_id,
                        "device_code": device_code,
                    },
                )

                if response.status_code == 200:
                    token_data = response.json()
                    self._id_token = token_data.get("id_token")
                    self._refresh_token = token_data.get("refresh_token")

                    # Extract email from id_token claims if present
                    self._email = token_data.get("email")

                    # Exchange id_token for API key
                    await self._exchange_for_api_key(self._id_token)
                    await self._save_token()
                    self._export_api_key()

                    self._device_code_status = DeviceCodeStatus.AUTHORIZED
                    logger.info("OpenAI OAuth authorized", email=self._email)
                    return

                error_data = response.json()
                error_code = error_data.get("error", "")

                if error_code == "authorization_pending":
                    continue
                elif error_code == "slow_down":
                    interval = min(interval + 5, 30)
                    continue
                elif error_code == "expired_token":
                    self._device_code_status = DeviceCodeStatus.EXPIRED
                    self._device_code_error = "Device code expired — please try again"
                    return
                else:
                    self._device_code_status = DeviceCodeStatus.ERROR
                    self._device_code_error = f"Authorization failed: {error_code}"
                    return

            except (httpx.HTTPError, Exception) as exc:
                logger.warning("Device code poll error", error=str(exc))
                continue

        self._device_code_status = DeviceCodeStatus.EXPIRED
        self._device_code_error = "Device code flow timed out"

    async def _exchange_for_api_key(self, id_token: str) -> None:
        """Exchange id_token for an OpenAI API key."""
        response = await self._http.post(
            self._config.api_key_exchange_url,
            headers={"Authorization": f"Bearer {id_token}"},
            json={"name": "lovely-assistant-oauth"},
        )

        if response.status_code != 200:
            raise OAuthRefreshError(
                f"API key exchange failed: {response.status_code} {response.text}"
            )

        data = response.json()
        self._api_key = data["key"]
        # API keys from exchange typically have their own expiry
        self._expires_at = time.time() + data.get("expires_in", 7 * 24 * 3600)

    def _export_api_key(self) -> None:
        """Set the API key in os.environ for Pydantic AI auto-detection."""
        if self._api_key:
            os.environ["OPENAI_API_KEY"] = self._api_key
            logger.debug("OPENAI_API_KEY exported to environment")

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt a string with Fernet."""
        if self._fernet is None:
            raise OAuthNotConfiguredError("Fernet cipher not initialized")
        return self._fernet.encrypt(plaintext.encode()).decode()

    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt a Fernet-encrypted string."""
        if self._fernet is None:
            raise OAuthNotConfiguredError("Fernet cipher not initialized")
        return self._fernet.decrypt(ciphertext.encode()).decode()

    async def _load_stored_token(self) -> None:
        """Load stored OAuth token from database and export to env."""
        try:
            async with self._db_service.session_context() as session:
                from lovely_assistant.services.database.repositories import OAuthTokenRepository

                repo = OAuthTokenRepository(session)
                token = await repo.get("openai")

                if token is None:
                    logger.debug("No stored OAuth token found")
                    return

                if token.encrypted_api_key:
                    self._api_key = self._decrypt(token.encrypted_api_key)
                if token.encrypted_refresh_token:
                    self._refresh_token = self._decrypt(token.encrypted_refresh_token)
                if token.encrypted_id_token:
                    self._id_token = self._decrypt(token.encrypted_id_token)
                self._expires_at = token.expires_at or 0.0
                self._email = token.email

                if self._api_key and not self.needs_refresh():
                    self._export_api_key()
                    self._device_code_status = DeviceCodeStatus.AUTHORIZED
                    logger.info("Loaded stored OpenAI OAuth token", email=self._email)
                elif self._api_key and self._refresh_token:
                    logger.info("Stored OAuth token needs refresh")
                    await self.refresh()

        except Exception as exc:
            logger.warning("Failed to load stored OAuth token", error=str(exc))

    async def _save_token(self) -> None:
        """Save current token state to database."""
        if self._db_service is None:
            return

        async with self._db_service.session_context() as session:
            from lovely_assistant.services.database.repositories import OAuthTokenRepository

            repo = OAuthTokenRepository(session)
            await repo.upsert(
                provider="openai",
                encrypted_api_key=self._encrypt(self._api_key) if self._api_key else None,
                encrypted_refresh_token=self._encrypt(self._refresh_token) if self._refresh_token else None,
                encrypted_id_token=self._encrypt(self._id_token) if self._id_token else None,
                expires_at=self._expires_at,
                email=self._email,
            )
            await session.commit()
