"""OAuth service — manages OpenAI ChatGPT/Codex subscription auth.

Uses the public Codex CLI OAuth client ID (PKCE, no secret required).
Flows:
- Device Code → user authorizes → OAuth access/id/refresh tokens
- Codex CLI sync → import tokens from ~/.codex/auth.json
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from cryptography.fernet import Fernet
from loguru import logger
from pydantic import BaseModel

from assistant_runtime.services.oauth.config import OPENAI_OAUTH_CLIENT_ID, OAuthConfig
from assistant_runtime.services.oauth.exceptions import (
    OAuthCodexSyncError,
    OAuthDeviceCodeError,
    OAuthNotConfiguredError,
    OAuthRefreshError,
    OAuthTokenExpiredError,
)


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


class OAuthService:
    """Manages OpenAI OAuth tokens — Device Code, Codex CLI sync, refresh."""

    def __init__(self, config: OAuthConfig) -> None:
        self._config = config
        self._fernet: Fernet | None = None
        self._http: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task | None = None

        # In-memory state
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._id_token: str | None = None
        self._account_id: str | None = None
        self._expires_at: float = 0.0
        self._email: str | None = None
        self._auth_source: AuthSource | None = None

        # Device code flow state
        self._device_code_status = DeviceCodeStatus.IDLE
        self._device_code_error: str | None = None
        self._pending_device_code: str | None = None

        # Database service (set after startup via set_database_service)
        self._db_service: Any = None

    @property
    def configured(self) -> bool:
        """Whether OAuth is configured (has encryption key)."""
        return bool(self._config.encryption_key)

    def set_database_service(self, db_service: Any) -> None:
        """Inject database service after lifecycle registration."""
        self._db_service = db_service

    async def start(self) -> None:
        """Initialize Fernet cipher and load stored token from DB."""
        if not self.configured:
            logger.info("OAuth not configured — set OAUTH__ENCRYPTION_KEY to enable")
            return

        self._fernet = Fernet(self._config.encryption_key.encode())
        self._http = httpx.AsyncClient(timeout=30.0)

        # Load stored token from DB if available
        loaded = False
        if self._db_service is not None:
            loaded = await self._load_stored_token()

        if not loaded and self._config.codex_auto_sync:
            try:
                await self.sync_from_codex_cli()
            except OAuthCodexSyncError as exc:
                logger.debug("Codex CLI auth sync unavailable", error=str(exc))

    async def stop(self) -> None:
        """Cancel polling task and close HTTP client."""
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None

        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def health_check(self) -> dict[str, Any]:
        """Report OAuth connection health."""
        if not self.configured:
            return {"status": "disabled", "reason": "not_configured"}

        connected = self.get_codex_session() is not None
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
            json={
                "client_id": OPENAI_OAUTH_CLIENT_ID,
            },
        )

        if response.status_code != 200:
            raise OAuthDeviceCodeError(
                f"Device code request failed: {response.status_code} {response.text}"
            )

        data = response.json()
        interval = self._parse_interval_seconds(data.get("interval"))
        self._pending_device_code = data["device_auth_id"]
        self._device_code_status = DeviceCodeStatus.POLLING
        self._device_code_error = None

        # Start background polling
        self._poll_task = asyncio.create_task(
            self._poll_for_token(
                device_auth_id=data["device_auth_id"],
                user_code=data["user_code"],
                interval=interval,
                expires_in=self._config.device_code_timeout,
            )
        )

        return DeviceCodeResponse(
            user_code=data["user_code"],
            verification_uri=self._config.device_verification_uri,
            expires_in=self._config.device_code_timeout,
        )

    def get_device_code_status(self) -> AuthStatus:
        """Get current authorization flow status."""
        connected = self.get_codex_session() is not None
        return AuthStatus(
            connected=connected,
            status=self._device_code_status if not connected else DeviceCodeStatus.AUTHORIZED,
            source=self._auth_source,
            email=self._email,
            api_key_preview=None,
            expires_at=self._expires_at if self._expires_at > 0 else None,
            error=self._device_code_error,
        )

    def get_codex_session(self) -> CodexSession | None:
        """Return the current ChatGPT/Codex session if it is usable."""
        if (
            self._access_token is None
            or self._account_id is None
            or self._is_access_token_expired()
        ):
            return None
        return CodexSession(
            access_token=self._access_token,
            account_id=self._account_id,
            expires_at=self._expires_at if self._expires_at > 0 else None,
            email=self._email,
            source=self._auth_source,
        )

    def get_openai_api_key(self) -> str | None:
        """Legacy API-key accessor.

        ChatGPT/Codex subscription auth does not yield a reusable OpenAI API key.
        """
        return None

    def needs_refresh(self) -> bool:
        """Check if the access token expires within the refresh buffer."""
        if self._expires_at <= 0:
            return self._access_token is None or self._account_id is None
        return time.time() > (self._expires_at - self._config.refresh_buffer_seconds)

    def _is_access_token_expired(self) -> bool:
        """Check whether the current access token has actually expired."""
        if self._expires_at <= 0:
            return self._access_token is None or self._account_id is None
        return time.time() > self._expires_at

    async def refresh(self) -> None:
        """Refresh the OAuth token chain: refresh_token → new access/id tokens."""
        if not self.configured:
            raise OAuthNotConfiguredError()
        if self._refresh_token is None:
            raise OAuthTokenExpiredError("No refresh token available — re-authenticate required")
        if self._http is None:
            raise OAuthNotConfiguredError("OAuth service not started")

        logger.info("Refreshing OpenAI OAuth token")

        # Step 1: Use refresh_token to get new access/id tokens
        response = await self._http.post(
            self._config.token_url,
            json={
                "grant_type": "refresh_token",
                "client_id": OPENAI_OAUTH_CLIENT_ID,
                "refresh_token": self._refresh_token,
            },
        )

        if response.status_code != 200:
            raise OAuthRefreshError(f"Token refresh failed: {response.status_code} {response.text}")

        token_data = response.json()
        self._access_token = token_data.get("access_token")
        self._id_token = token_data.get("id_token")
        if "refresh_token" in token_data:
            self._refresh_token = token_data["refresh_token"]
        self._sync_token_metadata()
        await self._save_token()

        logger.info("OpenAI OAuth token refreshed successfully")

    async def sync_from_codex_cli(self) -> AuthStatus:
        """Import ChatGPT/Codex OAuth tokens from the local Codex CLI."""
        if not self.configured:
            raise OAuthNotConfiguredError()
        if self._http is None:
            raise OAuthNotConfiguredError("OAuth service not started")

        tokens = self._read_codex_cli_auth()
        self._access_token = tokens.access_token
        self._refresh_token = tokens.refresh_token
        self._id_token = tokens.id_token
        self._email = tokens.email
        self._account_id = tokens.account_id
        self._sync_token_metadata()

        if self.needs_refresh():
            await self.refresh()
        else:
            await self._save_token()

        self._auth_source = AuthSource.CODEX_CLI
        self._device_code_status = DeviceCodeStatus.AUTHORIZED
        self._device_code_error = None

        logger.info(
            "Synced OpenAI OAuth from Codex CLI",
            auth_mode=tokens.auth_mode,
            email=self._email,
        )
        return self.get_device_code_status()

    async def disconnect(self) -> None:
        """Disconnect — delete tokens from DB and clear env var."""
        self._access_token = None
        self._refresh_token = None
        self._id_token = None
        self._account_id = None
        self._expires_at = 0.0
        self._email = None
        self._auth_source = None
        self._device_code_status = DeviceCodeStatus.IDLE
        self._device_code_error = None

        # Delete from DB
        if self._db_service is not None:
            async with self._db_service.session_context() as session:
                from assistant_runtime.services.database.repositories import OAuthTokenRepository

                repo = OAuthTokenRepository(session)
                await repo.delete("openai")
                await session.commit()

        logger.info("OpenAI OAuth disconnected")

    # --- Private methods ---

    async def _poll_for_token(
        self, device_auth_id: str, user_code: str, interval: int, expires_in: int
    ) -> None:
        """Poll OpenAI device-auth endpoint until authorization completes."""
        deadline = time.time() + expires_in

        while time.time() < deadline:
            await asyncio.sleep(interval)

            try:
                response = await self._http.post(
                    self._config.device_auth_token_url,
                    json={
                        "device_auth_id": device_auth_id,
                        "user_code": user_code,
                    },
                )

                if response.status_code == 200:
                    code_data = response.json()
                    await self._exchange_authorization_code_for_tokens(
                        authorization_code=code_data["authorization_code"],
                        code_verifier=code_data["code_verifier"],
                    )
                    await self._save_token()

                    self._auth_source = AuthSource.DEVICE_CODE
                    self._device_code_status = DeviceCodeStatus.AUTHORIZED
                    logger.info("OpenAI OAuth authorized", email=self._email)
                    return

                if response.status_code in {403, 404}:
                    continue
                self._device_code_status = DeviceCodeStatus.ERROR
                self._device_code_error = (
                    f"Authorization failed: {response.status_code} {response.text}"
                )
                return

            except Exception as exc:
                self._device_code_status = DeviceCodeStatus.ERROR
                self._device_code_error = str(exc)
                logger.warning("Device code poll error", error=str(exc))
                return

        self._device_code_status = DeviceCodeStatus.EXPIRED
        self._device_code_error = "Device code flow timed out"

    async def _exchange_authorization_code_for_tokens(
        self,
        *,
        authorization_code: str,
        code_verifier: str,
    ) -> None:
        """Exchange a device-auth authorization code for OAuth tokens."""
        if self._http is None:
            raise OAuthNotConfiguredError("OAuth service not started")

        response = await self._http.post(
            self._config.token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": self._config.device_callback_url,
                "client_id": OPENAI_OAUTH_CLIENT_ID,
                "code_verifier": code_verifier,
            },
        )

        if response.status_code != 200:
            raise OAuthRefreshError(
                f"Authorization code exchange failed: {response.status_code} {response.text}"
            )

        token_data = response.json()
        self._access_token = token_data.get("access_token")
        self._id_token = token_data.get("id_token")
        self._refresh_token = token_data.get("refresh_token")
        self._sync_token_metadata()

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

    async def _load_stored_token(self) -> bool:
        """Load stored OAuth token from database."""
        try:
            async with self._db_service.session_context() as session:
                from assistant_runtime.services.database.repositories import OAuthTokenRepository

                repo = OAuthTokenRepository(session)
                token = await repo.get("openai")

                if token is None:
                    logger.debug("No stored OAuth token found")
                    return False

                # Reuse the existing column for ChatGPT/Codex access tokens.
                if token.encrypted_api_key:
                    self._access_token = self._decrypt(token.encrypted_api_key)
                if token.encrypted_refresh_token:
                    self._refresh_token = self._decrypt(token.encrypted_refresh_token)
                if token.encrypted_id_token:
                    self._id_token = self._decrypt(token.encrypted_id_token)
                self._expires_at = token.expires_at or 0.0
                self._email = token.email
                self._sync_token_metadata()

                if self._access_token and self._account_id and not self.needs_refresh():
                    self._auth_source = AuthSource.DATABASE
                    self._device_code_status = DeviceCodeStatus.AUTHORIZED
                    logger.info("Loaded stored OpenAI OAuth token", email=self._email)
                    return True
                if self._refresh_token:
                    logger.info("Stored OAuth token needs refresh")
                    await self.refresh()
                    self._auth_source = AuthSource.DATABASE
                    self._device_code_status = DeviceCodeStatus.AUTHORIZED
                    return True

        except Exception as exc:
            logger.warning("Failed to load stored OAuth token", error=str(exc))
        return False

    async def _save_token(self) -> None:
        """Save current token state to database."""
        if self._db_service is None:
            return

        async with self._db_service.session_context() as session:
            from assistant_runtime.services.database.repositories import OAuthTokenRepository

            repo = OAuthTokenRepository(session)
            await repo.upsert(
                provider="openai",
                encrypted_api_key=self._encrypt(self._access_token) if self._access_token else None,
                encrypted_refresh_token=self._encrypt(self._refresh_token)
                if self._refresh_token
                else None,
                encrypted_id_token=self._encrypt(self._id_token) if self._id_token else None,
                expires_at=self._expires_at,
                email=self._email,
            )
            await session.commit()

    def _read_codex_cli_auth(self) -> CodexCliAuth:
        """Read ChatGPT/Codex OAuth state from the local Codex CLI auth file."""
        auth_path = Path(self._config.codex_auth_file).expanduser()
        if not auth_path.exists():
            raise OAuthCodexSyncError(f"Codex auth file not found: {auth_path}")

        try:
            data = json.loads(auth_path.read_text())
        except OSError as exc:
            raise OAuthCodexSyncError(f"Failed to read Codex auth file: {auth_path}") from exc
        except json.JSONDecodeError as exc:
            raise OAuthCodexSyncError(f"Invalid JSON in Codex auth file: {auth_path}") from exc

        auth_mode = data.get("auth_mode")
        if auth_mode == "apikey":
            raise OAuthCodexSyncError(
                "Codex CLI is using API key mode, not ChatGPT OAuth. Run `codex login` first."
            )

        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            raise OAuthCodexSyncError(f"Codex auth file is missing OAuth tokens: {auth_path}")

        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        id_token = tokens.get("id_token")
        account_id = tokens.get("account_id")

        if not isinstance(access_token, str) or not access_token:
            raise OAuthCodexSyncError(f"Codex auth file is missing access_token: {auth_path}")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise OAuthCodexSyncError(f"Codex auth file is missing refresh_token: {auth_path}")
        if id_token is not None and not isinstance(id_token, str):
            id_token = None
        if account_id is not None and not isinstance(account_id, str):
            account_id = None

        return CodexCliAuth(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            account_id=account_id,
            auth_mode=auth_mode if isinstance(auth_mode, str) else None,
            email=self._extract_email_from_id_token(id_token),
        )

    def _parse_interval_seconds(self, value: object) -> int:
        """Normalize device-auth polling intervals, which may arrive as strings."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            with contextlib.suppress(ValueError):
                return int(value.strip())
        return self._config.device_code_poll_interval

    def _sync_token_metadata(self) -> None:
        """Refresh derived auth metadata from the current token set."""
        self._email = self._extract_email_from_id_token(self._id_token) or self._email
        self._account_id = self._extract_account_id(self._id_token, self._access_token)
        expires_at = self._extract_token_expiry(self._access_token)
        if expires_at is not None:
            self._expires_at = expires_at
        elif self._access_token and self._expires_at <= 0:
            self._expires_at = time.time() + max(self._config.refresh_buffer_seconds + 300, 3600)

    def _decode_jwt_claims(self, token: str | None) -> dict[str, Any] | None:
        """Decode JWT claims without verification for local token metadata access."""
        if not token or token.count(".") < 2:
            return None

        payload = token.split(".")[1]
        padding = "=" * (-len(payload) % 4)
        try:
            raw = base64.urlsafe_b64decode(payload + padding)
            claims = json.loads(raw)
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error):
            return None

        return claims if isinstance(claims, dict) else None

    def _extract_email_from_id_token(self, id_token: str | None) -> str | None:
        """Best-effort decode of email claim from the OIDC id_token payload."""
        claims = self._decode_jwt_claims(id_token)
        if claims is None:
            return None

        email = claims.get("email")
        if isinstance(email, str) and email:
            return email

        profile = claims.get("https://api.openai.com/profile")
        if isinstance(profile, dict):
            email = profile.get("email")
            if isinstance(email, str) and email:
                return email

        return None

    def _extract_account_id(self, id_token: str | None, access_token: str | None) -> str | None:
        """Extract the ChatGPT workspace/account id from token claims."""
        for token in (id_token, access_token):
            claims = self._decode_jwt_claims(token)
            if claims is None:
                continue
            auth_claim = claims.get("https://api.openai.com/auth")
            if isinstance(auth_claim, dict):
                account_id = auth_claim.get("chatgpt_account_id")
                if isinstance(account_id, str) and account_id:
                    return account_id
        return self._account_id

    def _extract_token_expiry(self, token: str | None) -> float | None:
        """Extract expiry from a JWT token, if present."""
        claims = self._decode_jwt_claims(token)
        if claims is None:
            return None

        exp = claims.get("exp")
        if isinstance(exp, int | float):
            return float(exp)
        return None
