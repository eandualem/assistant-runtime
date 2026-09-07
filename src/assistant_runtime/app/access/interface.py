"""AccessService — establishes the trusted principal of every caller.

Public facade for the access module. Implements LifecycleAware.

Three modes, chosen by ``ACCESS__MODE``:

- ``trusted_local`` (default): every caller is ``LOCAL_PRINCIPAL``, an
  administrator. The single-operator installation bound to localhost.
- ``header``: a reverse proxy that already authenticated the caller sets
  the principal id header (and optionally a roles header). Anything that
  can reach the server directly can forge these headers, so the server
  must only be reachable through that proxy.
- ``host``: the host application's ``AssistantDefinition.authenticate``
  callback turns the transport's credentials into a principal, so an
  existing identity system plugs in without adopting a new one.

Ownership rules live in ``assistant_runtime.principal`` and are the same
for HTTP, Socket.IO and in-process callers.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from assistant_runtime.app.access.config import AccessConfig
from assistant_runtime.app.access.exceptions import AccessDeniedError, AuthenticationError
from assistant_runtime.principal import (
    LOCAL_PRINCIPAL,
    Credentials,
    Principal,
    can_access_session,
)

Authenticator = Callable[[Credentials], Principal | None | Awaitable[Principal | None]]


class AccessService:
    """Authenticate callers and apply the ownership and administration rules."""

    def __init__(self, config: AccessConfig, authenticator: Authenticator | None = None) -> None:
        self._config = config
        self._authenticator = authenticator
        self._started = False

    async def start(self) -> None:
        if self._config.mode == "host" and self._authenticator is None:
            raise AccessDeniedError(
                "ACCESS__MODE=host needs AssistantDefinition.authenticate", severity="critical"
            )
        self._started = True
        logger.info("Access service started", mode=self._config.mode)

    async def stop(self) -> None:
        self._started = False
        logger.info("Access service stopped")

    async def health_check(self) -> dict[str, Any]:
        return {"healthy": self._started, "mode": self._config.mode}

    @property
    def mode(self) -> str:
        return self._config.mode

    async def authenticate(self, credentials: Credentials) -> Principal:
        """The principal behind ``credentials``; ``AuthenticationError`` when there is none."""
        mode = self._config.mode
        if mode == "trusted_local":
            return LOCAL_PRINCIPAL
        if mode == "header":
            return self._from_headers(credentials)
        assert self._authenticator is not None
        try:
            result = self._authenticator(credentials)
            if inspect.isawaitable(result):
                result = await result
        except AuthenticationError:
            raise
        except Exception as e:
            logger.warning("Host authentication failed", error=str(e))
            raise AuthenticationError("Authentication failed") from e
        if result is None:
            raise AuthenticationError("Not authenticated")
        if not isinstance(result, Principal):
            raise AuthenticationError("Host authenticator returned no Principal")
        return result

    def _from_headers(self, credentials: Credentials) -> Principal:
        principal_id = (credentials.header(self._config.principal_header) or "").strip()
        if not principal_id:
            raise AuthenticationError(
                f"Missing {self._config.principal_header} header (set by the authenticating proxy)"
            )
        roles = credentials.header(self._config.roles_header) or ""
        return Principal(
            id=principal_id,
            roles=frozenset(role.strip() for role in roles.split(",") if role.strip()),
        )

    @staticmethod
    def require_admin(principal: Principal) -> None:
        if not principal.is_admin:
            raise AccessDeniedError("Administration requires the admin role")

    @staticmethod
    def check_session(principal: Principal, owner_id: str | None, session_id: str = "") -> None:
        """``AccessDeniedError`` unless ``principal`` may act on the session."""
        if not can_access_session(principal, owner_id):
            label = f" '{session_id}'" if session_id else ""
            raise AccessDeniedError(f"Session{label} belongs to another principal")
