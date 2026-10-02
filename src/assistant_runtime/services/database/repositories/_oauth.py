"""Oauth queries; the caller owns the transaction."""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.services.database.models import (
    OAuthTokenORM,
)


class OAuthTokenRepository:
    """CRUD operations for OAuth tokens. Uses flush() — caller owns commit.

    ``kind`` is "api_key" for the provider-key store and "login" for a
    ChatGPT/Codex login; every operation touches only the row of that kind.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, provider: str, kind: str) -> OAuthTokenORM | None:
        """Get the provider's token of one kind."""
        result = await self._session.execute(
            select(OAuthTokenORM).where(
                OAuthTokenORM.provider == provider, OAuthTokenORM.kind == kind
            )
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        provider: str,
        kind: str,
        *,
        encrypted_api_key: str | None = None,
        encrypted_refresh_token: str | None = None,
        encrypted_id_token: str | None = None,
        expires_at: float | None = None,
        email: str | None = None,
    ) -> None:
        """Insert or update the provider's token of one kind."""
        values: dict[str, object] = {
            "provider": provider,
            "kind": kind,
            "encrypted_api_key": encrypted_api_key,
            "encrypted_refresh_token": encrypted_refresh_token,
            "encrypted_id_token": encrypted_id_token,
            "expires_at": expires_at,
            "email": email,
        }

        stmt = pg_insert(OAuthTokenORM).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["provider", "kind"],
            set_={
                "encrypted_api_key": stmt.excluded.encrypted_api_key,
                "encrypted_refresh_token": stmt.excluded.encrypted_refresh_token,
                "encrypted_id_token": stmt.excluded.encrypted_id_token,
                "expires_at": stmt.excluded.expires_at,
                "email": stmt.excluded.email,
                "updated_at": func.now(),
            },
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def delete(self, provider: str, kind: str) -> bool:
        """Delete the provider's token of one kind. Returns True if deleted."""
        result = await self._session.execute(
            delete(OAuthTokenORM).where(
                OAuthTokenORM.provider == provider, OAuthTokenORM.kind == kind
            )
        )
        await self._session.flush()
        return (result.rowcount or 0) > 0
