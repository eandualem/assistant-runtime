"""FastAPI dependencies: the request's principal, and administration."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from assistant_runtime.app.access.exceptions import AccessDeniedError, AuthenticationError
from assistant_runtime.app.access.interface import AccessService
from assistant_runtime.principal import Credentials, Principal


def get_access_service(request: Request) -> AccessService:
    """Access AccessService from app.state (created during lifespan)."""
    service: AccessService | None = getattr(request.app.state, "access_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Access service not available")
    return service


async def get_principal(request: Request) -> Principal:
    """The authenticated caller, or 401."""
    service = get_access_service(request)
    credentials = Credentials.from_headers(
        "http",
        request.headers,
        client=request.client.host if request.client else None,
    )
    try:
        return await service.authenticate(credentials)
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


async def require_admin(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
    """The caller, provided it administers this installation; else 403."""
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="Administration requires the admin role")
    return principal


PrincipalDep = Annotated[Principal, Depends(get_principal)]
AdminDep = Annotated[Principal, Depends(require_admin)]


def http_error(exc: AccessDeniedError | AuthenticationError) -> HTTPException:
    status = 401 if isinstance(exc, AuthenticationError) else 403
    return HTTPException(status_code=status, detail=str(exc))
