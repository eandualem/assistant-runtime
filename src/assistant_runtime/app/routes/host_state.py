"""Host state endpoints — small versioned values a host keeps (see ``services/host_state``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from assistant_runtime.app.access.deps import AdminDep
from assistant_runtime.services.host_state.deps import HostStateServiceDep
from assistant_runtime.services.host_state.exceptions import (
    HostStateConflictError,
    HostStateError,
    HostStateNotFoundError,
)

router = APIRouter(prefix="/host-state", tags=["host-state"])

_STATUS = {HostStateNotFoundError: 404, HostStateConflictError: 409}


class PutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Any
    expected_version: int | None = Field(default=None, ge=0)
    """Write only over this version; ``0`` when the key must not exist yet."""


def _http_error(exc: HostStateError) -> HTTPException:
    return HTTPException(status_code=_STATUS.get(type(exc), 422), detail=str(exc))


@router.get("/{namespace}")
async def list_values(namespace: str, state: HostStateServiceDep, admin: AdminDep) -> dict:
    """Every value in the namespace, by key."""
    try:
        entries = await state.list(namespace)
    except HostStateError as e:
        raise _http_error(e) from e
    return {"namespace": namespace, "entries": [e.to_dict() for e in entries]}


@router.get("/{namespace}/{key}")
async def get_value(namespace: str, key: str, state: HostStateServiceDep, admin: AdminDep) -> dict:
    try:
        return (await state.get(namespace, key)).to_dict()
    except HostStateError as e:
        raise _http_error(e) from e


@router.put("/{namespace}/{key}")
async def put_value(
    namespace: str, key: str, body: PutRequest, state: HostStateServiceDep, admin: AdminDep
) -> dict:
    """Store the value; its version rises by one."""
    try:
        entry = await state.put(
            namespace, key, body.value, by=admin.id, expected_version=body.expected_version
        )
    except HostStateError as e:
        raise _http_error(e) from e
    return entry.to_dict()


@router.delete("/{namespace}/{key}", status_code=204)
async def delete_value(
    namespace: str,
    key: str,
    state: HostStateServiceDep,
    admin: AdminDep,
    expected_version: int | None = None,
) -> None:
    try:
        await state.delete(namespace, key, expected_version=expected_version)
    except HostStateError as e:
        raise _http_error(e) from e
