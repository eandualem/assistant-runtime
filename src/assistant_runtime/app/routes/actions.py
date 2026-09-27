"""Action endpoints — the records of actions a host proposes and carries out.

An administrator (the host) writes them: the action with its status history,
text revisions and per-recipient results, and the owner's confirmation for
each recipient, written before the send (see ``services/actions``).
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import UUID4, AwareDatetime, BaseModel, ConfigDict, Field

from assistant_runtime.app.access.deps import AdminDep
from assistant_runtime.services.actions.deps import ActionServiceDep
from assistant_runtime.services.actions.exceptions import (
    ActionConflictError,
    ActionError,
    ActionNotFoundError,
)

router = APIRouter(prefix="/actions", tags=["actions"])
confirmations_router = APIRouter(prefix="/action-confirmations", tags=["actions"])

_STATUS = {ActionNotFoundError: 404, ActionConflictError: 409}

Status = Literal["proposed", "scheduled", "sending", "sent", "failed", "discarded", "undone"]


class ActionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1, max_length=64)
    text: str | None = None
    arguments: dict[str, Any] | None = None
    profile: str | None = Field(default=None, max_length=64)
    subject: str | None = Field(default=None, max_length=128)
    status: Status = "proposed"


class ActionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Status | None = None
    text: str | None = None
    arguments: dict[str, Any] | None = None
    confirmed_by: str | None = Field(default=None, max_length=128)
    decided_at: AwareDatetime | None = None
    results: dict[str, Any] | None = None
    """Per recipient; merged into the stored results."""
    expected_status: list[Status] | None = Field(default=None, min_length=1)
    """Change only while the action is in one of these (``409`` otherwise)."""
    expected_revision: int | None = None
    """Change only while the text is at this revision (``409`` otherwise)."""


class ConfirmationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID4
    recipient: str = Field(min_length=1, max_length=200)
    kind: Literal["message", "steer"]
    revision: int = Field(ge=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: Literal["button", "typed", "voice"]
    confirmed_at: AwareDatetime
    key_epoch: int | None = None


class ConfirmationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["sent", "failed"] | None = None
    result: Any = None
    key_epoch: int | None = None
    sender: str | None = Field(default=None, max_length=200)
    audience: str | None = Field(default=None, max_length=200)
    reconciled: Literal["matched", "altered", "missing", "undelivered"] | None = None


def _http_error(exc: ActionError) -> HTTPException:
    return HTTPException(status_code=_STATUS.get(type(exc), 422), detail=str(exc))


def _given(body: BaseModel, *, exclude: set[str] = frozenset()) -> dict[str, Any]:
    """The fields the request named, so an explicit null differs from an absent one."""
    return {name: getattr(body, name) for name in body.model_fields_set - exclude}


@router.post("", status_code=201)
async def create_action(body: ActionCreate, actions: ActionServiceDep, admin: AdminDep) -> dict:
    try:
        return (await actions.create(**body.model_dump(), by=admin.id)).to_dict()
    except ActionError as e:
        raise _http_error(e) from e


@router.get("")
async def list_actions(
    actions: ActionServiceDep,
    admin: AdminDep,
    status: Status | None = None,
    kind: str | None = None,
    profile: str | None = None,
    subject: str | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """Newest first."""
    records = await actions.list(
        status=status, kind=kind, profile=profile, subject=subject, limit=limit
    )
    return {"actions": [r.to_dict() for r in records]}


@router.get("/{action_id}")
async def get_action(action_id: int, actions: ActionServiceDep, admin: AdminDep) -> dict:
    """The action and its confirmations, in insertion order."""
    try:
        record, confirmations = await actions.get(action_id)
    except ActionError as e:
        raise _http_error(e) from e
    return {**record.to_dict(), "confirmations": [c.to_dict() for c in confirmations]}


@router.patch("/{action_id}")
async def update_action(
    action_id: int, body: ActionUpdate, actions: ActionServiceDep, admin: AdminDep
) -> dict:
    try:
        record = await actions.update(
            action_id,
            _given(body, exclude={"expected_status", "expected_revision"}),
            by=admin.id,
            expected_status=body.expected_status,
            expected_revision=body.expected_revision,
        )
    except ActionError as e:
        raise _http_error(e) from e
    return record.to_dict()


@router.post("/{action_id}/confirmations", status_code=201)
async def confirm_action(
    action_id: int, body: ConfirmationCreate, actions: ActionServiceDep, admin: AdminDep
) -> dict:
    """The owner's confirmation for one recipient, recorded before the send."""
    values = body.model_dump()
    try:
        record = await actions.confirm(action_id, confirmation_id=str(values.pop("id")), **values)
    except ActionError as e:
        raise _http_error(e) from e
    return record.to_dict()


@router.patch("/{action_id}/confirmations/{confirmation_id}")
async def update_confirmation(
    action_id: int,
    confirmation_id: str,
    body: ConfirmationUpdate,
    actions: ActionServiceDep,
    admin: AdminDep,
) -> dict:
    """Sign it while ``confirmed``, settle it once, or note its reconciliation."""
    try:
        record = await actions.update_confirmation(
            confirmation_id, _given(body), action_id=action_id
        )
    except ActionError as e:
        raise _http_error(e) from e
    return record.to_dict()


@confirmations_router.get("")
async def list_confirmations(
    actions: ActionServiceDep,
    admin: AdminDep,
    after: int = Query(0, ge=0),
    action_id: int | None = None,
    status: Literal["confirmed", "sent", "failed"] | None = None,
    sender: str | None = None,
    audience: str | None = None,
    signed: bool | None = None,
    reconciled: bool | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """Confirmations after ``after`` (their ``seq``) in insertion order."""
    records = await actions.confirmations(
        after=after,
        limit=limit,
        action_id=action_id,
        status=status,
        sender=sender,
        audience=audience,
        signed=signed,
        reconciled=reconciled,
    )
    return {
        "confirmations": [r.to_dict() for r in records],
        "next_after": records[-1].seq if records else after,
    }
