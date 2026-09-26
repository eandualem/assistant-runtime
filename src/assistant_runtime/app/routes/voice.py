"""Authenticated voice browser negotiation and backend event/control endpoints."""

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from assistant_runtime.app.access.deps import PrincipalDep
from assistant_runtime.app.voice.deps import VoiceServiceDep
from assistant_runtime.app.voice.exceptions import VoiceError
from assistant_runtime.app.voice.models import VoiceContext, VoiceOffer, VoiceToolResult

router = APIRouter(prefix="/voice")


async def _http(awaitable):
    try:
        return await awaitable
    except VoiceError as exc:
        if exc.metadata:
            return JSONResponse({"detail": str(exc), **exc.metadata}, status_code=exc.status_code)
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.get("/status")
async def status(service: VoiceServiceDep, principal: PrincipalDep) -> dict:
    return await service.health_check()


@router.get("/usage")
async def usage(service: VoiceServiceDep, principal: PrincipalDep) -> dict:
    return await _http(service.usage())


@router.post("/calls", status_code=201)
async def create_call(
    offer: VoiceOffer, request: Request, service: VoiceServiceDep, principal: PrincipalDep
) -> dict:
    result = await _http(service.create(offer, principal))
    if isinstance(result, dict):
        # Reachable through the host's server when the runtime is mounted under a prefix.
        result["events_url"] = request.scope.get("root_path", "") + result["events_url"]
    return result


@router.get("/calls/{call_id}")
async def get_call(call_id: str, service: VoiceServiceDep, principal: PrincipalDep) -> dict:
    return await _http(service.get(call_id, principal))


@router.get("/calls/{call_id}/events")
async def call_events(
    call_id: str,
    service: VoiceServiceDep,
    principal: PrincipalDep,
    after: int = Query(default=0, ge=0),
) -> StreamingResponse:
    snapshot = await _http(service.get(call_id, principal))
    if after > snapshot["cursor"]:
        raise HTTPException(409, "Event cursor is ahead of this call")
    return StreamingResponse(
        service.events(call_id, principal, after),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/calls/{call_id}/close")
async def close_call(call_id: str, service: VoiceServiceDep, principal: PrincipalDep) -> dict:
    return await _http(service.close(call_id, principal))


@router.post("/calls/{call_id}/cancel")
async def cancel_work(call_id: str, service: VoiceServiceDep, principal: PrincipalDep) -> dict:
    return await _http(service.cancel_work(call_id, principal))


@router.patch("/calls/{call_id}/context")
async def update_context(
    call_id: str, body: VoiceContext, service: VoiceServiceDep, principal: PrincipalDep
) -> dict:
    return await _http(service.update_context(call_id, body, principal))


@router.post("/calls/{call_id}/delegations/{delegation_id}/tool-result", status_code=202)
async def tool_result(
    call_id: str,
    delegation_id: str,
    body: VoiceToolResult,
    service: VoiceServiceDep,
    principal: PrincipalDep,
) -> dict:
    return await _http(service.tool_result(call_id, delegation_id, body, principal))
