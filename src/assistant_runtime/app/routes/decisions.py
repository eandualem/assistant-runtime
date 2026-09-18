"""Authenticated typed decisions: state plus questions in, typed answers out."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from assistant_runtime.app.access.deps import PrincipalDep
from assistant_runtime.app.assistant.deps import AssistantServiceDep
from assistant_runtime.services.artifacts.exceptions import UnknownProfileError
from assistant_runtime.services.decisions.deps import DecisionServiceDep
from assistant_runtime.services.decisions.exceptions import DecisionError
from assistant_runtime.services.decisions.models import DecisionRequest, DecisionResponse

router = APIRouter(prefix="/decisions")


@router.get("/status")
async def status(service: DecisionServiceDep, principal: PrincipalDep) -> dict:
    return await service.health_check()


@router.post("", response_model=DecisionResponse)
async def decide(
    body: DecisionRequest,
    service: DecisionServiceDep,
    assistant: AssistantServiceDep,
    principal: PrincipalDep,
):
    if body.profile is not None:
        try:
            assistant.validate_profile(body.profile)
        except UnknownProfileError as exc:
            raise HTTPException(422, str(exc)) from exc
    try:
        return await service.decide(body)
    except DecisionError as exc:
        if exc.metadata:
            return JSONResponse({"detail": str(exc), **exc.metadata}, status_code=exc.status_code)
        raise HTTPException(exc.status_code, str(exc)) from exc
