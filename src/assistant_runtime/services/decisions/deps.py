"""Decision service access through application state."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from assistant_runtime.services.decisions.interface import DecisionService


def get_decision_service(request: Request) -> DecisionService:
    service = getattr(request.app.state, "decision_service", None)
    if service is None:
        raise HTTPException(503, "Decision service unavailable")
    return service


DecisionServiceDep = Annotated[DecisionService, Depends(get_decision_service)]
