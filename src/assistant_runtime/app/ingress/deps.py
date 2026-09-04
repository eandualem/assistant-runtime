"""Dependency injection for the ingress module."""

from typing import Annotated

from fastapi import Depends, Request

from assistant_runtime.app.ingress.interface import IngressService


def get_ingress_service(request: Request) -> IngressService:
    """Access IngressService from app.state (created during lifespan)."""
    return request.app.state.ingress_service


IngressServiceDep = Annotated[IngressService, Depends(get_ingress_service)]
