"""Dependency injection for the LLM service module."""

from typing import Annotated

from fastapi import Depends, Request

from assistant_runtime.services.llm.interface import LlmService


def get_llm_service(request: Request) -> LlmService:
    """Access LlmService from app.state (created during lifespan)."""
    return request.app.state.llm_service


LlmServiceDep = Annotated[LlmService, Depends(get_llm_service)]
