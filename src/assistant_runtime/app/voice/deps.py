"""Voice service access through application state."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from assistant_runtime.app.voice.interface import VoiceService


def get_optional_voice_service(request: Request) -> VoiceService | None:
    return getattr(request.app.state, "voice_service", None)


def get_voice_service(request: Request) -> VoiceService:
    service = get_optional_voice_service(request)
    if service is None:
        raise HTTPException(503, "Voice service unavailable")
    return service


VoiceServiceDep = Annotated[VoiceService, Depends(get_voice_service)]
OptionalVoiceServiceDep = Annotated[VoiceService | None, Depends(get_optional_voice_service)]
