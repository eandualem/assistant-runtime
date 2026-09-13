"""Register voice after the shared turn pipeline."""

from typing import Any

from assistant_runtime.app.voice.interface import VoiceService
from assistant_runtime.base.lifecycle import LifecycleManager


async def register_voice(app_state: Any, lifecycle: LifecycleManager, *, settings: Any) -> None:
    service = VoiceService(
        settings.voice,
        streaming_service=getattr(app_state, "streaming_service", None),
        database_service=getattr(app_state, "database_service", None),
    )
    app_state.voice_service = service
    await lifecycle.register("voice_service", service)
