"""HTTP route layer — thin delegation to service modules."""

from fastapi import APIRouter

from lovely_assistant.app.routes.chat import router as chat_router
from lovely_assistant.app.routes.media import router as media_router
from lovely_assistant.app.routes.models import router as models_router
from lovely_assistant.app.routes.sessions import router as sessions_router
from lovely_assistant.app.routes.settings import router as settings_router

router = APIRouter()
router.include_router(chat_router)
router.include_router(media_router)
router.include_router(models_router)
router.include_router(sessions_router)
router.include_router(settings_router)

__all__ = ["router"]
