"""HTTP route layer — thin delegation to service modules."""

from fastapi import APIRouter

from lovely_assistant.app.routes.chat import router as chat_router
from lovely_assistant.app.routes.sessions import router as sessions_router

router = APIRouter()
router.include_router(chat_router)
router.include_router(sessions_router)

__all__ = ["router"]
