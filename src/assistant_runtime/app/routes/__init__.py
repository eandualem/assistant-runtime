"""HTTP route layer — thin delegation to service modules."""

from fastapi import APIRouter

from assistant_runtime.app.routes.actions import confirmations_router
from assistant_runtime.app.routes.actions import router as actions_router
from assistant_runtime.app.routes.agents import router as agents_router
from assistant_runtime.app.routes.agui import router as agui_router
from assistant_runtime.app.routes.artifacts import prompt_router, proposals_router, subjects_router
from assistant_runtime.app.routes.artifacts import router as artifacts_router
from assistant_runtime.app.routes.chat import router as chat_router
from assistant_runtime.app.routes.counts import router as counts_router
from assistant_runtime.app.routes.debug import router as debug_router
from assistant_runtime.app.routes.decisions import router as decisions_router
from assistant_runtime.app.routes.events import router as events_router
from assistant_runtime.app.routes.host_state import router as host_state_router
from assistant_runtime.app.routes.inbox import router as inbox_router
from assistant_runtime.app.routes.inject import router as inject_router
from assistant_runtime.app.routes.media import router as media_router
from assistant_runtime.app.routes.models import router as models_router
from assistant_runtime.app.routes.oauth import router as oauth_router
from assistant_runtime.app.routes.providers import router as providers_router
from assistant_runtime.app.routes.sessions import router as sessions_router
from assistant_runtime.app.routes.settings import router as settings_router
from assistant_runtime.app.routes.tasks import router as tasks_router
from assistant_runtime.app.routes.voice import router as voice_router

router = APIRouter()
router.include_router(actions_router)
router.include_router(agents_router)
router.include_router(confirmations_router)
router.include_router(agui_router)
router.include_router(artifacts_router)
router.include_router(proposals_router)
router.include_router(subjects_router)
router.include_router(prompt_router)
router.include_router(chat_router)
router.include_router(counts_router)
router.include_router(debug_router)
router.include_router(decisions_router)
router.include_router(events_router)
router.include_router(host_state_router)
router.include_router(inbox_router)
router.include_router(inject_router)
router.include_router(media_router)
router.include_router(models_router)
router.include_router(oauth_router)
router.include_router(providers_router)
router.include_router(sessions_router)
router.include_router(settings_router)
router.include_router(tasks_router)
router.include_router(voice_router)

__all__ = ["router"]
