"""Application entrypoint — FastAPI app creation, lifespan, health endpoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import socketio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from assistant_runtime.app.assistant.exceptions import AssistantError
from assistant_runtime.app.assistant.factory import register_assistant
from assistant_runtime.app.heartbeat.factory import register_heartbeat
from assistant_runtime.app.routes import router
from assistant_runtime.app.settings import RuntimeSettings
from assistant_runtime.app.streaming.exceptions import StreamingError
from assistant_runtime.app.streaming.factory import register_streaming
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.logging_config import setup_logging
from assistant_runtime.services.database.factory import register_database
from assistant_runtime.services.history.factory import register_history
from assistant_runtime.services.llm._cache_control_patch import apply_patch as _apply_cache_patch
from assistant_runtime.services.llm.factory import register_llm
from assistant_runtime.services.mcp.factory import register_mcp
from assistant_runtime.services.media.factory import register_media
from assistant_runtime.services.oauth.factory import register_oauth
from assistant_runtime.services.tools.factory import register_tools
from assistant_runtime.services.tracing import initialize_tracing, shutdown_tracing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle — startup and shutdown."""
    # Patch pydantic-ai cache control bug before any agent creation
    _apply_cache_patch()

    # Load .env into os.environ before AppSettings or os.getenv() calls.
    # Pydantic Settings' env_file only populates model fields, not os.environ.
    # LLM providers use os.getenv() for API keys, so they need this.
    load_dotenv()

    if initialize_tracing():
        logger.info("Langfuse tracing enabled")

    settings = AppSettings()
    setup_logging(json_output=settings.log_json, level=settings.log_level)

    lifecycle = LifecycleManager()
    app.state.lifecycle = lifecycle

    # Register modules in dependency order (infrastructure first, then services, then app)
    await register_database(app.state, lifecycle)
    await register_oauth(app.state, lifecycle)
    await register_llm(app.state, lifecycle)
    await register_history(app.state, lifecycle)
    await register_media(app.state, lifecycle)
    await register_mcp(app.state, lifecycle)
    await register_tools(app.state, lifecycle)
    await register_assistant(app.state, lifecycle)
    await register_heartbeat(app.state, lifecycle)
    await register_streaming(app.state, lifecycle)

    await lifecycle.start_all()

    # Create runtime settings AFTER start_all — DatabaseService._healthy is now set
    app.state.runtime_settings = RuntimeSettings(
        frozen_config=settings.assistant,
        database_service=getattr(app.state, "database_service", None),
    )
    await app.state.runtime_settings.load_from_db()
    # Services are registered before runtime settings exists.
    # Attach the live runtime settings through public service APIs.
    rs = app.state.runtime_settings
    if getattr(app.state, "assistant_service", None) is not None:
        app.state.assistant_service.set_runtime_settings(rs)
    if getattr(app.state, "streaming_service", None) is not None:
        app.state.streaming_service.set_runtime_settings(rs)
    if getattr(app.state, "history_service", None) is not None:
        app.state.history_service.set_runtime_settings(rs)
    if getattr(app.state, "media_service", None) is not None:
        app.state.media_service.set_runtime_settings(rs)
    if getattr(app.state, "tool_service", None) is not None:
        app.state.tool_service.set_runtime_settings(rs)

    # Cleanup expired sessions on startup (before accepting requests)
    if getattr(app.state, "assistant_service", None) is not None:
        try:
            cleaned = await app.state.assistant_service.cleanup_expired_sessions()
            if cleaned:
                logger.info("Cleaned up expired sessions on startup", count=cleaned)
        except Exception as e:
            logger.warning("Session cleanup skipped on startup", error=str(e))

    logger.info("Application started", app=settings.app_name)

    yield

    await lifecycle.stop_all()
    shutdown_tracing()
    logger.info("Application stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Assistant Runtime",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(AssistantError)
    async def assistant_error_handler(request, exc: AssistantError):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": exc.__class__.__name__},
        )

    @app.exception_handler(StreamingError)
    async def streaming_error_handler(request, exc: StreamingError):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": exc.__class__.__name__},
        )

    # Routes
    app.include_router(router, prefix="/api")

    @app.get("/health")
    async def health() -> JSONResponse:
        lifecycle: LifecycleManager = app.state.lifecycle
        result = await lifecycle.health()
        status_code = 200 if result.get("healthy") else 503
        return JSONResponse(content=result, status_code=status_code)

    return app


def create_asgi_app() -> socketio.ASGIApp:
    """Create the full ASGI application with Socket.IO wrapper."""
    from assistant_runtime.app.socketio_server import create_sio

    fastapi_app = create_app()
    sio = create_sio()
    sio.fastapi_app = fastapi_app
    fastapi_app.state.sio = sio
    return socketio.ASGIApp(sio, fastapi_app)


app = create_asgi_app()
