"""Application entrypoint — FastAPI app creation, lifespan, health endpoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from assistant_runtime import RUNTIME_MARKER, __version__
from assistant_runtime.app.access import deps as access_deps
from assistant_runtime.app.access.exceptions import AccessDeniedError, AuthenticationError
from assistant_runtime.app.access.factory import register_access
from assistant_runtime.app.access.middleware import BrowserOriginMiddleware
from assistant_runtime.app.assistant.definition import AssistantDefinition
from assistant_runtime.app.assistant.exceptions import AssistantError, SessionError
from assistant_runtime.app.assistant.factory import register_assistant
from assistant_runtime.app.heartbeat.factory import register_heartbeat
from assistant_runtime.app.ingress.factory import register_ingress
from assistant_runtime.app.routes import router
from assistant_runtime.app.settings import RuntimeSettings
from assistant_runtime.app.streaming.exceptions import StreamingError
from assistant_runtime.app.streaming.factory import register_streaming
from assistant_runtime.app.streaming.interface import StreamingService
from assistant_runtime.app.voice.factory import register_voice
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.logging_config import setup_logging
from assistant_runtime.services.artifacts.factory import register_artifacts
from assistant_runtime.services.database.factory import register_database
from assistant_runtime.services.decisions.factory import register_decisions
from assistant_runtime.services.history.factory import register_history
from assistant_runtime.services.llm.factory import register_llm
from assistant_runtime.services.mcp.factory import register_mcp
from assistant_runtime.services.media.factory import register_media
from assistant_runtime.services.oauth.exceptions import (
    OAuthCodexSyncError,
    OAuthError,
    OAuthNotConfiguredError,
)
from assistant_runtime.services.oauth.factory import register_oauth
from assistant_runtime.services.tools.factory import register_tools
from assistant_runtime.services.tracing import initialize_tracing, shutdown_tracing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle — startup and shutdown.

    The environment is taken as it is: the ``assistant-runtime`` command loads
    ``.env`` before anything else, and a host embedding the runtime owns its
    own environment (see ``docs/composition.md``).
    """
    lifecycle = LifecycleManager()
    app.state.lifecycle = lifecycle
    try:
        if initialize_tracing():
            logger.info("Langfuse tracing enabled")

        settings = getattr(app.state, "settings", None)
        if settings is None:
            settings = AppSettings()
        app.state.settings = settings
        setup_logging(json_output=settings.log_json, level=settings.log_level)

        # Register modules in dependency order (infrastructure first, then services, then app)
        await register_access(app.state, lifecycle, settings=settings)
        await register_database(app.state, lifecycle, settings=settings)
        await register_oauth(app.state, lifecycle, settings=settings)
        await register_llm(app.state, lifecycle, settings=settings)
        await register_history(app.state, lifecycle, settings=settings)
        await register_media(app.state, lifecycle, settings=settings)
        await register_decisions(app.state, lifecycle, settings=settings)
        await register_mcp(app.state, lifecycle)
        await register_artifacts(app.state, lifecycle, settings=settings)
        await register_tools(app.state, lifecycle, settings=settings)
        await register_assistant(app.state, lifecycle, settings=settings)
        await register_streaming(app.state, lifecycle, settings=settings)
        await register_voice(app.state, lifecycle, settings=settings)
        await register_ingress(app.state, lifecycle)
        await register_heartbeat(app.state, lifecycle, settings=settings)

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
        if getattr(app.state, "history_service", None) is not None:
            app.state.history_service.set_runtime_settings(rs)
        if getattr(app.state, "media_service", None) is not None:
            app.state.media_service.set_runtime_settings(rs)
        if getattr(app.state, "tool_service", None) is not None:
            app.state.tool_service.set_runtime_settings(rs)

        # Cleanup expired sessions and old traces on startup (before accepting requests)
        if getattr(app.state, "assistant_service", None) is not None:
            try:
                cleaned = await app.state.assistant_service.cleanup_expired_sessions()
                if cleaned:
                    logger.info("Cleaned up expired sessions on startup", count=cleaned)
            except Exception as e:
                logger.warning("Session cleanup skipped on startup", error=str(e))
        if getattr(app.state, "streaming_service", None) is not None:
            try:
                cleaned = await app.state.streaming_service.cleanup_expired_traces()
                if cleaned:
                    logger.info("Cleaned up old traces on startup", count=cleaned)
            except Exception as e:
                logger.warning("Trace cleanup skipped on startup", error=str(e))

        logger.info("Application started", app=settings.app_name)

        yield
    finally:
        # Startup may fail or be cancelled after services have already started.
        # stop_all also safely handles the rollback performed by start_all.
        try:
            await lifecycle.stop_all()
        finally:
            shutdown_tracing()
            logger.info("Application stopped")


def create_app(
    *, assistant: AssistantDefinition | None = None, settings: AppSettings | None = None
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Assistant Runtime",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.assistant_definition = assistant
    app.state.settings = settings

    # CORS: the listed origins plus the origin regex (localhost on any port by
    # default). Socket.IO applies the same AccessConfig in create_asgi_app.
    access = (settings or AppSettings()).access
    origins = list(access.cors_origins)
    app.add_middleware(
        BrowserOriginMiddleware,
        allow_origins=origins,
        allow_origin_regex=access.cors_origin_regex or None,
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _detail(request: Request) -> bool:
        """Whether internal error text may reach the client (STREAMING__CLIENT_ERROR_DETAIL)."""
        app_settings = getattr(request.app.state, "settings", None)
        return app_settings is None or app_settings.streaming.client_error_detail

    # Exception handlers
    @app.exception_handler(OAuthError)
    async def oauth_error_handler(request, exc: OAuthError):
        if isinstance(exc, OAuthNotConfiguredError):
            status_code, message = 503, str(exc)
        elif isinstance(exc, OAuthCodexSyncError):
            status_code, message = 400, str(exc)
        else:
            # Provider error bodies may contain sensitive OAuth details.
            status_code, message = 502, "OAuth provider request failed; reconnect or retry"
        return JSONResponse(
            status_code=status_code,
            content={"error": message, "type": exc.__class__.__name__},
        )

    @app.exception_handler(AuthenticationError)
    async def authentication_error_handler(request, exc: AuthenticationError):
        return JSONResponse(status_code=401, content={"error": str(exc), "type": "Unauthorized"})

    @app.exception_handler(AccessDeniedError)
    async def access_denied_handler(request, exc: AccessDeniedError):
        return JSONResponse(status_code=403, content={"error": str(exc), "type": "Forbidden"})

    @app.exception_handler(SessionError)
    async def session_error_handler(request, exc: SessionError):
        # The request does not fit the session (unknown session or parent,
        # duplicate id, wrong tool call): the client's mistake, not ours.
        return JSONResponse(
            status_code=409,
            content={"error": str(exc), "type": exc.__class__.__name__},
        )

    @app.exception_handler(AssistantError)
    async def assistant_error_handler(request: Request, exc: AssistantError):
        message = str(exc) if _detail(request) else "The assistant request failed"
        return JSONResponse(
            status_code=500,
            content={"error": message, "type": exc.__class__.__name__},
        )

    @app.exception_handler(StreamingError)
    async def streaming_error_handler(request: Request, exc: StreamingError):
        message = str(exc) if _detail(request) else "The assistant request failed"
        return JSONResponse(
            status_code=500,
            content={"error": message, "type": exc.__class__.__name__},
        )

    # Routes
    app.include_router(router, prefix="/api")

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        """Liveness for anyone; component detail for an authenticated caller.

        The full report names providers, models, MCP servers and the database
        host, so an anonymous caller (a probe, a page on another site in
        ``header``/``host`` mode) gets only the per-component ``healthy`` flags.
        In ``trusted_local`` every caller is authenticated.
        """
        lifecycle: LifecycleManager = app.state.lifecycle
        result = await lifecycle.health()
        result["runtime"] = RUNTIME_MARKER
        if not await _authenticated(request):
            result = public_health(result)
        status_code = 200 if result.get("healthy") else 503
        return JSONResponse(content=result, status_code=status_code)

    return app


async def _authenticated(request: Request) -> bool:
    """Whether the route dependency would accept the caller (never raises)."""
    try:
        # The same dependency the routes use; looked up through the module
        # so a test may replace the access service.
        await access_deps.get_principal(request)
    except HTTPException:
        return False
    return True


def public_health(result: dict) -> dict:
    """The anonymous form of a health report: flags only, no configuration."""
    return {
        "healthy": result.get("healthy", False),
        "runtime": RUNTIME_MARKER,
        "components": {
            name: {"healthy": bool(component.get("healthy", False))}
            for name, component in (result.get("components") or {}).items()
            if isinstance(component, dict)
        },
    }


def create_asgi_app(
    *, assistant: AssistantDefinition | None = None, settings: AppSettings | None = None
) -> socketio.ASGIApp:
    """Create the full ASGI application with Socket.IO wrapper."""
    from assistant_runtime.app.socketio_server import create_sio

    if settings is None:
        # Both transports read the origin rule, so resolve settings once.
        settings = AppSettings()
    fastapi_app = create_app(assistant=assistant, settings=settings)
    sio = create_sio(settings.access)
    sio.fastapi_app = fastapi_app
    fastapi_app.state.sio = sio
    return socketio.ASGIApp(sio, fastapi_app)


@asynccontextmanager
async def create_runtime(
    *, assistant: AssistantDefinition | None = None, settings: AppSettings | None = None
) -> AsyncIterator[StreamingService]:
    """Open the server's runtime in-process, yielding its public turn interface.

    Call ``run_message`` or ``stream_message`` on the yielded service. The
    same factories, session policy and lifecycle are used by the server.
    """
    fastapi_app = create_app(assistant=assistant, settings=settings)
    async with fastapi_app.router.lifespan_context(fastapi_app):
        yield fastapi_app.state.streaming_service


__all__ = ["AssistantDefinition", "create_app", "create_asgi_app", "create_runtime"]
