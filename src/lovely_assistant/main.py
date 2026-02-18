"""Application entrypoint — FastAPI app creation, lifespan, health endpoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from lovely_assistant.app.assistant.factory import register_assistant
from lovely_assistant.app.streaming.factory import register_streaming
from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.config import AppSettings
from lovely_assistant.logging_config import setup_logging
from lovely_assistant.services.history.factory import register_history
from lovely_assistant.services.llm.factory import register_llm
from lovely_assistant.services.tools.factory import register_tools


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle — startup and shutdown."""
    settings = AppSettings()
    setup_logging(json_output=settings.log_json, level=settings.log_level)

    lifecycle = LifecycleManager()
    app.state.lifecycle = lifecycle

    # Register modules in dependency order (services before app)
    await register_llm(app.state, lifecycle)
    await register_history(app.state, lifecycle)
    await register_tools(app.state, lifecycle)
    await register_assistant(app.state, lifecycle)
    await register_streaming(app.state, lifecycle)

    await lifecycle.start_all()
    logger.info("Application started", app=settings.app_name)

    yield

    await lifecycle.stop_all()
    logger.info("Application stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Lovely Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict:
        lifecycle: LifecycleManager = app.state.lifecycle
        return await lifecycle.health()

    return app


app = create_app()
