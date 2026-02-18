"""Shared test fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient

from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.main import create_app


@pytest.fixture
def app():
    """Create a test application instance with lifecycle initialized."""
    test_app = create_app()
    # Initialize state that lifespan normally sets up, so /health works without
    # running the full ASGI lifespan (which ASGITransport doesn't trigger).
    test_app.state.lifecycle = LifecycleManager()
    return test_app


@pytest.fixture
async def client(app):
    """Async HTTP client for testing FastAPI endpoints."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
