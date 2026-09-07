"""Shared test fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient

from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.main import create_app


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


@pytest.fixture(autouse=True)
def trusted_local_access(monkeypatch):
    """Route tests act as the trusted local operator unless a test says otherwise."""
    from assistant_runtime.app.access import deps as access_deps
    from assistant_runtime.app.access.config import AccessConfig
    from assistant_runtime.app.access.interface import AccessService

    service = AccessService(AccessConfig())
    monkeypatch.setattr(access_deps, "get_access_service", lambda request: service)
    return service
