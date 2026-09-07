"""Tests for the /health endpoint."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.main import create_app


async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200


async def test_health_response_structure(client):
    response = await client.get("/health")
    data = response.json()
    assert "healthy" in data
    assert "components" in data
    assert data["healthy"] is True
    assert data["components"] == {}


async def test_health_returns_503_when_unhealthy():
    app = create_app()
    lifecycle = LifecycleManager()

    # Register a component that reports unhealthy
    unhealthy_component = AsyncMock()
    unhealthy_component.health_check = AsyncMock(
        return_value={"healthy": False, "error": "connection refused"}
    )
    unhealthy_component.start = AsyncMock()
    unhealthy_component.stop = AsyncMock()
    await lifecycle.register("database_service", unhealthy_component)

    app.state.lifecycle = lifecycle

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["healthy"] is False
        assert data["components"]["database_service"]["healthy"] is False


@pytest.mark.asyncio
async def test_health_is_200_when_the_optional_services_are_unavailable():
    """Postgres down and OAuth unconfigured is the documented default, not a failure.

    Both services are the real ones: this is the case a first-time user and
    every readiness probe hit, and it must answer 200 (docs/api.md#health).
    """
    from assistant_runtime.services.database.config import DatabaseConfig
    from assistant_runtime.services.database.interface import DatabaseService
    from assistant_runtime.services.oauth.config import OAuthConfig
    from assistant_runtime.services.oauth.interface import OAuthService

    app = create_app()
    lifecycle = LifecycleManager()
    # Port 1 has nothing listening, so the connectivity check fails immediately.
    database = DatabaseService(DatabaseConfig(port=1))
    await database.start()
    try:
        await lifecycle.register("database_service", database)
        await lifecycle.register("oauth_service", OAuthService(OAuthConfig()))
        app.state.lifecycle = lifecycle

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
    finally:
        await database.stop()

    assert response.status_code == 200
    data = response.json()
    assert data["healthy"] is True
    assert data["components"]["database_service"] == {
        "healthy": True,
        "reachable": False,
        "host": "localhost",
    }
    assert data["components"]["oauth_service"]["healthy"] is True
    assert data["components"]["oauth_service"]["status"] == "disabled"


@pytest.mark.asyncio
async def test_health_is_503_when_a_service_never_started():
    """A database service that was never started is a genuine failure."""
    from assistant_runtime.services.database.config import DatabaseConfig
    from assistant_runtime.services.database.interface import DatabaseService

    app = create_app()
    lifecycle = LifecycleManager()
    await lifecycle.register("database_service", DatabaseService(DatabaseConfig()))
    app.state.lifecycle = lifecycle

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 503
    assert response.json()["components"]["database_service"]["healthy"] is False
