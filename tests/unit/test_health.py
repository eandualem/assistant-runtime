"""Tests for the /health endpoint."""

from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient

from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.main import create_app


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
    unhealthy_component.health_check = AsyncMock(return_value={"healthy": False, "error": "connection refused"})
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
