"""Tests for the /health endpoint."""


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
