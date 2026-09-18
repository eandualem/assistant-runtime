"""Tests for the /api/decisions routes with a fake decision service."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.decisions import router
from assistant_runtime.services.artifacts.exceptions import UnknownProfileError
from assistant_runtime.services.decisions.exceptions import DecisionError
from assistant_runtime.services.decisions.models import DecisionResponse

BODY: dict[str, Any] = {
    "state": {"line": "can you wave"},
    "questions": {
        "start": {"type": "noul", "instructions": "Start a gesture now?"},
        "which": {
            "type": "choice",
            "instructions": "Which?",
            "criteria": {"wave": None, "nod": None},
        },
    },
}
RESPONSE = DecisionResponse(
    model="jev-1.13.0",
    answers={
        "start": {"type": "noul", "noul": 0.9},
        "which": {
            "type": "choice",
            "choice": "wave",
            "probabilities": {"wave": 0.8, "nod": 0.2},
            "confidence": 0.7,
        },
    },
    usage={"input_tokens": 12, "output_tokens": 0},
    timing={"total_ms": 3, "provider_ms": 2},
)


@pytest.fixture
def decision_service() -> MagicMock:
    service = MagicMock()
    service.health_check = AsyncMock(
        return_value={
            "healthy": True,
            "configured": True,
            "provider": "typesafe",
            "model": "jev-latest",
        }
    )
    service.decide = AsyncMock(return_value=RESPONSE)
    return service


@pytest.fixture
def assistant_service() -> MagicMock:
    service = MagicMock()

    def validate(name: str | None) -> None:
        if name == "missing":
            raise UnknownProfileError("Unknown profile 'missing'")

    service.validate_profile = MagicMock(side_effect=validate)
    return service


@pytest.fixture
async def client(decision_service: MagicMock, assistant_service: MagicMock):
    app = FastAPI()
    app.state.decision_service = decision_service
    app.state.assistant_service = assistant_service
    app.include_router(router, prefix="/api")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_status_reports_the_service_health(client: AsyncClient):
    response = await client.get("/api/decisions/status")
    assert response.status_code == 200
    assert response.json()["configured"] is True


async def test_decide_returns_typed_answers(client: AsyncClient, decision_service: MagicMock):
    response = await client.post("/api/decisions", json=BODY)
    assert response.status_code == 200
    data = response.json()
    assert data["answers"]["which"]["choice"] == "wave"
    assert data["timing"] == {"total_ms": 3, "provider_ms": 2}
    request = decision_service.decide.await_args.args[0]
    assert request.questions["start"].type == "noul"
    assert request.profile is None


async def test_profile_is_validated_before_the_call(
    client: AsyncClient, decision_service: MagicMock
):
    response = await client.post("/api/decisions", json={**BODY, "profile": "missing"})
    assert response.status_code == 422
    assert "missing" in response.json()["detail"]
    decision_service.decide.assert_not_awaited()

    response = await client.post("/api/decisions", json={**BODY, "profile": "avatar"})
    assert response.status_code == 200
    assert decision_service.decide.await_args.args[0].profile == "avatar"


async def test_invalid_body_is_422_without_a_call(client: AsyncClient, decision_service: MagicMock):
    response = await client.post("/api/decisions", json={"state": "s", "questions": {}})
    assert response.status_code == 422
    decision_service.decide.assert_not_awaited()


async def test_missing_key_is_a_clear_503(client: AsyncClient, decision_service: MagicMock):
    decision_service.decide.side_effect = DecisionError(
        "Decisions need TYPESAFE_API_KEY on the runtime; no fallback is used", 503
    )
    response = await client.post("/api/decisions", json=BODY)
    assert response.status_code == 503
    assert response.json() == {
        "detail": "Decisions need TYPESAFE_API_KEY on the runtime; no fallback is used"
    }


async def test_provider_errors_carry_the_provider_status(
    client: AsyncClient, decision_service: MagicMock
):
    decision_service.decide.side_effect = DecisionError(
        "Decision provider rate limit reached",
        429,
        provider_status_code=429,
        provider_detail="Rate limit exceeded",
    )
    response = await client.post("/api/decisions", json=BODY)
    assert response.status_code == 429
    assert response.json() == {
        "detail": "Decision provider rate limit reached",
        "provider_status_code": 429,
        "provider_detail": "Rate limit exceeded",
    }


async def test_absent_service_is_503():
    app = FastAPI()
    app.include_router(router, prefix="/api")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/api/decisions/status")
    assert response.status_code == 503
