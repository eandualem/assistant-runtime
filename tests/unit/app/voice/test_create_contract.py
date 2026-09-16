"""Live create HTTP contracts, with both HTTP boundaries exercised offline."""

import io
import json
from types import SimpleNamespace
from typing import Literal

import httpx
import pytest
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, ValidationError

from assistant_runtime.app.access.deps import get_principal
from assistant_runtime.app.routes.voice import router
from assistant_runtime.app.voice._transport import LiveTransport
from assistant_runtime.app.voice.config import VoiceConfig
from assistant_runtime.app.voice.exceptions import VoiceError
from assistant_runtime.app.voice.interface import VoiceService
from tests.unit.app.voice.test_voice import OWNER, Backend
from tests.voice_helpers import Transport


# Narrow fixture transcribed from the public Live DataChannelConfig contract:
# https://developers.openai.com/api/reference/typescript/resources/live
# Last verified 2026-09-14. This is a provider-side shape check, not a production model.
class ServerEventSelector(BaseModel, extra="forbid"):
    type: str
    response_event: str | None = None


class DataChannelConfig(BaseModel, extra="forbid"):
    allowed_client_events: Literal["all"] | list[str] = "all"
    allowed_server_events: Literal["all"] | list[ServerEventSelector] = "all"


class ClientConfig(BaseModel, extra="forbid"):
    data_channel: DataChannelConfig


@pytest.fixture
async def boundary(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key-never-log")
    backend, sideband = Backend(), Transport()
    transport = LiveTransport(1)
    await transport.http.aclose()
    state = SimpleNamespace(status=201, requests=[], failure=None, headers={}, malformed=False)

    async def provider(request):
        state.requests.append(request)
        if state.failure:
            raise state.failure
        if state.status != 201:
            return httpx.Response(
                state.status,
                headers=state.headers,
                json={"error": {"message": "secret-key-never-log private-history private-sdp"}},
            )
        if state.malformed:
            return httpx.Response(201, json={"session": {"id": "live_orphan"}})
        body = json.loads(request.content)
        if "client" in body["session"]:
            try:
                state.permissions = ClientConfig.model_validate(body["session"]["client"])
            except ValidationError:
                return httpx.Response(400, json={"error": {"message": "invalid data_channel"}})
        return httpx.Response(
            201,
            json={"session": {"id": "live_provider"}, "transport": {"sdp": "answer"}},
        )

    transport.http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    transport.attach = sideband.attach
    service = VoiceService(
        VoiceConfig(enabled=True, close_timeout_seconds=0.01), backend, transport=transport
    )
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.state.voice_service = service
    app.dependency_overrides[get_principal] = lambda: OWNER
    state.backend, state.service, state.sideband = backend, service, sideband
    state.transport = transport
    state.logs = io.StringIO()
    sink = logger.add(state.logs, format="{message}")
    await service.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            state.client = client
            yield state
    finally:
        await service.stop()
        logger.remove(sink)


async def create(boundary, **fields):
    return await boundary.client.post(
        "/api/voice/calls",
        json={
            "session_id": "thread",
            "sdp": "private-sdp",
            "mode": "conversation",
            "history": [{"role": "user", "content": "private-history"}],
            **fields,
        },
    )


async def test_conversation_payload_passes_documented_provider_shape(boundary):
    # The documented provider validator rejects the deployed regression.
    with pytest.raises(ValidationError):
        ClientConfig.model_validate({"data_channel": ["session.instructions.append"]})
    response = await create(boundary)
    assert response.status_code == 201, response.text
    permissions = boundary.permissions.data_channel
    assert permissions.allowed_client_events == [
        "session.instructions.append",
        "session.thinking.append",
        "session.input_audio.mute",
        "session.input_audio.unmute",
    ]
    assert permissions.allowed_server_events == "all"  # Keep acknowledgments/transcripts.
    body = json.loads(boundary.requests[0].content)
    assert body["session"]["delegation"] == {"type": "client"}
    assert body["transport"] == {"type": "webrtc", "sdp": "private-sdp"}
    assert len(boundary.requests) == 1


@pytest.mark.parametrize(
    "status", [400, 401, 402, 403, 404, 405, 408, 409, 413, 415, 418, 422, 429, 499, 500, 502, 503]
)
async def test_provider_rejection_metadata_and_reservation_cleanup(boundary, status):
    boundary.status = status
    request_id = "req_" + "a" * 32
    boundary.headers = {"x-request-id": request_id}
    response = await create(boundary)
    assert response.status_code == 502
    expected = (
        "rejected"
        if status in {400, 401, 402, 403, 404, 405, 409, 413, 415, 422, 429}
        else "unknown"
    )
    assert response.json() == {
        "detail": "GPT-Live rejected the API credentials or model access"
        if status in (401, 403)
        else "GPT-Live session creation failed",
        "allocation_status": expected,
        "provider_status_code": status,
        "provider_request_id": request_id,
    }
    assert not boundary.backend.leases
    assert (await boundary.service.health_check())["active_calls"] == 0
    boundary.sideband.attach.assert_not_awaited()
    assert len(boundary.requests) == 1
    logs = boundary.logs.getvalue()
    assert f"provider_status_code={status}" in logs
    assert f"allocation_status={expected}" in logs
    assert request_id in logs
    for private in ("secret-key-never-log", "private-sdp", "private-history"):
        assert private not in response.text + logs


@pytest.mark.parametrize("request_id", ["secret-key-never-log", "req_" + "a" * 500, ""])
async def test_unrecognized_request_id_is_not_exposed(boundary, request_id):
    boundary.status = 400
    boundary.headers = {"x-request-id": request_id}
    response = await create(boundary)
    assert "provider_request_id" not in response.json()
    assert "secret-key-never-log" not in response.text + boundary.logs.getvalue()


async def test_timeout_remains_unknown_and_is_not_retried(boundary):
    boundary.failure = httpx.ReadTimeout("private-sdp secret-key-never-log")
    response = await create(boundary)
    assert response.status_code == 502
    assert response.json()["allocation_status"] == "unknown"
    assert "provider_status_code" not in response.json()
    assert len(boundary.requests) == 1
    assert not boundary.backend.leases
    assert "private-sdp" not in response.text + boundary.logs.getvalue()


async def test_attach_failure_remains_unknown_after_successful_creation(boundary):
    boundary.sideband.attach.side_effect = RuntimeError("private-sdp secret-key-never-log")
    response = await create(boundary)
    assert response.status_code == 502
    assert response.json()["allocation_status"] == "unknown"
    assert not boundary.backend.leases
    assert len(boundary.requests) == 1
    assert "private-sdp" not in response.text + boundary.logs.getvalue()


async def test_local_policy_rejection_has_no_provider_request(boundary):
    boundary.service.config = boundary.service.config.model_copy(
        update={"delegation_enabled": False}
    )
    response = await create(boundary, mode="delegated")
    assert response.status_code == 409
    assert response.json()["allocation_status"] == "rejected"
    assert "provider_status_code" not in response.json()
    assert not boundary.requests
    assert not boundary.backend.leases


async def test_malformed_success_remains_unknown(boundary):
    boundary.malformed = True
    response = await create(boundary)
    assert response.status_code == 502
    assert response.json()["allocation_status"] == "unknown"
    assert len(boundary.requests) == 1
    assert not boundary.backend.leases
    boundary.sideband.attach.assert_not_awaited()


async def test_old_array_is_rejected_at_provider_http_boundary(boundary, monkeypatch):
    send_create = boundary.transport.create

    async def old_payload(key, session, sdp):
        session["client"]["data_channel"] = session["client"]["data_channel"][
            "allowed_client_events"
        ]
        return await send_create(key, session, sdp)

    monkeypatch.setattr(boundary.transport, "create", old_payload)
    response = await create(boundary)
    assert response.status_code == 502
    assert response.json()["provider_status_code"] == 400
    assert response.json()["allocation_status"] == "rejected"
    assert len(boundary.requests) == 1
    assert not boundary.backend.leases
    boundary.sideband.attach.assert_not_awaited()


async def test_attach_cannot_reclassify_created_session_as_rejected(boundary):
    boundary.sideband.attach.side_effect = VoiceError(
        "Sideband rejected", 502, allocation_status="rejected", provider_status_code=403
    )
    response = await create(boundary)
    assert response.status_code == 502
    assert response.json()["allocation_status"] == "unknown"
    assert "provider_status_code" not in response.json()
    assert len(boundary.requests) == 1
    assert not boundary.backend.leases


@pytest.mark.parametrize("instructions", ["", " \n\t", "x" * 16001])
async def test_invalid_call_instructions_rejected_before_allocation(boundary, instructions):
    response = await create(boundary, instructions=instructions)
    assert response.status_code == 422
    assert not boundary.requests
    assert not boundary.backend.leases


@pytest.mark.parametrize("profile", ["", "../profile", "/tmp/profile.toml", "Uppercase", "a" * 65])
async def test_invalid_profile_name_rejected_before_allocation(boundary, profile):
    response = await create(boundary, profile=profile)
    assert response.status_code == 422
    assert not boundary.requests
    assert not boundary.backend.leases


async def test_unknown_profile_rejected_before_allocation(boundary):
    response = await create(boundary, profile="unregistered")
    assert response.status_code == 422
    assert response.json()["allocation_status"] == "rejected"
    assert not boundary.requests
    assert not boundary.backend.leases


@pytest.mark.parametrize("instructions", [None, "x" * 16000])
async def test_valid_call_instructions_accepted_without_overriding_policy(boundary, instructions):
    response = await create(boundary, instructions=instructions)
    assert response.status_code == 201
    session = json.loads(boundary.requests[0].content)["session"]
    expected = instructions or boundary.service.config.conversation_instructions
    assert session["instructions"].startswith(expected)
    assert "Do not delegate work or call tools" in session["instructions"][len(expected) :]
    assert session["model"] == boundary.service.config.model
    assert session["audio"]["output"]["voice"] == boundary.service.config.voice
