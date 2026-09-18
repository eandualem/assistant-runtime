"""Decision service and TypeSafe provider, against a fake System One endpoint."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from assistant_runtime.services.decisions import (
    DecisionError,
    DecisionRequest,
    DecisionsConfig,
    DecisionService,
)
from assistant_runtime.services.decisions._typesafe import TypeSafeDecisions

QUESTIONS: dict[str, Any] = {
    "intent": {
        "type": "choice",
        "instructions": "What does the latest user line ask for?",
        "criteria": {"explicit": {"what": "a request", "examples": ["wave"]}, "none": None},
    },
    "start": {"type": "noul", "instructions": "Start a gesture now?"},
    "energy": {"type": "score", "instructions": "How lively?", "criteria": ["still", "lively"]},
}
ANSWERS: dict[str, Any] = {
    "intent": {
        "type": "choice",
        "choice": "explicit",
        "probabilities": {"explicit": 0.9, "none": 0.1},
        "confidence": 0.8,
    },
    "start": {"type": "noul", "noul": 0.93},
    "energy": {
        "type": "score",
        "score": 1,
        "legend": {"0": "still", "1": "lively"},
        "probabilities": {"0": 0.3, "1": 0.7},
        "confidence": 0.4,
    },
}


def _request(**overrides: Any) -> DecisionRequest:
    return DecisionRequest(state={"line": "can you wave"}, questions=QUESTIONS, **overrides)


def _provider(handler, **config: Any) -> TypeSafeDecisions:
    settings = DecisionsConfig(**config)
    return TypeSafeDecisions(
        base_url=settings.base_url,
        model=settings.model,
        timeout_seconds=settings.timeout_seconds,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def _service(handler, env: dict[str, str] | None, monkeypatch, **config: Any):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    service = DecisionService(DecisionsConfig(**config), provider=_provider(handler, **config))
    await service.start()
    return service


class TestModels:
    def test_question_types_and_shapes(self):
        request = _request()
        assert {q.type for q in request.questions.values()} == {"choice", "noul", "score"}

    @pytest.mark.parametrize(
        "question",
        [
            {"type": "choice", "instructions": "x", "criteria": {"only": None}},
            {"type": "score", "instructions": "x", "criteria": ["one"]},
            {"type": "noul", "instructions": "x", "criteria": {"maybe": "y"}},
            {"type": "guess", "instructions": "x"},
            {"type": "noul", "instructions": "x", "extra": 1},
        ],
    )
    def test_malformed_questions_are_rejected(self, question):
        with pytest.raises(ValidationError):
            DecisionRequest(state="s", questions={"q": question})

    def test_requests_need_a_question_and_forbid_credentials(self):
        with pytest.raises(ValidationError):
            DecisionRequest(state="s", questions={})
        with pytest.raises(ValidationError):
            DecisionRequest(state="s", questions=QUESTIONS, api_key="k")
        with pytest.raises(ValidationError):
            DecisionRequest(state="s", questions=QUESTIONS, profile="Not-Valid")


class TestConfig:
    @pytest.mark.parametrize(
        "url", ["https://api.typesafe.ai", "http://127.0.0.1:7199", "http://localhost/"]
    )
    def test_https_or_loopback_base_url(self, url):
        assert DecisionsConfig(base_url=url).base_url == url

    @pytest.mark.parametrize("url", ["http://decisions.example", "ftp://x", "api.typesafe.ai"])
    def test_cleartext_base_url_is_rejected(self, url):
        with pytest.raises(ValidationError, match="https"):
            DecisionsConfig(base_url=url)


class TestTypeSafeProvider:
    async def test_sends_every_question_in_one_bearer_request(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"model": "jev-1.13.0", "answers": ANSWERS})

        provider = _provider(handler, base_url="https://decisions.example/")
        data = await provider.decide(api_key="secret", state={"a": 1}, questions=QUESTIONS)

        assert data["answers"] == ANSWERS
        assert len(seen) == 1
        assert str(seen[0].url) == "https://decisions.example/v1/systemone"
        assert seen[0].headers["authorization"] == "Bearer secret"
        body = json.loads(seen[0].content)
        assert body == {"model": "jev-latest", "state": {"a": 1}, "questions": QUESTIONS}

    @pytest.mark.parametrize(
        ("provider_status", "status"),
        [(401, 502), (403, 502), (422, 422), (429, 429), (529, 502), (500, 502), (404, 502)],
    )
    async def test_provider_errors_map_to_runtime_status_with_a_bounded_reason(
        self, provider_status, status
    ):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                provider_status, json={"detail": "questions.energy: " + "x" * 600, "state": 1}
            )

        with pytest.raises(DecisionError) as info:
            await _provider(handler).decide(api_key="k", state="s", questions=QUESTIONS)
        assert info.value.status_code == status
        assert info.value.provider_status_code == provider_status
        if provider_status in (422, 429):
            assert info.value.provider_detail.startswith("questions.energy: ")
            assert len(info.value.provider_detail) == 500
            assert info.value.metadata == {
                "provider_status_code": provider_status,
                "provider_detail": info.value.provider_detail,
            }
        else:
            assert info.value.provider_detail is None
            assert info.value.metadata == {"provider_status_code": provider_status}

    @pytest.mark.parametrize(
        ("body", "detail"),
        [
            ({"error": {"message": "bad request"}}, "bad request"),
            ({"message": "  slow down "}, "slow down"),
            ({"error": ["not", "a", "string"]}, None),
            ("plain text", None),
        ],
    )
    async def test_provider_reason_is_read_from_common_fields(self, body, detail):
        def handler(request: httpx.Request) -> httpx.Response:
            if isinstance(body, str):
                return httpx.Response(429, text=body)
            return httpx.Response(429, json=body)

        with pytest.raises(DecisionError) as info:
            await _provider(handler).decide(api_key="k", state="s", questions=QUESTIONS)
        assert info.value.provider_detail == detail
        if detail is None:
            assert info.value.metadata == {"provider_status_code": 429}

    async def test_timeout_is_504_and_transport_failure_is_502(self):
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(DecisionError) as info:
            await _provider(timeout).decide(api_key="k", state="s", questions=QUESTIONS)
        assert info.value.status_code == 504
        assert info.value.provider_status_code is None
        with pytest.raises(DecisionError) as info:
            await _provider(down).decide(api_key="k", state="s", questions=QUESTIONS)
        assert info.value.status_code == 502

    async def test_non_json_success_is_502(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>")

        with pytest.raises(DecisionError) as info:
            await _provider(handler).decide(api_key="k", state="s", questions=QUESTIONS)
        assert info.value.status_code == 502

    async def test_any_success_status_is_accepted(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json={"answers": ANSWERS})

        data = await _provider(handler).decide(api_key="k", state="s", questions=QUESTIONS)
        assert data["answers"] == ANSWERS

    async def test_closed_client_is_a_502_not_a_crash(self):
        provider = _provider(lambda request: httpx.Response(200, json={"answers": ANSWERS}))
        await provider._client.aclose()
        with pytest.raises(DecisionError) as info:
            await provider.decide(api_key="k", state="s", questions=QUESTIONS)
        assert info.value.status_code == 502


class TestDecisionService:
    async def test_missing_key_reports_unconfigured_and_fails_the_call(self, monkeypatch):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={"answers": ANSWERS})

        service = await _service(handler, None, monkeypatch)
        assert (await service.health_check()) == {
            "healthy": True,
            "configured": False,
            "provider": "typesafe",
            "model": "jev-latest",
        }
        with pytest.raises(DecisionError) as info:
            await service.decide(_request())
        assert info.value.status_code == 503
        assert "TYPESAFE_API_KEY" in str(info.value)
        assert calls == 0

    async def test_key_variable_is_configurable(self, monkeypatch):
        service = await _service(
            lambda request: httpx.Response(200, json={"answers": ANSWERS}),
            {"DECIDER_KEY": "k"},
            monkeypatch,
            api_key_env="DECIDER_KEY",
        )
        assert service.configured is True
        assert (await service.decide(_request())).answers["start"].noul == 0.93

    async def test_answers_are_typed_and_timed(self, monkeypatch):
        service = await _service(
            lambda request: httpx.Response(
                200,
                json={"model": "jev-1.13.0", "answers": ANSWERS, "usage": {"input_tokens": 80}},
            ),
            {"TYPESAFE_API_KEY": "k"},
            monkeypatch,
        )
        response = await service.decide(_request())
        assert response.model == "jev-1.13.0"
        assert response.usage == {"input_tokens": 80}
        assert response.answers["intent"].choice == "explicit"
        assert response.answers["energy"].legend == {"0": "still", "1": "lively"}
        assert response.timing.total_ms >= response.timing.provider_ms >= 0
        dumped = response.model_dump()
        assert set(dumped["answers"]) == set(QUESTIONS)

    async def test_unset_question_fields_are_not_sent(self, monkeypatch):
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"answers": ANSWERS})

        service = await _service(handler, {"TYPESAFE_API_KEY": "k"}, monkeypatch)
        await service.decide(_request(profile="avatar"))
        assert seen[0]["questions"]["start"] == {
            "type": "noul",
            "instructions": "Start a gesture now?",
        }
        assert "profile" not in seen[0]

    @pytest.mark.parametrize(
        "answers",
        [
            {"intent": ANSWERS["intent"], "start": ANSWERS["start"]},
            {
                **ANSWERS,
                "start": {"type": "choice", "choice": "x", "probabilities": {}, "confidence": 1},
            },
            {**ANSWERS, "start": {"type": "noul", "noul": 1.5}},
            {**ANSWERS, "start": {"type": "noul", "noul": float("nan")}},
            {**ANSWERS, "intent": {**ANSWERS["intent"], "confidence": 2}},
            {**ANSWERS, "energy": {**ANSWERS["energy"], "score": float("inf")}},
            "not a map",
            None,
        ],
    )
    async def test_missing_or_mistyped_answers_are_502(self, monkeypatch, answers):
        service = await _service(
            lambda request: httpx.Response(200, json={"answers": answers}),
            {"TYPESAFE_API_KEY": "k"},
            monkeypatch,
        )
        with pytest.raises(DecisionError) as info:
            await service.decide(_request())
        assert info.value.status_code == 502

    async def test_question_count_is_bounded(self, monkeypatch):
        service = await _service(
            lambda request: httpx.Response(200, json={"answers": ANSWERS}),
            {"TYPESAFE_API_KEY": "k"},
            monkeypatch,
            max_questions=2,
        )
        with pytest.raises(DecisionError) as info:
            await service.decide(_request())
        assert info.value.status_code == 422
        assert "DECISIONS__MAX_QUESTIONS" in str(info.value)

    async def test_stopped_service_refuses_calls(self, monkeypatch):
        service = await _service(
            lambda request: httpx.Response(200, json={"answers": ANSWERS}),
            {"TYPESAFE_API_KEY": "k"},
            monkeypatch,
        )
        await service.stop()
        assert (await service.health_check())["healthy"] is False
        with pytest.raises(DecisionError) as info:
            await service.decide(_request())
        assert info.value.status_code == 503

    async def test_default_provider_owns_a_shared_client(self, monkeypatch):
        service = DecisionService(DecisionsConfig())
        await service.start()
        assert (await service.health_check())["provider"] == "typesafe"
        first = service._client
        assert first is not None
        await service.stop()
        assert service._client is None
        await service.start()
        assert service._client is not None
        assert service._client is not first
        assert service._provider._client is service._client
        await service.stop()

    async def test_blank_key_is_not_configured(self, monkeypatch):
        service = await _service(
            lambda request: httpx.Response(200, json={"answers": ANSWERS}),
            {"TYPESAFE_API_KEY": "   "},
            monkeypatch,
        )
        assert service.configured is False
        with pytest.raises(DecisionError) as info:
            await service.decide(_request())
        assert info.value.status_code == 503

    async def test_state_size_is_bounded(self, monkeypatch):
        service = await _service(
            lambda request: httpx.Response(200, json={"answers": ANSWERS}),
            {"TYPESAFE_API_KEY": "k"},
            monkeypatch,
            max_state_bytes=1024,
        )
        request = DecisionRequest(state={"transcript": "x" * 2000}, questions=QUESTIONS)
        with pytest.raises(DecisionError) as info:
            await service.decide(request)
        assert info.value.status_code == 422
        assert "DECISIONS__MAX_STATE_BYTES" in str(info.value)

    async def test_unasked_answers_are_dropped(self, monkeypatch):
        service = await _service(
            lambda request: httpx.Response(
                200, json={"answers": {**ANSWERS, "debug": {"type": "noul", "noul": 0.1}}}
            ),
            {"TYPESAFE_API_KEY": "k"},
            monkeypatch,
        )
        response = await service.decide(_request())
        assert set(response.answers) == set(QUESTIONS)
