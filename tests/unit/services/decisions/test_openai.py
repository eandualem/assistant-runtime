"""OpenAI's native decision answers through the existing runtime contract, offline."""

import copy
import json

import httpx
import pytest

from assistant_runtime.services.decisions import (
    DecisionError,
    DecisionRequest,
    DecisionsConfig,
    DecisionService,
)
from assistant_runtime.services.decisions._openai import OpenAIDecisions

from .test_decisions import QUESTIONS

ANSWERS = [
    {"name": "start", "type": "predicate", "probability": 0.93},
    {
        "name": "intent",
        "type": "choice",
        "choice": "explicit",
        "confidence": 0.8,
        "probabilities": [
            {"value": "explicit", "probability": 0.9},
            {"value": "none", "probability": 0.1},
        ],
    },
    {
        "name": "energy",
        "type": "score",
        "score": 0.7,
        "confidence": 0.4,
        "probabilities": [
            {"value": 0, "label": "0", "probability": 0.3},
            {"value": 1, "label": "1", "probability": 0.7},
        ],
    },
]


def test_explicit_provider_selects_defaults_and_retains_overrides():
    default = DecisionsConfig()
    assert (default.provider, default.api_key_env, default.model, default.base_url) == (
        "typesafe",
        "TYPESAFE_API_KEY",
        "jev-latest",
        "https://api.typesafe.ai",
    )
    selected = DecisionsConfig(provider="openai")
    assert (selected.api_key_env, selected.model, selected.base_url) == (
        "OPENAI_API_KEY",
        "gpt-6-luna",
        "https://api.openai.com",
    )
    explicit = DecisionsConfig(
        provider="openai", api_key_env="DECIDER_KEY", model="custom", base_url="http://localhost"
    )
    assert (explicit.api_key_env, explicit.model, explicit.base_url) == (
        "DECIDER_KEY",
        "custom",
        "http://localhost",
    )


async def test_selected_provider_maps_questions_and_answers_and_owns_its_client(monkeypatch):
    seen, clients = [], []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200, json={"model": "gpt-6-luna", "answers": ANSWERS, "usage": {"input_tokens": 80}}
        )

    original = httpx.AsyncClient

    def client():
        value = original(transport=httpx.MockTransport(handler))
        clients.append(value)
        return value

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "wrong-provider-key")
    service = DecisionService(DecisionsConfig(provider="openai"))
    await service.start()
    try:
        status = await service.health_check()
        assert status["provider"] == "openai"
        assert status["configured"]
        questions = copy.deepcopy(QUESTIONS)
        questions["start"]["criteria"] = {"true": {"examples": ["wave"]}, "false": "wait"}
        request = DecisionRequest(
            state={"image_url": "https://example.com/data"}, questions=questions
        )
        result = await service.decide(request)
        assert result.answers["start"].noul == 0.93
        assert result.answers["intent"].probabilities == {"explicit": 0.9, "none": 0.1}
        assert result.answers["intent"].confidence == 0.8
        assert result.answers["energy"].score == 0.7
        assert result.answers["energy"].probabilities == {"0": 0.3, "1": 0.7}
        assert result.answers["energy"].confidence == 0.4
        assert result.answers["energy"].legend == {"0": "still", "1": "lively"}
        assert result.model == "gpt-6-luna"
        assert result.usage == {"input_tokens": 80}
        assert str(seen[0].url) == "https://api.openai.com/v1/decisions"
        assert seen[0].headers["authorization"] == "Bearer test-openai-key"
        body = json.loads(seen[0].content)
        assert json.loads(body["input"]) == request.state  # evidence, not an image fetch
        mapped = {q["name"]: q for q in body["questions"]}
        assert mapped["intent"]["choices"] == [
            {
                "value": "explicit",
                "description": json.dumps(questions["intent"]["criteria"]["explicit"]),
            },
            {"value": "none", "description": ""},
        ]
        assert mapped["start"]["type"] == "predicate"
        assert json.dumps(questions["start"]["criteria"]) in mapped["start"]["instructions"]
        assert mapped["energy"]["levels"] == [
            {"label": "0", "description": "still"},
            {"label": "1", "description": "lively"},
        ]
        await service.stop()
        assert clients[0].is_closed
        await service.start()
        assert len(clients) == 2
        assert not clients[1].is_closed
        monkeypatch.delenv("OPENAI_API_KEY")
        assert not service.configured  # never uses the other provider's key
        with pytest.raises(DecisionError) as error:
            await service.decide(request)
        assert error.value.status_code == 503
        assert len(seen) == 1
    finally:
        await service.stop()


@pytest.mark.parametrize(
    "bad", ["refusal", "missing", "duplicate", "options", "type", "confidence"]
)
async def test_refused_or_malformed_answers_fail_without_fabricated_values(bad, monkeypatch):
    answers = copy.deepcopy(ANSWERS)
    if bad == "refusal":
        answers[0] = {"name": "start", "type": "refusal"}
    elif bad == "missing":
        answers.pop()
    elif bad == "duplicate":
        answers.append(answers[0])
    elif bad == "options":
        answers[1]["probabilities"][1]["value"] = "explicit"
    elif bad == "type":
        answers[0]["type"] = "score"
    else:
        answers[1]["confidence"] = float("inf")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=json.dumps({"answers": answers}))
        )
    ) as client:
        provider = OpenAIDecisions(
            base_url="https://api.openai.com", model="gpt-6-luna", timeout_seconds=10, client=client
        )
        service = DecisionService(DecisionsConfig(provider="openai"), provider=provider)
        await service.start()
        try:
            with pytest.raises(DecisionError) as error:
                await service.decide(DecisionRequest(state={}, questions=QUESTIONS))
            assert error.value.status_code == 502
            if bad == "refusal":
                assert "declined" in str(error.value)
        finally:
            await service.stop()


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (400, 422),
        (401, 502),
        (429, 429),
        (500, 502),
        ("timeout", 504),
        ("network", 502),
        ("html", 502),
    ],
)
async def test_provider_failures_keep_the_runtime_error_contract(failure, status):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        if failure == "network":
            raise httpx.ConnectError("down", request=request)
        if failure == "html":
            return httpx.Response(200, text="<html>")
        return httpx.Response(failure, json={"error": {"message": "reason " + "x" * 600}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIDecisions(
            base_url="https://api.openai.com", model="gpt-6-luna", timeout_seconds=10, client=client
        )
        with pytest.raises(DecisionError) as error:
            await provider.decide(api_key="test-key", state={}, questions=QUESTIONS)
        assert error.value.status_code == status
        if failure in (400, 429):
            assert len(error.value.provider_detail) == 500
        else:
            assert error.value.provider_detail is None
