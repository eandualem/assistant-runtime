"""Actual Pydantic AI/OpenAI serialization against an offline Codex HTTP endpoint."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError
from pydantic_ai import models
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from assistant_runtime.app.assistant.usage import merge_usage, usage_dict
from assistant_runtime.config import AppSettings
from assistant_runtime.services.llm.config import LLMConfig
from assistant_runtime.services.llm.interface import LlmService


def frames(tier, response_id="resp_test", tool=False, model="gpt-5.6-sol"):
    response = {
        "id": response_id,
        "object": "response",
        "model": model,
        "created_at": 1,
        "status": "in_progress",
        "output": [],
        "service_tier": "priority",
    }
    events = [
        {"type": "response.created", "response": response},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_test", "role": "assistant", "content": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_test",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hold.",
            "logprobs": [],
        },
        {
            "type": "response.completed",
            "response": {
                **response,
                "status": "completed",
                "service_tier": tier,
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            },
        },
    ]
    if tool:
        call = {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "move",
            "arguments": "{}",
            "status": "completed",
        }
        events[1:3] = [
            {"type": "response.output_item.added", "output_index": 0, "item": call},
            {"type": "response.output_item.done", "output_index": 0, "item": call},
        ]
        events[-1]["response"]["output"] = [call]
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


@pytest.fixture
async def codex(monkeypatch):
    # Real model requests are allowed only into this explicitly replaced HTTP client.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    state = SimpleNamespace(
        requests=[], actual="priority", status=200, streams=[], tool=False, model="gpt-5.6-sol"
    )

    class WireStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            for frame in frames(state.actual, tool=state.tool, model=state.model).split("\n\n"):
                await asyncio.sleep(0)
                yield (frame + "\n\n").encode()

        async def aclose(self):
            self.closed = True

    async def handler(request):
        assert str(request.url) == "https://chatgpt.com/backend-api/codex/responses"
        assert request.headers["authorization"] == "Bearer subscription-test"
        state.requests.append(json.loads(request.content))
        if state.status != 200:
            return httpx.Response(state.status, json={"error": {"message": "Unsupported tier"}})
        await asyncio.sleep(0)
        stream = WireStream()
        state.streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:

        def client(**kwargs):
            assert kwargs["api_key"] == "subscription-test"
            return AsyncOpenAI(**{**kwargs, "http_client": http, "max_retries": 0})

        monkeypatch.setattr("assistant_runtime.services.llm.interface.AsyncOpenAI", client)
        services = []

        def service(tier):
            instance = LlmService(
                LLMConfig(
                    codex_only=True,
                    codex_service_tier=tier,
                    primary_model=f"openai:{state.model}",
                )
            )
            instance.set_oauth_service(
                SimpleNamespace(
                    get_codex_session=lambda: SimpleNamespace(
                        access_token="subscription-test", account_id="test-account"
                    )
                )
            )
            services.append(instance)
            return instance

        state.service = service
        yield state
        for instance in services:
            await instance.stop()


@pytest.mark.parametrize(
    ("setting", "wire"), [(None, None), ("default", "default"), ("fast", "priority")]
)
@pytest.mark.parametrize(
    "actual", ["priority", "default", "fast", "ultrafast", None, "unrecognized"]
)
async def test_wire_and_actual_response_are_distinct(codex, setting, wire, actual):
    codex.actual = actual
    service = codex.service(setting)
    agent = service.build_agent(system_prompt="Return a short decision.", thinking_budget=4000)
    result = await agent.run("Wait.")
    assert result.output == "Hold."
    assert len(codex.requests) == 1
    payload = codex.requests[0]
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["reasoning"]["effort"] == "low"
    assert payload["store"] is False
    assert payload["stream"] is True
    if wire is None:
        assert "service_tier" not in payload
    else:
        assert payload["service_tier"] == wire
    assert "max_output_tokens" not in payload
    assert "temperature" not in payload
    actual = None if actual == "unrecognized" else actual
    response = result.response
    assert response.provider_details["codex_service_tier"] == {"requested": wire, "actual": actual}
    usage = usage_dict(result)
    assert usage["service_tiers"] == [
        {
            "provider_response_id": "resp_test",
            "model": "gpt-5.6-sol",
            "requested": wire,
            "actual": actual,
        }
    ]
    assert merge_usage(usage, {"requests": 0})["service_tiers"] == usage["service_tiers"]
    assert (await service.health_check())["codex_service_tier"] == setting


async def test_concurrent_calls_do_not_share_requested_tier(codex):
    service = codex.service("fast")
    agent = service.build_agent(system_prompt="Return a short decision.", thinking_budget=4000)
    results = await asyncio.gather(
        agent.run("Fast."),
        agent.run("Standard.", model_settings={"openai_service_tier": "default"}),
    )
    assert [r.response.provider_details["codex_service_tier"]["requested"] for r in results] == [
        "priority",
        "default",
    ]


async def test_tier_rejection_does_not_retry_with_standard_or_api(codex):
    codex.status = 400
    service = codex.service("fast")
    agent = service.build_agent(system_prompt="Decision.", thinking_budget=4000)
    with pytest.raises(ModelHTTPError):
        await agent.run("Wait.")
    assert len(codex.requests) == 1
    assert codex.requests[0]["service_tier"] == "priority"


def test_startup_config_validation(monkeypatch):
    monkeypatch.setenv("LLM__CODEX_SERVICE_TIER", "fast")
    assert AppSettings(_env_file=None).llm.codex_service_tier == "fast"
    with pytest.raises(ValidationError):
        LLMConfig(codex_service_tier="priority")


def test_fast_default_does_not_configure_api_key_requests():
    service = LlmService(LLMConfig(codex_service_tier="fast"))
    original = {"openai_reasoning_effort": "low"}
    assert service._apply_model_transport_defaults("openai:gpt-5.6-sol", original) is original


async def test_native_stream_cancellation_closes_transport_and_keeps_actual_unknown(codex):
    agent = codex.service("fast").build_agent(system_prompt="Decision.", thinking_budget=4000)
    async with agent.model.request_stream(
        [ModelRequest(parts=[UserPromptPart("Wait.")])],
        agent.model_settings,
        ModelRequestParameters(),
    ) as streamed:
        assert not codex.streams[0].closed
        await streamed.close_stream()
        assert codex.streams[0].closed
        assert streamed.provider_details["codex_service_tier"] == {
            "requested": "priority",
            "actual": None,
        }


async def test_tiers_survive_cumulative_followup_without_double_counting(codex):
    from pydantic_ai.usage import RunUsage

    from assistant_runtime.app.streaming._runner import TurnRunner, _RunState

    agent = codex.service("fast").build_agent(system_prompt="Decision.", thinking_budget=4000)
    cumulative = RunUsage()
    first = await agent.run("First.", usage=cumulative)
    plan = SimpleNamespace(accepted_tool_result=None, prior_usage=None)
    state = _RunState()
    TurnRunner._capture_result(plan, state, first)
    second = await agent.run("Follow up.", message_history=first.all_messages(), usage=cumulative)
    TurnRunner._capture_result(plan, state, second)
    assert state.usage["requests"] == 2
    assert len(state.usage["service_tiers"]) == 2
    assert state.usage["input_tokens"] == 6
    assert state.usage["output_tokens"] == 4


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-6-astra"])
@pytest.mark.parametrize("mode", ["host_tools", "text"])
async def test_http_host_continuations_retain_tiers(codex, isolated_services, mode, model):
    from assistant_runtime.main import create_app

    codex.tool = True
    codex.model = model
    configured = codex.service("fast")
    settings = isolated_services.model_copy(update={"llm": configured._config})
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        app.state.llm_service.set_oauth_service(configured._oauth_service)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/chat",
                json={
                    "id": "decision",
                    "session_id": "body",
                    "content": "Move.",
                    "output_mode": mode,
                    "config": {"thinking_budget": 4000},
                    "host_context": {
                        "actions": [
                            {
                                "name": "move",
                                "description": "Move the host",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ]
                    },
                },
            )
            assert response.status_code == 200, response.text
            payload = codex.requests[0]
            assert payload["model"] == model
            assert payload["reasoning"]["effort"] == "low"
            assert payload["service_tier"] == "priority"
            assert any(t["name"] == "move" for t in payload["tools"])
            for unsupported in ("temperature", "top_p", "top_logprobs", "logprobs"):
                assert unsupported not in payload
            assert "message.output_text.logprobs" not in payload.get("include", [])
            decision = response.json()
            if mode == "host_tools":
                assert decision["decision"] == "pending", decision
                assert decision["content"] is None
            assert decision["usage"]["service_tiers"][0]["actual"] == "priority"
            codex.tool = False
            receipt = await client.post(
                "/api/chat",
                json={
                    "id": "receipt",
                    "session_id": "body",
                    "content": "",
                    "tool_call_id": decision["pending_tool_call"]["call_id"],
                    "tool_result": {"executed": False},
                    "tool_outcome": "failed",
                },
            )
            assert receipt.status_code == 200, receipt.text
            tiers = receipt.json()["usage"]["service_tiers"]
            assert tiers[0] == decision["usage"]["service_tiers"][0]
            if mode == "host_tools":
                assert receipt.json()["decision"] == "completed"
                assert len(tiers) == len(codex.requests) == 1
            else:
                assert receipt.json()["content"] == "Hold."
                assert len(tiers) == len(codex.requests) == 2
                assert receipt.json()["usage"]["requests"] == 2
