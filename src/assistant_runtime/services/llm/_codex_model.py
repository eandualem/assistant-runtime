"""Codex-backed OpenAI responses model helpers.

OpenAI's ChatGPT/Codex backend currently requires streaming responses, even for
calls that the rest of the app treats as non-streaming. This adapter bridges
that mismatch by exhausting the stream and returning the accumulated response.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

from openai.resources.responses import AsyncResponses
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.settings import ModelSettings


@dataclass
class _TierObservation:
    requested: str | None = None
    actual: str | None = None
    response: StreamedResponse | None = None

    def publish(self) -> None:
        if self.response is not None:
            self.response.provider_details = {
                **(self.response.provider_details or {}),
                "codex_service_tier": {"requested": self.requested, "actual": self.actual},
            }


_tier_observation: ContextVar[_TierObservation | None] = ContextVar(
    "codex_tier_observation", default=None
)
_KNOWN_TIERS = {
    "auto",
    "default",
    "flex",
    "scale",
    "priority",
    "fast",
    "ultrafast",
    "standard",
    "on_demand",
}


class _ObservedCodexStream:
    """Observe SDK-decoded events; leave SSE parsing and execution to upstream."""

    def __init__(self, stream: Any, observation: _TierObservation):
        self.stream = stream
        self.observation = observation

    async def __aenter__(self):
        await self.stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self.stream.__aexit__(*args)

    async def close(self) -> None:
        await self.stream.close()

    async def __aiter__(self):
        async for event in self.stream:
            if event.type in {"response.completed", "response.incomplete"}:
                tier = event.response.service_tier
                self.observation.actual = (
                    tier if isinstance(tier, str) and tier in _KNOWN_TIERS else None
                )
                self.observation.publish()
            yield event


class CodexResponses(AsyncResponses):
    """Keep the actual subscription tier that Pydantic AI does not yet retain."""

    async def create(self, **kwargs: Any) -> Any:
        stream = await super().create(**kwargs)
        observation = _tier_observation.get()
        if kwargs.get("stream") and observation is not None:
            requested = kwargs.get("service_tier")
            observation.requested = requested if isinstance(requested, str) else None
            observation.publish()
            return _ObservedCodexStream(stream, observation)
        return stream


@dataclass(init=False)
class OpenAICodexResponsesModel(OpenAIResponsesModel):
    """OpenAI Responses model variant for ChatGPT/Codex subscription auth."""

    @asynccontextmanager
    async def request_stream(self, *args: Any, **kwargs: Any):
        observation = _TierObservation()
        token = _tier_observation.set(observation)
        try:
            async with super().request_stream(*args, **kwargs) as streamed:
                observation.response = streamed
                observation.publish()
                yield streamed
        finally:
            _tier_observation.reset(token)

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        async with self.request_stream(
            cast(list, messages),
            model_settings,
            model_request_parameters,
        ) as streamed:
            async for _event in streamed:
                pass
            return streamed.get()
