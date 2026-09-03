"""Codex-backed OpenAI responses model helpers.

OpenAI's ChatGPT/Codex backend currently requires streaming responses, even for
calls that the rest of the app treats as non-streaming. This adapter bridges
that mismatch by exhausting the stream and returning the accumulated response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.settings import ModelSettings


@dataclass(init=False)
class OpenAICodexResponsesModel(OpenAIResponsesModel):
    """OpenAI Responses model variant for ChatGPT/Codex subscription auth."""

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
