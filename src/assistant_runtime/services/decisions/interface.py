"""DecisionService: typed decisions from a configured decision provider."""

from __future__ import annotations

import os
import time
from typing import Any, Protocol

import httpx
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from assistant_runtime.services.decisions._typesafe import TypeSafeDecisions
from assistant_runtime.services.decisions.config import DecisionsConfig
from assistant_runtime.services.decisions.exceptions import DecisionError
from assistant_runtime.services.decisions.models import (
    Answer,
    DecisionRequest,
    DecisionResponse,
    Timing,
)


class DecisionProvider(Protocol):
    """What a decision provider offers: every question answered in one call."""

    name: str

    @property
    def model(self) -> str: ...

    async def decide(
        self, *, api_key: str, state: Any, questions: dict[str, Any]
    ) -> dict[str, Any]:
        """The provider's response body (``answers`` keyed like ``questions``); raises ``DecisionError``."""
        ...


_ANSWERS = TypeAdapter(dict[str, Answer])


class DecisionService:
    """Lifecycle-managed decision calls; the provider key stays server-side."""

    def __init__(self, config: DecisionsConfig, *, provider: DecisionProvider | None = None):
        self.config = config
        self._provider = provider
        self._client: httpx.AsyncClient | None = None
        self._started = False

    async def start(self) -> None:
        if self._provider is None:
            self._client = httpx.AsyncClient()
            self._provider = TypeSafeDecisions(
                base_url=self.config.base_url,
                model=self.config.model,
                timeout_seconds=self.config.timeout_seconds,
                client=self._client,
            )
        self._started = True
        logger.info(
            "Decision service started",
            provider=self._provider.name,
            configured=self.configured,
        )

    async def stop(self) -> None:
        self._started = False
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> dict:
        return {
            "healthy": self._started,
            "configured": self.configured,
            "provider": self._provider.name if self._provider is not None else None,
            "model": self._provider.model if self._provider is not None else None,
        }

    @property
    def configured(self) -> bool:
        return bool(os.getenv(self.config.api_key_env))

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Answer every question in one provider call. No fallback: an absent key fails."""
        started = time.perf_counter()
        if not self._started or self._provider is None:
            raise DecisionError("Decision service unavailable", 503)
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise DecisionError(
                f"Decisions need {self.config.api_key_env} on the runtime; no fallback is used",
                503,
            )
        if len(request.questions) > self.config.max_questions:
            raise DecisionError(
                f"At most {self.config.max_questions} questions per call "
                f"(DECISIONS__MAX_QUESTIONS)",
                422,
            )
        questions = {
            key: question.model_dump(exclude_none=True)
            for key, question in request.questions.items()
        }
        provider_started = time.perf_counter()
        data = await self._provider.decide(
            api_key=api_key, state=request.state, questions=questions
        )
        provider_ms = round((time.perf_counter() - provider_started) * 1000)
        answers = _validated_answers(data.get("answers"), request)
        usage = data.get("usage")
        model = data.get("model")
        return DecisionResponse(
            model=model if isinstance(model, str) else None,
            answers=answers,
            usage=usage if isinstance(usage, dict) else None,
            timing=Timing(
                total_ms=round((time.perf_counter() - started) * 1000), provider_ms=provider_ms
            ),
        )


def _validated_answers(raw: Any, request: DecisionRequest) -> dict[str, Answer]:
    """Every question answered, each with its own question's type; otherwise 502."""
    try:
        answers = _ANSWERS.validate_python(raw)
    except ValidationError as exc:
        logger.warning("Decision provider answers failed validation", errors=exc.error_count())
        raise DecisionError("Decision provider returned an unexpected answer", 502) from exc
    for key, question in request.questions.items():
        answer = answers.get(key)
        if answer is None or answer.type != question.type:
            logger.warning("Decision provider answer missing or mistyped", question=key)
            raise DecisionError("Decision provider returned an unexpected answer", 502)
    return answers
