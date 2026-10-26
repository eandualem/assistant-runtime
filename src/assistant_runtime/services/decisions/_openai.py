"""OpenAI's Decisions endpoint mapped to the runtime's existing typed answers."""

from __future__ import annotations

import json
from typing import Any

import httpx
from loguru import logger

from assistant_runtime.services.decisions._typesafe import _provider_detail, _provider_error
from assistant_runtime.services.decisions.exceptions import DecisionError


class OpenAIDecisions:
    """Implements ``DecisionProvider`` using the dedicated ``POST /v1/decisions`` API."""

    name = "openai"

    def __init__(
        self, *, base_url: str, model: str, timeout_seconds: float, client: httpx.AsyncClient
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/decisions"
        self._model = model
        self._timeout = timeout_seconds
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    async def decide(
        self, *, api_key: str, state: Any, questions: dict[str, Any]
    ) -> dict[str, Any]:
        body = {
            "model": self._model,
            # JSON stays evidence, including keys resembling message or image fields.
            "input": json.dumps(state, ensure_ascii=False, separators=(",", ":")),
            "questions": [_question(name, question) for name, question in questions.items()],
        }
        try:
            response = await self._client.post(
                self._url,
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise DecisionError(
                f"Decision provider did not answer within {self._timeout:g} seconds", 504
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("Decision provider unreachable", error=type(exc).__name__)
            raise DecisionError("Decision provider unreachable", 502) from exc
        except Exception as exc:
            logger.exception("Decision provider request could not be made")
            raise DecisionError("Decision provider request failed", 502) from exc
        if response.status_code == 400:
            raise DecisionError(
                "Decision provider rejected the state or questions",
                422,
                provider_status_code=400,
                provider_detail=_provider_detail(response),
            )
        if not response.is_success:
            raise _provider_error(response.status_code, response)
        try:
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("answers"), list):
                raise ValueError("Missing answer array")
            return {**data, "answers": _answers(data["answers"], questions)}
        except (ValueError, KeyError, TypeError) as exc:
            raise DecisionError("Decision provider returned an unexpected answer", 502) from exc


def _text(value: Any) -> str:
    """Retain structured instructions and criteria as JSON text, without guessing fields."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _question(name: str, question: dict[str, Any]) -> dict[str, Any]:
    kind = question["type"]
    result = {
        "name": name,
        "type": "predicate" if kind == "noul" else kind,
        "instructions": _text(question.get("instructions")),
    }
    criteria = question.get("criteria")
    if kind == "choice":
        if len(criteria) > 255:
            raise DecisionError("OpenAI Decisions accepts at most 255 choices per question", 422)
        result["choices"] = [
            {"value": key, "description": _text(value)} for key, value in criteria.items()
        ]
    elif kind == "score":
        result["levels"] = [
            {"label": str(index), "description": _text(value)}
            for index, value in enumerate(criteria)
        ]
    elif criteria:
        result["instructions"] += "\n\nTrue/false criteria: " + _text(criteria)
    return result


def _probabilities(values: Any, expected: set[str] | set[int]) -> dict[str, Any]:
    """Preserve each provider probability once under the supplied option or level id."""
    if not isinstance(values, list):
        raise ValueError("Missing probabilities")
    result = {}
    for entry in values:
        if not isinstance(entry, dict):
            raise ValueError("Invalid probability entry")
        value = entry["value"]
        if type(value) not in (str, int) or value not in expected or str(value) in result:
            raise ValueError("Unexpected or duplicate option")
        result[str(value)] = entry["probability"]
    if set(result) != {str(value) for value in expected}:
        raise ValueError("Incomplete probabilities")
    return result


def _answers(raw: list[Any], questions: dict[str, Any]) -> dict[str, Any]:
    answers = {}
    for answer in raw:
        if not isinstance(answer, dict) or not isinstance(answer.get("name"), str):
            raise ValueError("Missing answer name")
        name = answer["name"]
        if name not in questions:
            continue
        if name in answers:
            raise ValueError("Duplicate answer")
        if answer["type"] == "refusal":
            raise DecisionError("Decision provider declined to answer a question", 502)
        question = questions[name]
        kind = question["type"]
        if answer["type"] != ("predicate" if kind == "noul" else kind):
            raise ValueError("Wrong answer type")
        if kind == "noul":
            answers[name] = {"type": kind, "noul": answer["probability"]}
            continue
        criteria = question["criteria"]
        expected = set(criteria) if kind == "choice" else set(range(len(criteria)))
        result = {
            "type": kind,
            "confidence": answer["confidence"],
            "probabilities": _probabilities(answer["probabilities"], expected),
        }
        if kind == "choice":
            if not isinstance(answer["choice"], str) or answer["choice"] not in criteria:
                raise ValueError("Unknown choice")
            result["choice"] = answer["choice"]
        else:
            result["score"] = answer["score"]
            result["legend"] = {str(index): value for index, value in enumerate(criteria)}
        answers[name] = result
    if answers.keys() != questions.keys():
        raise ValueError("Missing answers")
    return answers
