"""TypeSafe's System One endpoint as a decision provider.

One request carries every question; the model evaluates them in parallel and
answers in one response (https://docs.typesafe.ai/api). The client is shared
for the service's lifetime so each call reuses the TLS connection.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from assistant_runtime.services.decisions.exceptions import DecisionError

SYSTEM_ONE_PATH = "/v1/systemone"


class TypeSafeDecisions:
    """Implements ``DecisionProvider`` against ``POST {base_url}/v1/systemone``."""

    name = "typesafe"

    def __init__(
        self, *, base_url: str, model: str, timeout_seconds: float, client: httpx.AsyncClient
    ) -> None:
        self._url = base_url.rstrip("/") + SYSTEM_ONE_PATH
        self._model = model
        self._timeout = timeout_seconds
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    async def decide(
        self, *, api_key: str, state: Any, questions: dict[str, Any]
    ) -> dict[str, Any]:
        body = {"model": self._model, "state": state, "questions": questions}
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
            # A closed client or an unusable base URL: still the documented contract.
            logger.exception("Decision provider request could not be made")
            raise DecisionError("Decision provider request failed", 502) from exc
        if not response.is_success:
            raise _provider_error(response.status_code, response)
        try:
            data = response.json()
        except ValueError as exc:
            raise DecisionError("Decision provider returned an unexpected answer", 502) from exc
        if not isinstance(data, dict):
            raise DecisionError("Decision provider returned an unexpected answer", 502)
        return data


PROVIDER_DETAIL_LIMIT = 500


def _provider_error(status: int, response: httpx.Response) -> DecisionError:
    """The provider's error status as the runtime's, with its bounded reason."""
    detail = _provider_detail(response)
    logger.warning(
        "Decision provider request failed", provider_status_code=status, provider_detail=detail
    )
    if status in (401, 403):
        message, runtime_status = "Decision provider rejected the runtime's key", 502
    elif status == 422:
        message, runtime_status = "Decision provider rejected the state or questions", 422
    elif status == 429:
        message, runtime_status = "Decision provider rate limit reached", 429
    else:
        message, runtime_status = "Decision provider request failed", 502
    # The caller gets the reason only when it is about its own request; what the
    # provider says about the runtime's key or its own failures stays in the log.
    return DecisionError(
        message,
        runtime_status,
        provider_status_code=status,
        provider_detail=detail if status in (422, 429) else None,
    )


def _provider_detail(response: httpx.Response) -> str | None:
    """The reason string from a provider error body (``detail``, ``error`` or ``message``), bounded."""
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    for key in ("detail", "error", "message"):
        value = body.get(key)
        if isinstance(value, dict):
            value = value.get("message")
        if isinstance(value, str) and value.strip():
            return value.strip()[:PROVIDER_DETAIL_LIMIT]
    return None
