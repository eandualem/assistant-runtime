"""Shared HTTP client for the agent-backbone REST API."""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

from lovely_assistant.base.resilience import retry_with_backoff

_BACKBONE_RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, ConnectionError, TimeoutError)


async def backbone_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
) -> tuple[int, Any]:
    """Make a backbone API request. Returns (status_code, parsed_json_body).

    Retries up to 3 times on timeouts and connection errors.
    Returns (-1, error_dict) on permanent network error.
    Env vars are read at call time (not import time) so load_dotenv() in lifespan works.
    """
    backbone_url = os.environ.get("BACKBONE_URL", "http://127.0.0.1:9877")
    backbone_api_key = os.environ.get("BACKBONE_API_KEY", "")

    headers: dict[str, str] = {"Accept": "application/json"}
    if backbone_api_key:
        headers["Authorization"] = f"Bearer {backbone_api_key}"

    url = f"{backbone_url}{path}"

    @retry_with_backoff(
        max_attempts=3,
        min_wait=0.5,
        max_wait=10.0,
        retry_on=_BACKBONE_RETRYABLE,
        name="backbone_request",
    )
    async def _request() -> tuple[int, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=json_body,
                params=params,
            )
            return (response.status_code, response.json())

    try:
        return await _request()
    except httpx.TimeoutException:
        logger.warning("Backbone request timed out after retries", method=method, path=path)
        return (
            -1,
            {
                "success": False,
                "error": f"Request timed out: {method} {path}",
                "error_code": "BACKBONE_TIMEOUT",
            },
        )
    except httpx.HTTPError as exc:
        logger.warning(
            "Backbone request failed after retries", method=method, path=path, error=str(exc)
        )
        return (
            -1,
            {
                "success": False,
                "error": f"HTTP error: {exc}",
                "error_code": "BACKBONE_HTTP_ERROR",
            },
        )
