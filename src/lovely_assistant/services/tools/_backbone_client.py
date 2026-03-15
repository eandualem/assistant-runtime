"""Shared HTTP client for the agent-backbone REST API."""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

from lovely_assistant.services.tools._http_client import request_json


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
    backbone_url = os.environ.get("BACKBONE_URL", "http://127.0.0.1:7120")
    backbone_api_key = os.environ.get("BACKBONE_API_KEY", "")

    headers: dict[str, str] = {"Accept": "application/json"}
    if backbone_api_key:
        headers["Authorization"] = f"Bearer {backbone_api_key}"

    status, data = await request_json(
        method,
        f"{backbone_url}{path}",
        headers=headers,
        json_body=json_body,
        params=params,
        timeout=30.0,
        retry_name="backbone_request",
        timeout_error=f"Request timed out: {method} {path}",
        timeout_error_code="BACKBONE_TIMEOUT",
        http_error_code="BACKBONE_HTTP_ERROR",
        client_factory=httpx.AsyncClient,
    )

    if status == -1 and isinstance(data, dict):
        if data.get("error_code") == "BACKBONE_TIMEOUT":
            logger.warning("Backbone request timed out after retries", method=method, path=path)
        elif data.get("error_code") == "BACKBONE_HTTP_ERROR":
            logger.warning(
                "Backbone request failed after retries",
                method=method,
                path=path,
                error=data.get("error", "unknown"),
            )

    return status, data


def backbone_error(payload: dict[str, Any]) -> str:
    """Extract normalized error text from a backbone transport payload."""
    return payload.get("error", payload.get("message", "Request failed"))
