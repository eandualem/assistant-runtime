"""Shared HTTP client for the agent-backbone REST API."""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

BACKBONE_URL = os.environ.get("BACKBONE_URL", "http://127.0.0.1:9877")
BACKBONE_API_KEY = os.environ.get("BACKBONE_API_KEY", "")


async def backbone_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
) -> tuple[int, Any]:
    """Make a backbone API request. Returns (status_code, parsed_json_body).

    Returns (-1, error_dict) on network error or timeout.
    """
    headers: dict[str, str] = {"Accept": "application/json"}
    if BACKBONE_API_KEY:
        headers["Authorization"] = f"Bearer {BACKBONE_API_KEY}"

    url = f"{BACKBONE_URL}{path}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=json_body,
                params=params,
            )
            return (response.status_code, response.json())
    except httpx.TimeoutException:
        logger.warning("Backbone request timed out", method=method, path=path)
        return (-1, {"message": f"Request timed out: {method} {path}"})
    except httpx.HTTPError as exc:
        logger.warning("Backbone request failed", method=method, path=path, error=str(exc))
        return (-1, {"message": f"HTTP error: {exc}"})
