"""Shared async JSON HTTP client helpers for tool modules."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

import httpx

from lovely_assistant.base.resilience import retry_with_backoff

RETRYABLE_HTTP_ERRORS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    ConnectionError,
    TimeoutError,
)

AsyncClientFactory: TypeAlias = Callable[..., Any]
RequestExecutor: TypeAlias = Callable[[Any], Awaitable[httpx.Response]]


async def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
    timeout: float = 30.0,
    retry_name: str = "http_request",
    timeout_error: str | None = None,
    timeout_error_code: str = "HTTP_TIMEOUT",
    http_error_code: str = "HTTP_ERROR",
    client_factory: AsyncClientFactory = httpx.AsyncClient,
    request_executor: RequestExecutor | None = None,
) -> tuple[int, Any]:
    """Make a retrying JSON request and normalize transport failures."""

    @retry_with_backoff(
        max_attempts=3,
        min_wait=0.5,
        max_wait=10.0,
        retry_on=RETRYABLE_HTTP_ERRORS,
        name=retry_name,
    )
    async def _request() -> tuple[int, Any]:
        async with client_factory(timeout=timeout) as client:
            response = (
                await request_executor(client)
                if request_executor is not None
                else await client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    params=params,
                )
            )
            return response.status_code, response.json()

    try:
        return await _request()
    except (httpx.TimeoutException, TimeoutError):
        return (
            -1,
            {
                "success": False,
                "error": timeout_error or f"Request timed out: {method} {url}",
                "error_code": timeout_error_code,
            },
        )
    except (httpx.HTTPStatusError, httpx.HTTPError, ConnectionError) as exc:
        return (
            -1,
            {
                "success": False,
                "error": f"HTTP error: {exc}",
                "error_code": http_error_code,
            },
        )
