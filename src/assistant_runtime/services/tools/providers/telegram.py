"""Telegram messaging tools -- send messages via the Telegram Bot API."""

from __future__ import annotations

import os
from typing import Any

import httpx

from assistant_runtime.services.tools.request_context import (
    get_current_assistant_session_id,
    record_current_telegram_chat_binding,
)

TELEGRAM_API_BASE = "https://api.telegram.org"


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


async def respond_telegram(message: str) -> dict[str, Any]:
    """Send a message to the operator via the Telegram Bot API.

    Reads TELEGRAM_TOKEN and TELEGRAM_CHAT_ID from environment at call time.
    Both must be set; there is no default chat.

    Returns {"success": True, "message_id": ...} on success,
    or {"success": False, "error": ..., "error_code": ...} on failure.
    """
    return await TelegramMessaging().respond_telegram(message=message)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TelegramMessaging:
    """The messaging capability served by this provider (see ``capabilities.messaging``)."""

    def __init__(self, *, token: str | None = None, chat_id: str | None = None) -> None:
        self._token = token
        self._chat_id = chat_id

    async def respond_telegram(self, message: str) -> dict[str, Any]:
        """Send with this instance's token and chat; omitted values use the environment."""
        if not message or not message.strip():
            return {"error": "Message cannot be empty", "success": False}

        token = os.environ.get("TELEGRAM_TOKEN", "") if self._token is None else self._token
        if not token:
            return {
                "success": False,
                "error": "TELEGRAM_TOKEN not configured. Set TELEGRAM_TOKEN in .env",
                "error_code": "TELEGRAM_AUTH_MISSING",
            }

        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "") if self._chat_id is None else self._chat_id
        if not chat_id:
            return {
                "success": False,
                "error": "TELEGRAM_CHAT_ID not configured. Set TELEGRAM_CHAT_ID in .env",
                "error_code": "TELEGRAM_CHAT_MISSING",
            }
        url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"

        # Sent once: a send that timed out may have been delivered, and a retry
        # would deliver the message twice.
        async def _send() -> tuple[int, Any]:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    json={"chat_id": chat_id, "text": message.strip()},
                )
                return (response.status_code, response.json())

        try:
            status, data = await _send()
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "Request timed out sending Telegram message",
                "error_code": "TELEGRAM_TIMEOUT",
            }
        except httpx.HTTPError as exc:
            return {
                "success": False,
                "error": f"HTTP error: {exc}",
                "error_code": "TELEGRAM_HTTP_ERROR",
            }

        if status != 200:
            description = (
                data.get("description", "Unknown error") if isinstance(data, dict) else str(data)
            )
            return {
                "success": False,
                "error": f"Telegram API error ({status}): {description}",
                "error_code": "TELEGRAM_API_ERROR",
            }

        result = data.get("result", {}) if isinstance(data, dict) else {}
        reply_session_id = get_current_assistant_session_id()
        record_current_telegram_chat_binding(chat_id)

        payload = {
            "success": True,
            "message_id": result.get("message_id"),
            "chat_id": chat_id,
        }
        if reply_session_id:
            payload["reply_session_id"] = reply_session_id
        return payload
