"""Telegram messaging tools -- send messages via the Telegram Bot API."""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

from lovely_assistant.services.tools._http_client import request_json
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition

TELEGRAM_API_BASE = "https://api.telegram.org"
_DEFAULT_CHAT_ID = "897573812"  # Elias's Telegram chat ID


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


async def respond_telegram(message: str) -> dict[str, Any]:
    """Send a message to Elias via the Telegram Bot API.

    Reads TELEGRAM_TOKEN and TELEGRAM_CHAT_ID from environment at call time.
    TELEGRAM_CHAT_ID defaults to Elias's chat ID if not set.

    Returns {"success": True, "message_id": ...} on success,
    or {"success": False, "error": ..., "error_code": ...} on failure.
    """
    if not message or not message.strip():
        return {"error": "Message cannot be empty", "success": False}

    token = os.environ.get("TELEGRAM_TOKEN", "")
    if not token:
        return {
            "success": False,
            "error": "TELEGRAM_TOKEN not configured. Set TELEGRAM_TOKEN in .env",
            "error_code": "TELEGRAM_AUTH_MISSING",
        }

    chat_id = os.environ.get("TELEGRAM_CHAT_ID", _DEFAULT_CHAT_ID)
    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message.strip()}
    status, data = await request_json(
        "POST",
        url,
        json_body=payload,
        timeout=30.0,
        retry_name="telegram_send",
        timeout_error="Request timed out sending Telegram message",
        timeout_error_code="TELEGRAM_TIMEOUT",
        http_error_code="TELEGRAM_HTTP_ERROR",
        client_factory=httpx.AsyncClient,
        request_executor=lambda client: client.post(url, json=payload),
    )

    if status == -1:
        return data

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
    return {
        "success": True,
        "message_id": result.get("message_id"),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_telegram_tools(registry: ToolRegistry) -> None:
    """Register Telegram messaging tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="respond_telegram",
            description=(
                "Send a message to Elias via Telegram. "
                "Use this when responding to messages that arrived via Telegram "
                "(indicated by [via:telegram] envelope tag). "
                "Compose your full response as the message text."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message text to send via Telegram",
                    },
                },
                "required": ["message"],
            },
            category=ToolCategory.BACKEND,
        ),
        respond_telegram,
    )

    logger.info("Registered Telegram messaging tools", count=1)
