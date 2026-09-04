"""Messaging: reach the person the assistant works for on their messaging channel.

A provider implements ``MessagingProvider``; each method returns the tool's result
dict (``{"success": False, "error": ...}`` on failure).
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


class MessagingProvider(Protocol):
    """What a provider of the messaging capability implements."""

    async def respond_telegram(self, message: str) -> dict[str, Any]: ...


def register_messaging_tools(registry: ToolRegistry, provider: MessagingProvider) -> None:
    """Register Telegram messaging tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="respond_telegram",
            description=(
                "Send a message to the operator via Telegram. "
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
        provider.respond_telegram,
    )

    logger.info("Registered Telegram messaging tools", count=1)
