"""Shared inbox injection helper for routes and background services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lovely_assistant.services.database.repositories import InboxRepository

if TYPE_CHECKING:
    from lovely_assistant.app.assistant.interface import AssistantService
    from lovely_assistant.services.database.interface import DatabaseService


async def inject_inbox_message(
    *,
    db: DatabaseService,
    assistant_service: AssistantService | None,
    from_agent: str,
    via: str,
    message: str,
    session_id: str | None = None,
    telegram_chat_id: str | None = None,
) -> dict[str, Any]:
    """Store an injected message using the same delivery semantics as `/assistant/inject`."""
    target_session_id = session_id
    session_found = False

    if assistant_service is not None:
        sessions = assistant_service.get_session_store()
        if sessions is not None:
            if target_session_id:
                ctx = await sessions.get_context_if_exists_async(target_session_id)
                session_found = ctx is not None
            elif via == "telegram" and telegram_chat_id:
                target_session_id = await sessions.get_session_id_for_telegram_chat_async(
                    telegram_chat_id
                )
                session_found = target_session_id is not None

    context: dict[str, Any] = {"via": via, "injected": True}
    if telegram_chat_id:
        context["telegram_chat_id"] = telegram_chat_id
    if target_session_id and session_found:
        context["session_id"] = target_session_id

    async with db.session_context() as session:
        repo = InboxRepository(session)
        row = await repo.create(
            from_agent=from_agent,
            message=message,
            severity="info",
            context=context,
        )

    response: dict[str, Any] = {
        "status": "delivered" if session_found else "deferred",
        "inbox_id": row.id,
    }
    if target_session_id and session_found:
        response["session_id"] = target_session_id
    return response
