"""Notes: a store of markdown notes the assistant keeps for the user.

A provider implements ``NotesStore``; every method returns the tool's
result dict (``{"success": False, "error": ...}`` on failure) so the
capability adds nothing but the tool schema and the action dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from loguru import logger

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


class NotesStore(Protocol):
    """What a notes provider offers. Paths are relative to the store's root."""

    async def create(
        self, *, title: str, content: str, tags: list[str] | None, folder: str
    ) -> dict[str, Any]: ...

    async def list(self, *, tag: str, limit: int, folder: str) -> dict[str, Any]: ...

    async def read(self, *, filename: str) -> dict[str, Any]: ...

    async def search(self, *, query: str, limit: int, folder: str) -> dict[str, Any]: ...

    async def update(
        self, *, filename: str, content: str, tags: list[str] | None
    ) -> dict[str, Any]: ...

    async def delete(self, *, filename: str) -> dict[str, Any]: ...

    async def create_folder(self, *, folder: str) -> dict[str, Any]: ...

    async def move(self, *, filename: str, folder: str) -> dict[str, Any]: ...


def build_manage_notes(store: NotesStore) -> Callable[..., Any]:
    """The ``manage_notes`` handler bound to a store."""

    async def manage_notes(
        action: str,
        title: str = "",
        content: str = "",
        tags: list[str] | None = None,
        filename: str = "",
        tag: str = "",
        query: str = "",
        limit: int = 20,
        folder: str = "",
    ) -> dict[str, Any]:
        """Manage notes: create, list, read, search, update, delete, create_folder, move_note."""
        match action:
            case "create":
                return await store.create(title=title, content=content, tags=tags, folder=folder)
            case "list":
                return await store.list(tag=tag, limit=limit, folder=folder)
            case "read":
                return await store.read(filename=filename)
            case "search":
                return await store.search(query=query, limit=limit, folder=folder)
            case "update":
                return await store.update(filename=filename, content=content, tags=tags)
            case "delete":
                return await store.delete(filename=filename)
            case "create_folder":
                return await store.create_folder(folder=folder)
            case "move_note":
                return await store.move(filename=filename, folder=folder)
        return {
            "error": f"Unknown action '{action}'. Valid actions: {', '.join(_ACTIONS)}",
            "success": False,
        }

    return manage_notes


_ACTIONS = ("create", "list", "read", "search", "update", "delete", "create_folder", "move_note")


def register_notes_tools(registry: ToolRegistry, store: NotesStore) -> None:
    """Register ``manage_notes`` against a store."""
    registry.register_backend_tool(
        ToolDefinition(
            name="manage_notes",
            description=(
                "Manage the user's notes: create, list, read, search, update, delete, "
                "create_folder and move_note. Notes are markdown with a title, a date and "
                "tags, and can be organised into folders."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": list(_ACTIONS),
                        "description": "The action to perform",
                    },
                    "title": {"type": "string", "description": "Note title (required for create)"},
                    "content": {
                        "type": "string",
                        "description": "Note content (required for create, optional for update)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization (optional for create/update)",
                    },
                    "filename": {
                        "type": "string",
                        "description": (
                            "Note filename or path: 'my-note.md' for a root note, "
                            "'projects/alpha/my-note.md' for one in a folder. "
                            "Required for read/update/delete/move_note."
                        ),
                    },
                    "folder": {
                        "type": "string",
                        "description": (
                            "Folder path relative to the notes root. For create: where the "
                            "note goes. For create_folder: the folder to create. For move_note: "
                            "the destination. For list/search: scope to this folder."
                        ),
                    },
                    "tag": {"type": "string", "description": "Filter by tag (optional for list)"},
                    "query": {
                        "type": "string",
                        "description": "Search query (required for search)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 20)",
                        "default": 20,
                    },
                },
                "required": ["action"],
            },
            category=ToolCategory.BACKEND,
        ),
        build_manage_notes(store),
    )
    logger.info("Registered notes capability")
