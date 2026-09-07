"""Library: reference documents the assistant can consult, read-only.

Documents are named entries in named collections (a directory of skills,
a team handbook, a repository's docs). A provider implements
``DocumentLibrary``.
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


class DocumentLibrary(Protocol):
    """What a library provider offers."""

    async def list_documents(self, *, collection: str | None) -> dict[str, Any]:
        """``{"success": True, "documents": [{"name", "description", "collection"}]}``."""
        ...

    async def read_document(self, *, name: str, collection: str | None) -> dict[str, Any]:
        """``{"success": True, "name", "collection", "content"}`` or an error dict."""
        ...


def register_library_tools(registry: ToolRegistry, library: DocumentLibrary) -> None:
    """Register ``list_documents`` and ``read_document`` against a library."""

    async def list_documents(collection: str = "") -> dict[str, Any]:
        """List the reference documents, optionally in one collection."""
        return await library.list_documents(collection=collection or None)

    async def read_document(name: str, collection: str = "") -> dict[str, Any]:
        """Read one reference document by name."""
        if not name:
            return {"success": False, "error": "Document name is required"}
        return await library.read_document(name=name, collection=collection or None)

    registry.register_backend_tool(
        ToolDefinition(
            name="list_documents",
            description=(
                "List the reference documents available to you (guides, procedures, "
                "skills), with each one's name, description and collection. "
                "Use 'collection' to list one collection only."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Optional collection name to filter by",
                    }
                },
            },
            category=ToolCategory.BACKEND,
        ),
        list_documents,
    )
    registry.register_backend_tool(
        ToolDefinition(
            name="read_document",
            description=(
                "Read the full content of a reference document by name. Give 'collection' "
                "when the same name exists in several collections."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Document name"},
                    "collection": {
                        "type": "string",
                        "description": "Optional collection the document belongs to",
                    },
                },
                "required": ["name"],
            },
            category=ToolCategory.BACKEND,
        ),
        read_document,
    )
    logger.info("Registered library capability")
