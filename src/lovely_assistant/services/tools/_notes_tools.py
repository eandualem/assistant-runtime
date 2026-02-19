"""Notes management tool -- CRUD operations on a markdown notes folder."""

from __future__ import annotations

import asyncio
import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition

NOTES_DIR = Path.home() / "notes"
FILENAME_PATTERN = re.compile(r"^[\w\-]+\.md$")
SLUG_INVALID_CHARS = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_notes_dir() -> Path:
    """Ensure the notes directory exists."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    return NOTES_DIR


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    slug = SLUG_INVALID_CHARS.sub("-", text.lower().strip())
    return slug.strip("-")[:80]


def _validate_filename(filename: str) -> str | None:
    """Validate a note filename. Returns error message or None if valid."""
    if not filename:
        return "Filename cannot be empty"
    if not FILENAME_PATTERN.match(filename):
        return "Invalid filename — must match pattern: alphanumeric, hyphens, underscores, ending in .md"
    # Prevent traversal even though the regex already blocks it
    if ".." in filename or "/" in filename:
        return "Invalid filename — path traversal not allowed"
    return None


def _parse_note(path: Path) -> dict[str, Any] | None:
    """Parse a markdown note with YAML frontmatter. Returns None on error."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None

    frontmatter: dict[str, Any] = {}
    body = content

    if content.startswith("---\n"):
        parts = content.split("---\n", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                frontmatter = {}
            body = parts[2].strip()

    return {
        "filename": path.name,
        "title": frontmatter.get("title", path.stem),
        "date": str(frontmatter.get("date", "")),
        "tags": frontmatter.get("tags", []),
        "content": body,
    }


def _build_note_content(title: str, content: str, tags: list[str] | None = None) -> str:
    """Build a markdown note with YAML frontmatter."""
    frontmatter = {
        "title": title,
        "date": str(date.today()),
        "tags": tags or [],
    }
    yaml_block = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).strip()
    return f"---\n{yaml_block}\n---\n\n{content}\n"


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


async def manage_notes(
    action: str,
    title: str = "",
    content: str = "",
    tags: list[str] | None = None,
    filename: str = "",
    tag: str = "",
    query: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """Manage notes in ~/notes/. Supports create, list, read, search, update, delete."""

    actions = {
        "create": _create_note,
        "list": _list_notes,
        "read": _read_note,
        "search": _search_notes,
        "update": _update_note,
        "delete": _delete_note,
    }

    if action not in actions:
        return {
            "error": f"Unknown action '{action}'. Valid actions: {', '.join(actions)}",
            "success": False,
        }

    return await actions[action](
        title=title,
        content=content,
        tags=tags,
        filename=filename,
        tag=tag,
        query=query,
        limit=limit,
    )


async def _create_note(
    title: str, content: str, tags: list[str] | None = None, **_kwargs: Any
) -> dict[str, Any]:
    if not title:
        return {"error": "Title is required for create", "success": False}
    if not content:
        return {"error": "Content is required for create", "success": False}

    notes_dir = _ensure_notes_dir()
    slug = _slugify(title)
    if not slug:
        return {"error": "Title produces empty slug", "success": False}

    filename = f"{date.today()}-{slug}.md"
    path = notes_dir / filename

    # Avoid overwriting
    if path.exists():
        counter = 1
        while path.exists():
            filename = f"{date.today()}-{slug}-{counter}.md"
            path = notes_dir / filename
            counter += 1

    note_content = _build_note_content(title, content, tags)
    await asyncio.to_thread(path.write_text, note_content, "utf-8")

    logger.info("Created note", filename=filename, title=title)
    return {"filename": filename, "title": title, "success": True}


async def _list_notes(tag: str = "", limit: int = 20, **_kwargs: Any) -> dict[str, Any]:
    notes_dir = _ensure_notes_dir()

    md_files = sorted(notes_dir.glob("*.md"), reverse=True)
    notes = []
    for path in md_files:
        parsed = await asyncio.to_thread(_parse_note, path)
        if parsed is None:
            continue
        if tag and tag not in parsed["tags"]:
            continue
        # Summary: first line of content
        first_line = parsed["content"].split("\n", 1)[0][:200] if parsed["content"] else ""
        notes.append(
            {
                "filename": parsed["filename"],
                "title": parsed["title"],
                "date": parsed["date"],
                "tags": parsed["tags"],
                "preview": first_line,
            }
        )
        if len(notes) >= limit:
            break

    return {"notes": notes, "count": len(notes), "success": True}


async def _read_note(filename: str = "", **_kwargs: Any) -> dict[str, Any]:
    error = _validate_filename(filename)
    if error:
        return {"error": error, "success": False}

    path = NOTES_DIR / filename
    parsed = await asyncio.to_thread(_parse_note, path)
    if parsed is None:
        return {"error": f"Note not found: {filename}", "success": False}

    return {**parsed, "success": True}


async def _search_notes(query: str = "", limit: int = 20, **_kwargs: Any) -> dict[str, Any]:
    if not query:
        return {"error": "Query is required for search", "success": False}

    notes_dir = _ensure_notes_dir()
    query_lower = query.lower()
    results = []

    md_files = sorted(notes_dir.glob("*.md"), reverse=True)
    for path in md_files:
        parsed = await asyncio.to_thread(_parse_note, path)
        if parsed is None:
            continue

        # Search in title, content, and tags
        searchable = f"{parsed['title']} {parsed['content']} {' '.join(parsed['tags'])}".lower()
        if query_lower not in searchable:
            continue

        # Extract context snippet around the match
        content_lower = parsed["content"].lower()
        idx = content_lower.find(query_lower)
        snippet = ""
        if idx >= 0:
            start = max(0, idx - 50)
            end = min(len(parsed["content"]), idx + len(query) + 50)
            snippet = parsed["content"][start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(parsed["content"]):
                snippet = snippet + "..."
        else:
            # Match was in title or tags, show first line
            snippet = parsed["content"].split("\n", 1)[0][:200] if parsed["content"] else ""

        results.append(
            {
                "filename": parsed["filename"],
                "title": parsed["title"],
                "date": parsed["date"],
                "tags": parsed["tags"],
                "snippet": snippet,
            }
        )
        if len(results) >= limit:
            break

    return {"results": results, "count": len(results), "query": query, "success": True}


async def _update_note(
    filename: str = "",
    content: str = "",
    tags: list[str] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    error = _validate_filename(filename)
    if error:
        return {"error": error, "success": False}

    path = NOTES_DIR / filename
    parsed = await asyncio.to_thread(_parse_note, path)
    if parsed is None:
        return {"error": f"Note not found: {filename}", "success": False}

    if not content and tags is None:
        return {"error": "Provide content or tags to update", "success": False}

    new_content = content if content else parsed["content"]
    new_tags = tags if tags is not None else parsed["tags"]
    note_content = _build_note_content(parsed["title"], new_content, new_tags)
    await asyncio.to_thread(path.write_text, note_content, "utf-8")

    logger.info("Updated note", filename=filename)
    return {"filename": filename, "title": parsed["title"], "updated": True, "success": True}


async def _delete_note(filename: str = "", **_kwargs: Any) -> dict[str, Any]:
    error = _validate_filename(filename)
    if error:
        return {"error": error, "success": False}

    path = NOTES_DIR / filename
    if not path.exists():
        return {"error": f"Note not found: {filename}", "success": False}

    await asyncio.to_thread(path.unlink)

    logger.info("Deleted note", filename=filename)
    return {"filename": filename, "deleted": True, "success": True}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_notes_tools(registry: ToolRegistry) -> None:
    """Register the notes management tool."""
    registry.register_backend_tool(
        ToolDefinition(
            name="manage_notes",
            description=(
                "Manage Elias's personal notes. Supports create, list, read, search, "
                "update, and delete operations on markdown notes with YAML frontmatter. "
                "Notes are stored as files in ~/notes/."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "read", "search", "update", "delete"],
                        "description": "The action to perform",
                    },
                    "title": {
                        "type": "string",
                        "description": "Note title (required for create)",
                    },
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
                        "description": "Note filename (required for read/update/delete)",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag (optional for list)",
                    },
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
        manage_notes,
    )

    logger.info("Registered notes management tool")
