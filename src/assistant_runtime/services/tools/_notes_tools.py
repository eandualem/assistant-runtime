"""Notes management tool -- CRUD operations on a markdown notes folder.

Supports flat notes and subdirectory organization (folders).
"""

from __future__ import annotations

import asyncio
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition

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


def _is_safe_path(target: Path) -> bool:
    """Check that *target* resolves within NOTES_DIR (traversal guard)."""
    try:
        return target.resolve().is_relative_to(NOTES_DIR.resolve())
    except (ValueError, OSError):
        return False


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


def _validate_note_path(note_path: str) -> str | None:
    """Validate a note path that may include subdirectories.

    Accepts ``folder/subfolder/note-name.md`` style paths.
    Returns error message or None if valid.
    """
    if not note_path:
        return "Note path cannot be empty"
    if "\\" in note_path or ".." in note_path:
        return "Invalid note path — path traversal not allowed"

    # Each segment must be safe
    parts = Path(note_path).parts
    if not parts:
        return "Note path cannot be empty"

    # Last part must be a valid .md filename
    basename = parts[-1]
    if not FILENAME_PATTERN.match(basename):
        return "Invalid filename — must match pattern: alphanumeric, hyphens, underscores, ending in .md"

    # Check the resolved path stays inside NOTES_DIR
    target = NOTES_DIR / note_path
    if not _is_safe_path(target):
        return "Invalid note path — resolves outside notes directory"

    return None


def _validate_folder(folder: str) -> str | None:
    """Validate a folder path. Returns error message or None if valid."""
    if not folder:
        return "Folder path cannot be empty"
    if "\\" in folder or ".." in folder:
        return "Invalid folder — path traversal not allowed"

    target = NOTES_DIR / folder
    if not _is_safe_path(target):
        return "Invalid folder — resolves outside notes directory"

    return None


def _relative_path(path: Path) -> str:
    """Return the path of a note relative to NOTES_DIR."""
    try:
        return str(path.resolve().relative_to(NOTES_DIR.resolve()))
    except ValueError:
        return path.name


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

    # Compute folder relative to NOTES_DIR
    rel = _relative_path(path)
    folder = str(Path(rel).parent)
    if folder == ".":
        folder = ""

    return {
        "filename": path.name,
        "path": rel,
        "folder": folder,
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
# Tool handlers
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
    folder: str = "",
) -> dict[str, Any]:
    """Manage notes in ~/notes/. Supports create, list, read, search, update, delete, create_folder, move_note."""

    actions = {
        "create": _create_note,
        "list": _list_notes,
        "read": _read_note,
        "search": _search_notes,
        "update": _update_note,
        "delete": _delete_note,
        "create_folder": _create_folder,
        "move_note": _move_note,
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
        folder=folder,
    )


async def _create_note(
    title: str, content: str, tags: list[str] | None = None, folder: str = "", **_kwargs: Any
) -> dict[str, Any]:
    if not title:
        return {"error": "Title is required for create", "success": False}
    if not content:
        return {"error": "Content is required for create", "success": False}

    notes_dir = _ensure_notes_dir()

    # Determine target directory
    target_dir = notes_dir
    if folder:
        error = _validate_folder(folder)
        if error:
            return {"error": error, "success": False}
        target_dir = notes_dir / folder
        target_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(title)
    if not slug:
        return {"error": "Title produces empty slug", "success": False}

    filename = f"{date.today()}-{slug}.md"
    path = target_dir / filename

    # Avoid overwriting
    if path.exists():
        counter = 1
        while path.exists():
            filename = f"{date.today()}-{slug}-{counter}.md"
            path = target_dir / filename
            counter += 1

    note_content = _build_note_content(title, content, tags)
    await asyncio.to_thread(path.write_text, note_content, "utf-8")

    rel_path = _relative_path(path)
    logger.info("Created note", path=rel_path, title=title)
    return {"filename": filename, "path": rel_path, "title": title, "success": True}


async def _list_notes(
    tag: str = "", limit: int = 20, folder: str = "", **_kwargs: Any
) -> dict[str, Any]:
    notes_dir = _ensure_notes_dir()

    # Search scope
    search_dir = notes_dir
    if folder:
        error = _validate_folder(folder)
        if error:
            return {"error": error, "success": False}
        search_dir = notes_dir / folder
        if not search_dir.is_dir():
            return {"notes": [], "count": 0, "success": True}

    md_files = sorted(search_dir.rglob("*.md"), reverse=True)
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
                "path": parsed["path"],
                "folder": parsed["folder"],
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
    # Support both bare filenames and paths with folders
    if "/" in filename:
        error = _validate_note_path(filename)
        if error:
            return {"error": error, "success": False}
        path = NOTES_DIR / filename
    else:
        error = _validate_filename(filename)
        if error:
            return {"error": error, "success": False}
        path = NOTES_DIR / filename

    parsed = await asyncio.to_thread(_parse_note, path)
    if parsed is None:
        return {"error": f"Note not found: {filename}", "success": False}

    return {**parsed, "success": True}


async def _search_notes(
    query: str = "", limit: int = 20, folder: str = "", **_kwargs: Any
) -> dict[str, Any]:
    if not query:
        return {"error": "Query is required for search", "success": False}

    notes_dir = _ensure_notes_dir()
    query_lower = query.lower()
    results = []

    search_dir = notes_dir
    if folder:
        error = _validate_folder(folder)
        if error:
            return {"error": error, "success": False}
        search_dir = notes_dir / folder
        if not search_dir.is_dir():
            return {"results": [], "count": 0, "query": query, "success": True}

    md_files = sorted(search_dir.rglob("*.md"), reverse=True)
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
                "path": parsed["path"],
                "folder": parsed["folder"],
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
    # Support both bare filenames and paths with folders
    if "/" in filename:
        error = _validate_note_path(filename)
        if error:
            return {"error": error, "success": False}
        path = NOTES_DIR / filename
    else:
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
    return {
        "filename": parsed["filename"],
        "path": _relative_path(path),
        "title": parsed["title"],
        "updated": True,
        "success": True,
    }


async def _delete_note(filename: str = "", **_kwargs: Any) -> dict[str, Any]:
    # Support both bare filenames and paths with folders
    if "/" in filename:
        error = _validate_note_path(filename)
        if error:
            return {"error": error, "success": False}
        path = NOTES_DIR / filename
    else:
        error = _validate_filename(filename)
        if error:
            return {"error": error, "success": False}
        path = NOTES_DIR / filename

    if not path.exists():
        return {"error": f"Note not found: {filename}", "success": False}

    await asyncio.to_thread(path.unlink)

    logger.info("Deleted note", filename=filename)
    return {"filename": path.name, "path": _relative_path(path), "deleted": True, "success": True}


async def _create_folder(folder: str = "", **_kwargs: Any) -> dict[str, Any]:
    if not folder:
        return {"error": "Folder path is required for create_folder", "success": False}

    error = _validate_folder(folder)
    if error:
        return {"error": error, "success": False}

    notes_dir = _ensure_notes_dir()
    target = notes_dir / folder

    if target.exists():
        return {
            "folder": folder,
            "created": False,
            "message": "Folder already exists",
            "success": True,
        }

    await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)

    logger.info("Created notes folder", folder=folder)
    return {"folder": folder, "created": True, "success": True}


async def _move_note(filename: str = "", folder: str = "", **_kwargs: Any) -> dict[str, Any]:
    if not filename:
        return {"error": "Filename is required for move_note", "success": False}
    if not folder:
        return {"error": "Destination folder is required for move_note", "success": False}

    # Validate source — can be bare filename or path
    if "/" in filename:
        error = _validate_note_path(filename)
        if error:
            return {"error": error, "success": False}
        src = NOTES_DIR / filename
    else:
        error = _validate_filename(filename)
        if error:
            return {"error": error, "success": False}
        src = NOTES_DIR / filename

    if not src.exists():
        return {"error": f"Note not found: {filename}", "success": False}

    # Validate destination folder
    error = _validate_folder(folder)
    if error:
        return {"error": error, "success": False}

    notes_dir = _ensure_notes_dir()
    dest_dir = notes_dir / folder
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / src.name
    if dest.exists():
        return {
            "error": f"A note named '{src.name}' already exists in '{folder}'",
            "success": False,
        }

    await asyncio.to_thread(shutil.move, str(src), str(dest))

    new_path = _relative_path(dest)
    logger.info("Moved note", source=filename, destination=new_path)
    return {
        "filename": src.name,
        "from": _relative_path(src) if src.exists() else filename,
        "to": new_path,
        "success": True,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_notes_tools(registry: ToolRegistry) -> None:
    """Register the notes management tool."""
    registry.register_backend_tool(
        ToolDefinition(
            name="manage_notes",
            description=(
                "Manage the user's personal notes. Supports create, list, read, search, "
                "update, delete, create_folder, and move_note operations on markdown notes "
                "with YAML frontmatter. Notes are stored as files in ~/notes/ and can be "
                "organized into subdirectories (folders)."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "create",
                            "list",
                            "read",
                            "search",
                            "update",
                            "delete",
                            "create_folder",
                            "move_note",
                        ],
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
                        "description": (
                            "Note filename or path. Bare filename (e.g., 'my-note.md') for root notes. "
                            "Path with folder (e.g., 'governance/tracks/my-note.md') for notes in subdirectories. "
                            "Required for read/update/delete/move_note."
                        ),
                    },
                    "folder": {
                        "type": "string",
                        "description": (
                            "Folder path relative to ~/notes/. "
                            "For create: target folder for the new note (e.g., 'governance/tracks'). "
                            "For create_folder: the folder to create. "
                            "For move_note: destination folder. "
                            "For list/search: scope results to this folder and its subfolders."
                        ),
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
