"""Filesystem providers: a folder of markdown notes and directories of documents."""

from __future__ import annotations

import asyncio
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

FILENAME_PATTERN = re.compile(r"^[\w\-]+\.md$")
_SLUG_INVALID_CHARS = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """A URL-safe slug for a title, at most 80 characters."""
    slug = _SLUG_INVALID_CHARS.sub("-", text.lower().strip())
    return slug.strip("-")[:80]


def validate_filename(filename: str) -> str | None:
    """An error message when ``filename`` is not a bare ``*.md`` name, else None."""
    if not filename:
        return "Filename cannot be empty"
    if not FILENAME_PATTERN.match(filename):
        return "Invalid filename — must match pattern: alphanumeric, hyphens, underscores, ending in .md"
    if ".." in filename or "/" in filename:
        return "Invalid filename — path traversal not allowed"
    return None


def build_note_content(title: str, content: str, tags: list[str] | None = None) -> str:
    """A markdown note with YAML frontmatter."""
    frontmatter = {"title": title, "date": str(date.today()), "tags": tags or []}
    yaml_block = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).strip()
    return f"---\n{yaml_block}\n---\n\n{content}\n"


def parse_frontmatter(content: str) -> dict[str, Any]:
    """The YAML frontmatter of a markdown document as a dict (empty when absent or invalid)."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    try:
        parsed = yaml.safe_load(content[3:end].strip())
    except yaml.YAMLError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class MarkdownNotes:
    """Notes as markdown files with YAML frontmatter under one root folder.

    Implements ``capabilities.notes.NotesStore``. Every path a caller gives
    is validated to stay inside the root.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    # --- validation ------------------------------------------------------------

    def _is_inside_root(self, target: Path) -> bool:
        try:
            return target.resolve().is_relative_to(self.root.resolve())
        except (ValueError, OSError):
            return False

    def validate_note_path(self, note_path: str) -> str | None:
        """An error for a ``folder/sub/note.md`` path that is unsafe, else None."""
        if not note_path:
            return "Note path cannot be empty"
        if "\\" in note_path or ".." in note_path:
            return "Invalid note path — path traversal not allowed"
        parts = Path(note_path).parts
        if not parts:
            return "Note path cannot be empty"
        if not FILENAME_PATTERN.match(parts[-1]):
            return "Invalid filename — must match pattern: alphanumeric, hyphens, underscores, ending in .md"
        if not self._is_inside_root(self.root / note_path):
            return "Invalid note path — resolves outside notes directory"
        return None

    def validate_folder(self, folder: str) -> str | None:
        """An error for a folder path that is unsafe, else None."""
        if not folder:
            return "Folder path cannot be empty"
        if "\\" in folder or ".." in folder:
            return "Invalid folder — path traversal not allowed"
        if not self._is_inside_root(self.root / folder):
            return "Invalid folder — resolves outside notes directory"
        return None

    def _resolve(self, filename: str) -> tuple[Path | None, str | None]:
        """``(path, None)`` for a bare filename or folder path, else ``(None, error)``."""
        error = (
            self.validate_note_path(filename) if "/" in filename else validate_filename(filename)
        )
        if error:
            return None, error
        return self.root / filename, None

    # --- files -----------------------------------------------------------------

    def _ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return path.name

    def parse_note(self, path: Path) -> dict[str, Any] | None:
        """A note file as a dict (filename, path, folder, title, date, tags, content), or None."""
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
        rel = self._relative(path)
        folder = str(Path(rel).parent)
        return {
            "filename": path.name,
            "path": rel,
            "folder": "" if folder == "." else folder,
            "title": frontmatter.get("title", path.stem),
            "date": str(frontmatter.get("date", "")),
            "tags": frontmatter.get("tags", []),
            "content": body,
        }

    def _scope(self, folder: str) -> tuple[Path | None, str | None]:
        """The directory to scan for ``folder`` (None when it does not exist), or an error."""
        root = self._ensure_root()
        if not folder:
            return root, None
        error = self.validate_folder(folder)
        if error:
            return None, error
        target = root / folder
        return (target if target.is_dir() else None), None

    # --- NotesStore --------------------------------------------------------------

    async def create(
        self, *, title: str, content: str, tags: list[str] | None, folder: str
    ) -> dict[str, Any]:
        if not title:
            return {"error": "Title is required for create", "success": False}
        if not content:
            return {"error": "Content is required for create", "success": False}
        target_dir = self._ensure_root()
        if folder:
            error = self.validate_folder(folder)
            if error:
                return {"error": error, "success": False}
            target_dir = target_dir / folder
            target_dir.mkdir(parents=True, exist_ok=True)
        slug = slugify(title)
        if not slug:
            return {"error": "Title produces empty slug", "success": False}
        filename = f"{date.today()}-{slug}.md"
        path = target_dir / filename
        counter = 1
        while path.exists():
            filename = f"{date.today()}-{slug}-{counter}.md"
            path = target_dir / filename
            counter += 1
        await asyncio.to_thread(path.write_text, build_note_content(title, content, tags), "utf-8")
        rel = self._relative(path)
        logger.info("Created note", path=rel, title=title)
        return {"filename": filename, "path": rel, "title": title, "success": True}

    async def list(self, *, tag: str, limit: int, folder: str) -> dict[str, Any]:
        scope, error = self._scope(folder)
        if error:
            return {"error": error, "success": False}
        notes: list[dict[str, Any]] = []
        for path in sorted(scope.rglob("*.md"), reverse=True) if scope else []:
            parsed = await asyncio.to_thread(self.parse_note, path)
            if parsed is None or (tag and tag not in parsed["tags"]):
                continue
            preview = parsed["content"].split("\n", 1)[0][:200] if parsed["content"] else ""
            notes.append(
                {k: parsed[k] for k in ("filename", "path", "folder", "title", "date", "tags")}
                | {"preview": preview}
            )
            if len(notes) >= limit:
                break
        return {"notes": notes, "count": len(notes), "success": True}

    async def read(self, *, filename: str) -> dict[str, Any]:
        path, error = self._resolve(filename)
        if error:
            return {"error": error, "success": False}
        parsed = await asyncio.to_thread(self.parse_note, path)
        if parsed is None:
            return {"error": f"Note not found: {filename}", "success": False}
        return {**parsed, "success": True}

    async def search(self, *, query: str, limit: int, folder: str) -> dict[str, Any]:
        if not query:
            return {"error": "Query is required for search", "success": False}
        scope, error = self._scope(folder)
        if error:
            return {"error": error, "success": False}
        query_lower = query.lower()
        results: list[dict[str, Any]] = []
        for path in sorted(scope.rglob("*.md"), reverse=True) if scope else []:
            parsed = await asyncio.to_thread(self.parse_note, path)
            if parsed is None:
                continue
            searchable = f"{parsed['title']} {parsed['content']} {' '.join(parsed['tags'])}".lower()
            if query_lower not in searchable:
                continue
            results.append(
                {k: parsed[k] for k in ("filename", "path", "folder", "title", "date", "tags")}
                | {"snippet": _snippet(parsed["content"], query)}
            )
            if len(results) >= limit:
                break
        return {"results": results, "count": len(results), "query": query, "success": True}

    async def update(
        self, *, filename: str, content: str, tags: list[str] | None
    ) -> dict[str, Any]:
        path, error = self._resolve(filename)
        if error:
            return {"error": error, "success": False}
        parsed = await asyncio.to_thread(self.parse_note, path)
        if parsed is None:
            return {"error": f"Note not found: {filename}", "success": False}
        if not content and tags is None:
            return {"error": "Provide content or tags to update", "success": False}
        new_content = content if content else parsed["content"]
        new_tags = tags if tags is not None else parsed["tags"]
        await asyncio.to_thread(
            path.write_text, build_note_content(parsed["title"], new_content, new_tags), "utf-8"
        )
        logger.info("Updated note", filename=filename)
        return {
            "filename": parsed["filename"],
            "path": self._relative(path),
            "title": parsed["title"],
            "updated": True,
            "success": True,
        }

    async def delete(self, *, filename: str) -> dict[str, Any]:
        path, error = self._resolve(filename)
        if error:
            return {"error": error, "success": False}
        if not path.exists():
            return {"error": f"Note not found: {filename}", "success": False}
        await asyncio.to_thread(path.unlink)
        logger.info("Deleted note", filename=filename)
        return {
            "filename": path.name,
            "path": self._relative(path),
            "deleted": True,
            "success": True,
        }

    async def create_folder(self, *, folder: str) -> dict[str, Any]:
        if not folder:
            return {"error": "Folder path is required for create_folder", "success": False}
        error = self.validate_folder(folder)
        if error:
            return {"error": error, "success": False}
        target = self._ensure_root() / folder
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

    async def move(self, *, filename: str, folder: str) -> dict[str, Any]:
        if not filename:
            return {"error": "Filename is required for move_note", "success": False}
        if not folder:
            return {"error": "Destination folder is required for move_note", "success": False}
        src, error = self._resolve(filename)
        if error:
            return {"error": error, "success": False}
        if not src.exists():
            return {"error": f"Note not found: {filename}", "success": False}
        error = self.validate_folder(folder)
        if error:
            return {"error": error, "success": False}
        dest_dir = self._ensure_root() / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            return {
                "error": f"A note named '{src.name}' already exists in '{folder}'",
                "success": False,
            }
        source_rel = self._relative(src)
        await asyncio.to_thread(shutil.move, str(src), str(dest))
        new_path = self._relative(dest)
        logger.info("Moved note", source=filename, destination=new_path)
        return {"filename": src.name, "from": source_rel, "to": new_path, "success": True}


def _snippet(content: str, query: str) -> str:
    """Text around the first match of ``query`` in ``content``, or the first line."""
    idx = content.lower().find(query.lower())
    if idx < 0:
        return content.split("\n", 1)[0][:200] if content else ""
    start = max(0, idx - 50)
    end = min(len(content), idx + len(query) + 50)
    snippet = content[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return snippet


class FilesystemLibrary:
    """Documents as directories under named roots, each with an index markdown file.

    ``roots`` maps a collection name to a directory; a document is a
    subdirectory holding one of ``index_names`` (``SKILL.md`` first, which is
    the Claude Code skills layout). Implements
    ``capabilities.library.DocumentLibrary``.
    """

    def __init__(
        self,
        roots: dict[str, Path],
        index_names: tuple[str, ...] = ("SKILL.md", "README.md", "index.md"),
    ) -> None:
        self.roots = roots
        self.index_names = index_names

    def _index_file(self, entry: Path) -> Path | None:
        for name in self.index_names:
            candidate = entry / name
            if candidate.is_file():
                return candidate
        return None

    def scan(self, collection: str, root: Path) -> list[dict[str, Any]]:
        """The documents in one root, with name, description, path and collection."""
        if not root.is_dir():
            return []
        documents: list[dict[str, Any]] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            index = self._index_file(entry)
            if index is None:
                continue
            try:
                content = index.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("Failed to read document", path=str(index), error=str(exc))
                continue
            meta = parse_frontmatter(content)
            documents.append(
                {
                    "name": meta.get("name") or entry.name,
                    "description": meta.get("description"),
                    "path": str(entry),
                    "collection": collection,
                }
            )
        return documents

    async def list_documents(self, *, collection: str | None) -> dict[str, Any]:
        if collection is not None and collection not in self.roots:
            return {"success": False, "error": f"Unknown collection '{collection}'"}
        roots = {collection: self.roots[collection]} if collection else self.roots
        documents: list[dict[str, Any]] = []
        for name, root in roots.items():
            documents.extend(await asyncio.to_thread(self.scan, name, root))
        return {"success": True, "documents": documents}

    async def read_document(self, *, name: str, collection: str | None) -> dict[str, Any]:
        if "/" in name or "\\" in name or ".." in name:
            return {"success": False, "error": f"Invalid name: '{name}'"}
        if collection is not None and collection not in self.roots:
            return {"success": False, "error": f"Unknown collection '{collection}'"}
        roots = {collection: self.roots[collection]} if collection else self.roots
        for coll, root in roots.items():
            index = self._index_file(root / name)
            if index is None:
                continue
            try:
                content = await asyncio.to_thread(index.read_text, "utf-8")
            except OSError as exc:
                return {"success": False, "error": f"Failed to read document '{name}': {exc}"}
            return {"success": True, "name": name, "collection": coll, "content": content}
        where = f" in collection '{collection}'" if collection else ""
        return {"success": False, "error": f"Document '{name}' not found{where}"}
