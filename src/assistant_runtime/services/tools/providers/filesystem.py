"""Filesystem providers: a folder of markdown notes and directories of documents."""

from __future__ import annotations

import asyncio
import errno
import os
import re
import threading
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from assistant_runtime.services.tools.providers._filesystem import (
    RootedDirectory,
    exclusive_text,
    rooted,
)

FILENAME_PATTERN = re.compile(r"^[\w\-]+\.md$")
_SLUG_INVALID_CHARS = re.compile(r"[^a-z0-9]+")


def _inside_root(target: Path, root: Path) -> bool:
    try:
        return target.resolve().is_relative_to(root.resolve())
    except (ValueError, OSError, RuntimeError):
        return False


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
        # The destination check and the move are separate steps; parallel
        # tool calls must not interleave them and overwrite a note.
        self._move_lock = threading.Lock()

    # --- validation ------------------------------------------------------------

    def _is_inside_root(self, target: Path) -> bool:
        return _inside_root(target, self.root)

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
        if not self._is_inside_root(self.root / filename):
            return None, "Invalid note path — resolves outside notes directory"
        return self.root / filename, None

    # --- files -----------------------------------------------------------------

    def parse_note(self, path: Path) -> dict[str, Any] | None:
        """A note file as a dict (filename, path, folder, title, date, tags, content), or None."""
        try:
            with rooted(self.root) as root:
                return self._read_note(root, path)
        except UnicodeError:
            raise
        except (OSError, ValueError, RuntimeError):
            return None

    def _read_note(self, root: RootedDirectory, path: Path) -> dict[str, Any] | None:
        try:
            relative = root.relative(path)
            content = root.read_text(root.path / relative)
        except UnicodeError:
            raise
        except (OSError, ValueError, RuntimeError):
            return None
        return self._parse_note(path, content, str(relative))

    @staticmethod
    def _parse_note(path: Path, content: str, relative: str) -> dict[str, Any]:
        frontmatter: dict[str, Any] = {}
        body = content
        if content.startswith("---\n"):
            parts = content.split("---\n", 2)
            if len(parts) >= 3:
                try:
                    loaded = yaml.safe_load(parts[1])
                except yaml.YAMLError:
                    loaded = None
                frontmatter = loaded if isinstance(loaded, dict) else {}
                body = parts[2].strip()
        folder = str(Path(relative).parent)
        return {
            "filename": path.name,
            "path": relative,
            "folder": "" if folder == "." else folder,
            "title": frontmatter.get("title", path.stem),
            "date": str(frontmatter.get("date", "")),
            "tags": frontmatter.get("tags", []),
            "content": body,
        }

    def _scan_notes(
        self, folder: str, limit: int, *, tag: str = "", query: str = ""
    ) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        with rooted(self.root, create=True) as root:
            try:
                paths = root.markdown_paths(self.root / folder) if limit > 0 else []
            except (OSError, ValueError, RuntimeError):
                return notes
            for path in paths:
                if len(notes) >= limit:
                    break
                parsed = self._read_note(root, path)
                if parsed is None or (tag and tag not in parsed["tags"]):
                    continue
                if query:
                    searchable = f"{parsed['title']} {parsed['content']} {' '.join(parsed['tags'])}"
                    if query.lower() not in searchable.lower():
                        continue
                notes.append(parsed)
        return notes

    # --- NotesStore --------------------------------------------------------------

    async def create(
        self, *, title: str, content: str, tags: list[str] | None, folder: str
    ) -> dict[str, Any]:
        if not title:
            return {"error": "Title is required for create", "success": False}
        if not content:
            return {"error": "Content is required for create", "success": False}
        if folder:
            error = self.validate_folder(folder)
            if error:
                return {"error": error, "success": False}
        slug = slugify(title)
        if not slug:
            return {"error": "Title produces empty slug", "success": False}
        note = build_note_content(title, content, tags)
        filename, rel = await asyncio.to_thread(self._create, folder, slug, note)
        logger.info("Created note", path=rel, title=title)
        return {"filename": filename, "path": rel, "title": title, "success": True}

    def _create(self, folder: str, slug: str, note: str) -> tuple[str, str]:
        with rooted(self.root, create=True) as root:
            relative = root.relative(self.root / folder)
            with root.directory(root.path / relative, create=True) as parent:
                filename = _create_exclusive(parent, slug, note)
            return filename, str(relative / filename)

    async def list(self, *, tag: str, limit: int, folder: str) -> dict[str, Any]:
        error = self.validate_folder(folder) if folder else None
        if error:
            return {"error": error, "success": False}
        notes: list[dict[str, Any]] = []
        for parsed in await asyncio.to_thread(self._scan_notes, folder, limit, tag=tag):
            preview = parsed["content"].split("\n", 1)[0][:200] if parsed["content"] else ""
            notes.append(
                {k: parsed[k] for k in ("filename", "path", "folder", "title", "date", "tags")}
                | {"preview": preview}
            )
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
        error = self.validate_folder(folder) if folder else None
        if error:
            return {"error": error, "success": False}
        results: list[dict[str, Any]] = []
        for parsed in await asyncio.to_thread(self._scan_notes, folder, limit, query=query):
            results.append(
                {k: parsed[k] for k in ("filename", "path", "folder", "title", "date", "tags")}
                | {"snippet": _snippet(parsed["content"], query)}
            )
        return {"results": results, "count": len(results), "query": query, "success": True}

    async def update(
        self, *, filename: str, content: str, tags: list[str] | None
    ) -> dict[str, Any]:
        path, error = self._resolve(filename)
        if error:
            return {"error": error, "success": False}
        return await asyncio.to_thread(self._update, path, filename, content, tags)

    def _update(
        self, path: Path, filename: str, content: str, tags: list[str] | None
    ) -> dict[str, Any]:
        with ExitStack() as resources:
            try:
                root = resources.enter_context(rooted(self.root))
                relative = root.relative(path)
                handle = resources.enter_context(root.text(root.path / relative, update=True))
                parsed = self._parse_note(path, handle.read(), str(relative))
            except (UnicodeError, PermissionError):
                raise
            except (OSError, ValueError, RuntimeError):
                return {"error": f"Note not found: {filename}", "success": False}
            if not content and tags is None:
                return {"error": "Provide content or tags to update", "success": False}
            new_content = content if content else parsed["content"]
            new_tags = tags if tags is not None else parsed["tags"]
            handle.seek(0)
            handle.write(build_note_content(parsed["title"], new_content, new_tags))
            handle.truncate()
        logger.info("Updated note", filename=filename)
        return {
            **{k: parsed[k] for k in ("filename", "path", "title")},
            "updated": True,
            "success": True,
        }

    async def delete(self, *, filename: str) -> dict[str, Any]:
        path, error = self._resolve(filename)
        if error:
            return {"error": error, "success": False}
        if not path.exists():
            return {"error": f"Note not found: {filename}", "success": False}
        return await asyncio.to_thread(self._delete, path, filename)

    def _delete(self, path: Path, filename: str) -> dict[str, Any]:
        try:
            with rooted(self.root) as root:
                root.relative(path)
                relative = root.relative(path.parent) / path.name
                with root.parent(root.path / relative) as parent:
                    os.unlink(path.name, dir_fd=parent)
        except FileNotFoundError:
            return {"error": f"Note not found: {filename}", "success": False}
        logger.info("Deleted note", filename=filename)
        return {
            "filename": path.name,
            "path": str(relative),
            "deleted": True,
            "success": True,
        }

    async def create_folder(self, *, folder: str) -> dict[str, Any]:
        if not folder:
            return {"error": "Folder path is required for create_folder", "success": False}
        error = self.validate_folder(folder)
        if error:
            return {"error": error, "success": False}
        created = await asyncio.to_thread(self._create_folder, folder)
        if not created:
            return {
                "folder": folder,
                "created": False,
                "message": "Folder already exists",
                "success": True,
            }
        logger.info("Created notes folder", folder=folder)
        return {"folder": folder, "created": True, "success": True}

    def _create_folder(self, folder: str) -> bool:
        with rooted(self.root, create=True) as root:
            target = root.path / root.relative(self.root / folder)
            if target == root.path:
                return False
            with root.parent(target, create=True) as parent:
                try:
                    os.mkdir(target.name, dir_fd=parent)
                except FileExistsError:
                    with root.directory(target):
                        pass
                    return False
            return True

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
        return await asyncio.to_thread(self._move, src, filename, folder)

    def _move(self, source: Path, filename: str, folder: str) -> dict[str, Any]:
        try:
            with self._move_lock, rooted(self.root) as root:
                source_rel = str(root.relative(source))
                destination_rel = root.relative(self.root / folder) / source.name
                moved = root.move(source, root.path / destination_rel)
        except FileNotFoundError:
            return {"error": f"Note not found: {filename}", "success": False}
        if not moved:
            return {
                "error": f"A note named '{source.name}' already exists in '{folder}'",
                "success": False,
            }
        new_path = str(destination_rel)
        logger.info("Moved note", source=filename, destination=new_path)
        return {"filename": source.name, "from": source_rel, "to": new_path, "success": True}


def _create_exclusive(target_dir: int, slug: str, note: str) -> str:
    """Write ``note`` under a dated slug filename that does not exist yet, atomically."""
    counter = 0
    while True:
        suffix = f"-{counter}" if counter else ""
        filename = f"{date.today()}-{slug}{suffix}.md"
        try:
            with exclusive_text(target_dir, filename) as handle:
                handle.write(note)
        except FileExistsError:
            counter += 1
            continue
        return filename


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

    def _read_index(self, entry: Path, root: RootedDirectory) -> str | None:
        for name in self.index_names:
            try:
                return root.read_text(entry / name)
            except UnicodeError:
                raise
            except (ValueError, RuntimeError):
                continue
            except OSError as exc:
                if exc.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EINVAL):
                    raise
        return None

    def scan(self, collection: str, root: Path) -> list[dict[str, Any]]:
        """The documents in one root, with name, description, path and collection."""
        try:
            with rooted(root) as directory:
                return self._scan(collection, root, directory)
        except (FileNotFoundError, NotADirectoryError):
            return []

    def _scan(
        self, collection: str, root: Path, directory: RootedDirectory
    ) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for name in sorted(directory.entry_names()):
            entry = root / name
            try:
                content = self._read_index(entry, directory)
            except OSError as exc:
                logger.warning("Failed to read document", path=str(entry), error=str(exc))
                continue
            if content is None:
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
        return await asyncio.to_thread(self._read_document, name, collection)

    def _read_document(self, name: str, collection: str | None) -> dict[str, Any]:
        roots = {collection: self.roots[collection]} if collection else self.roots
        for coll, root in roots.items():
            try:
                with rooted(root) as directory:
                    content = self._read_index(root / name, directory)
            except (FileNotFoundError, NotADirectoryError):
                continue
            except OSError as exc:
                return {"success": False, "error": f"Failed to read document '{name}': {exc}"}
            if content is None:
                continue
            return {"success": True, "name": name, "collection": coll, "content": content}
        where = f" in collection '{collection}'" if collection else ""
        return {"success": False, "error": f"Document '{name}' not found{where}"}
