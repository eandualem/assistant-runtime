"""Descriptor-backed operations for the notes and document providers."""

from __future__ import annotations

import errno
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

_DIRECTORY_FLAGS = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_SEARCH_FLAGS = getattr(os, "O_SEARCH", getattr(os, "O_PATH", os.O_RDONLY)) | _DIRECTORY_FLAGS


@contextmanager
def _directory(parent: int, parts: tuple[str, ...], *, create: bool = False) -> Iterator[int]:
    fd = os.dup(parent)
    try:
        for name in parts:
            if create:
                with suppress(FileExistsError):
                    os.mkdir(name, dir_fd=fd)
            child = os.open(name, _SEARCH_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _file(parent: int, name: str, flags: int) -> Iterator[int]:
    fd = os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, 0o666, dir_fd=parent)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "Expected a regular markdown file", name)
        yield fd
    finally:
        os.close(fd)


@contextmanager
def exclusive_text(parent: int, name: str) -> Iterator[TextIO]:
    with (
        _file(parent, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL) as fd,
        os.fdopen(fd, "w", encoding="utf-8", closefd=False) as handle,
    ):
        yield handle


@dataclass(frozen=True)
class RootedDirectory:
    path: Path
    fd: int

    def relative(self, path: Path) -> Path:
        """Admit an alias target, then access only its canonical components."""
        return path.resolve().relative_to(self.path)

    @contextmanager
    def directory(self, path: Path, *, create: bool = False) -> Iterator[int]:
        with _directory(self.fd, self.relative(path).parts, create=create) as fd:
            yield fd

    @contextmanager
    def parent(self, path: Path, *, create: bool = False) -> Iterator[int]:
        # Entry operations must not resolve the final symlink itself.
        with self.directory(path.parent, create=create) as fd:
            yield fd

    @contextmanager
    def text(self, path: Path, *, update: bool = False) -> Iterator[TextIO]:
        relative = self.relative(path)
        with (
            _directory(self.fd, relative.parent.parts) as parent,
            _file(parent, relative.name, os.O_RDWR if update else os.O_RDONLY) as fd,
            os.fdopen(fd, "r+" if update else "r", encoding="utf-8", closefd=False) as handle,
        ):
            yield handle

    def read_text(self, path: Path) -> str:
        with self.text(path) as handle:
            return handle.read()

    def entry_names(self) -> list[str]:
        # Traversal needs only search permission; enumeration separately needs read.
        fd = os.open(".", os.O_RDONLY | _DIRECTORY_FLAGS, dir_fd=self.fd)
        try:
            return os.listdir(fd)
        finally:
            os.close(fd)

    def markdown_paths(self, scope: Path) -> list[Path]:
        with self.directory(scope) as fd:
            return sorted(
                (
                    scope / directory / name
                    for directory, _, files, _ in os.fwalk(".", dir_fd=fd, follow_symlinks=False)
                    for name in files
                    if name.endswith(".md")
                ),
                reverse=True,
            )

    def move(self, source: Path, destination: Path) -> bool:
        """Move the source entry; False means the destination already exists."""
        self.relative(source)
        with self.parent(source) as src:
            os.stat(source.name, dir_fd=src, follow_symlinks=False)
            return self._move_entry(src, source.name, destination)

    def _move_entry(self, src: int, source: str, destination: Path) -> bool:
        with self.parent(destination, create=True) as dst:
            try:
                os.stat(destination.name, dir_fd=dst, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                return False
            try:
                os.rename(source, destination.name, src_dir_fd=src, dst_dir_fd=dst)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                _move_across_devices(src, source, dst, destination.name)
        return True


@contextmanager
def rooted(root: Path, *, create: bool = False) -> Iterator[RootedDirectory]:
    """Pin one operation's configured root, including a configured-root alias."""
    canonical = root.resolve()
    anchor = os.open(canonical.anchor, _SEARCH_FLAGS)
    try:
        with _directory(anchor, canonical.parts[1:], create=create) as fd:
            yield RootedDirectory(canonical, fd)
    finally:
        os.close(anchor)


def _copy_metadata(source: int, destination: int, metadata: os.stat_result) -> None:
    os.utime(destination, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    if hasattr(os, "listxattr"):
        # Match copy2's best-effort Linux xattrs, using only pinned descriptors.
        ignored = {errno.EPERM, errno.ENOTSUP, errno.ENODATA, errno.EINVAL}
        try:
            names = os.listxattr(source)
        except OSError as exc:
            if exc.errno not in ignored - {errno.EPERM}:
                raise
        else:
            for name in names:
                try:
                    os.setxattr(destination, name, os.getxattr(source, name))
                except OSError as exc:
                    if exc.errno not in ignored:
                        raise
    os.fchmod(destination, stat.S_IMODE(metadata.st_mode))


def _move_across_devices(src: int, source: str, dst: int, destination: str) -> None:
    metadata = os.stat(source, dir_fd=src, follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode):
        os.symlink(os.readlink(source, dir_fd=src), destination, dir_fd=dst)
    else:
        with _file(src, source, os.O_RDONLY) as source_fd:
            metadata = os.fstat(source_fd)
            if getattr(metadata, "st_flags", 0):
                raise OSError(
                    errno.ENOTSUP,
                    "Cross-filesystem note moves cannot preserve file flags on this platform",
                )
            with _file(dst, destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL) as destination_fd:
                try:
                    with (
                        os.fdopen(source_fd, "rb", closefd=False) as reader,
                        os.fdopen(destination_fd, "wb", closefd=False) as writer,
                    ):
                        shutil.copyfileobj(reader, writer)
                    _copy_metadata(source_fd, destination_fd, metadata)
                except BaseException:
                    # A partial copy would make every retry report "already exists".
                    with suppress(OSError):
                        os.unlink(destination, dir_fd=dst)
                    raise
    os.unlink(source, dir_fd=src)
