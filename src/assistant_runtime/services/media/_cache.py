"""In-memory TTL image cache for generated images."""

from __future__ import annotations

import time
import uuid


class ImageCache:
    """Thread-safe in-memory cache with TTL eviction for generated images.

    Stores raw image bytes keyed by UUID. Entries expire after `ttl_seconds`.
    When the cache reaches `max_items`, expired entries are cleaned up on the
    next `store()` call. If still at capacity after cleanup, the oldest entry
    is evicted.
    """

    def __init__(self, ttl_seconds: int, max_items: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_items = max_items
        # Maps image_id -> (image_bytes, mime_type, monotonic_expiry)
        self._entries: dict[str, tuple[bytes, str, float]] = {}

    def store(self, image_bytes: bytes, mime_type: str) -> str:
        """Store image bytes and return a UUID image_id."""
        if len(self._entries) >= self._max_items:
            self.cleanup()
        # If still at capacity after cleanup, evict the oldest
        if len(self._entries) >= self._max_items:
            self._evict_oldest()

        image_id = uuid.uuid4().hex
        expiry = time.monotonic() + self._ttl_seconds
        self._entries[image_id] = (image_bytes, mime_type, expiry)
        return image_id

    def get(self, image_id: str) -> tuple[bytes, str] | None:
        """Retrieve cached image. Returns (bytes, mime_type) or None if missing/expired."""
        entry = self._entries.get(image_id)
        if entry is None:
            return None
        image_bytes, mime_type, expiry = entry
        if time.monotonic() > expiry:
            del self._entries[image_id]
            return None
        return (image_bytes, mime_type)

    def cleanup(self) -> int:
        """Remove all expired entries. Returns count of evicted items."""
        now = time.monotonic()
        expired = [k for k, (_, _, exp) in self._entries.items() if now > exp]
        for k in expired:
            del self._entries[k]
        return len(expired)

    def _evict_oldest(self) -> None:
        """Evict the entry with the earliest expiry time."""
        if not self._entries:
            return
        oldest_key = min(self._entries, key=lambda k: self._entries[k][2])
        del self._entries[oldest_key]

    def __len__(self) -> int:
        """Current number of entries (including potentially expired ones)."""
        return len(self._entries)
