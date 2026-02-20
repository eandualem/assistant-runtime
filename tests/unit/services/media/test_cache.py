"""Tests for the in-memory TTL image cache."""

import re

from lovely_assistant.services.media._cache import ImageCache


class TestStoreAndRetrieve:
    """Basic store/get round-trip tests."""

    def test_store_returns_hex_string(self):
        cache = ImageCache(ttl_seconds=60, max_items=10)
        image_id = cache.store(b"png-data", "image/png")
        assert isinstance(image_id, str)
        assert re.fullmatch(r"[0-9a-f]{32}", image_id)

    def test_store_returns_unique_ids(self):
        cache = ImageCache(ttl_seconds=60, max_items=10)
        id_a = cache.store(b"img-a", "image/png")
        id_b = cache.store(b"img-b", "image/png")
        assert id_a != id_b

    def test_get_returns_stored_data(self):
        cache = ImageCache(ttl_seconds=60, max_items=10)
        image_id = cache.store(b"png-data", "image/png")
        result = cache.get(image_id)
        assert result == (b"png-data", "image/png")

    def test_get_preserves_mime_type(self):
        cache = ImageCache(ttl_seconds=60, max_items=10)
        image_id = cache.store(b"jpeg-data", "image/jpeg")
        result = cache.get(image_id)
        assert result is not None
        assert result[1] == "image/jpeg"

    def test_get_missing_id_returns_none(self):
        cache = ImageCache(ttl_seconds=60, max_items=10)
        assert cache.get("nonexistent") is None

    def test_get_missing_id_on_empty_cache(self):
        cache = ImageCache(ttl_seconds=60, max_items=10)
        assert cache.get("abc123") is None


class TestLen:
    """Tests for __len__ reflecting current entry count."""

    def test_empty_cache(self):
        cache = ImageCache(ttl_seconds=60, max_items=10)
        assert len(cache) == 0

    def test_after_stores(self):
        cache = ImageCache(ttl_seconds=60, max_items=10)
        cache.store(b"a", "image/png")
        cache.store(b"b", "image/png")
        cache.store(b"c", "image/png")
        assert len(cache) == 3

    def test_includes_expired_entries(self, monkeypatch):
        """__len__ counts all entries, including expired ones (no lazy cleanup)."""
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=10, max_items=10)
        cache.store(b"data", "image/png")
        assert len(cache) == 1

        # Advance past TTL — len still counts it (not cleaned up yet)
        current_time = 1020.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        assert len(cache) == 1


class TestTTLExpiry:
    """Tests for time-based expiration using monkeypatched time.monotonic."""

    def test_get_within_ttl_succeeds(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=60, max_items=10)
        image_id = cache.store(b"data", "image/png")

        # Advance time but stay within TTL
        current_time = 1059.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        assert cache.get(image_id) == (b"data", "image/png")

    def test_get_at_exact_expiry_succeeds(self, monkeypatch):
        """At exactly the expiry boundary (now == expiry), entry is still valid."""
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=60, max_items=10)
        image_id = cache.store(b"data", "image/png")

        # Advance to exactly the expiry time (monotonic() == expiry)
        # The condition is `time.monotonic() > expiry`, so equal is NOT expired
        current_time = 1060.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        assert cache.get(image_id) == (b"data", "image/png")

    def test_get_past_ttl_returns_none(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=60, max_items=10)
        image_id = cache.store(b"data", "image/png")

        # Advance past TTL
        current_time = 1061.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        assert cache.get(image_id) is None

    def test_get_removes_expired_entry_from_cache(self, monkeypatch):
        """Expired entry is deleted on get (lazy cleanup)."""
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=10, max_items=10)
        image_id = cache.store(b"data", "image/png")
        assert len(cache) == 1

        # Expire and access
        current_time = 1011.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        cache.get(image_id)

        # Entry should be removed from internal storage
        assert len(cache) == 0

    def test_expired_entry_not_retrievable_after_lazy_cleanup(self, monkeypatch):
        """After get returns None for an expired entry, a second get also returns None."""
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=10, max_items=10)
        image_id = cache.store(b"data", "image/png")

        current_time = 1011.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        assert cache.get(image_id) is None
        # Second access — entry was deleted, so it hits the "entry is None" path
        assert cache.get(image_id) is None


class TestCleanup:
    """Tests for the cleanup method that removes all expired entries."""

    def test_cleanup_removes_expired_entries(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=10, max_items=10)
        cache.store(b"a", "image/png")
        cache.store(b"b", "image/png")
        assert len(cache) == 2

        # Expire all
        current_time = 1011.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        cache.cleanup()
        assert len(cache) == 0

    def test_cleanup_returns_count_of_evicted(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=10, max_items=10)
        cache.store(b"a", "image/png")
        cache.store(b"b", "image/png")
        cache.store(b"c", "image/png")

        current_time = 1011.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        count = cache.cleanup()
        assert count == 3

    def test_cleanup_returns_zero_when_nothing_expired(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=60, max_items=10)
        cache.store(b"a", "image/png")

        count = cache.cleanup()
        assert count == 0

    def test_cleanup_on_empty_cache_returns_zero(self):
        cache = ImageCache(ttl_seconds=60, max_items=10)
        assert cache.cleanup() == 0

    def test_cleanup_only_removes_expired(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=10, max_items=10)
        id_a = cache.store(b"a", "image/png")

        # Store second entry later
        current_time = 1005.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        id_b = cache.store(b"b", "image/png")

        # Advance so only the first entry is expired (expiry=1010) but not the second (expiry=1015)
        current_time = 1011.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        count = cache.cleanup()

        assert count == 1
        assert len(cache) == 1
        assert cache.get(id_b) == (b"b", "image/png")
        assert cache.get(id_a) is None


class TestMaxItemsEviction:
    """Tests for capacity-based eviction when max_items is reached."""

    def test_store_at_capacity_triggers_cleanup(self, monkeypatch):
        """When at capacity, store calls cleanup before evicting."""
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=10, max_items=2)
        cache.store(b"a", "image/png")
        cache.store(b"b", "image/png")
        assert len(cache) == 2

        # Expire all entries, then store a new one — cleanup frees space
        current_time = 1011.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        id_c = cache.store(b"c", "image/png")

        # Cleanup removed both expired entries; only new one remains
        assert len(cache) == 1
        assert cache.get(id_c) == (b"c", "image/png")

    def test_evict_oldest_when_cleanup_insufficient(self, monkeypatch):
        """When cleanup doesn't free space, the oldest (earliest expiry) is evicted."""
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=60, max_items=2)
        id_a = cache.store(b"a", "image/png")

        current_time = 1001.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        id_b = cache.store(b"b", "image/png")

        # At capacity, no expired entries — oldest (id_a, expiry=1060) gets evicted
        current_time = 1002.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        id_c = cache.store(b"c", "image/png")

        assert len(cache) == 2
        assert cache.get(id_a) is None  # evicted
        assert cache.get(id_b) == (b"b", "image/png")
        assert cache.get(id_c) == (b"c", "image/png")

    def test_evict_oldest_picks_earliest_expiry(self, monkeypatch):
        """Eviction targets the entry with the smallest expiry timestamp."""
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=100, max_items=3)
        id_first = cache.store(b"first", "image/png")  # expiry = 1100

        current_time = 1010.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        id_second = cache.store(b"second", "image/png")  # expiry = 1110

        current_time = 1020.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        id_third = cache.store(b"third", "image/png")  # expiry = 1120

        # Store a 4th — should evict id_first (expiry=1100, the earliest)
        current_time = 1030.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        id_fourth = cache.store(b"fourth", "image/png")

        assert len(cache) == 3
        assert cache.get(id_first) is None
        assert cache.get(id_second) == (b"second", "image/png")
        assert cache.get(id_third) == (b"third", "image/png")
        assert cache.get(id_fourth) == (b"fourth", "image/png")

    def test_max_items_one(self, monkeypatch):
        """Cache with max_items=1 always has at most one entry."""
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=60, max_items=1)
        id_a = cache.store(b"a", "image/png")
        assert len(cache) == 1

        current_time = 1001.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        id_b = cache.store(b"b", "image/png")

        assert len(cache) == 1
        assert cache.get(id_a) is None
        assert cache.get(id_b) == (b"b", "image/png")

    def test_store_below_capacity_no_eviction(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=60, max_items=5)
        id_a = cache.store(b"a", "image/png")
        id_b = cache.store(b"b", "image/png")

        assert len(cache) == 2
        assert cache.get(id_a) == (b"a", "image/png")
        assert cache.get(id_b) == (b"b", "image/png")


class TestEvictOldest:
    """Tests for the _evict_oldest internal method."""

    def test_evict_oldest_on_empty_cache_is_noop(self):
        cache = ImageCache(ttl_seconds=60, max_items=10)
        # Should not raise
        cache._evict_oldest()
        assert len(cache) == 0

    def test_evict_oldest_removes_one_entry(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )

        cache = ImageCache(ttl_seconds=60, max_items=10)
        cache.store(b"a", "image/png")

        current_time = 1001.0
        monkeypatch.setattr(
            "lovely_assistant.services.media._cache.time.monotonic", lambda: current_time
        )
        id_b = cache.store(b"b", "image/png")

        assert len(cache) == 2
        cache._evict_oldest()
        assert len(cache) == 1
        # b should survive (later expiry)
        assert cache.get(id_b) == (b"b", "image/png")
