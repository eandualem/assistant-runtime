"""Tests for media service configuration."""

import pytest
from pydantic import ValidationError

from lovely_assistant.services.media.config import MediaConfig


class TestMediaConfigDefaults:
    def test_defaults(self):
        config = MediaConfig()
        assert config.default_image_model == "openai:gpt-image-1"
        assert config.default_size == "1024x1024"
        assert config.default_quality == "medium"
        assert config.cache_ttl_seconds == 3600
        assert config.cache_max_items == 100
        assert config.generation_timeout_seconds == 60.0

    def test_override_default_image_model(self):
        config = MediaConfig(default_image_model="anthropic:claude-image-1")
        assert config.default_image_model == "anthropic:claude-image-1"

    def test_override_default_size(self):
        config = MediaConfig(default_size="512x512")
        assert config.default_size == "512x512"

    def test_override_default_quality(self):
        config = MediaConfig(default_quality="high")
        assert config.default_quality == "high"

    def test_override_cache_ttl_seconds(self):
        config = MediaConfig(cache_ttl_seconds=120)
        assert config.cache_ttl_seconds == 120

    def test_override_cache_max_items(self):
        config = MediaConfig(cache_max_items=500)
        assert config.cache_max_items == 500

    def test_override_generation_timeout_seconds(self):
        config = MediaConfig(generation_timeout_seconds=120.0)
        assert config.generation_timeout_seconds == 120.0

    def test_override_all_fields(self):
        config = MediaConfig(
            default_image_model="custom:model",
            default_size="256x256",
            default_quality="low",
            cache_ttl_seconds=300,
            cache_max_items=50,
            generation_timeout_seconds=30.0,
        )
        assert config.default_image_model == "custom:model"
        assert config.default_size == "256x256"
        assert config.default_quality == "low"
        assert config.cache_ttl_seconds == 300
        assert config.cache_max_items == 50
        assert config.generation_timeout_seconds == 30.0


class TestMediaConfigValidation:
    def test_frozen(self):
        config = MediaConfig()
        with pytest.raises(ValidationError):
            config.default_image_model = "other"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MediaConfig(nonexistent_field="value")

    # --- cache_ttl_seconds boundaries (ge=60, le=86400) ---

    def test_cache_ttl_below_min_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(cache_ttl_seconds=59)

    def test_cache_ttl_above_max_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(cache_ttl_seconds=86401)

    def test_cache_ttl_boundary_min(self):
        config = MediaConfig(cache_ttl_seconds=60)
        assert config.cache_ttl_seconds == 60

    def test_cache_ttl_boundary_max(self):
        config = MediaConfig(cache_ttl_seconds=86400)
        assert config.cache_ttl_seconds == 86400

    # --- cache_max_items boundaries (ge=1, le=1000) ---

    def test_cache_max_items_below_min_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(cache_max_items=0)

    def test_cache_max_items_above_max_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(cache_max_items=1001)

    def test_cache_max_items_boundary_min(self):
        config = MediaConfig(cache_max_items=1)
        assert config.cache_max_items == 1

    def test_cache_max_items_boundary_max(self):
        config = MediaConfig(cache_max_items=1000)
        assert config.cache_max_items == 1000

    # --- generation_timeout_seconds boundaries (ge=5.0, le=300.0) ---

    def test_generation_timeout_below_min_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(generation_timeout_seconds=4.9)

    def test_generation_timeout_above_max_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(generation_timeout_seconds=300.1)

    def test_generation_timeout_boundary_min(self):
        config = MediaConfig(generation_timeout_seconds=5.0)
        assert config.generation_timeout_seconds == 5.0

    def test_generation_timeout_boundary_max(self):
        config = MediaConfig(generation_timeout_seconds=300.0)
        assert config.generation_timeout_seconds == 300.0
