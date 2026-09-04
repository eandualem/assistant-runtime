"""Tests for the model catalog leaf module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from assistant_runtime.model_catalog import (
    ALLOWED_PROVIDERS,
    MEDIA_PROVIDER_ENV_VARS,
    MODEL_CATALOG,
    PROVIDER_DEFAULT_MODELS,
    PROVIDER_DEFAULT_SUMMARIZATION_MODELS,
    PROVIDER_ENV_VARS,
    ModelEntry,
    ProviderInfo,
    get_models,
    get_provider_info,
)
from assistant_runtime.services.media.interface import _PROVIDER_GENERATORS, _VIDEO_PROVIDERS


class TestModelCatalog:
    def test_catalog_not_empty(self):
        assert len(MODEL_CATALOG) > 0

    def test_all_entries_are_model_entry(self):
        for entry in MODEL_CATALOG:
            assert isinstance(entry, ModelEntry)

    def test_all_entries_have_required_fields(self):
        for entry in MODEL_CATALOG:
            assert entry.id
            assert entry.provider
            assert entry.name
            assert entry.capability in ("flagship", "balanced", "fast")
            assert len(entry.capabilities) > 0
            assert entry.description

    def test_entries_are_frozen(self):
        entry = MODEL_CATALOG[0]
        with pytest.raises(ValidationError, match="frozen"):
            entry.name = "Modified"  # type: ignore[misc]

    def test_has_anthropic_models(self):
        anthropic = [m for m in MODEL_CATALOG if m.provider == "anthropic"]
        assert len(anthropic) >= 4

    def test_has_openai_models(self):
        openai = [m for m in MODEL_CATALOG if m.provider == "openai"]
        assert len(openai) >= 3

    def test_has_google_models(self):
        google = [m for m in MODEL_CATALOG if m.provider == "google"]
        assert len(google) >= 2

    def test_has_image_generation_models(self):
        image_gen = [m for m in MODEL_CATALOG if "image-generation" in m.capabilities]
        assert len(image_gen) >= 3

    def test_has_video_generation_models(self):
        video_gen = [m for m in MODEL_CATALOG if "video-generation" in m.capabilities]
        assert len(video_gen) >= 3

    def test_text_models_have_context_window(self):
        text = [m for m in MODEL_CATALOG if "text" in m.capabilities]
        for m in text:
            assert m.context_window is not None
            assert m.context_window > 0

    def test_image_video_models_have_no_context_window(self):
        non_text = [m for m in MODEL_CATALOG if "text" not in m.capabilities]
        for m in non_text:
            assert m.context_window is None


class TestGetModels:
    def test_no_filter_returns_all(self):
        result = get_models()
        assert len(result) == len(MODEL_CATALOG)

    def test_filter_by_text_capability(self):
        result = get_models(capability="text")
        assert len(result) > 0
        assert all("text" in m.capabilities for m in result)

    def test_filter_by_image_generation(self):
        result = get_models(capability="image-generation")
        assert len(result) > 0
        assert all("image-generation" in m.capabilities for m in result)

    def test_filter_by_video_generation(self):
        result = get_models(capability="video-generation")
        assert len(result) > 0
        assert all("video-generation" in m.capabilities for m in result)

    def test_filter_by_provider_anthropic(self):
        result = get_models(provider="anthropic")
        assert len(result) > 0
        assert all(m.provider == "anthropic" for m in result)

    def test_filter_by_provider_openai(self):
        result = get_models(provider="openai")
        assert len(result) > 0
        assert all(m.provider == "openai" for m in result)

    def test_media_generation_models_use_supported_providers(self):
        image_providers = {
            m.provider for m in MODEL_CATALOG if "image-generation" in m.capabilities
        }
        video_providers = {
            m.provider for m in MODEL_CATALOG if "video-generation" in m.capabilities
        }

        assert image_providers <= set(_PROVIDER_GENERATORS)
        assert video_providers <= set(_VIDEO_PROVIDERS)

    def test_combined_filter(self):
        result = get_models(capability="text", provider="anthropic")
        assert len(result) > 0
        assert all(m.provider == "anthropic" and "text" in m.capabilities for m in result)

    def test_no_match_returns_empty(self):
        result = get_models(capability="nonexistent")
        assert result == []

    def test_no_match_provider_returns_empty(self):
        result = get_models(provider="nonexistent")
        assert result == []


class TestGetProviderInfo:
    def test_returns_all_providers(self):
        info = get_provider_info()
        assert "anthropic" in info
        assert "openai" in info
        assert "google" in info
        assert "openrouter" in info
        assert "runway" in info
        assert "luma" in info

    def test_provider_info_structure(self):
        info = get_provider_info()
        for provider_info in info.values():
            assert isinstance(provider_info, ProviderInfo)
            assert provider_info.name
            assert provider_info.env_var

    def test_not_configured_when_no_env(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        info = get_provider_info()
        assert info["anthropic"].configured is False

    def test_configured_when_env_set_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        info = get_provider_info()
        assert info["anthropic"].configured is True

    def test_configured_when_env_set_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        info = get_provider_info()
        assert info["openai"].configured is True

    def test_configured_when_env_set_runway(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "runway-test")
        info = get_provider_info()
        assert info["runway"].configured is True

    def test_configured_when_env_set_luma(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test")
        info = get_provider_info()
        assert info["luma"].configured is True


class TestProviderKnowledge:
    def test_allowed_providers_are_the_ones_with_env_vars(self):
        assert list(PROVIDER_ENV_VARS) == ALLOWED_PROVIDERS

    def test_every_llm_provider_has_a_default_model_in_the_catalog(self):
        ids = {m.id for m in MODEL_CATALOG}
        for provider in PROVIDER_ENV_VARS:
            assert PROVIDER_DEFAULT_MODELS[provider] in ids
            assert PROVIDER_DEFAULT_SUMMARIZATION_MODELS[provider] in ids

    def test_catalog_providers_are_known(self):
        known = set(PROVIDER_ENV_VARS) | set(MEDIA_PROVIDER_ENV_VARS)
        assert {m.provider for m in MODEL_CATALOG} <= known
