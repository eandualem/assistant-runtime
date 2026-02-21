"""Tests for the static model registry."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lovely_assistant.app.models_registry import (
    MODEL_CATALOG,
    ModelEntry,
    ProviderInfo,
    get_defaults,
    get_models,
    get_provider_info,
)


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

    def test_filter_by_provider_fal(self):
        result = get_models(provider="fal")
        assert len(result) > 0
        assert all(m.provider == "fal" for m in result)

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
        assert "fal" in info
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

    def test_configured_when_env_set_fal(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "fal-test")
        info = get_provider_info()
        assert info["fal"].configured is True

    def test_configured_when_env_set_runway(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "runway-test")
        info = get_provider_info()
        assert info["runway"].configured is True

    def test_configured_when_env_set_luma(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test")
        info = get_provider_info()
        assert info["luma"].configured is True


class TestGetDefaults:
    def test_returns_expected_keys(self):
        defaults = get_defaults()
        expected_keys = {
            "primary_model",
            "summarization_model",
            "working_memory_model",
            "default_image_model",
            "default_video_model",
            "subagent_model",
            "subagent_thinking_budget",
        }
        assert set(defaults.keys()) == expected_keys

    def test_values_have_correct_types(self):
        defaults = get_defaults()
        # String fields from LLM config
        assert isinstance(defaults["primary_model"], str)
        assert isinstance(defaults["summarization_model"], str)
        # working_memory_model defaults to None (str | None in config)
        assert defaults["working_memory_model"] is None or isinstance(
            defaults["working_memory_model"], str
        )
        # Media model defaults are strings
        assert isinstance(defaults["default_image_model"], str)
        assert isinstance(defaults["default_video_model"], str)
        # Subagent fields are always None (no config backing yet)
        assert defaults["subagent_model"] is None
        assert defaults["subagent_thinking_budget"] is None
