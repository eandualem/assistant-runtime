"""Startup registers trusted profiles; requests select only their names."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from assistant_runtime.artifacts import neutral_profile
from assistant_runtime.config import AppSettings
from assistant_runtime.services.artifacts.factory import register_artifacts


async def test_profiles_from_environment_are_loaded_once_at_startup(tmp_path, monkeypatch):
    profile_file = tmp_path / "support.toml"
    profile_file.write_text(
        'name = "support"\n[[artifacts]]\nname = "instructions"\n'
        'required = true\ndefault = "Help with support"\n'
    )
    monkeypatch.setenv("ASSISTANT__PROFILES", f'["{profile_file}", "technical_operator"]')
    settings = AppSettings(_env_file=None)
    state = SimpleNamespace()
    lifecycle = SimpleNamespace(register=AsyncMock())
    await register_artifacts(state, lifecycle, settings=settings)
    service = state.artifact_service
    assert service.available_profiles == ("neutral", "support", "technical_operator")
    lifecycle.register.assert_awaited_once_with("artifact_service", service)
    profile_file.unlink()
    await service.start()
    assert (await service.for_profile("support").active_texts())[
        "instructions"
    ] == "Help with support"
    await service.stop()


async def test_definition_default_preserved_alongside_additional_profiles():
    state = SimpleNamespace(assistant_definition=SimpleNamespace(profile=neutral_profile()))
    settings = AppSettings(assistant={"profile": "not-a-file", "profiles": ["technical_operator"]})
    await register_artifacts(state, SimpleNamespace(register=AsyncMock()), settings=settings)
    assert state.artifact_service.profile.name == "neutral"
    assert (
        state.artifact_service.for_profile("technical_operator").profile.name
        == "technical_operator"
    )


async def test_duplicate_profile_names_fail_before_lifecycle_registration():
    state = SimpleNamespace()
    lifecycle = SimpleNamespace(register=AsyncMock())
    with pytest.raises(ValueError, match="Duplicate assistant profile name"):
        await register_artifacts(
            state, lifecycle, settings=AppSettings(assistant={"profiles": ["neutral"]})
        )
    lifecycle.register.assert_not_awaited()
    assert not hasattr(state, "artifact_service")
