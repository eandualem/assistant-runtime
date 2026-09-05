"""Assistant profiles: schema validation, built-ins, file loading and precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant_runtime.artifacts import (
    BUILTIN_PROFILE_NAMES,
    PROFILES_DIR,
    ArtifactDefinition,
    ArtifactPolicy,
    AssistantProfile,
    load_profile_file,
    neutral_profile,
    profile_from_mapping,
    resolve_profile,
    technical_operator_profile,
)


class TestSchema:
    def test_policy_rejects_unknown_edit_mode(self):
        with pytest.raises(ValueError, match="assistant_edit must be one of"):
            ArtifactPolicy(assistant_edit="anything")

    @pytest.mark.parametrize("name", ["Soul", "1st", "with-dash", "", "a" * 65])
    def test_artifact_name_must_be_an_identifier(self, name):
        with pytest.raises(ValueError, match="Artifact name"):
            ArtifactDefinition(name=name)

    def test_required_artifact_needs_a_default(self):
        with pytest.raises(ValueError, match="needs a non-empty default"):
            ArtifactDefinition(name="instructions", required=True, default="  ")

    def test_profile_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="Duplicate artifact name"):
            AssistantProfile(artifacts=[ArtifactDefinition(name="a"), ArtifactDefinition(name="a")])

    def test_profile_name_must_be_an_identifier(self):
        with pytest.raises(ValueError, match="Profile name"):
            AssistantProfile(name="My Profile")

    def test_profile_snapshots_artifacts_and_orders_them(self):
        profile = AssistantProfile(
            artifacts=[
                ArtifactDefinition(name="b", required=True, default="B"),
                ArtifactDefinition(name="a", role="first"),
            ]
        )
        assert isinstance(profile.artifacts, tuple)
        assert profile.names == ("b", "a")
        assert profile.required_names == ("b",)
        assert profile.defaults == {"b": "B", "a": ""}
        assert profile.get("a").role == "first"
        assert profile.get("zzz") is None
        assert sorted(["zzz", "a", "b"], key=profile.sort_key) == ["b", "a", "zzz"]
        assert profile.names_text() == "b, a"
        assert profile.roles_text() == "a = first"


class TestBuiltinProfiles:
    def test_neutral_is_minimal_and_carries_no_operator_worldview(self):
        profile = neutral_profile()
        assert profile.names == ("instructions", "scratchpad")
        assert profile.required_names == ("instructions",)
        assert profile.get("scratchpad").policy.assistant_edit == "autonomous"
        assert profile.get("instructions").policy.assistant_edit == "propose"
        text = profile.defaults["instructions"].lower()
        for word in ("operator", "agent", "tmux", "backbone", "telegram"):
            assert word not in text

    def test_technical_operator_keeps_the_original_artifacts(self):
        profile = technical_operator_profile()
        assert profile.names == (
            "soul",
            "persona",
            "communication_protocol",
            "ecosystem",
            "scratchpad",
        )
        assert profile.required_names == profile.names[:-1]
        for name in profile.required_names:
            assert (PROFILES_DIR / "technical_operator" / f"{name}.md").is_file()
            text = profile.defaults[name]
            assert text
            assert text == text.strip()

    def test_technical_operator_texts_carry_no_private_names(self):
        blob = "\n".join(technical_operator_profile().defaults.values()).lower()
        for private in ("lovely", "jarvis", "loveble"):
            assert private not in blob

    def test_both_builtins_share_the_default_store_scope(self):
        # Switching between them keeps previously stored versions addressable.
        assert neutral_profile().name == technical_operator_profile().name == "default"


class TestProfileFiles:
    def test_toml_profile_with_inline_and_file_defaults(self, tmp_path: Path):
        (tmp_path / "policies.md").write_text("Refund within 30 days.\n", encoding="utf-8")
        toml = tmp_path / "profile.toml"
        toml.write_text(
            """
name = "support"

[[artifacts]]
name = "instructions"
role = "what the assistant is for"
required = true
default = "You help customers of the shop."

[[artifacts]]
name = "policies"
required = true
default_file = "policies.md"

[artifacts.policy]
assistant_edit = "none"
host_edit = true

[[artifacts]]
name = "notes"

[artifacts.policy]
assistant_edit = "autonomous"
assistant_activate = true
""",
            encoding="utf-8",
        )
        profile = load_profile_file(toml)
        assert profile.name == "support"
        assert profile.names == ("instructions", "policies", "notes")
        assert profile.defaults["policies"] == "Refund within 30 days."
        assert profile.get("policies").policy == ArtifactPolicy(
            assistant_edit="none", assistant_activate=False, host_edit=True
        )
        assert profile.get("notes").policy.assistant_activate is True

    def test_missing_file_is_a_clear_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Profile file not found"):
            load_profile_file(tmp_path / "absent.toml")

    def test_invalid_toml_is_a_clear_error(self, tmp_path: Path):
        toml = tmp_path / "profile.toml"
        toml.write_text("name = [unclosed", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid TOML"):
            load_profile_file(toml)

    def test_missing_default_file_is_a_clear_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="default_file not found"):
            profile_from_mapping(
                {"artifacts": [{"name": "a", "default_file": "gone.md"}]}, base_dir=tmp_path
            )

    @pytest.mark.parametrize(
        "data",
        [
            {"artifacts": "nope"},
            {"artifacts": [{"role": "no name"}]},
            {"artifacts": [{"name": "a", "policy": "text"}]},
            {"artifacts": [{"name": "a", "policy": {"assistant_edit": "sometimes"}}]},
        ],
    )
    def test_malformed_mappings_are_rejected(self, data):
        with pytest.raises(ValueError, match="artifact|policy|assistant_edit"):
            profile_from_mapping(data)


class TestResolvePrecedence:
    def test_nothing_configured_means_neutral(self):
        assert resolve_profile(None, None) == neutral_profile()

    @pytest.mark.parametrize("name", BUILTIN_PROFILE_NAMES)
    def test_setting_names_a_builtin(self, name):
        assert resolve_profile(name, None).names

    def test_setting_can_be_a_file_path(self, tmp_path: Path):
        toml = tmp_path / "p.toml"
        toml.write_text('name = "filed"\n[[artifacts]]\nname = "x"\n', encoding="utf-8")
        assert resolve_profile(str(toml), None).name == "filed"

    def test_definition_profile_wins_over_setting(self):
        own = AssistantProfile(name="own")
        assert resolve_profile("technical_operator", own) is own
