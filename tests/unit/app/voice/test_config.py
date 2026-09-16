import pytest


class TestInstructionFiles:
    def test_files_replace_the_inline_instructions(self, tmp_path):
        from assistant_runtime.app.voice.config import VoiceConfig

        live = tmp_path / "live.md"
        live.write_text("You are the voice of the application.\n", encoding="utf-8")
        config = VoiceConfig(conversation_instructions_file=str(live))
        assert config.conversation_instructions == "You are the voice of the application."
        assert config.instructions.startswith("You are a helpful voice assistant")

    def test_a_missing_or_empty_file_fails_startup(self, tmp_path):
        from pydantic import ValidationError

        from assistant_runtime.app.voice.config import VoiceConfig

        with pytest.raises(ValidationError, match="cannot be read"):
            VoiceConfig(instructions_file=str(tmp_path / "missing.md"))
        empty = tmp_path / "empty.md"
        empty.write_text("  \n", encoding="utf-8")
        with pytest.raises(ValidationError, match="1 to 16000"):
            VoiceConfig(instructions_file=str(empty))
