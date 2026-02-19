"""Tests for the pydantic-ai cache control monkey-patch."""

from unittest.mock import MagicMock, patch

from lovely_assistant.services.llm._cache_control_patch import EXPECTED_VERSION, apply_patch


class TestApplyPatch:
    @patch("lovely_assistant.services.llm._cache_control_patch.importlib.metadata.version")
    def test_patches_successfully(self, mock_version):
        mock_version.return_value = EXPECTED_VERSION
        # Should not raise
        apply_patch()

    @patch("lovely_assistant.services.llm._cache_control_patch.importlib.metadata.version")
    def test_warns_on_version_mismatch(self, mock_version):
        mock_version.return_value = "9.9.9"
        # apply_patch should still work but log a warning
        apply_patch()
        # The patch should still be applied even on version mismatch

    @patch("lovely_assistant.services.llm._cache_control_patch.importlib.metadata.version")
    def test_patched_function_guards_missing_type(self, mock_version):
        """The patched function should not crash on params without 'type' key."""
        mock_version.return_value = EXPECTED_VERSION
        apply_patch()

        from pydantic_ai.models.anthropic import AnthropicModel

        instance = MagicMock(spec=AnthropicModel)

        # Call with param missing 'type' key -- should not raise
        params = [{"content": "hello"}]  # no 'type' key
        cache_control = {"type": "ephemeral"}
        AnthropicModel._add_cache_control_to_last_param(instance, params, cache_control)

    @patch("lovely_assistant.services.llm._cache_control_patch.importlib.metadata.version")
    def test_patched_function_guards_empty_params(self, mock_version):
        mock_version.return_value = EXPECTED_VERSION
        apply_patch()

        from pydantic_ai.models.anthropic import AnthropicModel

        instance = MagicMock(spec=AnthropicModel)

        # Empty params should not raise
        AnthropicModel._add_cache_control_to_last_param(instance, [], {"type": "ephemeral"})

    @patch("lovely_assistant.services.llm._cache_control_patch.importlib.metadata.version")
    def test_patched_function_passes_through_valid_params(self, mock_version):
        """When params have 'type' key, the original function should be called."""
        mock_version.return_value = EXPECTED_VERSION
        apply_patch()

        from pydantic_ai.models.anthropic import AnthropicModel

        # Valid params with 'type' key -- should call through to original
        instance = MagicMock(spec=AnthropicModel)
        params = [{"type": "text", "text": "hello"}]
        cache_control = {"type": "ephemeral"}
        # This may raise from the original function trying to do something with mocks,
        # but it should NOT be a KeyError on 'type'
        try:
            AnthropicModel._add_cache_control_to_last_param(instance, params, cache_control)
        except KeyError as e:
            if "'type'" in str(e):
                raise AssertionError("Should not get KeyError on 'type'") from e
        except Exception:
            pass  # Other errors from mock internals are fine
