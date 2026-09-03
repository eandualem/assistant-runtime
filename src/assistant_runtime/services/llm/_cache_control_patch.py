"""Defensive patch for pydantic-ai AnthropicModel._add_cache_control_to_last_param.

The original method does a bare `last_param['type']` access which KeyErrors on
malformed params (missing 'type' key). This patch adds a guard.
"""

from __future__ import annotations

import functools
import importlib.metadata

from loguru import logger

EXPECTED_VERSION = "2.38.0"


def apply_patch() -> None:
    """Apply the defensive cache control patch to pydantic-ai's AnthropicModel."""
    version = importlib.metadata.version("pydantic-ai-slim")
    if version != EXPECTED_VERSION:
        logger.warning(
            "pydantic-ai version changed — check if cache control patch still needed",
            installed=version,
            expected=EXPECTED_VERSION,
        )

    try:
        from pydantic_ai.models.anthropic import AnthropicModel
    except ImportError:
        logger.debug("pydantic-ai anthropic model not available, skipping cache control patch")
        return

    original = AnthropicModel._add_cache_control_to_last_param

    @functools.wraps(original)
    def patched(self, params, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not params:
            return None
        last_param = params[-1]
        if not isinstance(last_param, dict) or "type" not in last_param:
            logger.debug("Skipping cache control — last param has no 'type' key")
            return None
        return original(self, params, *args, **kwargs)

    AnthropicModel._add_cache_control_to_last_param = patched  # type: ignore[assignment]
    logger.debug("Applied cache control patch for pydantic-ai")
