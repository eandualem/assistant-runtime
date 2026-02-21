"""LLM-backed conversation summarization and working memory extraction.

Internal module — only accessed through HistoryService (interface.py).
Translated from arclio-assistant's history/summarizer.py with key differences:
- Takes LlmService instead of creating Agent directly (protocol boundary)
- Prompts co-located in prompts/ subdirectory
- No StructuredLogger — uses loguru directly
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from lovely_assistant.services.history.config import HistoryConfig
from lovely_assistant.services.history.models import (
    CompactionResult,
    MemoryDeltaResult,
    WorkingMemory,
)

if TYPE_CHECKING:
    from lovely_assistant.services.llm.interface import LlmService


def _load_prompt_file(prompt_name: str) -> tuple[str, str]:
    """Load prompts from external file.

    Returns:
        Tuple of (user_prompt_template, system_prompt)
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        f"{prompt_name}.txt",
    )

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(
            f"Required prompt file not found: {prompt_path}. "
            "Prompt files must exist — fallback prompts are not allowed."
        )

    with open(prompt_path) as f:
        content = f.read()

    system_marker = "## System Prompt"
    user_marker = "## User Prompt Template"

    system_idx = content.find(system_marker)
    user_idx = content.find(user_marker)

    if system_idx == -1 or user_idx == -1:
        raise ValueError(
            f"Malformed prompt file: {prompt_path}. "
            "Required sections '## System Prompt' and '## User Prompt Template' not found."
        )

    system_section = content[system_idx + len(system_marker) : user_idx].strip()
    user_section = content[user_idx + len(user_marker) :].strip()

    if not system_section or not user_section:
        raise ValueError(
            f"Malformed prompt file: {prompt_path}. "
            "System prompt and user prompt sections must not be empty."
        )

    return user_section, system_section


# Load prompts at module level
_STRUCTURED_SUMMARIZE_PROMPT, _STRUCTURED_SUMMARIZE_SYSTEM = _load_prompt_file(
    "structured_summarization_prompt"
)
_DELTA_EXTRACTION_PROMPT, _DELTA_EXTRACTION_SYSTEM = _load_prompt_file(
    "delta_memory_extraction_prompt"
)


class HistorySummarizer:
    """LLM-backed summarization and working memory delta extraction.

    Uses LlmService.build_agent() for all LLM calls, maintaining the protocol
    boundary: this module doesn't know about providers or model configuration.
    """

    STRUCTURED_SUMMARIZE_PROMPT = _STRUCTURED_SUMMARIZE_PROMPT
    STRUCTURED_SUMMARIZE_SYSTEM = _STRUCTURED_SUMMARIZE_SYSTEM
    DELTA_EXTRACTION_PROMPT = _DELTA_EXTRACTION_PROMPT
    DELTA_EXTRACTION_SYSTEM = _DELTA_EXTRACTION_SYSTEM

    def __init__(
        self,
        config: HistoryConfig,
        llm_service: LlmService,
        runtime_settings: Any | None = None,
    ) -> None:
        self.config = config
        self._llm = llm_service
        self._runtime_settings = runtime_settings

    def _format_messages_for_summarization(self, messages: list[dict[str, Any]]) -> str:
        """Format dict-format messages into a string for the summarization prompt."""
        truncation_limit = self.config.message_truncation_limit
        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > truncation_limit:
                content = content[:truncation_limit] + "... [truncated]"
            formatted_parts.append(f"[{role}]: {content}")

        return "\n\n".join(formatted_parts)

    async def summarize_structured(
        self,
        messages: list[dict[str, Any]],
        existing_summary: str | None = None,
    ) -> CompactionResult:
        """Summarize messages into a structured CompactionResult using LLM."""
        if not messages:
            return CompactionResult(
                summary=existing_summary or "",
                working_memory=WorkingMemory(),
            )

        start_time = time.time()

        formatted_messages = self._format_messages_for_summarization(messages)

        existing_summary_section = ""
        if existing_summary:
            existing_summary_section = f"Previous summary to incorporate:\n{existing_summary}\n"

        prompt = self.STRUCTURED_SUMMARIZE_PROMPT.format(
            existing_summary_section=existing_summary_section,
            messages=formatted_messages,
        )

        model = (
            self._runtime_settings.get("summarization_model", self.config.summarization_model)
            if self._runtime_settings
            else self.config.summarization_model
        )

        try:
            agent = self._llm.build_agent(
                system_prompt=self.STRUCTURED_SUMMARIZE_SYSTEM,
                output_type=CompactionResult,
                **({"model": model} if model else {}),
            )

            result = await agent.run(prompt)
            compaction_result = result.output

            duration_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"[HISTORY] Structured summarization of {len(messages)} messages "
                f"in {duration_ms:.0f}ms"
            )

            return compaction_result

        except Exception as e:
            logger.error(f"[HISTORY] Structured summarization failed: {e}")
            return self._create_fallback_compaction_result(messages, existing_summary)

    async def extract_memory_delta(
        self,
        current_wm: WorkingMemory,
        recent_messages: list[dict[str, Any]],
        turn_number: int,
    ) -> WorkingMemory:
        """Extract working memory updates using delta-based approach.

        On failure, returns current_wm unchanged (safe fallback).
        """
        if not recent_messages:
            return current_wm

        start_time = time.time()

        formatted_messages = self._format_messages_for_summarization(recent_messages)

        entries_json = json.dumps(
            [e.model_dump() for e in current_wm.entries],
            indent=2,
        )

        prompt = self.DELTA_EXTRACTION_PROMPT.format(
            current_entries_json=entries_json,
            formatted_messages=formatted_messages,
            turn_number=turn_number,
        )

        wm_model = (
            self._runtime_settings.get("working_memory_model", self.config.working_memory_model)
            if self._runtime_settings
            else self.config.working_memory_model
        )
        sm_model = (
            self._runtime_settings.get("summarization_model", self.config.summarization_model)
            if self._runtime_settings
            else self.config.summarization_model
        )
        model = wm_model or sm_model

        agent = self._llm.build_agent(
            system_prompt=self.DELTA_EXTRACTION_SYSTEM,
            output_type=MemoryDeltaResult,
            **({"model": model} if model else {}),
        )

        max_attempts = 2
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                result = await agent.run(prompt)
                delta_result = result.output

                updated_wm = WorkingMemory(
                    active_goal=delta_result.active_goal,
                    progress=delta_result.progress,
                    next_steps=delta_result.next_steps,
                    key_decisions=delta_result.key_decisions,
                    open_loops=delta_result.open_loops,
                    entries=[e.model_copy() for e in current_wm.entries],
                    turn_number=turn_number,
                )

                updated_wm.apply_deltas(
                    delta_result.deltas, turn_number, max_entries=self.config.max_memory_entries
                )

                duration_ms = (time.time() - start_time) * 1000

                delta_counts = {"add": 0, "update": 0, "delete": 0}
                for delta in delta_result.deltas:
                    if delta.operation.value in delta_counts:
                        delta_counts[delta.operation.value] += 1

                logger.info(
                    f"[WORKING_MEMORY] Delta extraction in {duration_ms:.0f}ms: "
                    f"{len(delta_result.deltas)} operations "
                    f"(+{delta_counts['add']} ~{delta_counts['update']} -{delta_counts['delete']}), "
                    f"{len(updated_wm.entries)} total entries"
                )

                return updated_wm

            except Exception as e:
                last_error = e
                if attempt < max_attempts:
                    logger.warning(
                        f"[WORKING_MEMORY] Delta extraction attempt {attempt} failed, retrying: {e}"
                    )
                    continue

        logger.warning(
            f"[WORKING_MEMORY] Delta extraction failed, keeping current WM: {last_error}"
        )
        return current_wm

    def _create_fallback_summary(self, messages: list[dict[str, Any]]) -> str:
        """Create a basic summary without LLM when summarization fails."""
        user_count = 0
        assistant_count = 0
        first_user_content: str | None = None

        for m in messages:
            role = m.get("role", "")
            if role == "user":
                user_count += 1
                if first_user_content is None:
                    content = m.get("content", "")
                    if isinstance(content, str) and content:
                        first_user_content = content
            elif role == "assistant":
                assistant_count += 1

        summary_parts = [
            f"Previous conversation: {len(messages)} messages",
            f"({user_count} user, {assistant_count} assistant)",
        ]

        if first_user_content:
            summary_parts.append(f"Started with: {first_user_content[:200]}...")

        return " ".join(summary_parts)

    def _create_fallback_compaction_result(
        self,
        messages: list[dict[str, Any]],
        existing_summary: str | None = None,
    ) -> CompactionResult:
        """Create a fallback CompactionResult when LLM fails."""
        fallback_summary = self._create_fallback_summary(messages)
        if existing_summary:
            fallback_summary = f"{existing_summary}\n\n{fallback_summary}"

        first_goal = ""
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    first_goal = content[:200]
                break

        return CompactionResult(
            summary=fallback_summary,
            user_goal=first_goal,
            current_state="Unknown (summarization failed)",
            working_memory=WorkingMemory(
                active_goal=first_goal,
            ),
        )
