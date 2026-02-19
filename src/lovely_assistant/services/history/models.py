"""Context engineering models — working memory, compaction, and memory deltas.

Adapted from arclio-assistant's models/context_engineering.py for the agent
operations domain. Drops OperationalLearning and learnings migration (greenfield).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class HistoryPreparationResult:
    """Debug metadata from history preparation."""

    history: list[Any] = field(default_factory=list)
    was_compacted: bool = False
    message_count: int = 0
    estimated_tokens: int = 0
    compacted_from: int = 0
    message_summaries: list[dict[str, Any]] = field(default_factory=list)


MEMORY_CATEGORIES = [
    "agent_behavior",
    "backbone_pattern",
    "tool_behavior",
    "workflow_pattern",
    "user_preference",
    "system_quirk",
]


class MemoryOperation(StrEnum):
    """Delta operation types for memory management."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"


class MemoryEntry(BaseModel):
    """A single memory entry with UUID for delta tracking."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(description="The memory content/insight")
    category: str = Field(
        description=(
            "One of: agent_behavior, backbone_pattern, tool_behavior, "
            "workflow_pattern, user_preference, system_quirk"
        ),
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="0.0-1.0, decreases when contradicted, increases when confirmed",
    )
    evidence: str = Field(default="", description="The observation that produced this entry")
    created_turn: int = Field(default=0, description="Turn number when created")
    updated_turn: int = Field(default=0, description="Turn number when last updated")


class MemoryDelta(BaseModel):
    """A single delta operation on memory."""

    operation: MemoryOperation
    entry_id: str | None = Field(
        default=None, description="ID of entry to UPDATE/DELETE, None for ADD"
    )
    entry: MemoryEntry | None = Field(default=None, description="New/updated entry for ADD/UPDATE")
    reason: str = Field(default="", description="Why this operation was chosen")


class MemoryDeltaResult(BaseModel):
    """LLM output: updated task state plus delta operations."""

    active_goal: str = ""
    progress: str = ""
    next_steps: str = ""
    key_decisions: list[str] = Field(default_factory=list)
    open_loops: list[str] = Field(default_factory=list)
    deltas: list[MemoryDelta] = Field(default_factory=list)


class WorkingMemory(BaseModel):
    """Structured state persisted in session context across turns.

    Contains two layers:
    - Task state (ephemeral): tracks the current goal, progress, decisions
    - Memory entries (durable): accumulates reusable insights via delta operations
    """

    active_goal: str = Field(default="", description="The user's current primary goal")
    progress: str = Field(default="", description="What has been accomplished so far")
    next_steps: str = Field(default="", description="What should happen next")
    key_decisions: list[str] = Field(
        default_factory=list, description="Important decisions made during the conversation"
    )
    open_loops: list[str] = Field(
        default_factory=list, description="Started but unfinished items or unanswered questions"
    )
    entries: list[MemoryEntry] = Field(
        default_factory=list,
        description="Durable memory entries with IDs for delta updates",
    )
    turn_number: int = Field(default=0, description="Current turn number for tracking entry ages")

    def apply_deltas(
        self, deltas: list[MemoryDelta], current_turn: int, max_entries: int = 15
    ) -> None:
        """Apply delta operations to memory entries."""
        for delta in deltas:
            if delta.operation == MemoryOperation.ADD and delta.entry:
                new_entry = delta.entry.model_copy()
                new_entry.id = str(uuid4())
                new_entry.created_turn = current_turn
                new_entry.updated_turn = current_turn
                self.entries.append(new_entry)

            elif delta.operation == MemoryOperation.UPDATE and delta.entry_id and delta.entry:
                for i, entry in enumerate(self.entries):
                    if entry.id == delta.entry_id:
                        updated_entry = delta.entry.model_copy()
                        updated_entry.id = entry.id
                        updated_entry.created_turn = entry.created_turn
                        updated_entry.updated_turn = current_turn
                        self.entries[i] = updated_entry
                        break

            elif delta.operation == MemoryOperation.DELETE and delta.entry_id:
                self.entries = [e for e in self.entries if e.id != delta.entry_id]

        if len(self.entries) > max_entries:
            self.entries.sort(key=lambda e: e.confidence, reverse=True)
            self.entries = self.entries[:max_entries]

    def is_empty(self) -> bool:
        """Check if working memory has any meaningful content."""
        return (
            not self.active_goal
            and not self.progress
            and not self.next_steps
            and not self.key_decisions
            and not self.open_loops
            and not self.entries
        )

    def to_prompt_section(self) -> str:
        """Format working memory for injection into the system prompt."""
        if self.is_empty():
            return "(No working memory from previous turns)"

        lines: list[str] = []

        if self.active_goal:
            lines.append(f"**Active Goal:** {self.active_goal}")
        if self.progress:
            lines.append(f"**Progress:** {self.progress}")
        if self.next_steps:
            lines.append(f"**Next Steps:** {self.next_steps}")
        if self.key_decisions:
            lines.append("**Key Decisions:**")
            for decision in self.key_decisions:
                lines.append(f"- {decision}")
        if self.open_loops:
            lines.append("**Open Loops:**")
            for loop in self.open_loops:
                lines.append(f"- {loop}")

        if self.entries:
            lines.append("")
            lines.append("**Operational Learnings:**")
            for entry in self.entries:
                lines.append(
                    f"- [{entry.category}] {entry.content} (confidence: {entry.confidence:.1f})"
                )

        return "\n".join(lines)


class CompactionResult(BaseModel):
    """Structured output from conversation compaction."""

    summary: str = Field(description="Narrative summary of the conversation (max 300 words)")
    user_goal: str = Field(default="", description="The user's primary goal or intent")
    current_state: str = Field(default="", description="Current state of the conversation/task")
    actions_taken: list[str] = Field(
        default_factory=list, description="Actions taken during the conversation"
    )
    errors_encountered: list[str] = Field(
        default_factory=list, description="Errors encountered and their resolution status"
    )
    working_memory: WorkingMemory = Field(
        default_factory=WorkingMemory,
        description="Structured working memory for the agent",
    )

    def to_summary_message(self) -> str:
        """Format the compaction result for storage as a history summary message."""
        parts: list[str] = [self.summary]

        if self.user_goal:
            parts.append(f"\n**Goal:** {self.user_goal}")
        if self.current_state:
            parts.append(f"**State:** {self.current_state}")
        if self.actions_taken:
            parts.append("**Actions:** " + "; ".join(self.actions_taken))
        if self.errors_encountered:
            parts.append("**Errors:** " + "; ".join(self.errors_encountered))

        return "\n".join(parts)
