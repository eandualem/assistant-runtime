"""History service — conversation history management and context engineering."""

from lovely_assistant.services.history.config import HistoryConfig
from lovely_assistant.services.history.exceptions import (
    CompactionError,
    HistoryError,
    SummarizationError,
)
from lovely_assistant.services.history.interface import HistoryService
from lovely_assistant.services.history.models import (
    CompactionResult,
    MemoryDelta,
    MemoryDeltaResult,
    MemoryEntry,
    MemoryOperation,
    WorkingMemory,
)

__all__ = [
    "CompactionError",
    "CompactionResult",
    "HistoryConfig",
    "HistoryError",
    "HistoryService",
    "MemoryDelta",
    "MemoryDeltaResult",
    "MemoryEntry",
    "MemoryOperation",
    "SummarizationError",
    "WorkingMemory",
]
