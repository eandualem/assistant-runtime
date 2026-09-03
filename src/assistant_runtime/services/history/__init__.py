"""History service — conversation history management and context engineering."""

from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.history.exceptions import (
    CompactionError,
    HistoryError,
    SummarizationError,
)
from assistant_runtime.services.history.interface import HistoryService
from assistant_runtime.services.history.models import (
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
