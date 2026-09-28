"""Data types of the host state module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class HostStateEntry:
    """One value under a namespace and key; ``version`` rises with each write."""

    namespace: str
    key: str
    value: Any
    version: int
    updated_by: str
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "key": self.key,
            "value": self.value,
            "version": self.version,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
