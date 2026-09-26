"""Errors safe to expose at the voice HTTP boundary."""

from typing import Literal


class VoiceError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 409,
        *,
        allocation_status: Literal["rejected", "unknown"] | None = None,
        provider_status_code: int | None = None,
        provider_request_id: str | None = None,
        reason: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.metadata = {
            key: value
            for key, value in {
                "allocation_status": allocation_status,
                "provider_status_code": provider_status_code,
                "provider_request_id": provider_request_id,
                "reason": reason,
            }.items()
            if value is not None
        }
