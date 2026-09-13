"""Host-facing voice requests. Provider credentials and policy are never request fields."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from assistant_runtime.app.assistant.config import TunableOverrides
from assistant_runtime.host_context import HostContext


class VoiceOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=64)
    sdp: str = Field(min_length=1, max_length=65536)
    host_context: HostContext | None = None
    config: TunableOverrides | None = None

    @field_validator("sdp", "session_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class VoiceContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host_context: HostContext


class VoiceToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(min_length=1, max_length=256)
    tool_result: Any
    tool_outcome: Literal["success", "failed"] = "success"
