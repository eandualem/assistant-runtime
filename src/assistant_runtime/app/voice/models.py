"""Host-facing voice requests. Provider credentials and policy are never request fields."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from assistant_runtime.app.assistant.config import TunableOverrides
from assistant_runtime.host_context import HostContext


class VoiceHistoryMessage(BaseModel):
    """Visible conversation text supplied by the host, never system instructions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=7000)


class VoiceOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=64)
    sdp: str = Field(min_length=1, max_length=65536)
    host_context: HostContext | None = None
    config: TunableOverrides | None = None
    mode: Literal["delegated", "conversation"] | None = None
    instructions: str | None = Field(default=None, min_length=1, max_length=16000)
    profile: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    history: list[VoiceHistoryMessage] | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def bounded_history(self):
        if (
            self.history is not None
            and sum(len(item.content.encode("utf-8")) for item in self.history) > 7000
        ):
            raise ValueError("history must contain at most 7000 UTF-8 bytes of text")
        return self

    @field_validator("sdp", "session_id", "instructions")
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
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
