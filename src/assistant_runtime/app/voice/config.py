"""Startup policy for voice sessions; independent of backend model routing."""

from pydantic import BaseModel, ConfigDict, Field


class VoiceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    delegation_enabled: bool = True
    model: str = Field(default="gpt-live-1", min_length=1, max_length=128)
    voice: str = Field(default="marin", min_length=1, max_length=128)
    api_key_env: str = Field(default="OPENAI_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")
    instructions: str = Field(
        default=(
            "You are a helpful voice assistant. Keep spoken replies concise. "
            "Delegate tasks, factual questions, and application actions to the backend. "
            "Do not claim an action succeeded until the backend confirms it. "
            "You can continue listening while the backend works."
        ),
        min_length=1,
        max_length=16000,
    )
    conversation_instructions: str = Field(
        default="You are a helpful voice assistant. Keep spoken replies concise.",
        min_length=1,
        max_length=16000,
    )
    max_sessions: int = Field(default=4, ge=1, le=100)
    max_duration_seconds: int = Field(default=1800, ge=15, le=7200)
    connect_timeout_seconds: float = Field(default=20, gt=0, le=60)
    close_timeout_seconds: float = Field(default=10, gt=0, le=60)
    max_transcript_chars: int = Field(default=200000, ge=1000, le=1000000)
    context_chars: int = Field(default=16000, ge=1000, le=24000)
    retained_calls: int = Field(default=100, ge=1, le=1000)
    event_buffer_size: int = Field(default=512, ge=16, le=4096)
