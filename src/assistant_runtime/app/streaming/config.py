"""Configuration for the streaming module."""

from pydantic import BaseModel, ConfigDict, Field


class StreamingConfig(BaseModel):
    """Streaming module configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_events_per_stream: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Safety limit on events per stream.",
    )
    stream_timeout_seconds: float = Field(
        default=300.0,
        ge=10.0,
        le=600.0,
        description="Maximum time for a single streaming request.",
    )
    emit_debug_events: bool = Field(
        default=False,
        description=(
            "Emit assistant:debug events exposing the system prompt, history and tool "
            "selection. Off by default: any client of the unauthenticated socket could read them."
        ),
    )
    part_start_chunk_size: int = Field(
        default=100,
        ge=20,
        le=2000,
        description="Max chars per event when chunking large PartStartEvent content.",
    )
    part_start_chunk_threshold: int = Field(
        default=200,
        ge=50,
        le=5000,
        description="PartStartEvent content above this length gets chunked.",
    )
