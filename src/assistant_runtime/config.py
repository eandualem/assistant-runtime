"""Root application settings. Composes all module configs."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from assistant_runtime.app.assistant.config import AssistantConfig
from assistant_runtime.app.heartbeat.config import HeartbeatConfig
from assistant_runtime.app.streaming.config import StreamingConfig
from assistant_runtime.services.database.config import DatabaseConfig
from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.llm.config import LLMConfig
from assistant_runtime.services.media.config import MediaConfig
from assistant_runtime.services.oauth.config import OAuthConfig
from assistant_runtime.services.tools.config import ToolConfig


class AppSettings(BaseSettings):
    """Root application settings. Reads from environment variables and .env file."""

    app_name: str = "assistant-runtime"
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool = False

    database: DatabaseConfig = DatabaseConfig()
    llm: LLMConfig = LLMConfig()
    history: HistoryConfig = HistoryConfig()
    media: MediaConfig = MediaConfig()
    oauth: OAuthConfig = OAuthConfig()
    tools: ToolConfig = ToolConfig()
    assistant: AssistantConfig = AssistantConfig()
    heartbeat: HeartbeatConfig = HeartbeatConfig()
    streaming: StreamingConfig = StreamingConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )
