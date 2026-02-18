"""Root application settings. Composes all module configs."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from lovely_assistant.app.assistant.config import AssistantConfig
from lovely_assistant.app.streaming.config import StreamingConfig
from lovely_assistant.services.history.config import HistoryConfig
from lovely_assistant.services.llm.config import LLMConfig
from lovely_assistant.services.tools.config import ToolConfig


class AppSettings(BaseSettings):
    """Root application settings. Reads from environment variables and .env file."""

    app_name: str = "lovely-assistant"
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool = False

    llm: LLMConfig = LLMConfig()
    history: HistoryConfig = HistoryConfig()
    tools: ToolConfig = ToolConfig()
    assistant: AssistantConfig = AssistantConfig()
    streaming: StreamingConfig = StreamingConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )
