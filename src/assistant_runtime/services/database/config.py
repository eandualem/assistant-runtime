"""Configuration for the database service module."""

from pydantic import BaseModel, ConfigDict, Field, computed_field


class DatabaseConfig(BaseModel):
    """Database module configuration. Nested into AppSettings as `database`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(default="localhost", description="Database hostname")
    port: int = Field(default=5434, description="Database port")
    user: str = Field(default="assistant_runtime", description="Database user")
    password: str = Field(default="assistant_runtime", description="Database password")
    name: str = Field(default="assistant_runtime", description="Database name")
    pool_size: int = Field(default=5, ge=1, le=20, description="Connection pool size")
    pool_overflow: int = Field(default=10, ge=0, le=40, description="Max pool overflow connections")
    echo: bool = Field(default=False, description="Echo SQL statements to log")

    @computed_field
    @property
    def async_url(self) -> str:
        """Async database URL for asyncpg driver."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        )

    @computed_field
    @property
    def sync_url(self) -> str:
        """Sync database URL for Alembic migrations."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
