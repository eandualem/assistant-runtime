"""Configuration for the database service module."""

from pydantic import BaseModel, ConfigDict, Field, computed_field
from sqlalchemy.engine import URL


class DatabaseConfig(BaseModel):
    """Database module configuration. Nested into AppSettings as `database`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(
        default="localhost",
        description="Database hostname, or a Unix-socket directory when it starts with '/'",
    )
    port: int = Field(default=5434, description="Database port")
    user: str = Field(default="assistant_runtime", description="Database user")
    password: str = Field(
        default="assistant_runtime",
        description="Database password; empty for trust or peer authentication",
    )
    name: str = Field(default="assistant_runtime", description="Database name")
    pool_size: int = Field(default=5, ge=1, le=20, description="Connection pool size")
    pool_overflow: int = Field(default=10, ge=0, le=40, description="Max pool overflow connections")
    echo: bool = Field(default=False, description="Echo SQL statements to log")
    required: bool = Field(
        default=False,
        description="Fail startup when Postgres is unreachable instead of keeping state in memory",
    )
    connect_timeout: float = Field(
        default=5.0,
        gt=0,
        description=(
            "Seconds to wait for a new database connection (asyncpg's connect timeout). "
            "Bounds a call against a Postgres that drops packets instead of refusing "
            "connections, instead of asyncpg's 60-second default. Must be positive: "
            "asyncpg reads 0 as an immediate timeout."
        ),
    )
    migrate_on_start: bool = Field(
        default=False,
        description="Upgrade the schema to the packaged head before the services start",
    )

    def url(self, drivername: str = "postgresql+asyncpg") -> URL:
        """The connection URL, escaped; a socket directory goes in the query."""
        socket = self.host.startswith("/")
        return URL.create(
            drivername,
            username=self.user,
            password=self.password or None,
            host=None if socket else self.host,
            port=self.port,
            database=self.name,
            query={"host": self.host} if socket else {},
        )

    @computed_field
    @property
    def async_url(self) -> str:
        """Async database URL for asyncpg driver."""
        return self.url().render_as_string(hide_password=False)

    @computed_field
    @property
    def sync_url(self) -> str:
        """Sync database URL for Alembic migrations."""
        return self.url("postgresql").render_as_string(hide_password=False)
