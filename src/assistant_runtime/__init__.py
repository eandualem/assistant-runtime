"""Assistant Runtime: an assistant backend that plugs into any work environment."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("assistant-runtime")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
