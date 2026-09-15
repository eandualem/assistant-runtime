"""Assistant Runtime: an assistant backend that plugs into any work environment."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("assistant-runtime")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

RUNTIME_MARKER = "assistant-runtime"
"""The ``runtime`` value in this runtime's ``/health`` body; ``serve`` replaces only such a listener."""

__all__ = ["RUNTIME_MARKER", "__version__"]
