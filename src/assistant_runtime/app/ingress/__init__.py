"""Ingress module — messages from other systems delivered into sessions."""

from assistant_runtime.app.ingress.deps import IngressServiceDep
from assistant_runtime.app.ingress.exceptions import IngressError
from assistant_runtime.app.ingress.interface import IngressService

__all__ = ["IngressError", "IngressService", "IngressServiceDep"]
