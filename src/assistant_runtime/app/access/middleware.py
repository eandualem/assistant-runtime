"""Browser origin enforcement for requests that can change runtime state."""

from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send


class BrowserOriginMiddleware(CORSMiddleware):
    """Apply the CORS allowlist before simple requests can cause side effects.

    CORS alone only prevents a browser from reading a disallowed response;
    bodyless and form POSTs can still reach the application without preflight.
    Calls without an Origin remain available to non-browser hosts.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
            origin = Headers(scope=scope).get("origin")
            if origin is not None and not self.is_allowed_origin(origin):
                response = JSONResponse(
                    {"detail": "Browser origin is not allowed"}, status_code=403
                )
                await response(scope, receive, send)
                return
        await super().__call__(scope, receive, send)
