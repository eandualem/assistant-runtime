"""``assistant-runtime serve``: run the HTTP + Socket.IO server with uvicorn."""

from __future__ import annotations

import argparse
import os


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the ASGI app. Blocks until the server stops."""
    import uvicorn

    if args.verbose:
        os.environ.setdefault("LOG_LEVEL", "DEBUG")
    uvicorn.run(
        "assistant_runtime.main:app",
        host=args.host,
        port=args.port,
        reload=bool(args.reload),
        log_level="info",
    )
    return 0
