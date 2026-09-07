"""``assistant-runtime`` command-line interface.

assistant-runtime chat [--model M]    talk to the assistant in the terminal (no server, no database)
assistant-runtime serve [--port P]    run the HTTP + Socket.IO server (replaces a previous one on that port)
assistant-runtime doctor              check provider keys, model ids, database, optional extras
assistant-runtime migrate             create or update the Postgres schema
assistant-runtime docs [page]         the documentation shipped with this install
assistant-runtime help [page]         the same pages (alias of docs)
assistant-runtime version             print the installed version

``chat`` runs the runtime in-process: the same services, tools and prompt as
the server, streamed to the terminal instead of a Socket.IO client. Only a
provider key is required; Postgres is optional (without it sessions live in
memory and the bundled default prompt artifacts are used).
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from dotenv import find_dotenv, load_dotenv

from assistant_runtime.cli.chat import cmd_chat
from assistant_runtime.cli.docs import cmd_docs
from assistant_runtime.cli.doctor import cmd_doctor
from assistant_runtime.cli.migrate import cmd_migrate
from assistant_runtime.cli.serve import cmd_serve

DEFAULT_PORT = 7100


def package_version() -> str:
    """The installed package version, or ``unknown`` in a source-only checkout."""
    try:
        return version("assistant-runtime")
    except PackageNotFoundError:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``assistant-runtime`` command."""
    parser = argparse.ArgumentParser(
        prog="assistant-runtime",
        description="Assistant Runtime: an assistant backend that plugs into any work environment.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show INFO-level logs")
    parser.add_argument(
        "--version",
        action="version",
        version=package_version(),
        help="print the installed version and exit",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    p = sub.add_parser("chat", help="talk to the assistant in the terminal")
    p.add_argument("--model", help="model id for this chat, e.g. anthropic:claude-sonnet-5")
    p.add_argument("--session", help="session id to resume or create (default: a new one)")
    p.add_argument("--show-thinking", action="store_true", help="print the model's thinking stream")
    p.add_argument("--message", "-m", help="send one message, print the reply, and exit")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("serve", help="run the HTTP + Socket.IO server")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default: {DEFAULT_PORT})")
    p.add_argument("--reload", action="store_true", help="restart on source changes")
    p.add_argument(
        "--no-replace",
        action="store_true",
        help="fail if the port is busy instead of replacing a previous assistant-runtime there",
    )
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="check the environment and configuration")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("migrate", help="create or update the Postgres schema")
    p.add_argument("--revision", default="head", help="target revision (default: head)")
    p.set_defaults(func=cmd_migrate)

    for name, help_text in (
        ("docs", "print the documentation shipped with this install"),
        ("help", "same as docs"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("page", nargs="?", help="page name; omit to list the pages")
        p.set_defaults(func=cmd_docs)

    p = sub.add_parser("version", help="print the installed version")
    p.set_defaults(func=lambda _args: (print(package_version()), 0)[1])

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``assistant-runtime`` console script."""
    # Before anything reads settings: ``AppSettings`` resolves its own
    # ``.env`` against the working directory only, so a run from a nested
    # directory would otherwise see the provider keys but not the nested
    # ``SECTION__FIELD`` values.
    load_dotenv(find_dotenv(usecwd=True))
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
