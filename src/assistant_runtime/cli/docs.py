"""``assistant-runtime docs [page]`` and its alias ``help``: the shipped documentation."""

from __future__ import annotations

import argparse

DOCS_URL = "https://github.com/eandualem/assistant-runtime/tree/main/docs"


def cmd_docs(args: argparse.Namespace) -> int:
    """List the pages, or print one."""
    from assistant_runtime.help import get_doc, list_docs

    pages = list_docs()
    if not pages:
        print(f"no documentation shipped with this install; read it at {DOCS_URL}")
        return 1
    page = getattr(args, "page", None)
    if not page:
        print("assistant-runtime documentation; `assistant-runtime docs <page>` prints one page:\n")
        for entry in pages:
            print(f"  {entry['name']:<18s} {entry['summary']}")
        return 0
    content = get_doc(page)
    if content is None:
        known = ", ".join(entry["name"] for entry in pages)
        print(f"unknown page '{page}'; try: {known}")
        return 1
    print(content, end="" if content.endswith("\n") else "\n")
    return 0
