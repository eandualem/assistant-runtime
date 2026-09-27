"""What an agent received: a compact record of a turn's system prompt.

The stable part of a prompt (the profile's artifacts, included profiles,
kept documents, MCP connections) repeats from turn to turn, so it is stored
once as a snapshot keyed by its hash. The record on each assistant message
names that snapshot and carries the rest inline: the profile and subject,
the artifact versions the text came from, the dynamic fragments (time,
host context, working memory) and any instruction appended after them.
``prompt_text`` rebuilds the exact text the model received.
"""

from __future__ import annotations

import hashlib
from typing import Any

from assistant_runtime.app.assistant.models import PromptResult

DYNAMIC_FRAGMENTS = frozenset({"datetime", "host_context", "working_memory"})
_SEPARATOR = "\n\n"


def prompt_record(
    prompt: PromptResult,
    *,
    profile: str,
    subject: str | None,
    artifact_versions: dict[str, int | None],
) -> tuple[dict[str, Any], str]:
    """The record for a message, and the stable snapshot text it points at."""
    stable = _SEPARATOR.join(
        f["content"] for f in prompt.fragments if f["name"] not in DYNAMIC_FRAGMENTS
    )
    dynamic = [
        [f["name"], f["content"]] for f in prompt.fragments if f["name"] in DYNAMIC_FRAGMENTS
    ]
    joined = _join(stable, dynamic)
    suffix = prompt.content[len(joined) :] if prompt.content.startswith(joined) else ""
    record = {
        "profile": profile,
        "subject": subject,
        "artifact_versions": artifact_versions,
        "snapshot_hash": snapshot_hash(stable),
        "dynamic": dynamic,
        "suffix": suffix,
        "fragments": [{"name": f["name"], "chars": f["char_count"]} for f in prompt.fragments],
    }
    return record, stable


def prompt_text(record: dict[str, Any], snapshot: str) -> str:
    """The exact system prompt a record describes, given its snapshot text."""
    return _join(snapshot, record.get("dynamic") or []) + (record.get("suffix") or "")


def snapshot_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _join(stable: str, dynamic: list[list[str]]) -> str:
    return _SEPARATOR.join(([stable] if stable else []) + [content for _, content in dynamic])
