"""Offline compatibility checks for the Codex realtime protocol."""

import json
from pathlib import Path

REALTIME_VERSION = "v3"
# What this transport calls and listens for; a CLI without them is incompatible.
_REQUIRED_METHODS = {
    "ClientRequest.json": {
        "account/read",
        "thread/start",
        "thread/realtime/start",
        "thread/realtime/stop",
        "thread/realtime/appendSpeech",
        "thread/realtime/appendText",
        "turn/interrupt",
    },
    "ServerNotification.json": {
        "thread/realtime/sdp",
        "thread/realtime/started",
        "thread/realtime/closed",
        "thread/realtime/error",
        "thread/realtime/transcript/delta",
        "thread/realtime/transcript/done",
        "thread/realtime/itemAdded",
        "turn/started",
    },
}
_REQUIRED_THREAD_PARAMS = {"approvalPolicy", "cwd", "developerInstructions", "ephemeral", "sandbox"}
_REQUIRED_START_PARAMS = {
    "clientManagedHandoffs",
    "includeStartupContext",
    "initialItems",
    "outputModality",
    "prompt",
    "transport",
    "version",
    "voice",
}


def _methods(schema: dict) -> set[str]:
    methods = set()
    for variant in schema.get("oneOf", []):
        if not isinstance(variant, dict):
            continue
        properties = variant.get("properties")
        method = properties.get("method") if isinstance(properties, dict) else None
        if not isinstance(method, dict):
            continue
        # One unusual variant must not hide the others.
        names = method.get("enum") or ([method["const"]] if "const" in method else [])
        methods.update(name for name in names if isinstance(name, str))
    return methods


def missing_from_schema(directory: Path) -> list[str]:
    """Names this transport needs that the generated app-server schema lacks."""
    missing = []
    for file, needed in _REQUIRED_METHODS.items():
        try:
            present = _methods(json.loads((directory / file).read_text()))
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            present = set()
        missing += sorted(needed - present)
    try:
        start = json.loads((directory / "v2" / "ThreadRealtimeStartParams.json").read_text())
    except (OSError, ValueError):
        start = {}
    try:
        params = set(start.get("properties", {}))
    except (AttributeError, TypeError):
        params = set()
    try:
        versions = list(start["definitions"]["RealtimeConversationVersion"]["enum"])
    except (KeyError, TypeError):
        versions = []
    missing += [f"thread/realtime/start.{name}" for name in sorted(_REQUIRED_START_PARAMS - params)]
    if REALTIME_VERSION not in versions:
        missing.append(f"thread/realtime/start.version={REALTIME_VERSION}")
    try:
        started = json.loads(
            (directory / "v2" / "ThreadRealtimeStartedNotification.json").read_text()
        )
        started_fields = set(started.get("properties", {}))
    except (OSError, ValueError, TypeError, AttributeError):
        started_fields = set()
    if "version" not in started_fields:  # the session's version is checked on start
        missing.append("thread/realtime/started.version")
    for file, method, needed in (
        ("ThreadStartParams.json", "thread/start", _REQUIRED_THREAD_PARAMS),
        ("ThreadRealtimeAppendTextParams.json", "thread/realtime/appendText", {"role"}),
    ):
        try:
            fields = set(json.loads((directory / "v2" / file).read_text()).get("properties", {}))
        except (OSError, ValueError, AttributeError, TypeError):
            fields = set()
        missing += [f"{method}.{name}" for name in sorted(needed - fields)]
    return missing
