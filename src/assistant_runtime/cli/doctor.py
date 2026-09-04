"""``assistant-runtime doctor``: report what is configured and what is reachable."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from assistant_runtime.model_catalog import PROVIDER_ENV_VARS

OK, WARN, FAIL = "ok  ", "warn", "FAIL"

Line = tuple[str, str]  # (status, message)


def _python() -> Line:
    v = sys.version_info
    status = OK if (v.major, v.minor) >= (3, 12) else FAIL
    return status, f"python {v.major}.{v.minor}.{v.micro} (3.12+ required)"


def _env_file() -> Line:
    path = Path(".env")
    if path.is_file():
        return OK, f".env found at {path.resolve()}"
    return WARN, "no .env in the current directory (environment variables are used as-is)"


def configured_providers(env: dict[str, str] | None = None) -> list[str]:
    """Provider prefixes that have an API key set."""
    source = os.environ if env is None else env
    found = [provider for provider, var in PROVIDER_ENV_VARS.items() if source.get(var, "").strip()]
    if source.get("LLM__PROVIDERS_JSON", "").strip():
        found.append("providers-json")
    return found


def _providers() -> Line:
    found = configured_providers()
    if found:
        return OK, "provider keys: " + ", ".join(found)
    return FAIL, "no provider key set (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, ...)"


def _models() -> list[Line]:
    from assistant_runtime.config import AppSettings
    from assistant_runtime.services.llm._settings import validate_model_id

    config = AppSettings().llm
    providers = set(configured_providers())
    lines: list[Line] = []
    for label, model_id in (
        ("primary model", config.primary_model),
        ("summarization model", config.summarization_model),
    ):
        try:
            validate_model_id(model_id)
        except Exception as exc:
            lines.append((FAIL, f"{label} {model_id}: {exc}"))
            continue
        prefix = model_id.split(":", 1)[0]
        if providers and prefix not in providers and "providers-json" not in providers:
            lines.append((WARN, f"{label} {model_id}: no key for provider '{prefix}'"))
        else:
            lines.append((OK, f"{label} {model_id}"))
    return lines


async def _database_async() -> Line:
    from assistant_runtime.config import AppSettings
    from assistant_runtime.services.database.interface import DatabaseService

    config = AppSettings().database
    service = DatabaseService(config=config)
    try:
        await asyncio.wait_for(service.start(), timeout=5.0)
        healthy = service.healthy
    except Exception:
        healthy = False
    finally:
        with contextlib.suppress(Exception):
            await service.stop()
    where = f"{config.host}:{config.port}/{config.name}"
    if healthy:
        return OK, f"postgres reachable at {where}"
    return WARN, f"postgres not reachable at {where} (sessions stay in memory)"


def _database() -> Line:
    return asyncio.run(_database_async())


def _codex() -> Line:
    """The ChatGPT/Codex subscription path: enabled by the encryption key, fed by a login."""
    from assistant_runtime.config import AppSettings

    config = AppSettings().oauth
    if not config.encryption_key:
        return OK, "chatgpt/codex subscription auth: off (set OAUTH__ENCRYPTION_KEY to enable)"
    auth_file = Path(config.codex_auth_file).expanduser()
    if auth_file.is_file():
        return OK, f"chatgpt/codex subscription auth: enabled; codex cli login found at {auth_file}"
    return (
        WARN,
        "chatgpt/codex subscription auth: enabled but no codex cli login found; run the device "
        "flow (POST /api/oauth/openai/device-code) unless a token is already stored in Postgres",
    )


def _extras() -> list[Line]:
    lines: list[Line] = []
    for module, extra, purpose in (
        ("langfuse", "tracing", "Langfuse tracing"),
        ("runwayml", "video", "video generation"),
    ):
        installed = importlib.util.find_spec(module) is not None
        lines.append(
            (
                OK if installed else WARN,
                f"{purpose}: {'installed' if installed else f'not installed (uv sync --extra {extra})'}",
            )
        )
    return lines


def run_checks(checks: list[Callable[[], Line | list[Line]]]) -> list[Line]:
    """Run each check, folding an exception into a FAIL line."""
    lines: list[Line] = []
    for check in checks:
        try:
            result = check()
        except Exception as exc:
            result = (FAIL, f"{check.__name__.lstrip('_')}: {exc}")
        lines.extend(result if isinstance(result, list) else [result])
    return lines


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Print one line per check. Exit 1 when any check FAILs."""
    load_dotenv()
    lines = run_checks([_python, _env_file, _providers, _models, _codex, _database, _extras])
    for status, message in lines:
        print(f"[{status}] {message}")
    return 1 if any(status == FAIL for status, _ in lines) else 0
