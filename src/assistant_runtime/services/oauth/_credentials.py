"""Read local Codex credentials and decode their metadata without authenticating it."""

import base64
import binascii
import json
from pathlib import Path
from typing import Any

from assistant_runtime.services.oauth.exceptions import OAuthCodexSyncError
from assistant_runtime.services.oauth.models import CodexCliAuth


def read_codex_cli_auth(auth_file: str) -> CodexCliAuth:
    """Read ChatGPT/Codex OAuth state from the local Codex CLI auth file."""
    auth_path = Path(auth_file).expanduser()
    if not auth_path.exists():
        raise OAuthCodexSyncError(f"Codex auth file not found: {auth_path}")

    try:
        data = json.loads(auth_path.read_text())
    except OSError as exc:
        raise OAuthCodexSyncError(f"Failed to read Codex auth file: {auth_path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OAuthCodexSyncError(f"Invalid JSON in Codex auth file: {auth_path}") from exc

    if not isinstance(data, dict):
        raise OAuthCodexSyncError("Codex auth file must contain a JSON object")
    auth_mode = data.get("auth_mode")
    if auth_mode == "apikey":
        raise OAuthCodexSyncError(
            "Codex CLI is using API key mode, not ChatGPT OAuth. Run `codex login` first."
        )

    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        raise OAuthCodexSyncError(f"Codex auth file is missing OAuth tokens: {auth_path}")

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    id_token = tokens.get("id_token")
    account_id = tokens.get("account_id")

    if not isinstance(access_token, str) or not access_token:
        raise OAuthCodexSyncError(f"Codex auth file is missing access_token: {auth_path}")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise OAuthCodexSyncError(f"Codex auth file is missing refresh_token: {auth_path}")
    if id_token is not None and not isinstance(id_token, str):
        id_token = None
    if account_id is not None and not isinstance(account_id, str):
        account_id = None

    return CodexCliAuth(
        access_token=access_token,
        refresh_token=refresh_token,
        id_token=id_token,
        account_id=account_id,
        auth_mode=auth_mode if isinstance(auth_mode, str) else None,
        email=extract_email(id_token),
    )


def _decode_jwt_claims(token: str | None) -> dict[str, Any] | None:
    """Decode JWT claims without verification for local token metadata access."""
    if not token or token.count(".") < 2:
        return None

    payload = token.split(".")[1]
    padding = "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload + padding)
        claims = json.loads(raw)
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error):
        return None

    return claims if isinstance(claims, dict) else None


def extract_email(id_token: str | None) -> str | None:
    """Best-effort decode of email claim from the OIDC id_token payload."""
    claims = _decode_jwt_claims(id_token)
    if claims is None:
        return None

    email = claims.get("email")
    if isinstance(email, str) and email:
        return email

    profile = claims.get("https://api.openai.com/profile")
    if isinstance(profile, dict):
        email = profile.get("email")
        if isinstance(email, str) and email:
            return email

    return None


def extract_account_id(id_token: str | None, access_token: str | None) -> str | None:
    """Extract the ChatGPT workspace/account id from token claims."""
    for token in (id_token, access_token):
        claims = _decode_jwt_claims(token)
        if claims is None:
            continue
        auth_claim = claims.get("https://api.openai.com/auth")
        if isinstance(auth_claim, dict):
            account_id = auth_claim.get("chatgpt_account_id")
            if isinstance(account_id, str) and account_id:
                return account_id
    return None


def extract_token_expiry(token: str | None) -> float | None:
    """Extract expiry from a JWT token, if present."""
    claims = _decode_jwt_claims(token)
    if claims is None:
        return None

    exp = claims.get("exp")
    if isinstance(exp, int | float):
        return float(exp)
    return None
