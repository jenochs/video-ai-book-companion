"""Localhost as a security boundary: tokens and path validation.

Accompanies Chapter 8 (From Models to Systems) of "Video Generation with AI"
(O'Reilly) by Joseph Enochs.

Implements Pattern 4: per-session bearer tokens generated at every app start,
constant-time token comparison, and whitelist path validation that is
case-insensitive on Windows and case-sensitive elsewhere. The token code
mirrors backend/app_factory.py and the path logic is a Python port of
electron/path-validation.ts from LTX Desktop (pinned commit 81f7f45 in the
chapter).
"""

from __future__ import annotations

import hmac
import os
import secrets
import sys
from dataclasses import dataclass
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class SessionTokens:
    """The two-token split: ordinary inference vs administrative endpoints.

    An exfiltrated auth token can drive generations but cannot shut the
    backend down or grant filesystem access.
    """

    auth_token: str
    admin_token: str


def generate_session_tokens() -> SessionTokens:
    """Two random 32-byte tokens, regenerated at every app start.

    The Electron original uses randomBytes(32).toString('base64url');
    secrets.token_urlsafe(32) is the Python equivalent.
    """
    return SessionTokens(
        auth_token=secrets.token_urlsafe(32),
        admin_token=secrets.token_urlsafe(32),
    )


def verify_token(provided: str, expected: str) -> bool:
    """Constant-time comparison: a naive == leaks timing information."""
    return hmac.compare_digest(provided.encode(), expected.encode())


# ---------------------------------------------------------------------------
# Path validation across platforms
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform == "win32"


def _strip_file_url(p: str) -> str:
    parsed = urlparse(p)
    return unquote(parsed.path)


def _normalize(p: str) -> str:
    resolved = os.path.abspath(p)
    return resolved.lower() if _IS_WINDOWS else resolved


def validate_path(input_path: str, allowed_roots: list[str]) -> str:
    """Return the resolved path if it falls under an allowed root, else raise.

    Strips file:// URLs, resolves relative segments before checking, and
    folds case only on Windows, which closes the case-folding class of
    path-traversal tricks.
    """
    cleaned = (
        _strip_file_url(input_path)
        if input_path.startswith("file://")
        else input_path
    )
    resolved = os.path.abspath(cleaned)
    norm = _normalize(resolved)

    for root in (_normalize(r) for r in allowed_roots):
        if norm == root or norm.startswith(root + os.sep):
            return resolved
    raise ValueError(f"Path not allowed: {input_path}")
