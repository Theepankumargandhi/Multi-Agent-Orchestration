"""Shared heuristic credential detection for admission and post-patch gates.

These patterns catch common literal formats, not arbitrary secrets or encoded
credentials. Decisions and receipts report labels/counts, never matched values.
"""
from __future__ import annotations

import json
import re

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:ghp_|github_pat_|gsk_)[A-Za-z0-9_]{16,}\b"),
    # Retain the legacy fixture spelling for backward-compatible policy behavior.
    re.compile(r"\b(?:ghp|github_pat)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|password|secret|token)['\"]?\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
)
_CREDENTIAL_FIELD = re.compile(r"(?i)(?:api[_-]?key|password|secret|token)$")


def require_secret_free_prompt(messages) -> None:
    """Hold known credential formats before a coding-provider call, without logging them.

    JSON evidence is decoded as well as checked literally so string escaping
    cannot hide assignment-shaped credentials. This is heuristic, not a general
    guarantee that arbitrary or encoded secrets can be recognized.
    """
    pending = [(getattr(message, "content", message), 0) for message in messages]
    while pending:
        value, depth = pending.pop()
        if depth > 20:
            raise ValueError("prompt structure exceeds security inspection limits")
        if isinstance(value, str):
            if any(pattern.search(value) for pattern in SECRET_PATTERNS):
                raise ValueError("credential-like material withheld from coding provider")
            if value.lstrip().startswith(("{", "[", '"')):
                try:
                    decoded = json.loads(value)
                except (ValueError, RecursionError):
                    continue
                if decoded != value:
                    pending.append((decoded, depth + 1))
        elif isinstance(value, dict):
            # Inspect the association too: scanning keys and values separately
            # loses the meaning of generic credentials without a vendor prefix.
            if any(isinstance(key, str) and _CREDENTIAL_FIELD.search(key)
                   and isinstance(item, str) and len(item.strip()) >= 12
                   for key, item in value.items()):
                raise ValueError("credential-like material withheld from coding provider")
            pending.extend((item, depth + 1) for pair in value.items() for item in pair)
        elif isinstance(value, (list, tuple)):
            pending.extend((item, depth + 1) for item in value)
