"""Shared heuristic credential detection for admission and post-patch gates.

These patterns catch common literal formats, not arbitrary secrets or encoded
credentials. Decisions and receipts report labels/counts, never matched values.
"""
from __future__ import annotations

import re

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:ghp_|github_pat_|gsk_)[A-Za-z0-9_]{16,}\b"),
    # Retain the legacy fixture spelling for backward-compatible policy behavior.
    re.compile(r"\b(?:ghp|github_pat)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
)
