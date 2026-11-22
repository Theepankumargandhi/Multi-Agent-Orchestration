"""Public-demo support: question quotas and serving the web UI from the API's port.

A public demo spends the operator's model quota on strangers' questions, so DEMO_MODE caps
questions per account per hour and for the whole server per day. Counts live in memory: a
demo keeps no data across restarts anyway.
"""

import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.responses import PlainTextResponse
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


DEMO_MODE = _flag("DEMO_MODE")
DEMO_QUESTIONS_PER_HOUR = _positive_int("DEMO_QUESTIONS_PER_HOUR", 15)
DEMO_QUESTIONS_PER_DAY = _positive_int("DEMO_QUESTIONS_PER_DAY", 300)
DEMO_MAX_DOCUMENTS = _positive_int("DEMO_MAX_DOCUMENTS", 3)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


class QuestionQuota:
    """Sliding one-hour limit per account plus a fixed limit per UTC day for everyone."""

    def __init__(self, per_hour: int, per_day: int, clock=time.time):
        self.per_hour = per_hour
        self.per_day = per_day
        self._clock = clock
        self._lock = threading.Lock()
        self._accounts: dict[str, deque[float]] = {}
        self._day = ""
        self._day_count = 0

    def spend(self, account: str) -> str | None:
        """Count one question; return why it is refused instead, without counting it."""
        now = self._clock()
        today = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if today != self._day:
                self._day, self._day_count = today, 0
                self._accounts = {key: times for key, times in self._accounts.items() if times and times[-1] > now - 3600}
            if self._day_count >= self.per_day:
                return (
                    "The demo has answered all the questions it can today. "
                    "It resets at midnight UTC, or you can run CiteRight yourself from GitHub."
                )
            times = self._accounts.setdefault(account, deque())
            while times and times[0] <= now - 3600:
                times.popleft()
            if len(times) >= self.per_hour:
                minutes = max(1, round((times[0] + 3600 - now) / 60))
                return (
                    f"The demo allows {_count(self.per_hour, 'question')} an hour per account. "
                    f"Try again in about {minutes} minute{'s' if minutes != 1 else ''}."
                )
            times.append(now)
            self._day_count += 1
            return None


_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (
        b"content-security-policy",
        b"default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        b"font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'self' https://huggingface.co",
    ),
]


class SinglePortApp:
    """Serve the built web UI and, under /api, the API: one port, as hosts like Hugging Face Spaces expect.

    The API sees the same paths as behind the nginx proxy in compose.yaml ("/api/stream" -> "/stream").
    """

    def __init__(self, api: ASGIApp, web_dir: str | Path):
        self.api = api
        self.web_dir = Path(web_dir)
        self.static = StaticFiles(directory=self.web_dir, html=True)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.api(scope, receive, send)
            return
        path = scope.get("path", "")
        if path == "/api" or path.startswith("/api/"):
            inner = dict(scope)
            inner["path"] = path[4:] or "/"
            if isinstance(scope.get("raw_path"), bytes) and scope["raw_path"].startswith(b"/api"):
                inner["raw_path"] = scope["raw_path"][4:] or b"/"
            await self.api(inner, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(_SECURITY_HEADERS)
                # index.html must not be cached across deploys; hashed assets can be.
                cache = b"public, max-age=31536000, immutable" if path.startswith("/assets/") else b"no-store"
                headers.append((b"cache-control", cache))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.static(scope, receive, send_with_headers)
        except HTTPException as error:  # not found, or a method other than GET/HEAD
            await PlainTextResponse(error.detail, status_code=error.status_code)(scope, receive, send_with_headers)
