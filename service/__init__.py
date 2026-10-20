"""Service adapters; load the full application only when explicitly requested."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from service.service import app as app

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from service.service import app

    globals()[name] = app
    return app
