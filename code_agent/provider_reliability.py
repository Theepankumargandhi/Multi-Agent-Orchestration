"""Content-free provider failure receipts and per-run inference admission control."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal

from pydantic import BaseModel, Field


class ProviderFailure(BaseModel):
    category: Literal["quota", "authentication", "permission", "policy", "invalid_request", "rate_limit", "transient", "unknown"]
    http_status: int | None = None
    code: str = "other"
    retry_after_seconds: float | None = Field(default=None, ge=0, le=86400)
    stage: str = "model"

    @property
    def retryable(self) -> bool:
        return self.category in {"rate_limit", "transient"}

    @property
    def terminal(self) -> bool:
        return self.category in {"quota", "authentication", "permission", "policy", "invalid_request"}


_QUOTA = {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached", "usage_limit_reached", "organization_usage_limit_exceeded", "organization_quota_exceeded", "project_quota_exceeded", "organization_spend_limit_exceeded", "project_spend_limit_exceeded"}
_POLICY = {"content_policy_violation", "content_filter"}
_CODES = _QUOTA | _POLICY | {"invalid_api_key", "rate_limit_exceeded", "slow_down", "server_error", "server_is_overloaded", "model_not_found"}


def classify_provider_error(exc: Exception, *, stage: str = "model", now: datetime | None = None) -> ProviderFailure:
    """Never copy exception messages, response bodies, request IDs or arbitrary codes."""
    if isinstance(exc, ProviderCircuitOpen):
        return exc.failure
    body = getattr(exc, "body", None)
    details = body.get("error", body) if isinstance(body, dict) else {}
    details = details if isinstance(details, dict) else {}
    codes = [getattr(exc, "code", None), getattr(exc, "type", None), details.get("code"), details.get("type")]
    codes = [value for value in codes if isinstance(value, str)]
    status = getattr(exc, "status_code", None)
    status = status if isinstance(status, int) and 100 <= status <= 599 else None
    name = type(exc).__name__.lower()
    if any(code in _QUOTA for code in codes):
        category = "quota"
    elif status == 401 or (status is None and "authentication" in name) or "invalid_api_key" in codes:
        category = "authentication"
    elif any(code in _POLICY for code in codes):
        category = "policy"
    elif status == 403 or (status is None and "permissiondenied" in name):
        category = "permission"
    elif status == 429 or (status is None and "ratelimit" in name):
        category = "rate_limit"
    elif status in {408, 409, 500, 502, 503, 504} or (status is None and any(value in name for value in ("timeout", "connection", "internalserver", "serviceunavailable"))):
        category = "transient"
    elif (status is not None and 400 <= status < 500) or name == "structuredoutputerror":
        category = "invalid_request"
    else:
        category = "unknown"
    delay = None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    if value is not None:
        try:
            delay = float(value)
        except (ValueError, TypeError):
            try:
                date = parsedate_to_datetime(str(value))
                delay = (date - (now or datetime.now(UTC))).total_seconds()
            except (ValueError, TypeError, OverflowError):
                pass
        if delay is not None:
            delay = min(86400.0, max(0.0, delay)) if math.isfinite(delay) else None
    # Stages are controlled by callers, not provider payloads.
    safe_stage = stage if stage in {"model", "analysis", "implementation", "test_author", "review", "repair", "tournament", "benchmark"} else "model"
    return ProviderFailure(category=category, http_status=status,
                           code=next((code for code in codes if code in _CODES), "other"),
                           retry_after_seconds=delay, stage=safe_stage)


def retry_delay(failure: ProviderFailure, attempt: int, remaining_seconds: float) -> float | None:
    """Three total attempts, max eight-second wait, always respect server minimum."""
    if not failure.retryable or attempt >= 2:
        return None
    delay = max(0.5 * 2**attempt + random.uniform(0.0, 0.25), failure.retry_after_seconds or 0.0)
    if delay > 8.0 or delay + 1.0 >= remaining_seconds:
        return None
    return delay


class ProviderCircuitOpen(RuntimeError):
    def __init__(self, failure: ProviderFailure):
        self.failure = failure
        super().__init__(f"provider admission held: {failure.category}")


class ProviderGuard:
    """One solve's shared hold. Already in-flight requests cannot be recalled."""

    def __init__(self):
        self.failures: list[ProviderFailure] = []
        self.hold: ProviderFailure | None = None
        self.calls = 0

    def check(self) -> None:
        if self.hold is not None:
            raise ProviderCircuitOpen(self.hold)

    async def invoke(self, model, messages, *, config=None, stage="model"):
        self.check()
        self.calls += 1
        try:
            return await model.ainvoke(messages, config=config)
        except ProviderCircuitOpen as exc:
            # A shared outer budget can hold another team's nested guard.
            self.calls -= 1
            self.hold = exc.failure
            self.failures.append(exc.failure)
            raise
        except Exception as exc:
            failure = classify_provider_error(exc, stage=stage)
            self.failures.append(failure)
            if failure.terminal:
                self.hold = failure
            raise


class GuardedModel:
    def __init__(self, model, guard: ProviderGuard, stage: str):
        self.model, self.guard, self.stage = model, guard, stage

    async def ainvoke(self, messages, config=None):
        return await self.guard.invoke(self.model, messages, config=config, stage=self.stage)
