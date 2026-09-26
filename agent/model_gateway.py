"""Policy-driven LLM inference gateway with budgets, caching, failover, and shadowing."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Literal, Protocol

from pydantic import BaseModel, Field, model_validator


def _hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Stable credential-free approximation used only for admission control."""
    return max(1, math.ceil(len(text) / 4))


def _embedding(text: str, dimensions: int = 256) -> dict[int, float]:
    tokens = re.findall(r"[a-z0-9_]+", text.casefold())
    features = tokens + [f"{left}:{right}" for left, right in zip(tokens, tokens[1:])]
    counts: Counter[int] = Counter()
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        counts[int.from_bytes(digest[:4], "big") % dimensions] += 1
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return {index: value / norm for index, value in counts.items()}


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    return sum(value * right.get(index, 0.0) for index, value in left.items())


class ProviderSpec(BaseModel):
    name: str = Field(min_length=2, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")
    model: str = Field(min_length=1, max_length=160)
    input_cost_per_million: float = Field(ge=0)
    output_cost_per_million: float = Field(ge=0)
    timeout_ms: int = Field(default=30_000, ge=10, le=300_000)
    failure_threshold: int = Field(default=3, ge=1, le=20)
    cooldown_seconds: float = Field(default=30, ge=0, le=3600)


class GatewayPolicy(BaseModel):
    version: str = Field(default="inference-policy-v1", min_length=2, max_length=80)
    daily_budget_usd: float = Field(default=5, gt=0)
    max_prompt_tokens: int = Field(default=16_000, ge=16, le=2_000_000)
    max_completion_tokens: int = Field(default=1024, ge=1, le=100_000)
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = Field(default=0.92, ge=0.5, le=1)
    semantic_cache_ttl_seconds: float = Field(default=900, ge=0, le=86_400)
    semantic_cache_max_entries: int = Field(default=1000, ge=1, le=100_000)
    canary_provider: str = ""
    canary_percentage: float = Field(default=0, ge=0, le=100)
    shadow_provider: str = ""
    shadow_enabled: bool = False
    fallback_providers: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_rollout(self) -> "GatewayPolicy":
        if self.canary_percentage and not self.canary_provider:
            raise ValueError("canary_provider is required when canary_percentage is non-zero")
        if self.shadow_enabled and not self.shadow_provider:
            raise ValueError("shadow_provider is required when shadowing is enabled")
        return self


class InferenceRequest(BaseModel):
    request_id: str = Field(min_length=3, max_length=160)
    tenant_id: str = Field(min_length=1, max_length=120)
    system: str = Field(default="", max_length=100_000)
    prompt: str = Field(min_length=1, max_length=1_000_000)
    preferred_provider: str = Field(min_length=2, max_length=80)
    high_risk: bool = False
    allow_cache: bool = True
    allow_canary: bool = True
    allow_shadow: bool = True
    max_completion_tokens: int | None = Field(default=None, ge=1, le=100_000)


class ProviderResult(BaseModel):
    content: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: float = Field(default=0, ge=0)


class ProviderAttempt(BaseModel):
    provider: str
    model: str
    outcome: Literal[
        "success", "error", "timeout", "circuit_open", "budget_rejected"
    ]
    latency_ms: float = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    error_type: str = ""


class ShadowEvidence(BaseModel):
    provider: str
    executed: bool
    agreement: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    reason: str = ""
    response_fingerprint: str = ""


class InferenceReceipt(BaseModel):
    schema_version: str = "1.0"
    request_id: str
    tenant_fingerprint: str
    policy_version: str
    policy_fingerprint: str
    prompt_fingerprint: str
    response_fingerprint: str = ""
    selected_provider: str = ""
    selected_model: str = ""
    cache_hit: bool = False
    cache_similarity: float = 0
    canary_selected: bool = False
    high_risk: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    completion_token_limit: int = 0
    cost_usd: float = 0
    tenant_daily_spend_usd: float = 0
    attempts: list[ProviderAttempt] = Field(default_factory=list)
    shadow: ShadowEvidence | None = None
    outcome: Literal["success", "blocked", "unavailable"]
    generated_at: str
    receipt_fingerprint: str = ""


class InferenceResponse(BaseModel):
    content: str
    receipt: InferenceReceipt


class InferenceProvider(Protocol):
    async def generate(
        self, request: InferenceRequest, spec: ProviderSpec, max_completion_tokens: int
    ) -> ProviderResult: ...


@dataclass
class _Circuit:
    failures: int = 0
    state: Literal["closed", "open", "half_open"] = "closed"
    opened_at: float = 0


@dataclass
class _CacheEntry:
    tenant_fingerprint: str
    system_fingerprint: str
    policy_fingerprint: str
    cohort_provider: str
    signature: dict[int, float]
    response: str
    response_fingerprint: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    created_at: float


class GatewayRejectedError(RuntimeError):
    def __init__(self, message: str, receipt: InferenceReceipt):
        super().__init__(message)
        self.receipt = receipt


class InferenceGateway:
    """In-process control plane; providers remain replaceable adapters."""

    def __init__(
        self,
        policy: GatewayPolicy,
        specs: list[ProviderSpec],
        providers: dict[str, InferenceProvider],
        *,
        clock: Callable[[], float] = time.time,
        fingerprint_key: bytes | None = None,
    ) -> None:
        self.policy = policy
        self.specs = {item.name: item for item in specs}
        if set(self.specs) != set(providers):
            raise ValueError("provider adapters and specifications must have identical names")
        declared = set(policy.fallback_providers)
        declared.update(filter(None, [policy.canary_provider, policy.shadow_provider]))
        unknown = declared - set(self.specs)
        if unknown:
            raise ValueError(f"gateway policy references unknown providers: {sorted(unknown)}")
        self.providers = providers
        self.clock = clock
        self.fingerprint_key = fingerprint_key
        self._circuits = defaultdict(_Circuit)
        self._spend: dict[tuple[str, str], float] = defaultdict(float)
        self._cache: list[_CacheEntry] = []

    def _day(self) -> str:
        return datetime.fromtimestamp(self.clock(), UTC).date().isoformat()

    def spend(self, tenant_id: str) -> float:
        return self._spend[(self._day(), self._content_fingerprint(tenant_id))]

    def _content_fingerprint(self, value: object) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        if self.fingerprint_key:
            return hmac.new(self.fingerprint_key, payload, hashlib.sha256).hexdigest()
        return hashlib.sha256(payload).hexdigest()

    def circuit_state(self, provider: str) -> str:
        circuit = self._circuits[provider]
        if circuit.state == "open":
            spec = self.specs[provider]
            if self.clock() - circuit.opened_at >= spec.cooldown_seconds:
                circuit.state = "half_open"
        return circuit.state

    def _cost(self, spec: ProviderSpec, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * spec.input_cost_per_million
            + completion_tokens * spec.output_cost_per_million
        ) / 1_000_000

    def _canary_selected(self, request: InferenceRequest) -> bool:
        if request.high_risk or not request.allow_canary or not self.policy.canary_provider:
            return False
        bucket = int(_hash(request.request_id)[:8], 16) % 10_000
        return bucket < round(self.policy.canary_percentage * 100)

    def _receipt(self, request: InferenceRequest, outcome: str) -> InferenceReceipt:
        policy_payload = self.policy.model_dump(mode="json")
        completion_limit = min(
            self.policy.max_completion_tokens,
            request.max_completion_tokens or self.policy.max_completion_tokens,
        )
        receipt = InferenceReceipt(
            request_id=request.request_id,
            tenant_fingerprint=self._content_fingerprint(request.tenant_id),
            policy_version=self.policy.version,
            policy_fingerprint=_hash(policy_payload),
            prompt_fingerprint=self._content_fingerprint(
                {"system": request.system, "prompt": request.prompt}
            ),
            high_risk=request.high_risk,
            completion_token_limit=completion_limit,
            outcome=outcome,
            generated_at=datetime.fromtimestamp(self.clock(), UTC).isoformat(),
        )
        return receipt

    @staticmethod
    def _seal(receipt: InferenceReceipt) -> None:
        receipt.receipt_fingerprint = _hash(
            receipt.model_dump(mode="json", exclude={"receipt_fingerprint"})
        )

    def verify_receipt(self, receipt: InferenceReceipt) -> bool:
        return receipt.receipt_fingerprint == _hash(
            receipt.model_dump(mode="json", exclude={"receipt_fingerprint"})
        )

    def _cache_lookup(
        self, request: InferenceRequest, cohort_provider: str
    ) -> tuple[_CacheEntry | None, float]:
        if (
            not self.policy.semantic_cache_enabled
            or not request.allow_cache
            or request.high_risk
        ):
            return None, 0
        now = self.clock()
        tenant = self._content_fingerprint(request.tenant_id)
        system = self._content_fingerprint(request.system)
        policy = _hash(self.policy.model_dump(mode="json"))
        signature = _embedding(request.prompt)
        self._cache = [
            item
            for item in self._cache
            if now - item.created_at <= self.policy.semantic_cache_ttl_seconds
        ]
        candidates = [
            item
            for item in self._cache
            if item.tenant_fingerprint == tenant
            and item.system_fingerprint == system
            and item.policy_fingerprint == policy
            and item.cohort_provider == cohort_provider
            and item.completion_tokens
            <= min(
                self.policy.max_completion_tokens,
                request.max_completion_tokens or self.policy.max_completion_tokens,
            )
        ]
        if not candidates:
            return None, 0
        scored = [(item, _cosine(signature, item.signature)) for item in candidates]
        best, similarity = max(scored, key=lambda item: item[1])
        return (best, similarity) if similarity >= self.policy.semantic_cache_threshold else (None, similarity)

    def _cache_store(
        self,
        request: InferenceRequest,
        result: ProviderResult,
        spec: ProviderSpec,
        cohort_provider: str,
    ) -> None:
        if not self.policy.semantic_cache_enabled or not request.allow_cache or request.high_risk:
            return
        self._cache.append(
            _CacheEntry(
                tenant_fingerprint=self._content_fingerprint(request.tenant_id),
                system_fingerprint=self._content_fingerprint(request.system),
                policy_fingerprint=_hash(self.policy.model_dump(mode="json")),
                cohort_provider=cohort_provider,
                signature=_embedding(request.prompt),
                response=result.content,
                response_fingerprint=self._content_fingerprint(result.content),
                provider=spec.name,
                model=spec.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                created_at=self.clock(),
            )
        )
        del self._cache[: -self.policy.semantic_cache_max_entries]

    def _record_failure(self, spec: ProviderSpec) -> None:
        circuit = self._circuits[spec.name]
        circuit.failures += 1
        if circuit.failures >= spec.failure_threshold:
            circuit.state = "open"
            circuit.opened_at = self.clock()

    def _record_success(self, spec: ProviderSpec) -> None:
        self._circuits[spec.name] = _Circuit()

    async def _shadow(
        self, request: InferenceRequest, production: ProviderResult
    ) -> ShadowEvidence | None:
        if not self.policy.shadow_enabled or request.high_risk or not request.allow_shadow:
            return None
        name = self.policy.shadow_provider
        spec = self.specs[name]
        if self.circuit_state(name) == "open":
            return ShadowEvidence(provider=name, executed=False, reason="circuit_open")
        completion_limit = min(
            self.policy.max_completion_tokens,
            request.max_completion_tokens or self.policy.max_completion_tokens,
        )
        projected = self._cost(
            spec, estimate_tokens(request.system + request.prompt), completion_limit
        )
        if self.spend(request.tenant_id) + projected > self.policy.daily_budget_usd:
            return ShadowEvidence(provider=name, executed=False, reason="budget_rejected")
        try:
            result = await asyncio.wait_for(
                self.providers[name].generate(request, spec, completion_limit),
                timeout=spec.timeout_ms / 1000,
            )
        except Exception as exc:
            self._record_failure(spec)
            return ShadowEvidence(
                provider=name, executed=False, reason=f"provider_{type(exc).__name__}"
            )
        self._record_success(spec)
        cost = self._cost(spec, result.prompt_tokens, result.completion_tokens)
        self._spend[(self._day(), self._content_fingerprint(request.tenant_id))] += cost
        agreement = _cosine(_embedding(production.content), _embedding(result.content))
        return ShadowEvidence(
            provider=name,
            executed=True,
            agreement=max(0, min(1, agreement)),
            latency_ms=result.latency_ms,
            cost_usd=cost,
            response_fingerprint=self._content_fingerprint(result.content),
        )

    async def execute(self, request: InferenceRequest) -> InferenceResponse:
        prompt_tokens = estimate_tokens(request.system + request.prompt)
        completion_limit = min(
            self.policy.max_completion_tokens,
            request.max_completion_tokens or self.policy.max_completion_tokens,
        )
        if prompt_tokens > self.policy.max_prompt_tokens:
            receipt = self._receipt(request, "blocked")
            receipt.prompt_tokens = prompt_tokens
            receipt.tenant_daily_spend_usd = self.spend(request.tenant_id)
            self._seal(receipt)
            raise GatewayRejectedError("prompt token budget exceeded", receipt)

        canary = self._canary_selected(request)
        cohort_provider = self.policy.canary_provider if canary else request.preferred_provider
        cached, similarity = self._cache_lookup(request, cohort_provider)
        if cached:
            receipt = self._receipt(request, "success")
            receipt.selected_provider = "semantic-cache"
            receipt.selected_model = cached.model
            receipt.response_fingerprint = cached.response_fingerprint
            receipt.cache_hit = True
            receipt.cache_similarity = similarity
            receipt.canary_selected = canary
            receipt.prompt_tokens = prompt_tokens
            receipt.completion_tokens = cached.completion_tokens
            receipt.tenant_daily_spend_usd = self.spend(request.tenant_id)
            self._seal(receipt)
            return InferenceResponse(content=cached.response, receipt=receipt)

        candidates = [
            self.policy.canary_provider if canary else request.preferred_provider,
            request.preferred_provider,
            *self.policy.fallback_providers,
        ]
        candidates = list(dict.fromkeys(filter(None, candidates)))
        receipt = self._receipt(request, "unavailable")
        receipt.canary_selected = canary
        receipt.prompt_tokens = prompt_tokens
        for name in candidates:
            spec = self.specs.get(name)
            if spec is None:
                continue
            if self.circuit_state(name) == "open":
                receipt.attempts.append(
                    ProviderAttempt(
                        provider=name,
                        model=spec.model,
                        outcome="circuit_open",
                        latency_ms=0,
                        estimated_cost_usd=0,
                    )
                )
                continue
            projected = self._cost(spec, prompt_tokens, completion_limit)
            if self.spend(request.tenant_id) + projected > self.policy.daily_budget_usd:
                receipt.attempts.append(
                    ProviderAttempt(
                        provider=name,
                        model=spec.model,
                        outcome="budget_rejected",
                        latency_ms=0,
                        estimated_cost_usd=projected,
                    )
                )
                continue
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    self.providers[name].generate(request, spec, completion_limit),
                    timeout=spec.timeout_ms / 1000,
                )
            except TimeoutError:
                latency = (time.perf_counter() - started) * 1000
                self._record_failure(spec)
                receipt.attempts.append(
                    ProviderAttempt(
                        provider=name,
                        model=spec.model,
                        outcome="timeout",
                        latency_ms=latency,
                        estimated_cost_usd=0,
                        error_type="TimeoutError",
                    )
                )
                continue
            except Exception as exc:
                latency = (time.perf_counter() - started) * 1000
                self._record_failure(spec)
                receipt.attempts.append(
                    ProviderAttempt(
                        provider=name,
                        model=spec.model,
                        outcome="error",
                        latency_ms=latency,
                        estimated_cost_usd=0,
                        error_type=type(exc).__name__,
                    )
                )
                continue
            self._record_success(spec)
            cost = self._cost(spec, result.prompt_tokens, result.completion_tokens)
            self._spend[(self._day(), self._content_fingerprint(request.tenant_id))] += cost
            receipt.attempts.append(
                ProviderAttempt(
                    provider=name,
                    model=spec.model,
                    outcome="success",
                    latency_ms=result.latency_ms,
                    estimated_cost_usd=cost,
                )
            )
            receipt.outcome = "success"
            receipt.selected_provider = name
            receipt.selected_model = spec.model
            receipt.response_fingerprint = self._content_fingerprint(result.content)
            receipt.completion_tokens = result.completion_tokens
            receipt.cost_usd = cost
            self._cache_store(request, result, spec, cohort_provider)
            receipt.shadow = await self._shadow(request, result)
            receipt.tenant_daily_spend_usd = self.spend(request.tenant_id)
            self._seal(receipt)
            return InferenceResponse(content=result.content, receipt=receipt)

        receipt.tenant_daily_spend_usd = self.spend(request.tenant_id)
        self._seal(receipt)
        raise GatewayRejectedError("no provider satisfied availability and budget policy", receipt)
