import asyncio
import importlib

import pytest

from agent.model_gateway import (
    GatewayPolicy,
    GatewayRejectedError,
    InferenceGateway,
    InferenceReceipt,
    InferenceRequest,
    ProviderResult,
    ProviderSpec,
)


class ScriptedProvider:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.completion_limits = []

    async def generate(self, request, spec, max_completion_tokens):
        self.calls += 1
        self.completion_limits.append(max_completion_tokens)
        value = self.script.pop(0) if self.script else f"response from {spec.name}"
        if isinstance(value, Exception):
            raise value
        if callable(value):
            value = await value()
        return ProviderResult(
            content=str(value),
            prompt_tokens=20,
            completion_tokens=10,
            latency_ms=12,
        )


def _spec(name, *, input_price=1.0, output_price=2.0, failure_threshold=2, cooldown=30):
    return ProviderSpec(
        name=name,
        model=f"{name}-model",
        input_cost_per_million=input_price,
        output_cost_per_million=output_price,
        failure_threshold=failure_threshold,
        cooldown_seconds=cooldown,
        timeout_ms=20,
    )


def _request(request_id="request-1", tenant="tenant-a", prompt="Explain durable checkpoints"):
    return InferenceRequest(
        request_id=request_id,
        tenant_id=tenant,
        system="Answer using approved evidence.",
        prompt=prompt,
        preferred_provider="primary",
    )


@pytest.mark.asyncio
async def test_per_request_completion_limit_caps_provider_generation():
    primary = ScriptedProvider(["bounded response"])
    gateway = InferenceGateway(
        GatewayPolicy(max_completion_tokens=1024),
        [_spec("primary")],
        {"primary": primary},
    )
    request = _request()
    request.max_completion_tokens = 128
    response = await gateway.execute(request)
    assert primary.completion_limits == [128]
    assert response.receipt.completion_token_limit == 128


@pytest.mark.asyncio
async def test_semantic_cache_is_tenant_scoped_and_cost_free():
    primary = ScriptedProvider(["Durable checkpoints preserve state.", "tenant B response"])
    gateway = InferenceGateway(
        GatewayPolicy(semantic_cache_threshold=0.5),
        [_spec("primary")],
        {"primary": primary},
    )
    first = await gateway.execute(_request())
    cached = await gateway.execute(
        _request("request-2", prompt="Explain the durable checkpoints")
    )
    isolated = await gateway.execute(
        _request("request-3", tenant="tenant-b", prompt="Explain the durable checkpoints")
    )

    assert not first.receipt.cache_hit
    assert cached.receipt.cache_hit
    assert cached.receipt.selected_provider == "semantic-cache"
    assert cached.receipt.cost_usd == 0
    assert isolated.content == "tenant B response"
    assert primary.calls == 2


@pytest.mark.asyncio
async def test_high_risk_requests_bypass_cache_canary_and_shadow():
    primary = ScriptedProvider(["first", "second"])
    canary = ScriptedProvider(["canary"])
    shadow = ScriptedProvider(["shadow"])
    policy = GatewayPolicy(
        semantic_cache_threshold=0.5,
        canary_provider="canary",
        canary_percentage=100,
        shadow_provider="shadow",
        shadow_enabled=True,
    )
    gateway = InferenceGateway(
        policy,
        [_spec("primary"), _spec("canary"), _spec("shadow")],
        {"primary": primary, "canary": canary, "shadow": shadow},
    )
    request = _request(prompt="Review this credential security incident")
    request.high_risk = True
    one = await gateway.execute(request)
    request.request_id = "request-high-risk-2"
    two = await gateway.execute(request)

    assert [one.content, two.content] == ["first", "second"]
    assert not one.receipt.canary_selected and not two.receipt.cache_hit
    assert one.receipt.shadow is None
    assert canary.calls == shadow.calls == 0


@pytest.mark.asyncio
async def test_circuit_breaker_falls_back_then_allows_half_open_probe():
    now = [1_700_000_000.0]
    primary = ScriptedProvider([RuntimeError("provider down"), "primary recovered"])
    fallback = ScriptedProvider(["fallback one", "fallback two"])
    gateway = InferenceGateway(
        GatewayPolicy(semantic_cache_enabled=False, fallback_providers=["fallback"]),
        [_spec("primary", failure_threshold=1, cooldown=30), _spec("fallback")],
        {"primary": primary, "fallback": fallback},
        clock=lambda: now[0],
    )
    first = await gateway.execute(_request())
    second = await gateway.execute(_request("request-2"))
    now[0] += 31
    third = await gateway.execute(_request("request-3"))

    assert first.receipt.selected_provider == "fallback"
    assert second.receipt.attempts[0].outcome == "circuit_open"
    assert third.receipt.selected_provider == "primary"
    assert gateway.circuit_state("primary") == "closed"


@pytest.mark.asyncio
async def test_budget_admission_chooses_affordable_fallback_and_then_fails_closed():
    expensive = ScriptedProvider(["should not execute"])
    cheap = ScriptedProvider(["affordable"])
    gateway = InferenceGateway(
        GatewayPolicy(
            daily_budget_usd=0.00002,
            max_completion_tokens=100,
            semantic_cache_enabled=False,
            fallback_providers=["cheap"],
        ),
        [
            _spec("primary", input_price=10, output_price=10),
            _spec("cheap", input_price=0.01, output_price=0.01),
        ],
        {"primary": expensive, "cheap": cheap},
    )
    response = await gateway.execute(_request())
    assert response.receipt.attempts[0].outcome == "budget_rejected"
    assert response.receipt.selected_provider == "cheap"
    assert expensive.calls == 0

    gateway.policy.daily_budget_usd = gateway.spend("tenant-a")
    with pytest.raises(GatewayRejectedError) as caught:
        await gateway.execute(_request("request-budget-blocked"))
    assert caught.value.receipt.outcome == "unavailable"
    assert all(item.outcome == "budget_rejected" for item in caught.value.receipt.attempts)


@pytest.mark.asyncio
async def test_shadow_result_is_measured_but_never_served():
    primary = ScriptedProvider(["checkpoint state is durable", "stream-safe response"])
    shadow = ScriptedProvider(["checkpoint state remains durable"])
    gateway = InferenceGateway(
        GatewayPolicy(
            semantic_cache_enabled=False,
            shadow_provider="shadow",
            shadow_enabled=True,
        ),
        [_spec("primary"), _spec("shadow")],
        {"primary": primary, "shadow": shadow},
    )
    response = await gateway.execute(_request())
    assert response.content == "checkpoint state is durable"
    assert response.receipt.selected_provider == "primary"
    assert response.receipt.shadow and response.receipt.shadow.executed
    assert response.receipt.shadow.agreement > 0.5
    assert response.receipt.tenant_daily_spend_usd > response.receipt.cost_usd
    stream_request = _request("request-stream")
    stream_request.allow_shadow = False
    stream_response = await gateway.execute(stream_request)
    assert stream_response.receipt.shadow is None


@pytest.mark.asyncio
async def test_timeout_is_contained_by_fallback():
    async def slow():
        await asyncio.sleep(0.05)
        return "late"

    primary = ScriptedProvider([slow])
    fallback = ScriptedProvider(["on time"])
    gateway = InferenceGateway(
        GatewayPolicy(semantic_cache_enabled=False, fallback_providers=["fallback"]),
        [_spec("primary", failure_threshold=1), _spec("fallback")],
        {"primary": primary, "fallback": fallback},
    )
    response = await gateway.execute(_request())
    assert response.content == "on time"
    assert response.receipt.attempts[0].outcome == "timeout"
    assert response.receipt.selected_provider == "fallback"


def test_canary_assignment_is_deterministic_and_high_risk_is_excluded():
    providers = {
        "primary": ScriptedProvider([]),
        "canary": ScriptedProvider([]),
    }
    gateway = InferenceGateway(
        GatewayPolicy(canary_provider="canary", canary_percentage=20),
        [_spec("primary"), _spec("canary")],
        providers,
    )
    assignments = [gateway._canary_selected(_request(f"request-{index}")) for index in range(1000)]
    repeated = [gateway._canary_selected(_request(f"request-{index}")) for index in range(1000)]
    high_risk = _request("request-risk")
    high_risk.high_risk = True
    opted_out = _request("request-opted-out")
    opted_out.allow_canary = False
    assert assignments == repeated
    assert 170 <= sum(assignments) <= 230
    assert not gateway._canary_selected(high_risk)
    assert not gateway._canary_selected(opted_out)


@pytest.mark.asyncio
async def test_semantic_cache_does_not_cross_canary_cohorts():
    primary = ScriptedProvider(["production response"])
    canary = ScriptedProvider(["canary response"])
    gateway = InferenceGateway(
        GatewayPolicy(
            canary_provider="canary",
            canary_percentage=50,
            semantic_cache_threshold=0.5,
        ),
        [_spec("primary"), _spec("canary")],
        {"primary": primary, "canary": canary},
    )
    assigned_id = next(
        f"cohort-{index}"
        for index in range(100)
        if gateway._canary_selected(_request(f"cohort-{index}"))
    )
    control_id = next(
        f"control-{index}"
        for index in range(100)
        if not gateway._canary_selected(_request(f"control-{index}"))
    )
    assigned = await gateway.execute(_request(assigned_id))
    control = await gateway.execute(
        _request(control_id, prompt="Explain the durable checkpoints")
    )
    assert assigned.receipt.selected_provider == "canary"
    assert control.receipt.selected_provider == "primary"
    assert not control.receipt.cache_hit
    assert assigned.content != control.content


@pytest.mark.asyncio
async def test_receipts_exclude_content_and_detect_modification():
    secret_prompt = "private customer prompt marker-7812"
    secret_response = "private model response marker-9931"
    provider = ScriptedProvider([secret_response])
    gateway = InferenceGateway(
        GatewayPolicy(semantic_cache_enabled=False),
        [_spec("primary")],
        {"primary": provider},
    )
    response = await gateway.execute(_request(prompt=secret_prompt))
    encoded = response.receipt.model_dump_json()
    assert secret_prompt not in encoded
    assert secret_response not in encoded
    assert gateway.verify_receipt(response.receipt)
    tampered = InferenceReceipt.model_validate(response.receipt.model_dump())
    tampered.cost_usd = 99
    assert not gateway.verify_receipt(tampered)


@pytest.mark.asyncio
async def test_oversized_prompt_is_rejected_before_provider_execution():
    provider = ScriptedProvider(["unused"])
    gateway = InferenceGateway(
        GatewayPolicy(max_prompt_tokens=16),
        [_spec("primary")],
        {"primary": provider},
    )
    with pytest.raises(GatewayRejectedError) as caught:
        await gateway.execute(_request(prompt="x" * 1000))
    assert caught.value.receipt.outcome == "blocked"
    assert gateway.verify_receipt(caught.value.receipt)
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_research_agent_model_calls_use_gateway_receipt_and_online_event(monkeypatch, tmp_path):
    research = importlib.import_module("agent.research_assistant")

    provider = ScriptedProvider(["gateway-controlled response"])
    gateway = InferenceGateway(
        GatewayPolicy(semantic_cache_enabled=False),
        [_spec("primary")],
        {"primary": provider},
    )
    monkeypatch.setattr(research, "MODEL_GATEWAY_ENABLED", True)
    monkeypatch.setattr(research, "_inference_gateway", gateway)
    online_events = tmp_path / "online-events.jsonl"
    monkeypatch.setattr(research, "MODEL_GATEWAY_ONLINE_EVENT_PATH", str(online_events))
    monkeypatch.setattr(research, "ONLINE_EVAL_INTEGRITY_KEY", b"online-integrity-test-key")
    monkeypatch.setattr(research, "_gateway_provider_for_model", lambda model: "primary")
    config = {
        "configurable": {"model": "test-model", "user_id": "user-1", "thread_id": "thread-1"},
        "run_id": "run-1",
    }
    result = await research._call_llm("system policy", "user request", config)
    receipts = config["configurable"]["model_gateway_receipts"]
    assert result == "gateway-controlled response"
    assert len(receipts) == 1
    assert receipts[0]["selected_provider"] == "primary"
    assert "user request" not in str(receipts[0])
    exported = online_events.read_text(encoding="utf-8")
    assert '"selected_provider":"primary"' in exported
    assert "user request" not in exported
    assert "gateway-controlled response" not in exported


@pytest.mark.asyncio
async def test_research_agent_exports_blocked_gateway_decision_before_raising(monkeypatch, tmp_path):
    research = importlib.import_module("agent.research_assistant")
    provider = ScriptedProvider(["must not run"])
    gateway = InferenceGateway(
        GatewayPolicy(max_prompt_tokens=16, semantic_cache_enabled=False),
        [_spec("primary")],
        {"primary": provider},
    )
    output = tmp_path / "blocked-events.jsonl"
    monkeypatch.setattr(research, "MODEL_GATEWAY_ENABLED", True)
    monkeypatch.setattr(research, "_inference_gateway", gateway)
    monkeypatch.setattr(research, "_gateway_provider_for_model", lambda model: "primary")
    monkeypatch.setattr(research, "MODEL_GATEWAY_ONLINE_EVENT_PATH", str(output))
    monkeypatch.setattr(research, "ONLINE_EVAL_INTEGRITY_KEY", b"online-integrity-test-key")
    config = {"configurable": {"user_id": "user-1", "thread_id": "thread-1"}}

    with pytest.raises(GatewayRejectedError):
        await research._call_llm("system", "x" * 1000, config)

    assert provider.calls == 0
    assert config["configurable"]["model_gateway_receipts"][0]["outcome"] == "blocked"
    assert '"outcome":"blocked"' in output.read_text(encoding="utf-8")


def test_live_gateway_requires_a_dedicated_fingerprint_key(monkeypatch):
    research = importlib.import_module("agent.research_assistant")
    monkeypatch.setattr(research, "_inference_gateway", None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-provider-key")
    monkeypatch.delenv("MODEL_GATEWAY_FINGERPRINT_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_GATEWAY_FINGERPRINT_KEY"):
        research._get_inference_gateway()
