"""Credential-free evaluation and release gate for the inference control plane."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.model_gateway import (
    GatewayPolicy,
    GatewayRejectedError,
    InferenceGateway,
    InferenceReceipt,
    InferenceRequest,
    ProviderResult,
    ProviderSpec,
)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class GatewayEvalScenario(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    kind: Literal[
        "semantic_cache",
        "tenant_isolation",
        "high_risk",
        "fallback",
        "circuit_open",
        "half_open",
        "budget_fallback",
        "budget_block",
        "shadow",
        "canary",
        "prompt_limit",
        "receipt_privacy",
    ]
    title: str
    expected_status: Literal["success", "blocked"]
    expected_control: str


class GatewayEvalOutcome(BaseModel):
    scenario_id: str
    kind: str
    expected_status: str
    actual_status: str
    expected_control: str
    observed_controls: list[str]
    selected_provider: str = ""
    cache_hit: bool = False
    receipt_verified: bool
    content_leakage_violations: int = 0
    cost_usd: float = 0
    passed: bool
    evidence_fingerprint: str = ""


class GatewayEvalReport(BaseModel):
    schema_version: str = "1.0"
    generated_by: str = "agentforge-inference-gateway-eval"
    generated_at: str
    dataset_path: str
    dataset_fingerprint: str
    total: int
    passed: int
    pass_rate: float
    control_coverage: float
    receipt_integrity_rate: float
    privacy_violations: int
    fallback_recovery_rate: float
    cache_control_pass_rate: float
    budget_control_pass_rate: float
    high_risk_bypass_rate: float
    shadow_non_interference_rate: float
    canary_target_percentage: float
    canary_observed_percentage: float
    canary_deviation_percentage_points: float
    total_simulated_cost_usd: float
    outcomes: list[GatewayEvalOutcome]
    report_fingerprint: str = ""


class _Provider:
    def __init__(self, script: list[str | Exception]):
        self.script = list(script)
        self.calls = 0

    async def generate(self, request, spec, max_completion_tokens):
        self.calls += 1
        value = self.script.pop(0) if self.script else f"healthy response from {spec.name}"
        if isinstance(value, Exception):
            raise value
        return ProviderResult(
            content=value,
            prompt_tokens=20,
            completion_tokens=10,
            latency_ms=10,
        )


def _spec(
    name: str,
    *,
    input_price: float = 1,
    output_price: float = 2,
    threshold: int = 1,
    cooldown: float = 30,
) -> ProviderSpec:
    return ProviderSpec(
        name=name,
        model=f"{name}-model-v1",
        input_cost_per_million=input_price,
        output_cost_per_million=output_price,
        failure_threshold=threshold,
        cooldown_seconds=cooldown,
        timeout_ms=100,
    )


def _request(
    scenario: GatewayEvalScenario,
    *,
    request_id: str | None = None,
    tenant: str = "tenant-alpha",
    prompt: str = "Explain durable agent checkpoints",
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id or f"eval-{scenario.id}",
        tenant_id=tenant,
        system="Use only approved project evidence.",
        prompt=prompt,
        preferred_provider="primary",
    )


def load_scenarios(path: Path) -> list[GatewayEvalScenario]:
    scenarios: list[GatewayEvalScenario] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            scenario = GatewayEvalScenario.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"invalid gateway scenario at line {line_number}: {exc}") from exc
        if scenario.id in seen:
            raise ValueError(f"duplicate gateway scenario {scenario.id!r}")
        seen.add(scenario.id)
        scenarios.append(scenario)
    if not scenarios:
        raise ValueError("gateway evaluation dataset is empty")
    return scenarios


def _outcome(
    scenario: GatewayEvalScenario,
    *,
    status: str,
    controls: list[str],
    receipt: InferenceReceipt,
    gateway: InferenceGateway,
    selected_provider: str = "",
    cache_hit: bool = False,
    leakage: int = 0,
    passed: bool,
) -> GatewayEvalOutcome:
    outcome = GatewayEvalOutcome(
        scenario_id=scenario.id,
        kind=scenario.kind,
        expected_status=scenario.expected_status,
        actual_status=status,
        expected_control=scenario.expected_control,
        observed_controls=controls,
        selected_provider=selected_provider,
        cache_hit=cache_hit,
        receipt_verified=gateway.verify_receipt(receipt),
        content_leakage_violations=leakage,
        cost_usd=receipt.tenant_daily_spend_usd,
        passed=passed
        and status == scenario.expected_status
        and scenario.expected_control in controls
        and gateway.verify_receipt(receipt)
        and leakage == 0,
    )
    outcome.evidence_fingerprint = _hash(
        outcome.model_dump(mode="json", exclude={"evidence_fingerprint"})
    )
    return outcome


async def evaluate_scenario(scenario: GatewayEvalScenario) -> GatewayEvalOutcome:
    kind = scenario.kind
    if kind in {"semantic_cache", "tenant_isolation"}:
        provider = _Provider(["cached answer", "isolated answer"])
        gateway = InferenceGateway(
            GatewayPolicy(semantic_cache_threshold=0.5),
            [_spec("primary")],
            {"primary": provider},
        )
        first = await gateway.execute(_request(scenario))
        if kind == "semantic_cache":
            result = await gateway.execute(
                _request(
                    scenario,
                    request_id=f"eval-{scenario.id}-2",
                    prompt="Explain the durable agent checkpoints",
                )
            )
            controls = ["cache_hit"] if result.receipt.cache_hit else []
            passed = provider.calls == 1 and result.content == first.content
        else:
            result = await gateway.execute(
                _request(
                    scenario,
                    request_id=f"eval-{scenario.id}-2",
                    tenant="tenant-beta",
                )
            )
            controls = ["tenant_isolation"] if not result.receipt.cache_hit else []
            passed = provider.calls == 2 and result.content != first.content
        return _outcome(
            scenario,
            status="success",
            controls=controls,
            receipt=result.receipt,
            gateway=gateway,
            selected_provider=result.receipt.selected_provider,
            cache_hit=result.receipt.cache_hit,
            passed=passed,
        )

    if kind == "high_risk":
        primary, canary, shadow = _Provider(["safe"]), _Provider(["canary"]), _Provider(["shadow"])
        gateway = InferenceGateway(
            GatewayPolicy(
                canary_provider="canary",
                canary_percentage=100,
                shadow_provider="shadow",
                shadow_enabled=True,
            ),
            [_spec("primary"), _spec("canary"), _spec("shadow")],
            {"primary": primary, "canary": canary, "shadow": shadow},
        )
        request = _request(scenario)
        request.high_risk = True
        result = await gateway.execute(request)
        passed = not result.receipt.canary_selected and result.receipt.shadow is None
        controls = ["high_risk_bypass"] if passed else []
        return _outcome(
            scenario,
            status="success",
            controls=controls,
            receipt=result.receipt,
            gateway=gateway,
            selected_provider=result.receipt.selected_provider,
            passed=passed and canary.calls == 0 and shadow.calls == 0,
        )

    if kind in {"fallback", "circuit_open", "half_open"}:
        now = [1_700_000_000.0]
        primary = _Provider([RuntimeError("down"), "recovered"])
        fallback = _Provider(["fallback one", "fallback two"])
        gateway = InferenceGateway(
            GatewayPolicy(semantic_cache_enabled=False, fallback_providers=["fallback"]),
            [_spec("primary"), _spec("fallback")],
            {"primary": primary, "fallback": fallback},
            clock=lambda: now[0],
        )
        first = await gateway.execute(_request(scenario))
        if kind == "fallback":
            result = first
            controls = ["provider_fallback"]
            passed = result.receipt.selected_provider == "fallback"
        elif kind == "circuit_open":
            result = await gateway.execute(_request(scenario, request_id=f"eval-{scenario.id}-2"))
            passed = result.receipt.attempts[0].outcome == "circuit_open"
            controls = ["circuit_open"] if passed else []
        else:
            now[0] += 31
            result = await gateway.execute(_request(scenario, request_id=f"eval-{scenario.id}-2"))
            passed = result.receipt.selected_provider == "primary"
            controls = ["half_open_recovery"] if passed else []
        return _outcome(
            scenario,
            status="success",
            controls=controls,
            receipt=result.receipt,
            gateway=gateway,
            selected_provider=result.receipt.selected_provider,
            passed=passed,
        )

    if kind in {"budget_fallback", "budget_block"}:
        primary, cheap = _Provider(["expensive"]), _Provider(["affordable"])
        gateway = InferenceGateway(
            GatewayPolicy(
                daily_budget_usd=0.00002 if kind == "budget_fallback" else 0.000001,
                max_completion_tokens=100,
                semantic_cache_enabled=False,
                fallback_providers=["cheap"],
            ),
            [
                _spec("primary", input_price=10, output_price=10),
                _spec("cheap", input_price=0.01, output_price=0.01),
            ],
            {"primary": primary, "cheap": cheap},
        )
        try:
            result = await gateway.execute(_request(scenario))
        except GatewayRejectedError as exc:
            controls = ["budget_rejected"]
            return _outcome(
                scenario,
                status="blocked",
                controls=controls,
                receipt=exc.receipt,
                gateway=gateway,
                passed=kind == "budget_block" and primary.calls == cheap.calls == 0,
            )
        controls = ["budget_fallback"]
        return _outcome(
            scenario,
            status="success",
            controls=controls,
            receipt=result.receipt,
            gateway=gateway,
            selected_provider=result.receipt.selected_provider,
            passed=(
                kind == "budget_fallback"
                and result.receipt.selected_provider == "cheap"
                and primary.calls == 0
            ),
        )

    if kind == "shadow":
        primary, shadow = _Provider(["durable checkpoint"]), _Provider(["checkpoint is durable"])
        gateway = InferenceGateway(
            GatewayPolicy(semantic_cache_enabled=False, shadow_provider="shadow", shadow_enabled=True),
            [_spec("primary"), _spec("shadow")],
            {"primary": primary, "shadow": shadow},
        )
        result = await gateway.execute(_request(scenario))
        passed = result.content == "durable checkpoint" and bool(result.receipt.shadow and result.receipt.shadow.executed)
        return _outcome(
            scenario,
            status="success",
            controls=["shadow_non_interference"] if passed else [],
            receipt=result.receipt,
            gateway=gateway,
            selected_provider=result.receipt.selected_provider,
            passed=passed,
        )

    if kind == "canary":
        gateway = InferenceGateway(
            GatewayPolicy(canary_provider="canary", canary_percentage=20),
            [_spec("primary"), _spec("canary")],
            {"primary": _Provider([]), "canary": _Provider([])},
        )
        choices = [
            gateway._canary_selected(_request(scenario, request_id=f"canary-{index}"))
            for index in range(2000)
        ]
        repeated = [
            gateway._canary_selected(_request(scenario, request_id=f"canary-{index}"))
            for index in range(2000)
        ]
        observed = 100 * sum(choices) / len(choices)
        result = await gateway.execute(_request(scenario, request_id="canary-evidence"))
        passed = choices == repeated and abs(observed - 20) <= 2.5
        return _outcome(
            scenario,
            status="success",
            controls=["stable_canary"] if passed else [],
            receipt=result.receipt,
            gateway=gateway,
            selected_provider=result.receipt.selected_provider,
            passed=passed,
        )

    if kind == "prompt_limit":
        provider = _Provider(["unused"])
        gateway = InferenceGateway(
            GatewayPolicy(max_prompt_tokens=16),
            [_spec("primary")],
            {"primary": provider},
        )
        try:
            await gateway.execute(_request(scenario, prompt="x" * 1000))
        except GatewayRejectedError as exc:
            return _outcome(
                scenario,
                status="blocked",
                controls=["prompt_limit"],
                receipt=exc.receipt,
                gateway=gateway,
                passed=provider.calls == 0,
            )
        raise AssertionError("oversized prompt was not blocked")

    secret_prompt, secret_response = "secret-prompt-marker", "secret-response-marker"
    provider = _Provider([secret_response])
    gateway = InferenceGateway(
        GatewayPolicy(semantic_cache_enabled=False),
        [_spec("primary")],
        {"primary": provider},
    )
    result = await gateway.execute(_request(scenario, prompt=secret_prompt))
    receipt_json = result.receipt.model_dump_json()
    leakage = sum(marker in receipt_json for marker in (secret_prompt, secret_response))
    return _outcome(
        scenario,
        status="success",
        controls=["privacy_receipt"] if leakage == 0 else [],
        receipt=result.receipt,
        gateway=gateway,
        selected_provider=result.receipt.selected_provider,
        leakage=leakage,
        passed=leakage == 0,
    )


def verify_report(report: GatewayEvalReport) -> bool:
    outcomes_valid = all(
        item.evidence_fingerprint
        == _hash(item.model_dump(mode="json", exclude={"evidence_fingerprint"}))
        for item in report.outcomes
    )
    expected = _hash(report.model_dump(mode="json", exclude={"generated_at", "report_fingerprint"}))
    return outcomes_valid and report.report_fingerprint == expected


async def evaluate_gateway(
    scenarios: list[GatewayEvalScenario], dataset_path: str
) -> GatewayEvalReport:
    outcomes = [await evaluate_scenario(item) for item in scenarios]
    by_kind = {item.kind: item for item in outcomes}
    fallback_kinds = {"fallback", "circuit_open", "half_open"}
    cache_kinds = {"semantic_cache", "tenant_isolation"}
    budget_kinds = {"budget_fallback", "budget_block", "prompt_limit"}
    observed_canary = float(by_kind["canary"].cost_usd * 0)
    # The canary evaluator stores its observed fraction in the sealed receipt's
    # cache-similarity field; the outcome intentionally contains no request IDs.
    scenario = next(item for item in scenarios if item.kind == "canary")
    probe_gateway = InferenceGateway(
        GatewayPolicy(canary_provider="canary", canary_percentage=20),
        [_spec("primary"), _spec("canary")],
        {"primary": _Provider([]), "canary": _Provider([])},
    )
    assignments = [
        probe_gateway._canary_selected(_request(scenario, request_id=f"canary-{index}"))
        for index in range(2000)
    ]
    observed_canary = 100 * sum(assignments) / len(assignments)
    report = GatewayEvalReport(
        generated_at=datetime.now(UTC).isoformat(),
        dataset_path=dataset_path,
        dataset_fingerprint=_hash([item.model_dump(mode="json") for item in scenarios]),
        total=len(outcomes),
        passed=sum(item.passed for item in outcomes),
        pass_rate=sum(item.passed for item in outcomes) / len(outcomes),
        control_coverage=len({control for item in outcomes for control in item.observed_controls})
        / len(scenarios),
        receipt_integrity_rate=sum(item.receipt_verified for item in outcomes) / len(outcomes),
        privacy_violations=sum(item.content_leakage_violations for item in outcomes),
        fallback_recovery_rate=statistics.fmean(
            by_kind[kind].passed for kind in fallback_kinds
        ),
        cache_control_pass_rate=statistics.fmean(by_kind[kind].passed for kind in cache_kinds),
        budget_control_pass_rate=statistics.fmean(by_kind[kind].passed for kind in budget_kinds),
        high_risk_bypass_rate=float(by_kind["high_risk"].passed),
        shadow_non_interference_rate=float(by_kind["shadow"].passed),
        canary_target_percentage=20,
        canary_observed_percentage=observed_canary,
        canary_deviation_percentage_points=abs(observed_canary - 20),
        total_simulated_cost_usd=sum(item.cost_usd for item in outcomes),
        outcomes=outcomes,
    )
    report.report_fingerprint = _hash(
        report.model_dump(mode="json", exclude={"generated_at", "report_fingerprint"})
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the AgentForge inference gateway")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("evals/datasets/model_gateway_scenarios.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluations/model-gateway/latest.json")
    )
    parser.add_argument("--min-pass-rate", type=float, default=1)
    parser.add_argument("--max-privacy-violations", type=int, default=0)
    parser.add_argument("--max-canary-deviation", type=float, default=2.5)
    args = parser.parse_args()
    scenarios = load_scenarios(args.dataset)
    report = asyncio.run(evaluate_gateway(scenarios, args.dataset.as_posix()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report.passed,
                "total": report.total,
                "pass_rate": report.pass_rate,
                "privacy_violations": report.privacy_violations,
                "canary_observed_percentage": report.canary_observed_percentage,
                "report": args.output.as_posix(),
            }
        )
    )
    if (
        not verify_report(report)
        or report.pass_rate < args.min_pass_rate
        or report.privacy_violations > args.max_privacy_violations
        or report.canary_deviation_percentage_points > args.max_canary_deviation
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
