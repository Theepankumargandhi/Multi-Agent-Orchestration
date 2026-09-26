import json
from pathlib import Path

import pytest

from agent.model_gateway import InferenceReceipt, ProviderAttempt
from agent.online_evaluation import (
    DelayedFeedback,
    OnlineEvalPolicy,
    OnlineEvaluationController,
    OnlineInferenceEvent,
    append_online_event,
    event_from_gateway_receipt,
)
from evals.online_monitor_evaluation import (
    DEFAULT_DATASET,
    evaluate_dataset,
    load_dataset,
    verify_report,
)


def _event(
    event_id: str,
    variant: str = "control",
    *,
    quality: float | None = 0.9,
    safety: bool | None = False,
    latency: float = 100,
    cost: float = 0.002,
) -> OnlineInferenceEvent:
    return OnlineInferenceEvent(
        event_id=event_id,
        request_id=f"request-{event_id}",
        tenant_fingerprint="tenant-fingerprint-001",
        occurred_at=float(event_id.rsplit("-", 1)[-1]),
        variant=variant,
        selected_provider="primary" if variant == "control" else "candidate",
        outcome="success",
        quality_score=quality,
        safety_violation=safety,
        latency_ms=latency,
        cost_usd=cost,
        prompt_tokens=20,
        completion_tokens=10,
        receipt_fingerprint="a" * 64,
    )


def test_ingestion_is_exactly_once_and_conflicts_fail_closed():
    controller = OnlineEvaluationController()
    original = _event("event-001")
    assert controller.ingest(original)
    assert not controller.ingest(original)
    assert controller.duplicate_events == 1

    conflicting = original.model_copy(update={"cost_usd": 99, "event_fingerprint": ""})
    with pytest.raises(ValueError, match="conflicting duplicate"):
        controller.ingest(conflicting)


def test_delayed_feedback_is_tenant_safe_and_reseals_event():
    controller = OnlineEvaluationController(integrity_key=b"monitor-test-integrity-key")
    controller.ingest(_event("event-001", quality=None, safety=None))
    feedback = DelayedFeedback(
        feedback_id="feedback-001",
        event_id="event-001",
        tenant_fingerprint="tenant-fingerprint-001",
        quality_score=0.84,
        safety_violation=False,
        reviewer_fingerprint="reviewer-fingerprint-001",
        occurred_at=10,
    )
    assert controller.join_feedback(feedback)
    assert not controller.join_feedback(feedback)
    assert controller.events[0].quality_score == 0.84
    assert controller.verify_event(controller.events[0])

    mismatched = feedback.model_copy(
        update={
            "feedback_id": "feedback-002",
            "tenant_fingerprint": "different-tenant-001",
            "feedback_fingerprint": "",
        }
    )
    with pytest.raises(ValueError, match="tenant"):
        controller.join_feedback(mismatched)


def test_checked_in_stream_detects_drift_burn_rate_and_rolls_back():
    report = evaluate_dataset(DEFAULT_DATASET)
    windows = {item.name: item for item in report.windows}
    assert report.accepted_events == 20
    assert report.duplicate_events == 1
    assert report.joined_feedback == 2
    assert report.event_integrity_rate == 1
    assert report.privacy_violations == 0
    assert report.drift_detected
    assert report.provider_mix_js_divergence == 1
    assert windows["control_long"].quality_mean == pytest.approx(0.901)
    assert windows["canary_long"].quality_mean == pytest.approx(0.595)
    assert {item.metric for item in report.alerts if item.variant == "canary"} >= {
        "quality",
        "safety",
        "latency",
    }
    assert report.canary_decision.action == "rollback"
    assert report.canary_decision.quality_delta_ci_high < -0.28
    assert verify_report(report)


def test_healthy_canary_is_promoted_and_small_sample_is_held():
    policy = OnlineEvalPolicy(short_window=2, long_window=4, min_canary_samples=4)
    controller = OnlineEvaluationController(policy)
    for index in range(1, 5):
        controller.ingest(_event(f"event-b-{index:03d}", quality=0.9))
        controller.ingest(_event(f"event-c-{index:03d}", "canary", quality=0.9))
    assert controller.evaluate().canary_decision.action == "promote"

    held = OnlineEvaluationController(policy)
    held.ingest(_event("event-b-001"))
    held.ingest(_event("event-b-002"))
    held.ingest(_event("event-c-001", "canary"))
    assert held.evaluate().canary_decision.action == "hold"


def test_report_and_event_tampering_is_detected():
    controller = OnlineEvaluationController()
    controller.ingest(_event("event-001"))
    report = controller.evaluate()
    assert controller.verify_report(report)
    report.canary_decision.reason = "silently promote"
    assert not controller.verify_report(report)

    event = controller.events[0]
    event.cost_usd = 100
    assert not controller.verify_event(event)


def test_gateway_receipt_conversion_and_jsonl_export_never_include_content(tmp_path: Path):
    receipt = InferenceReceipt(
        request_id="request-receipt-001",
        tenant_fingerprint="tenant-fingerprint-001",
        policy_version="policy-v1",
        policy_fingerprint="b" * 64,
        prompt_fingerprint="c" * 64,
        response_fingerprint="d" * 64,
        selected_provider="fallback",
        selected_model="model-v2",
        canary_selected=True,
        high_risk=False,
        prompt_tokens=50,
        completion_tokens=20,
        cost_usd=0.004,
        attempts=[
            ProviderAttempt(
                provider="primary",
                model="model-v1",
                outcome="timeout",
                latency_ms=25,
                estimated_cost_usd=0,
                error_type="TimeoutError",
            ),
            ProviderAttempt(
                provider="fallback",
                model="model-v2",
                outcome="success",
                latency_ms=30,
                estimated_cost_usd=0.004,
            ),
        ],
        outcome="success",
        generated_at="2026-09-13T00:00:00+00:00",
        receipt_fingerprint="e" * 64,
    )
    integrity_key = b"online-monitor-integrity-key"
    event = event_from_gateway_receipt(receipt, integrity_key=integrity_key)
    assert event.variant == "canary"
    assert event.fallback_used
    assert event.latency_ms == 55
    verifier = OnlineEvaluationController(integrity_key=integrity_key)
    assert verifier.ingest(event)
    output = tmp_path / "online-events.jsonl"
    append_online_event(event, output)
    payload = output.read_text(encoding="utf-8")
    assert "prompt" not in json.loads(payload)
    assert "response" not in json.loads(payload)
    assert "model-v2" not in payload
    loaded, feedback, _, violations = load_dataset(output)
    assert len(loaded) == 1 and not feedback and violations == 0
    live_report = evaluate_dataset(output, integrity_key=integrity_key)
    assert live_report.accepted_events == 1
    assert verify_report(live_report, integrity_key=integrity_key)
