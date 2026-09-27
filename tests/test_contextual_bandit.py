import json
from pathlib import Path

import pytest

from evals.contextual_bandit import (
    BanditArtifact,
    ContextualBanditPolicy,
    default_actions,
    evaluate_policy,
    load_events,
    train_policy,
)

DATASET = Path("evals/datasets/contextual_bandit_feedback.jsonl")


def _artifact():
    return train_policy(load_events(DATASET), default_actions())


def test_policy_is_reproducible_integrity_checked_and_round_trips(tmp_path: Path):
    first = _artifact()
    second = _artifact()
    assert first.model_dump() == second.model_dump()
    path = tmp_path / "policy.json"
    first.save(path)
    assert BanditArtifact.load(path).artifact_fingerprint == first.artifact_fingerprint

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"]["economy"]["theta"][0] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        BanditArtifact.load(path)


def test_runtime_policy_enforces_risk_and_resource_constraints():
    policy = ContextualBanditPolicy(_artifact())
    simple = policy.decide("write a brief greeting message", request_id="simple")
    assert simple.action == "economy"
    assert simple.propensity == 1

    risky = policy.decide(
        "delete production credentials", request_id="risk", high_risk=True
    )
    assert risky.action != "economy"
    assert "economy" not in risky.feasible_actions

    constrained = policy.decide(
        "debug this API function", request_id="budget", max_cost_usd=0.01
    )
    assert constrained.action == "economy"
    with pytest.raises(ValueError, match="no bandit action"):
        policy.decide("hello", request_id="none", max_cost_usd=0.0001)


def test_epsilon_policy_emits_replayable_propensity():
    policy = ContextualBanditPolicy(_artifact())
    first = policy.decide("analyze repository architecture", request_id="stable", epsilon=0.2)
    second = policy.decide("analyze repository architecture", request_id="stable", epsilon=0.2)
    assert first == second
    assert any(first.propensity == pytest.approx(value) for value in [0.2 / 3, 0.8 + 0.2 / 3])


def test_offline_policy_evaluation_uses_ips_snips_and_doubly_robust_gate():
    events = load_events(DATASET)
    report = evaluate_policy(_artifact(), events)
    assert report.events == 12
    assert report.matched_events == 4
    assert report.effective_sample_size == pytest.approx(4)
    assert report.ips_utility > report.behavior_utility
    assert report.snips_utility > report.behavior_utility
    assert report.doubly_robust_utility > report.behavior_utility
    assert report.target_safety_violations == 0
    assert report.promoted is True


def test_gate_rejects_insufficient_action_support():
    events = load_events(DATASET)
    unsupported = [item.model_copy(update={"propensity": 1.0}) for item in events]
    report = evaluate_policy(
        _artifact(), unsupported, minimum_effective_sample_size=10
    )
    assert report.promoted is False
    assert any("sample size" in reason for reason in report.reasons)
