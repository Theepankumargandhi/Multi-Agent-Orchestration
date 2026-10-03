import json
import sys
from pathlib import Path

import pytest

from evals.contextual_bandit import (
    BanditArtifact,
    ContextualBanditPolicy,
    check_policy_reproducibility,
    default_actions,
    evaluate_policy,
    load_events,
    main,
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


def test_checked_in_policy_reproduces_with_strict_lineage_and_bounded_roundoff():
    artifact = _artifact()
    before = artifact.model_dump()
    comparison = check_policy_reproducibility(artifact, Path("evals/experiments/contextual_bandit_policy.json"))
    assert comparison["reproducible"] and comparison["absolute_tolerance"] == 1e-12
    assert comparison["maximum_coefficient_delta"] <= 1e-12
    assert comparison["generated_fingerprint"] == artifact.artifact_fingerprint
    assert artifact.model_dump() == before


@pytest.mark.parametrize("component", ["theta", "covariance"])
def test_roundoff_acceptance_keeps_independent_exact_digests(tmp_path, component):
    expected = _artifact()
    target = tmp_path / "reference.json"
    expected.save(target)
    actual = expected.model_copy(deep=True)
    if component == "theta":
        actual.models["economy"].theta[0] += 5e-13
    else:
        actual.models["economy"].covariance[0][0] += 5e-13
    actual.seal()
    comparison = check_policy_reproducibility(actual, target)
    assert not comparison["exact_match"] and 0 < comparison["maximum_coefficient_delta"] <= 1e-12
    assert actual.verify() and expected.verify()
    assert comparison["generated_fingerprint"] != comparison["reference_fingerprint"]
    assert BanditArtifact.load(target) == expected


@pytest.mark.parametrize("component,value", [
    ("theta", 1e-9), ("covariance", 1e-9), ("theta", float("nan")),
    ("theta", float("inf")), ("covariance", float("inf")),
])
def test_real_coefficient_drift_or_nonfinite_values_fail_closed(tmp_path, component, value):
    expected = _artifact()
    target = tmp_path / "reference.json"
    expected.save(target)
    actual = expected.model_copy(deep=True)
    if component == "theta":
        actual.models["economy"].theta[0] += value
    else:
        actual.models["economy"].covariance[0][0] += value
    actual.seal()
    with pytest.raises(ValueError, match="coefficients"):
        check_policy_reproducibility(actual, target)


@pytest.mark.parametrize("field", ["training_fingerprint", "cost_weight", "actions", "observations", "shape"])
def test_even_resealed_lineage_configuration_and_structure_changes_fail(tmp_path, field):
    expected = _artifact()
    target = tmp_path / "reference.json"
    expected.save(target)
    actual = expected.model_copy(deep=True)
    if field == "training_fingerprint":
        actual.training_fingerprint = "changed-dataset"
    elif field == "cost_weight":
        actual.cost_weight += 5e-13  # Configuration has no numerical tolerance.
    elif field == "actions":
        actual.actions[0].allow_high_risk = not actual.actions[0].allow_high_risk
    elif field == "observations":
        actual.models["economy"].observations += 1
    else:
        actual.models["economy"].covariance.pop()
    actual.seal()
    with pytest.raises(ValueError, match="identity|structure"):
        check_policy_reproducibility(actual, target)


@pytest.mark.parametrize("which", ["generated", "reference"])
def test_unsealed_tiny_changes_still_fail_exact_integrity(tmp_path, which):
    expected = _artifact()
    target = tmp_path / "reference.json"
    expected.save(target)
    actual = expected.model_copy(deep=True)
    if which == "generated":
        actual.models["economy"].theta[0] += 5e-13
    else:
        payload = json.loads(target.read_text())
        payload["models"]["economy"]["theta"][0] += 5e-13
        target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity"):
        check_policy_reproducibility(actual, target)


def test_cli_reports_roundoff_without_rebinding_hashes_or_skipping_promotion(tmp_path, monkeypatch, capsys):
    expected = _artifact()
    reference = tmp_path / "reference.json"
    expected.save(reference)
    actual = expected.model_copy(deep=True)
    actual.models["economy"].theta[0] += 5e-13
    actual.seal()
    monkeypatch.setattr("evals.contextual_bandit.train_policy", lambda *args: actual)
    output, report = tmp_path / "actual.json", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["bandit", str(DATASET), "--check", str(reference),
        "--artifact", str(output), "--report", str(report), "--require-promotion"])
    assert main() == 0
    comparison = json.loads(capsys.readouterr().out.splitlines()[0])["reproducibility"]
    assert not comparison["exact_match"]
    assert BanditArtifact.load(output).artifact_fingerprint == actual.artifact_fingerprint != expected.artifact_fingerprint
    assert json.loads(report.read_text())["policy_fingerprint"] == actual.artifact_fingerprint
    monkeypatch.setattr("evals.contextual_bandit.evaluate_policy", lambda *args: evaluate_policy(
        actual, load_events(DATASET), minimum_effective_sample_size=100))
    assert main() == 2  # Numerical reproducibility is not policy promotion.


def test_cli_rejects_real_drift_before_writing_outputs(tmp_path, monkeypatch):
    expected = _artifact()
    reference = tmp_path / "reference.json"
    expected.save(reference)
    actual = expected.model_copy(deep=True)
    actual.models["economy"].theta[0] += 1e-6
    actual.seal()
    monkeypatch.setattr("evals.contextual_bandit.train_policy", lambda *args: actual)
    output, report = tmp_path / "actual.json", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["bandit", str(DATASET), "--check", str(reference),
        "--artifact", str(output), "--report", str(report)])
    with pytest.raises(SystemExit, match="not reproducible.*coefficients"):
        main()
    assert not output.exists() and not report.exists()
