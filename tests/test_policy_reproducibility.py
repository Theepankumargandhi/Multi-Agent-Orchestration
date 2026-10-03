"""Cross-runtime numeric agreement never substitutes for exact integrity or behavior."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from agent.process_reward import ProcessRewardArtifact, load_traces, train_process_reward_model
from agent.verifier_ensemble import ProcessRewardEnsembleArtifact, train_process_reward_ensemble
from evals.process_reward_evaluation import compare_process_artifacts
from evals.process_reward_evaluation import main as reward_main
from evals.verifier_uncertainty_evaluation import (
    VerifierUncertaintyReport,
    compare_ensemble_artifacts,
    compare_uncertainty_reports,
)
from evals.verifier_uncertainty_evaluation import (
    main as ensemble_main,
)

REWARD = Path("evals/experiments/process_reward_model.json")
ENSEMBLE = Path("evals/experiments/process_reward_ensemble.json")
REPORT = Path("evals/experiments/verifier_uncertainty.report.json")
DATASET = Path("evals/datasets/process_reward_trajectories.jsonl")


def _seal_report(report):
    report.report_fingerprint = hashlib.sha256(json.dumps(
        report.model_dump(mode="json", exclude={"report_fingerprint"}),
        sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@pytest.fixture(scope="module")
def frozen():
    return (ProcessRewardArtifact.load(REWARD), ProcessRewardEnsembleArtifact.load(ENSEMBLE),
            VerifierUncertaintyReport.model_validate_json(REPORT.read_text()))


def test_real_trainers_agree_with_frozen_references_across_runtimes(frozen):
    traces = load_traces(DATASET)
    actual = train_process_reward_model(traces)
    ensemble = train_process_reward_ensemble(traces)
    assert compare_process_artifacts(actual, frozen[0])["reproducible"]
    assert compare_ensemble_artifacts(ensemble, frozen[1])["reproducible"]


def test_roundoff_preserves_distinct_exact_artifact_and_report_bindings(frozen):
    reference, ensemble_reference, report_reference = frozen
    actual = reference.model_copy(deep=True)
    actual.weights[0] += 5e-13
    actual.seal()
    assert not compare_process_artifacts(actual, reference)["exact_match"]
    ensemble = ensemble_reference.model_copy(deep=True)
    ensemble.members[0].weights[0] += 5e-13
    ensemble.members[0].seal()
    ensemble.seal()
    assert not compare_ensemble_artifacts(ensemble, ensemble_reference)["exact_match"]
    report = report_reference.model_copy(deep=True)
    report.ensemble_fingerprint = ensemble.artifact_fingerprint
    report.outcomes[0].selected_uncertainty += 5e-13
    _seal_report(report)
    comparison = compare_uncertainty_reports(report, report_reference, ensemble, ensemble_reference)
    assert 0 < comparison["maximum_uncertainty_delta"] <= 1e-12
    assert comparison["generated_fingerprint"] != comparison["reference_fingerprint"]
    assert ensemble.verify() and ensemble_reference.verify()


@pytest.mark.parametrize("change", ["unsealed", "weights", "nonfinite", "dataset", "threshold", "shape"])
def test_reward_reproduction_rejects_tampering_drift_and_metadata_changes(frozen, change):
    reference = frozen[0]
    actual = reference.model_copy(deep=True)
    if change in {"unsealed", "weights", "nonfinite"}:
        actual.weights[0] += {"unsealed": 5e-13, "weights": 1e-6, "nonfinite": float("inf")}[change]
    elif change == "dataset":
        actual.dataset_fingerprint = "changed"
    elif change == "threshold":
        actual.stopping_threshold += 5e-13
    else:
        actual.weights.pop()
    if change != "unsealed":
        actual.seal()
    with pytest.raises(ValueError):
        compare_process_artifacts(actual, reference)


@pytest.mark.parametrize("change", ["parent_digest", "member_digest", "member_lineage", "member_weight", "risk", "count"])
def test_ensemble_reproduction_checks_parent_and_each_member(frozen, change):
    reference = frozen[1]
    actual = reference.model_copy(deep=True)
    if change == "risk":
        actual.risk_multiplier += 5e-13
    elif change == "count":
        actual.members.pop()
    elif change == "member_lineage":
        actual.members[0].dataset_fingerprint = "changed"
        actual.members[0].seal()
    elif change == "member_weight":
        actual.members[0].weights[0] += 1e-6
        actual.members[0].seal()
    else:
        actual.members[0].weights[0] += 5e-13
        if change == "parent_digest":
            actual.members[0].seal()
    if change != "parent_digest":
        actual.seal()
    with pytest.raises(ValueError):
        compare_ensemble_artifacts(actual, reference)


@pytest.mark.parametrize("change", ["digest", "binding", "behavior", "dataset", "metric", "promotion", "uncertainty", "nan"])
def test_report_reproduction_never_tolerates_behavior_or_lineage_changes(frozen, change):
    artifact, reference = frozen[1:]
    actual = reference.model_copy(deep=True)
    if change in {"digest", "uncertainty", "nan"}:
        actual.outcomes[0].selected_uncertainty += {"digest": 5e-13, "uncertainty": 1e-6, "nan": float("nan")}[change]
    elif change == "binding":
        actual.ensemble_fingerprint = "changed"
    elif change == "behavior":
        actual.outcomes[0].ensemble_unsafe = not actual.outcomes[0].ensemble_unsafe
    elif change == "dataset":
        actual.dataset_fingerprint = "changed"
    elif change == "metric":
        actual.normal_selective_accuracy -= 5e-13
    else:
        actual.promoted = not actual.promoted
    if change != "digest":
        _seal_report(actual)
    with pytest.raises(ValueError):
        compare_uncertainty_reports(actual, reference, artifact)


def test_report_only_check_requires_same_exact_artifact_binding(frozen):
    artifact, reference = frozen[1:]
    actual = reference.model_copy(deep=True)
    changed = artifact.model_copy(deep=True)
    changed.members[0].weights[0] += 5e-13
    changed.members[0].seal()
    changed.seal()
    actual.ensemble_fingerprint = changed.artifact_fingerprint
    _seal_report(actual)
    with pytest.raises(ValueError, match="binding"):
        compare_uncertainty_reports(actual, reference, changed)


def test_reward_cli_uses_own_digest_and_keeps_promotion_required(frozen, tmp_path, monkeypatch):
    artifact = frozen[0].model_copy(deep=True)
    artifact.weights[0] += 5e-13
    artifact.seal()
    monkeypatch.setattr("evals.process_reward_evaluation.train_process_reward_model", lambda *args: artifact)
    output, report = tmp_path / "artifact.json", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["reward", str(DATASET), "--check", str(REWARD),
        "--artifact", str(output), "--output", str(report), "--require-promotion"])
    assert reward_main() == 0
    assert ProcessRewardArtifact.load(output).artifact_fingerprint == artifact.artifact_fingerprint
    from evals.process_reward_evaluation import evaluate_process_reward
    monkeypatch.setattr("evals.process_reward_evaluation.evaluate_process_reward", lambda *args:
        evaluate_process_reward(artifact, load_traces(DATASET), minimum_success_delta=0.9))
    assert reward_main() == 2


def test_ensemble_cli_checks_report_before_persisting_any_outputs(frozen, tmp_path, monkeypatch):
    actual = frozen[1].model_copy(deep=True)
    actual.members[0].weights[0] += 5e-13
    actual.members[0].seal()
    actual.seal()
    monkeypatch.setattr("evals.verifier_uncertainty_evaluation.train_process_reward_ensemble", lambda *args: actual)
    output, report = tmp_path / "artifact.json", tmp_path / "report.json"
    argv = ["ensemble", "--check-artifact", str(ENSEMBLE), "--check-report", str(REPORT),
        "--artifact", str(output), "--output", str(report), "--require-promotion"]
    monkeypatch.setattr(sys, "argv", argv)
    assert ensemble_main() == 0
    assert ProcessRewardEnsembleArtifact.load(output).artifact_fingerprint == actual.artifact_fingerprint
    assert json.loads(report.read_text())["ensemble_fingerprint"] == actual.artifact_fingerprint
    bad = frozen[2].model_copy(deep=True)
    bad.outcomes[0].ensemble_unsafe = not bad.outcomes[0].ensemble_unsafe
    _seal_report(bad)
    bad_reference = tmp_path / "bad-reference.json"
    bad_reference.write_text(bad.model_dump_json())
    monkeypatch.setattr(sys, "argv", ["ensemble", "--check-artifact", str(ENSEMBLE), "--check-report", str(bad_reference),
        "--artifact", str(tmp_path / "must-not-exist.json"), "--output", str(tmp_path / "must-not-exist-report.json")])
    with pytest.raises(SystemExit, match="behavioral"):
        ensemble_main()
    assert not (tmp_path / "must-not-exist.json").exists()
    assert not (tmp_path / "must-not-exist-report.json").exists()
