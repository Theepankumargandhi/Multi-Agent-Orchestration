import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.process_reward import (
    ProcessRewardArtifact,
    ProcessRewardScorer,
    load_traces,
    train_process_reward_model,
)
from evals.process_reward_evaluation import evaluate_process_reward, verify_report

DATASET = Path("evals/datasets/process_reward_trajectories.jsonl")


def _trained():
    traces = load_traces(DATASET)
    return traces, train_process_reward_model(traces)


def test_training_is_reproducible_integrity_checked_and_content_free(tmp_path: Path):
    traces, first = _trained()
    second = train_process_reward_model(traces)
    assert first.model_dump() == second.model_dump()
    assert first.human_labeled_steps == 0
    assert "Explain the indexed architecture" not in first.model_dump_json()

    path = tmp_path / "process-reward.json"
    first.save(path)
    assert ProcessRewardArtifact.load(path).artifact_fingerprint == first.artifact_fingerprint
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["weights"][0] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        ProcessRewardArtifact.load(path)


def test_step_verifier_prunes_failures_and_assigns_counterfactual_credit():
    traces, artifact = _trained()
    scorer = ProcessRewardScorer(artifact)
    good = next(trace for trace in traces if trace.trace_id == "te-code-grounded")
    unsafe = next(trace for trace in traces if trace.trace_id == "te-code-unsafe")

    good_receipt = scorer.evaluate_trace(good)
    unsafe_receipt = scorer.evaluate_trace(unsafe)
    assert good_receipt.accepted is True
    assert unsafe_receipt.accepted is False
    assert unsafe_receipt.pruned_at_step == "tool"
    assert any(abs(item.credit) > 0 for item in good_receipt.step_rewards)
    assert scorer.verify_receipt(good_receipt)


def test_scoring_never_reads_held_out_outcome_labels():
    traces, artifact = _trained()
    scorer = ProcessRewardScorer(artifact)
    trace = next(item for item in traces if item.trace_id == "te-medical-grounded")
    changed = trace.model_copy(update={"outcome_quality": 0.0, "safe": False})
    assert scorer.evaluate_trace(trace).trajectory_score == pytest.approx(
        scorer.evaluate_trace(changed).trajectory_score
    )


def test_held_out_gate_measures_selection_lift_safety_and_pruning():
    traces, artifact = _trained()
    report = evaluate_process_reward(artifact, traces)
    assert report.test_groups == 5
    assert report.baseline_success_rate == pytest.approx(0.2)
    assert report.verifier_success_rate == 1
    assert report.baseline_unsafe_selection_rate == pytest.approx(0.8)
    assert report.verifier_unsafe_selection_rate == 0
    assert report.failed_trajectory_prune_rate == 1
    assert report.promoted is True
    assert verify_report(report)

    report.success_rate_delta = -1
    assert not verify_report(report)


def test_gate_rejects_unachievable_lift_requirement():
    traces, artifact = _trained()
    report = evaluate_process_reward(artifact, traces, minimum_success_delta=0.9)
    assert report.promoted is False
    assert any("lift" in reason for reason in report.reasons)


def test_runtime_candidate_scoring_hot_loads_the_verified_artifact(tmp_path, monkeypatch):
    research = importlib.import_module("agent.research_assistant")

    _, artifact = _trained()
    path = tmp_path / "process-reward.json"
    artifact.save(path)
    monkeypatch.setattr(research, "PROCESS_REWARD_MODEL_ENABLED", True)
    monkeypatch.setattr(research, "PROCESS_REWARD_MODEL_PATH", path)
    monkeypatch.setattr(research, "_process_reward_scorer", None)
    monkeypatch.setattr(research, "_process_reward_mtime_ns", -1)
    report = SimpleNamespace(
        evidence_count=3,
        citation_precision=1.0,
        claims=[],
        action="pass",
        confidence=0.9,
        claim_coverage=1.0,
    )
    score = research._candidate_process_reward(report, high_risk=False)
    assert score is not None and score > 0.8
