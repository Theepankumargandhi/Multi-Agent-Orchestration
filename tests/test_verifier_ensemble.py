import importlib
import json
from pathlib import Path

import pytest

from agent.process_reward import load_traces
from agent.search_planner import SearchRequest, VerifierGuidedMCTS
from agent.verifier_active_learning import VerifierActiveLearningQueue
from agent.verifier_ensemble import (
    EnsembleProcessRewardScorer,
    ProcessRewardEnsembleArtifact,
    train_process_reward_ensemble,
)
from evals.verifier_uncertainty_evaluation import (
    corrupt_one_member,
    evaluate_verifier_uncertainty,
    load_scenarios,
    verify_report,
)

TRACES = Path("evals/datasets/process_reward_trajectories.jsonl")
SCENARIOS = Path("evals/datasets/verifier_uncertainty_scenarios.jsonl")


@pytest.fixture(scope="module")
def trained():
    traces = load_traces(TRACES)
    return traces, train_process_reward_ensemble(traces)


def _request():
    return SearchRequest(
        request_id="uncertainty-aware-search",
        route="rag",
        evidence_count=2,
        confidence=0.55,
        token_budget=1000,
    )


def test_ensemble_training_is_reproducible_and_integrity_checked(tmp_path: Path):
    traces = load_traces(TRACES)
    first = train_process_reward_ensemble(traces, member_count=3, epochs=60)
    second = train_process_reward_ensemble(traces, member_count=3, epochs=60)
    assert first.model_dump() == second.model_dump()

    path = tmp_path / "ensemble.json"
    first.save(path)
    assert ProcessRewardEnsembleArtifact.load(path).verify()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["risk_multiplier"] += 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        ProcessRewardEnsembleArtifact.load(path)


def test_ensemble_exposes_epistemic_uncertainty_and_detects_shift(trained):
    traces, artifact = trained
    trace = next(item for item in traces if item.trace_id == "te-medical-grounded")
    clean = EnsembleProcessRewardScorer(artifact).score_steps_with_uncertainty(
        trace.steps, trace.high_risk
    )
    shifted = EnsembleProcessRewardScorer(
        corrupt_one_member(artifact)
    ).score_steps_with_uncertainty(trace.steps, trace.high_risk)
    assert clean.out_of_distribution is False
    assert shifted.out_of_distribution is True
    assert shifted.standard_deviation > clean.standard_deviation
    assert clean.lower_confidence_bound < clean.mean


def test_risk_sensitive_mcts_answers_clean_case_and_abstains_under_shift(trained):
    _, artifact = trained
    clean = VerifierGuidedMCTS(EnsembleProcessRewardScorer(artifact)).plan(_request())
    shifted = VerifierGuidedMCTS(
        EnsembleProcessRewardScorer(corrupt_one_member(artifact))
    ).plan(_request())
    assert clean.terminal_action == "answer"
    assert clean.verifier_ood is False
    assert clean.risk_adjusted_reward <= clean.verifier_mean
    assert shifted.terminal_action == "abstain"
    assert shifted.verifier_ood is True
    assert shifted.verify()


def test_active_learning_queue_is_private_deduplicated_and_reviewable(
    trained, tmp_path: Path
):
    _, artifact = trained
    request = _request()
    plan = VerifierGuidedMCTS(
        EnsembleProcessRewardScorer(corrupt_one_member(artifact))
    ).plan(request)
    queue = VerifierActiveLearningQueue(tmp_path / "review.sqlite3")
    first = queue.enqueue(
        plan,
        request,
        uncertainty_threshold=0.08,
        created_at="2026-01-01T00:00:00+00:00",
    )
    second = queue.enqueue(
        plan,
        request,
        uncertainty_threshold=0.08,
        created_at="2026-01-01T00:00:00+00:00",
    )
    assert first is not None and first.verify()
    assert second is not None and second.event_id == first.event_id
    assert len(queue.pending()) == 1
    assert "uncertainty-aware-search" not in first.model_dump_json()
    assert queue.review(
        first.event_id,
        "ambiguous",
        "reviewer@example.com",
        reviewed_at="2026-01-02T00:00:00+00:00",
    )
    assert queue.pending() == []
    assert queue.counts() == {"reviewed": 1}
    exported = queue.export_reviewed(tmp_path / "reviewed.jsonl")
    assert len(exported) == 1
    assert exported[0].review_status == "human_reviewed"
    assert exported[0].query == "metadata-only active-learning trajectory"
    assert "reviewer@example.com" not in (tmp_path / "reviewed.jsonl").read_text(
        encoding="utf-8"
    )


def test_shift_ablation_measures_detection_containment_and_review_capture(trained):
    traces, artifact = trained
    report = evaluate_verifier_uncertainty(
        artifact, traces, load_scenarios(SCENARIOS)
    )
    assert report.point_shift_unsafe_rate == 1
    assert report.ensemble_shift_unsafe_rate == 0
    assert report.normal_selective_accuracy == 1
    assert report.shift_detection_rate == 1
    assert report.shift_containment_rate == 1
    assert report.active_learning_capture_rate == 1
    assert report.promoted
    assert verify_report(report)


def test_runtime_hot_loads_ensemble_and_conservative_candidate_score(
    trained, tmp_path: Path, monkeypatch
):
    research = importlib.import_module("agent.research_assistant")
    _, artifact = trained
    path = tmp_path / "ensemble.json"
    artifact.save(path)
    monkeypatch.setattr(research, "VERIFIER_ENSEMBLE_ENABLED", True)
    monkeypatch.setattr(research, "VERIFIER_ENSEMBLE_PATH", path)
    monkeypatch.setattr(research, "_process_reward_scorer", None)
    monkeypatch.setattr(research, "_process_reward_mtime_ns", -1)
    scorer = research._load_process_reward_scorer()
    assert isinstance(scorer, EnsembleProcessRewardScorer)
    assert scorer.artifact.artifact_fingerprint == artifact.artifact_fingerprint
