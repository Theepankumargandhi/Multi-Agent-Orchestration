import importlib
import json
from pathlib import Path

import pytest

from agent.offline_rl import (
    ConservativePlanningPolicy,
    OfflineRLArtifact,
    PlanningState,
    load_planning_replay,
    safe_actions,
    train_offline_rl_policy,
)
from agent.process_reward import ProcessRewardScorer
from agent.search_planner import SearchRequest, VerifierGuidedMCTS
from evals.offline_rl_evaluation import evaluate_offline_rl, verify_report

DATASET = Path("evals/datasets/planning_replay.jsonl")
PROCESS_REWARD_ARTIFACT = Path("evals/experiments/process_reward_model.json")


@pytest.fixture(scope="module")
def trained_policy():
    transitions = load_planning_replay(DATASET)
    artifact = train_offline_rl_policy(transitions)
    return transitions, artifact, ConservativePlanningPolicy(artifact)


def test_cql_training_is_reproducible_and_integrity_sealed(tmp_path: Path):
    transitions = load_planning_replay(DATASET)
    first = train_offline_rl_policy(transitions, epochs=50)
    second = train_offline_rl_policy(transitions, epochs=50)
    assert first.model_dump() == second.model_dump()

    path = tmp_path / "offline-rl-policy.json"
    first.save(path)
    assert OfflineRLArtifact.load(path).verify()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["conservative_alpha"] += 0.01
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        OfflineRLArtifact.load(path)


def test_sequential_off_policy_gate_promotes_only_supported_policy(trained_policy):
    transitions, artifact, _ = trained_policy
    report = evaluate_offline_rl(artifact, transitions)
    assert report.doubly_robust_return > report.behavior_return
    assert report.confidence_low > report.behavior_return
    assert report.effective_sample_size >= 2
    assert report.support_violations == 0
    assert report.heldout_action_accuracy >= 0.8
    assert report.unsafe_target_action_rate == 0
    assert report.supported_planning_success
    assert report.ood_uniform_fallback
    assert report.promoted
    assert verify_report(report)


def test_policy_uses_safety_mask_conservative_values_and_uniform_ood_fallback(
    trained_policy,
):
    _, _, policy = trained_policy
    high_risk_state = PlanningState(
        evidence_count=1,
        confidence=0.47,
    )
    actions = safe_actions(high_risk_state, high_risk=True)
    assert "answer" not in actions
    estimate = policy.priors(
        route="web",
        high_risk=True,
        state=high_risk_state,
        actions=actions,
    )
    probabilities = {item.action: item.probability for item in estimate.actions}
    assert probabilities["retrieve"] > probabilities["abstain"]
    assert sum(probabilities.values()) == pytest.approx(1)

    shifted = policy.priors(
        route="unseen-route",
        high_risk=False,
        state=PlanningState(evidence_count=1, confidence=0.5),
        actions=["reason", "abstain"],
    )
    assert shifted.out_of_distribution
    assert [item.probability for item in shifted.actions] == pytest.approx([0.5, 0.5])


def test_offline_rl_puct_is_auditable_and_preserves_safe_completion(trained_policy):
    _, artifact, policy = trained_policy
    plan = VerifierGuidedMCTS(
        ProcessRewardScorer.load(PROCESS_REWARD_ARTIFACT),
        planning_policy=policy,
    ).plan(
        SearchRequest(
            request_id="offline-rl-puct-test",
            route="rag",
            evidence_count=1,
            confidence=0.55,
            token_budget=1000,
        )
    )
    assert plan.terminal_action == "answer"
    assert plan.tree_policy_strategy == "offline_rl_puct"
    assert plan.offline_rl_policy_fingerprint == artifact.artifact_fingerprint
    assert not plan.offline_rl_ood
    assert plan.verify()


def test_runtime_policy_loader_hot_reloads_and_rejects_tampering(
    monkeypatch, tmp_path: Path, trained_policy
):
    _, artifact, _ = trained_policy
    path = tmp_path / "offline-rl-policy.json"
    artifact.save(path)
    research = importlib.import_module("agent.research_assistant")
    monkeypatch.setattr(research, "OFFLINE_RL_POLICY_PATH", path)
    monkeypatch.setattr(research, "_offline_rl_policy", None)
    monkeypatch.setattr(research, "_offline_rl_policy_mtime_ns", -1)
    first = research._load_offline_rl_policy()
    assert research._load_offline_rl_policy() is first

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["policy_temperature"] += 0.01
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(research, "_offline_rl_policy", None)
    with pytest.raises(ValueError, match="integrity"):
        research._load_offline_rl_policy()
