import importlib
import json
from pathlib import Path

import pytest

from agent.process_reward import ProcessRewardScorer
from agent.search_planner import SearchRequest, VerifierGuidedMCTS
from agent.world_model import (
    AgentWorldModel,
    WorldModelArtifact,
    load_transitions,
    train_world_model,
)
from evals.world_model_evaluation import (
    corrupt_one_world_model_member,
    evaluate_world_model,
    verify_report,
)

DATASET = Path("evals/datasets/world_model_transitions.jsonl")
PROCESS_REWARD_ARTIFACT = Path("evals/experiments/process_reward_model.json")


@pytest.fixture(scope="module")
def trained_world_model():
    examples = load_transitions(DATASET)
    artifact = train_world_model(examples)
    return examples, artifact, AgentWorldModel(artifact)


def test_training_is_reproducible_and_artifact_is_integrity_sealed(tmp_path: Path):
    examples = load_transitions(DATASET)
    first = train_world_model(examples, epochs=60)
    second = train_world_model(examples, epochs=60)
    assert first.model_dump() == second.model_dump()

    path = tmp_path / "world-model.json"
    first.save(path)
    assert WorldModelArtifact.load(path).verify()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["success_floor"] += 0.01
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        WorldModelArtifact.load(path)


def test_held_out_ablation_promotes_learned_dynamics(trained_world_model):
    examples, artifact, _ = trained_world_model
    report = evaluate_world_model(artifact, examples)
    assert report.learned_confidence_mae < report.heuristic_confidence_mae
    assert report.learned_success_brier < report.heuristic_success_brier
    assert report.ood_detection_rate == 1
    assert report.clean_planning_success
    assert report.ood_planning_abstained
    assert report.shifted_planning_abstained
    assert report.promoted
    assert verify_report(report)


def test_predictions_are_action_conditioned_and_detect_unknown_routes(
    trained_world_model,
):
    _, _, model = trained_world_model
    valid = model.predict(
        route="web",
        action="answer",
        high_risk=True,
        evidence_count=2,
        confidence=0.76,
        reasoned=True,
        verified=True,
    )
    invalid = model.predict(
        route="web",
        action="answer",
        high_risk=True,
        evidence_count=1,
        confidence=0.80,
        reasoned=True,
        verified=False,
    )
    shifted = model.predict(
        route="unseen-route",
        action="reason",
        high_risk=False,
        evidence_count=1,
        confidence=0.50,
        reasoned=False,
        verified=False,
    )
    assert valid.success_probability > invalid.success_probability
    assert invalid.confidence_delta_mean < valid.confidence_delta_mean
    assert shifted.out_of_distribution


def test_model_predictive_mcts_emits_auditable_metadata_and_abstains_on_ood(
    trained_world_model,
):
    _, artifact, model = trained_world_model
    scorer = ProcessRewardScorer.load(PROCESS_REWARD_ARTIFACT)
    clean = VerifierGuidedMCTS(scorer, world_model=model).plan(
        SearchRequest(
            request_id="supported-world-model-plan",
            route="rag",
            evidence_count=1,
            confidence=0.55,
            token_budget=1000,
        )
    )
    shifted = VerifierGuidedMCTS(scorer, world_model=model).plan(
        SearchRequest(
            request_id="shifted-world-model-plan",
            route="unseen-route",
            evidence_count=1,
            confidence=0.55,
            token_budget=1000,
        )
    )
    assert clean.terminal_action == "answer"
    assert clean.dynamics_strategy == "learned_world_model"
    assert clean.world_model_fingerprint == artifact.artifact_fingerprint
    assert clean.transition_success_lcb >= artifact.success_floor
    assert clean.verify()
    assert shifted.terminal_action == "abstain"
    assert shifted.world_model_ood
    assert shifted.verify()

    member_shift = VerifierGuidedMCTS(
        scorer,
        world_model=AgentWorldModel(corrupt_one_world_model_member(artifact)),
    ).plan(
        SearchRequest(
            request_id="member-shift-world-model-plan",
            route="rag",
            evidence_count=1,
            confidence=0.55,
            token_budget=1000,
        )
    )
    assert member_shift.terminal_action == "abstain"
    assert member_shift.world_model_ood


def test_runtime_loader_hot_reloads_and_fails_closed(
    monkeypatch, tmp_path: Path, trained_world_model
):
    _, artifact, _ = trained_world_model
    path = tmp_path / "world-model.json"
    artifact.save(path)
    research = importlib.import_module("agent.research_assistant")
    monkeypatch.setattr(research, "WORLD_MODEL_PATH", path)
    monkeypatch.setattr(research, "_world_model", None)
    monkeypatch.setattr(research, "_world_model_mtime_ns", -1)
    first = research._load_world_model()
    assert research._load_world_model() is first

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["success_floor"] += 0.01
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(research, "_world_model", None)
    with pytest.raises(ValueError, match="integrity"):
        research._load_world_model()
