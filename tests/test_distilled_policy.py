import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.distilled_policy import (
    DistillationContext,
    DistilledPlanningPolicy,
    DistilledPolicyArtifact,
    fingerprint,
    train_distilled_policy,
)
from agent.process_reward import ProcessRewardScorer
from agent.search_planner import SearchPolicy, SearchRequest, VerifierGuidedMCTS
from evals.distillation_evaluation import (
    audit_plan,
    evaluate_distillation,
    generate_teacher_examples,
    verify_report,
)
from evals.search_planning_evaluation import load_scenarios

SCORER = Path("evals/experiments/process_reward_model.json")


@pytest.fixture(scope="module")
def trained():
    scorer = ProcessRewardScorer.load(SCORER)
    examples = generate_teacher_examples(scorer)
    artifact = train_distilled_policy(
        examples, teacher_policy_fingerprint=fingerprint(SearchPolicy().model_dump(mode="json"))
    )
    return scorer, examples, artifact


def test_distillation_rejects_test_rows_and_overlapping_contexts(trained):
    _, examples, _ = trained
    contaminated = examples[0].model_copy(update={"split": "test"})
    with pytest.raises(ValueError, match="test examples"):
        train_distilled_policy([*examples, contaminated], teacher_policy_fingerprint="teacher")
    duplicate = examples[0].model_copy(update={"example_id": "overlap", "split": "validation"})
    with pytest.raises(ValueError, match="contexts overlap"):
        train_distilled_policy([*examples, duplicate], teacher_policy_fingerprint="teacher")


def test_training_reproduces_and_improves_validation_targets(trained):
    _, examples, artifact = trained
    first = train_distilled_policy(examples, teacher_policy_fingerprint="teacher", epochs=30)
    second = train_distilled_policy(examples, teacher_policy_fingerprint="teacher", epochs=30)
    assert first.model_dump() == second.model_dump()
    assert artifact.validation_kl < artifact.validation_uniform_kl
    assert artifact.training_examples == 35
    assert artifact.validation_examples == 35


def test_artifact_load_rejects_modified_or_nonfinite_weights(tmp_path, trained):
    _, _, artifact = trained
    path = tmp_path / "policy.json"
    artifact.save(path)
    assert DistilledPolicyArtifact.load(path).verify()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["weights"]["answer"][0] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        DistilledPolicyArtifact.load(path)
    payload["weights"]["answer"][0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        DistilledPolicyArtifact.model_validate(payload)


def test_action_mask_unknown_routes_and_uncertain_policy_fallback(trained):
    _, examples, artifact = trained
    policy = DistilledPlanningPolicy(artifact)
    context = DistillationContext(
        route="rag",
        confidence=0.5,
        evidence_count=0,
        remaining_tokens=1000,
        remaining_retrievals=2,
        legal_actions=["retrieve", "abstain"],
    )
    prior = policy.priors(context)
    assert set(prior.probabilities) == {"retrieve", "abstain"}
    assert sum(prior.probabilities.values()) == pytest.approx(1)
    unknown = policy.priors(context.model_copy(update={"route": "unseen-route"}))
    assert unknown.fallback_reason == "unsupported_route"
    assert list(unknown.probabilities.values()) == pytest.approx([0.5, 0.5])
    uncertain = artifact.model_copy(deep=True)
    uncertain.weights = {action: [0.0] * len(values) for action, values in uncertain.weights.items()}
    uncertain.seal()
    estimate = DistilledPlanningPolicy(uncertain).priors(context)
    assert estimate.fallback_reason == "high_policy_entropy"
    assert estimate.normalized_entropy == 1
    terminal = policy.priors(examples[5].context)
    assert terminal.probabilities == {"abstain": 1}


def test_compute_curves_preserve_quality_and_verify_receipts(trained):
    scorer, _, artifact = trained
    report = evaluate_distillation(artifact, scorer, load_scenarios())
    assert report.promoted
    assert verify_report(report)
    assert len(report.points) == 8
    assert all(point.success_rate == 1 for point in report.points)
    assert all(point.unsafe_releases == point.budget_violations == 0 for point in report.points)
    assert all(point.all_receipts_valid for point in report.points)
    report.points[0].success_rate = 0.0
    assert not verify_report(report)


def test_budget_audit_detects_forced_unsafe_answer(trained):
    scorer, _, artifact = trained
    request = SearchRequest(
        request_id="no-evidence-mask",
        route="rag",
        confidence=0.95,
        evidence_count=0,
        retrieval_available=False,
        token_budget=400,
    )
    plan = VerifierGuidedMCTS(scorer, distilled_policy=DistilledPlanningPolicy(artifact)).plan(request)
    assert plan.terminal_action == "abstain"
    assert audit_plan(plan, request) == (False, False)
    plan.planned_actions = ["answer"]
    plan.terminal_action = "answer"
    plan.tokens_planned = 160
    plan.seal()
    assert plan.verify()
    assert audit_plan(plan, request)[0]


def test_policy_rejects_incompatible_teacher_scorer(trained):
    scorer, _, artifact = trained
    incompatible = artifact.model_copy(deep=True)
    incompatible.teacher_model_fingerprints = ["another-model"]
    incompatible.seal()
    with pytest.raises(ValueError, match="teacher scorer mismatch"):
        VerifierGuidedMCTS(scorer, distilled_policy=DistilledPlanningPolicy(incompatible))


def test_runtime_distillation_loads_and_missing_artifact_abstains(monkeypatch, tmp_path, trained):
    _, _, artifact = trained
    research = importlib.import_module("agent.research_assistant")
    path = tmp_path / "policy.json"
    artifact.save(path)
    monkeypatch.setattr(research, "SEARCH_DISTILLATION_ENABLED", True)
    monkeypatch.setattr(research, "SEARCH_DISTILLATION_PATH", path)
    monkeypatch.setattr(research, "OFFLINE_RL_POLICY_ENABLED", False)
    monkeypatch.setattr(research, "WORLD_MODEL_ENABLED", False)
    monkeypatch.setattr(research, "VERIFIER_ENSEMBLE_ENABLED", False)
    monkeypatch.setattr(research, "VERIFIER_MCTS_ENABLED", True)
    monkeypatch.setattr(research, "PROCESS_REWARD_MODEL_PATH", SCORER)
    monkeypatch.setattr(research, "_process_reward_scorer", None)
    monkeypatch.setattr(research, "_process_reward_mtime_ns", -1)
    monkeypatch.setattr(research, "_distilled_policy", None)
    monkeypatch.setattr(research, "_distilled_policy_mtime_ns", -1)
    loaded = research._load_distilled_policy()
    assert research._load_distilled_policy() is loaded
    report = SimpleNamespace(
        route="rag",
        action="repair",
        confidence=0.55,
        evidence_count=2,
        claims=[],
        report_fingerprint="distilled-runtime-request",
    )
    plan = research._adaptive_compute_plan(report, {"decision": "abstain"})
    assert plan.search_plan.tree_policy_strategy == "distilled_puct"
    assert plan.search_plan.distilled_policy_fingerprint == artifact.artifact_fingerprint
    monkeypatch.setattr(research, "SEARCH_DISTILLATION_PATH", tmp_path / "absent.json")
    failed = research._adaptive_compute_plan(report, {"decision": "abstain"})
    assert failed.action == "abstain"
    assert failed.reason == "verifier_guided_search_unavailable"
