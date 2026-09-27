import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.adaptive_compute import (
    ComputeSignals,
    attach_search_plan,
    plan_compute,
    verify_plan,
)
from agent.process_reward import ProcessRewardScorer
from agent.search_planner import SearchPolicy, SearchRequest, VerifierGuidedMCTS
from evals.search_planning_evaluation import (
    evaluate_search_planning,
    load_scenarios,
    verify_report,
)

ARTIFACT = Path("evals/experiments/process_reward_model.json")
DATASET = Path("evals/datasets/search_planning_scenarios.jsonl")
KEY = b"verifier-mcts-test-key"


def _planner(**policy_overrides):
    return VerifierGuidedMCTS(
        ProcessRewardScorer.load(ARTIFACT), SearchPolicy(**policy_overrides)
    )


def test_search_is_deterministic_budgeted_and_replayable():
    request = SearchRequest(
        request_id="low-confidence-request",
        route="rag",
        evidence_count=1,
        confidence=0.48,
        token_budget=900,
    )
    first = _planner().plan(request)
    second = _planner().plan(request)
    assert first.model_dump() == second.model_dump()
    assert first.terminal_action == "answer"
    assert first.planned_actions[-3:] == ["reason", "verify", "answer"]
    assert first.tokens_planned <= request.token_budget
    assert first.nodes_expanded <= 128
    assert first.verify()


def test_high_risk_search_requires_two_sources_and_verification():
    plan = _planner().plan(
        SearchRequest(
            request_id="high-risk-request",
            route="web",
            evidence_count=1,
            confidence=0.5,
            high_risk=True,
            token_budget=1100,
        )
    )
    assert plan.terminal_action == "answer"
    assert plan.planned_actions.count("retrieve") >= 1
    assert plan.planned_actions[-3:] == ["reason", "verify", "answer"]
    assert plan.unsafe_branches_pruned > 0


def test_search_abstains_when_safe_completion_is_impossible():
    plan = _planner().plan(
        SearchRequest(
            request_id="no-evidence-request",
            route="rag",
            evidence_count=0,
            confidence=0.93,
            retrieval_available=False,
            token_budget=900,
        )
    )
    assert plan.terminal_action == "abstain"
    assert plan.planned_actions == ["abstain"]


def test_search_receipt_tampering_and_compute_attachment_are_detected():
    search = _planner().plan(
        SearchRequest(
            request_id="attach-search-plan",
            route="rag",
            evidence_count=2,
            confidence=0.55,
            token_budget=1000,
        )
    )
    compute = plan_compute(
        ComputeSignals(
            route="rag",
            grounding_action="repair",
            grounding_confidence=0.55,
            uncertainty_decision="abstain",
            evidence_count=2,
        ),
        integrity_key=KEY,
    )
    attached = attach_search_plan(compute, search, KEY)
    assert attached.reasoning_strategy == "verifier_mcts"
    assert attached.planned_actions == search.planned_actions
    assert attached.search_plan is not None
    assert attached.search_plan.nodes_expanded > 0
    assert verify_plan(attached, KEY)

    attached.search_plan.tokens_planned += 1
    assert not verify_plan(attached, KEY)

    search.tokens_planned += 1
    assert not search.verify()
    with pytest.raises(ValueError, match="search plan integrity"):
        attach_search_plan(compute, search, KEY)


def test_held_out_ablation_promotes_search_only_with_safety_and_integrity():
    report = evaluate_search_planning(
        ProcessRewardScorer.load(ARTIFACT), load_scenarios(DATASET)
    )
    assert report.baseline_success_rate == pytest.approx(0.3)
    assert report.search_success_rate == 1
    assert report.success_rate_delta == pytest.approx(0.7)
    assert report.baseline_unsafe_release_rate == pytest.approx(0.3)
    assert report.search_unsafe_release_rate == 0
    assert report.recovery_rate == 1
    assert report.budget_violation_rate == 0
    assert report.plan_integrity_rate == 1
    assert report.promoted
    assert verify_report(report)


def test_runtime_adaptive_compute_uses_search_and_fails_closed(monkeypatch, tmp_path):
    research = importlib.import_module("agent.research_assistant")
    report = SimpleNamespace(
        route="rag",
        action="repair",
        confidence=0.55,
        evidence_count=2,
        claims=[],
        report_fingerprint="0123456789abcdef0123456789abcdef",
    )
    monkeypatch.setattr(research, "VERIFIER_MCTS_ENABLED", True)
    monkeypatch.setattr(research, "PROCESS_REWARD_MODEL_PATH", ARTIFACT)
    monkeypatch.setattr(research, "_process_reward_scorer", None)
    monkeypatch.setattr(research, "_process_reward_mtime_ns", -1)
    plan = research._adaptive_compute_plan(report, {"decision": "abstain"})
    assert plan.action == "deliberate"
    assert plan.reasoning_strategy == "verifier_mcts"
    assert plan.planned_actions[-1] == "answer"
    assert verify_plan(plan, research.ADAPTIVE_COMPUTE_INTEGRITY_KEY)

    monkeypatch.setattr(research, "PROCESS_REWARD_MODEL_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(research, "_process_reward_scorer", None)
    failed = research._adaptive_compute_plan(report, {"decision": "abstain"})
    assert failed.action == "abstain"
    assert failed.reason == "verifier_guided_search_unavailable"
    assert verify_plan(failed, research.ADAPTIVE_COMPUTE_INTEGRITY_KEY)
