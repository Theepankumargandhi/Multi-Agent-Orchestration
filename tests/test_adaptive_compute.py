import pytest

from agent.adaptive_compute import (
    ComputePolicy,
    ComputeSignals,
    candidate_assessment,
    plan_compute,
    select_candidate,
    verify_plan,
    verify_receipt,
)

KEY = b"adaptive-compute-test-key"


def _signals(**overrides):
    payload = {
        "route": "rag",
        "grounding_action": "pass",
        "grounding_confidence": 0.9,
        "uncertainty_decision": "release",
        "evidence_count": 3,
    }
    payload.update(overrides)
    return ComputeSignals.model_validate(payload)


def _candidate(candidate_id, confidence, keys, **overrides):
    payload = {
        "candidate_id": candidate_id,
        "answer": f"private answer {candidate_id}",
        "confidence": confidence,
        "grounded": True,
        "conformal_decision": "release",
        "claim_keys": keys,
        "token_count": 100,
        "latency_ms": 100,
    }
    payload.update(overrides)
    return candidate_assessment(**payload)


def test_planner_early_exits_escalates_and_fails_closed():
    early = plan_compute(_signals(), integrity_key=KEY)
    uncertain = plan_compute(
        _signals(uncertainty_decision="abstain", grounding_confidence=0.6),
        integrity_key=KEY,
    )
    risky = plan_compute(
        _signals(
            grounding_action="abstain",
            high_risk=True,
            uncertainty_decision="not_evaluated",
        ),
        integrity_key=KEY,
    )
    unavailable = plan_compute(
        _signals(uncertainty_decision="unavailable"), integrity_key=KEY
    )
    assert early.action == "early_exit" and early.candidate_budget == 0
    assert uncertain.action == "deliberate" and uncertain.candidate_budget == 2
    assert risky.action == "abstain"
    assert unavailable.action == "abstain"
    assert verify_plan(uncertain, KEY)


def test_selector_releases_only_grounded_conformal_consensus():
    plan = plan_compute(
        _signals(uncertainty_decision="abstain", grounding_confidence=0.6),
        integrity_key=KEY,
    )
    candidates = [
        _candidate("a", 0.88, ["ev-1", "ev-2"]),
        _candidate("b", 0.91, ["ev-1", "ev-2"]),
    ]
    receipt = select_candidate(plan, candidates, integrity_key=KEY)
    assert receipt.status == "released"
    assert receipt.selected_candidate_id == "b"
    assert receipt.selected_consensus == 1
    assert verify_receipt(receipt, KEY)
    serialized = receipt.model_dump_json()
    assert "private answer" not in serialized


def test_selector_abstains_on_disagreement_or_failed_conformal_filter():
    plan = plan_compute(
        _signals(uncertainty_decision="abstain", grounding_confidence=0.5),
        integrity_key=KEY,
    )
    disagreement = select_candidate(
        plan,
        [_candidate("a", 0.9, ["ev-1"]), _candidate("b", 0.9, ["ev-2"])],
        integrity_key=KEY,
    )
    filtered = select_candidate(
        plan,
        [
            _candidate("a", 0.9, ["ev-1"], conformal_decision="abstain"),
            _candidate("b", 0.9, ["ev-1"], grounded=False),
        ],
        integrity_key=KEY,
    )
    assert disagreement.status == "abstained"
    assert filtered.status == "abstained"


def test_selector_enforces_token_and_latency_budgets():
    policy = ComputePolicy(max_extra_tokens=150, max_latency_ms=150)
    plan = plan_compute(
        _signals(uncertainty_decision="abstain", grounding_confidence=0.5),
        policy,
        KEY,
    )
    receipt = select_candidate(
        plan,
        [_candidate("a", 0.9, ["ev-1"]), _candidate("b", 0.9, ["ev-1"])],
        policy,
        KEY,
    )
    assert receipt.status == "budget_exhausted"
    assert receipt.evaluated_candidates == 1
    assert receipt.extra_tokens <= plan.token_budget
    assert receipt.latency_ms <= plan.latency_budget_ms


def test_tampered_plan_is_rejected_before_selection():
    plan = plan_compute(
        _signals(uncertainty_decision="abstain", grounding_confidence=0.5),
        integrity_key=KEY,
    )
    plan.token_budget += 1
    assert not verify_plan(plan, KEY)
    with pytest.raises(ValueError, match="integrity"):
        select_candidate(plan, [], integrity_key=KEY)


def test_no_evidence_is_not_mistaken_for_more_reasoning_need():
    plan = plan_compute(
        _signals(
            grounding_action="repair",
            uncertainty_decision="not_evaluated",
            evidence_count=0,
        )
    )
    assert plan.action == "abstain"
    assert plan.reason == "no_evidence_for_additional_compute"
