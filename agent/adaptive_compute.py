"""Budgeted test-time compute planning and consensus-based answer selection."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Literal

from pydantic import BaseModel, Field

from agent.search_planner import SearchPlan


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _fingerprint(value: object, key: bytes | None = None) -> str:
    payload = _canonical(value)
    return (
        hmac.new(key, payload, hashlib.sha256).hexdigest()
        if key
        else hashlib.sha256(payload).hexdigest()
    )


class ComputePolicy(BaseModel):
    version: str = "adaptive-compute-v1"
    max_candidates: int = Field(default=3, ge=2, le=5)
    max_extra_tokens: int = Field(default=1800, ge=128, le=8192)
    max_latency_ms: float = Field(default=15000, ge=100, le=120000)
    min_confidence_gain: float = Field(default=0.04, ge=0, le=1)
    min_consensus: float = Field(default=0.5, ge=0, le=1)
    high_risk_min_consensus: float = Field(default=0.67, ge=0, le=1)


class ComputeSignals(BaseModel):
    route: str
    grounding_action: Literal["pass", "repair", "abstain", "not_required"]
    grounding_confidence: float = Field(ge=0, le=1)
    uncertainty_decision: Literal["release", "abstain", "not_evaluated", "unavailable"]
    out_of_distribution: bool = False
    high_risk: bool = False
    evidence_count: int = Field(default=0, ge=0)


class ComputePlan(BaseModel):
    schema_version: str = "1.0"
    policy_version: str
    route: str
    action: Literal["early_exit", "deliberate", "abstain"]
    candidate_budget: int
    token_budget: int
    latency_budget_ms: float
    initial_confidence: float
    high_risk: bool
    reason: str
    reasoning_strategy: Literal["policy", "verifier_mcts"] = "policy"
    planned_actions: list[str] = Field(default_factory=list, max_length=12)
    search_plan_fingerprint: str = ""
    search_plan: SearchPlan | None = None
    plan_fingerprint: str = ""


class CandidateAssessment(BaseModel):
    candidate_id: str
    confidence: float = Field(ge=0, le=1)
    grounded: bool
    conformal_decision: Literal["release", "abstain", "not_evaluated"]
    claim_keys: list[str] = Field(default_factory=list)
    token_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    answer_fingerprint: str
    process_reward: float | None = Field(default=None, ge=0, le=1)


class CandidateSummary(BaseModel):
    candidate_id: str
    confidence: float
    grounded: bool
    conformal_decision: str
    token_count: int
    latency_ms: float
    consensus: float
    eligible: bool
    answer_fingerprint: str
    process_reward: float | None = None


class DeliberationReceipt(BaseModel):
    schema_version: str = "1.0"
    policy_version: str
    plan_fingerprint: str
    status: Literal["released", "abstained", "budget_exhausted"]
    selected_candidate_id: str = ""
    attempted_candidates: int
    evaluated_candidates: int
    eligible_candidates: int
    extra_tokens: int
    latency_ms: float
    selected_confidence: float = 0
    selected_consensus: float = 0
    reason: str
    candidate_summaries: list[CandidateSummary]
    receipt_fingerprint: str = ""


def plan_compute(
    signals: ComputeSignals,
    policy: ComputePolicy | None = None,
    integrity_key: bytes | None = None,
) -> ComputePlan:
    policy = policy or ComputePolicy()
    if signals.uncertainty_decision == "unavailable":
        action, reason = "abstain", "uncertainty_control_unavailable"
    elif signals.evidence_count == 0 and signals.grounding_action != "not_required":
        action, reason = "abstain", "no_evidence_for_additional_compute"
    elif signals.grounding_action == "abstain" and signals.high_risk:
        action, reason = "abstain", "unsupported_high_risk_claim"
    elif (
        signals.grounding_action in {"pass", "not_required"}
        and signals.uncertainty_decision in {"release", "not_evaluated"}
        and not signals.out_of_distribution
    ):
        action, reason = "early_exit", "verified_answer_needs_no_extra_compute"
    else:
        action, reason = "deliberate", "uncertain_answer_can_use_bounded_compute"

    candidate_budget = policy.max_candidates if action == "deliberate" else 0
    if action == "deliberate" and not signals.high_risk:
        candidate_budget = max(2, policy.max_candidates - 1)
    plan = ComputePlan(
        policy_version=policy.version,
        route=signals.route.casefold(),
        action=action,
        candidate_budget=candidate_budget,
        token_budget=policy.max_extra_tokens if action == "deliberate" else 0,
        latency_budget_ms=policy.max_latency_ms if action == "deliberate" else 0,
        initial_confidence=round(signals.grounding_confidence, 6),
        high_risk=signals.high_risk,
        reason=reason,
    )
    plan.plan_fingerprint = _fingerprint(
        plan.model_dump(mode="json", exclude={"plan_fingerprint"}), integrity_key
    )
    return plan


def verify_plan(plan: ComputePlan, integrity_key: bytes | None = None) -> bool:
    expected = _fingerprint(
        plan.model_dump(mode="json", exclude={"plan_fingerprint"}), integrity_key
    )
    if not hmac.compare_digest(plan.plan_fingerprint, expected):
        return False
    if plan.reasoning_strategy == "verifier_mcts":
        return bool(
            plan.search_plan
            and plan.search_plan.verify()
            and plan.search_plan_fingerprint == plan.search_plan.plan_fingerprint
            and plan.planned_actions == plan.search_plan.planned_actions
        )
    return (
        plan.search_plan is None
        and not plan.search_plan_fingerprint
        and not plan.planned_actions
    )


def seal_plan(plan: ComputePlan, integrity_key: bytes | None = None) -> ComputePlan:
    """Return a copy with an integrity fingerprint covering every planning field."""
    sealed = plan.model_copy(deep=True)
    sealed.plan_fingerprint = _fingerprint(
        sealed.model_dump(mode="json", exclude={"plan_fingerprint"}), integrity_key
    )
    return sealed


def attach_search_plan(
    plan: ComputePlan,
    search_plan: SearchPlan,
    integrity_key: bytes | None = None,
) -> ComputePlan:
    """Attach a verified search receipt and fail closed when search chooses abstention."""
    if not verify_plan(plan, integrity_key):
        raise ValueError("adaptive compute plan integrity verification failed")
    if not search_plan.verify():
        raise ValueError("reasoning search plan integrity verification failed")
    updated = plan.model_copy(deep=True)
    updated.reasoning_strategy = "verifier_mcts"
    updated.planned_actions = list(search_plan.planned_actions)
    updated.search_plan_fingerprint = search_plan.plan_fingerprint
    updated.search_plan = search_plan.model_copy(deep=True)
    if search_plan.terminal_action == "abstain":
        updated.action = "abstain"
        updated.candidate_budget = 0
        updated.token_budget = 0
        updated.latency_budget_ms = 0
        updated.reason = "verifier_guided_search_abstained"
    else:
        updated.reason = "verifier_guided_search_selected_deliberation"
    return seal_plan(updated, integrity_key)


def candidate_assessment(
    *,
    candidate_id: str,
    answer: str,
    confidence: float,
    grounded: bool,
    conformal_decision: Literal["release", "abstain", "not_evaluated"],
    claim_keys: list[str],
    token_count: int,
    latency_ms: float,
    process_reward: float | None = None,
) -> CandidateAssessment:
    return CandidateAssessment(
        candidate_id=candidate_id,
        confidence=round(confidence, 6),
        grounded=grounded,
        conformal_decision=conformal_decision,
        claim_keys=sorted(set(claim_keys)),
        token_count=token_count,
        latency_ms=round(latency_ms, 3),
        answer_fingerprint=_fingerprint(answer),
        process_reward=process_reward,
    )


def _agreement(left: CandidateAssessment, right: CandidateAssessment) -> float:
    left_keys, right_keys = set(left.claim_keys), set(right.claim_keys)
    if not left_keys and not right_keys:
        return 0
    return len(left_keys & right_keys) / len(left_keys | right_keys)


def select_candidate(
    plan: ComputePlan,
    candidates: list[CandidateAssessment],
    policy: ComputePolicy | None = None,
    integrity_key: bytes | None = None,
    attempted_candidates: int | None = None,
) -> DeliberationReceipt:
    policy = policy or ComputePolicy()
    if not verify_plan(plan, integrity_key):
        raise ValueError("adaptive compute plan integrity verification failed")
    if plan.action != "deliberate":
        raise ValueError("candidate selection requires a deliberation plan")

    evaluated: list[CandidateAssessment] = []
    extra_tokens = 0
    latency_ms = 0.0
    exhausted = False
    attempted = min(
        attempted_candidates if attempted_candidates is not None else len(candidates),
        plan.candidate_budget,
    )
    for candidate in candidates[: plan.candidate_budget]:
        next_tokens = extra_tokens + candidate.token_count
        next_latency = latency_ms + candidate.latency_ms
        if next_tokens > plan.token_budget or next_latency > plan.latency_budget_ms:
            exhausted = True
            continue
        evaluated.append(candidate)
        extra_tokens = next_tokens
        latency_ms = next_latency

    eligible = [
        item
        for item in evaluated
        if item.grounded
        and item.conformal_decision in {"release", "not_evaluated"}
        and item.confidence >= min(1.0, plan.initial_confidence + policy.min_confidence_gain)
    ]
    consensus = {
        item.candidate_id: max(
            (_agreement(item, other) for other in eligible if other is not item),
            default=0.0,
        )
        for item in evaluated
    }
    required_consensus = (
        policy.high_risk_min_consensus if plan.high_risk else policy.min_consensus
    )
    releasable = [
        item for item in eligible if consensus[item.candidate_id] >= required_consensus
    ]
    selected = max(
        releasable,
        key=lambda item: (
            item.process_reward if item.process_reward is not None else item.confidence,
            item.confidence,
            consensus[item.candidate_id],
            -item.token_count,
            item.candidate_id,
        ),
        default=None,
    )
    if selected is not None:
        status, reason = "released", "grounded_conformal_consensus_reached"
    elif exhausted:
        status, reason = "budget_exhausted", "compute_budget_reached_without_consensus"
    else:
        status, reason = "abstained", "no_candidate_met_release_policy"

    summaries = [
        CandidateSummary(
            candidate_id=item.candidate_id,
            confidence=item.confidence,
            grounded=item.grounded,
            conformal_decision=item.conformal_decision,
            token_count=item.token_count,
            latency_ms=item.latency_ms,
            consensus=round(consensus[item.candidate_id], 6),
            eligible=item in eligible,
            answer_fingerprint=item.answer_fingerprint,
            process_reward=item.process_reward,
        )
        for item in evaluated
    ]
    receipt = DeliberationReceipt(
        policy_version=policy.version,
        plan_fingerprint=plan.plan_fingerprint,
        status=status,
        selected_candidate_id=selected.candidate_id if selected else "",
        attempted_candidates=attempted,
        evaluated_candidates=len(evaluated),
        eligible_candidates=len(eligible),
        extra_tokens=extra_tokens,
        latency_ms=round(latency_ms, 3),
        selected_confidence=selected.confidence if selected else 0,
        selected_consensus=round(consensus[selected.candidate_id], 6) if selected else 0,
        reason=reason,
        candidate_summaries=summaries,
    )
    receipt.receipt_fingerprint = _fingerprint(
        receipt.model_dump(mode="json", exclude={"receipt_fingerprint"}), integrity_key
    )
    return receipt


def verify_receipt(receipt: DeliberationReceipt, integrity_key: bytes | None = None) -> bool:
    expected = _fingerprint(
        receipt.model_dump(mode="json", exclude={"receipt_fingerprint"}), integrity_key
    )
    return hmac.compare_digest(receipt.receipt_fingerprint, expected)
