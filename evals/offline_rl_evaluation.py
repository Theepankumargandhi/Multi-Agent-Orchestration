"""Sequential off-policy evaluation and promotion gate for planning priors."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from pydantic import BaseModel, Field

from agent.offline_rl import (
    ConservativePlanningPolicy,
    LoggedPlanningTransition,
    OfflineRLArtifact,
    PlanningState,
    load_planning_replay,
    safe_actions,
    train_offline_rl_policy,
)
from agent.process_reward import ProcessRewardScorer
from agent.search_planner import SearchPolicy, SearchRequest, VerifierGuidedMCTS

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "planning_replay.jsonl"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class EpisodeEstimate(BaseModel):
    episode_id: str
    behavior_return: float
    importance_weight: float = Field(ge=0)
    doubly_robust_return: float
    matched_actions: int = Field(ge=0)
    steps: int = Field(ge=1)


class OfflineRLEvaluationReport(BaseModel):
    schema_version: str = "1.0"
    generated_by: str = "agentforge-offline-rl-eval"
    dataset_fingerprint: str
    policy_fingerprint: str
    test_episodes: int
    test_transitions: int
    behavior_return: float
    ips_return: float
    snips_return: float
    doubly_robust_return: float
    confidence_low: float
    confidence_high: float
    effective_sample_size: float
    support_violations: int = Field(ge=0)
    unsupported_decisions: list[str]
    heldout_action_accuracy: float
    unsafe_target_action_rate: float
    supported_planning_success: bool
    ood_uniform_fallback: bool
    policy_integrity: bool
    promoted: bool
    reasons: list[str]
    episodes: list[EpisodeEstimate]
    report_fingerprint: str = ""


def _prior_map(
    policy: ConservativePlanningPolicy,
    item: LoggedPlanningTransition,
    state: PlanningState,
) -> dict[str, tuple[float, float]]:
    estimate = policy.priors(
        route=item.route,
        high_risk=item.high_risk,
        state=state,
        actions=safe_actions(state, high_risk=item.high_risk),
    )
    return {
        action.action: (action.probability, action.q_mean)
        for action in estimate.actions
    }


def _state_value(priors: dict[str, tuple[float, float]]) -> float:
    return sum(probability * value for probability, value in priors.values())


def _support_key(item: LoggedPlanningTransition) -> tuple[str, bool, bool, bool, int]:
    return (
        item.route.casefold(),
        item.high_risk,
        item.state.reasoned,
        item.state.verified,
        min(item.state.evidence_count, 2),
    )


def _bootstrap_interval(values: list[float], *, seed: int = 991) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(1000)
    ]
    samples.sort()
    return samples[24], samples[974]


def evaluate_offline_rl(
    artifact: OfflineRLArtifact,
    transitions: list[LoggedPlanningTransition],
    *,
    process_reward_artifact: Path = Path("evals/experiments/process_reward_model.json"),
) -> OfflineRLEvaluationReport:
    policy = ConservativePlanningPolicy(artifact)
    test = [item for item in transitions if item.split == "test"]
    grouped: dict[str, list[LoggedPlanningTransition]] = {}
    for item in test:
        grouped.setdefault(item.episode_id, []).append(item)
    if not grouped:
        raise ValueError("offline-RL evaluation has no held-out episodes")
    episode_estimates = []
    positive_action_matches = 0
    positive_actions = 0
    unsafe_target_actions = 0
    target_actions = 0
    training_support: dict[tuple[str, bool, bool, bool, int], set[str]] = {}
    for item in transitions:
        if item.split == "train":
            training_support.setdefault(_support_key(item), set()).add(item.action)
    support_violations = 0
    unsupported_decisions = []
    for episode_id, raw_items in sorted(grouped.items()):
        items = sorted(raw_items, key=lambda item: item.timestep)
        behavior_return = 0.0
        cumulative_ratio = 1.0
        doubly_robust = _state_value(_prior_map(policy, items[0], items[0].state))
        matched = 0
        for index, item in enumerate(items):
            current = _prior_map(policy, item, item.state)
            selected = max(current, key=lambda action: (current[action][0], action))
            if selected not in training_support.get(_support_key(item), set()):
                support_violations += 1
                unsupported_decisions.append(f"{item.event_id}:{selected}")
            target_actions += 1
            if selected not in safe_actions(item.state, high_risk=item.high_risk):
                unsafe_target_actions += 1
            if item.reward > 0:
                positive_actions += 1
                positive_action_matches += int(selected == item.action)
            target_probability, q_logged = current.get(item.action, (0.0, 0.0))
            matched += int(selected == item.action)
            cumulative_ratio *= target_probability / item.behavior_propensity
            behavior_return += (artifact.gamma**index) * item.reward
            next_value = 0.0
            if not item.done:
                next_value = _state_value(_prior_map(policy, item, item.next_state))
            temporal_difference = (
                item.reward + artifact.gamma * next_value - q_logged
            )
            doubly_robust += (
                artifact.gamma**index
            ) * cumulative_ratio * temporal_difference
        episode_estimates.append(
            EpisodeEstimate(
                episode_id=episode_id,
                behavior_return=behavior_return,
                importance_weight=cumulative_ratio,
                doubly_robust_return=doubly_robust,
                matched_actions=matched,
                steps=len(items),
            )
        )
    weights = [item.importance_weight for item in episode_estimates]
    returns = [item.behavior_return for item in episode_estimates]
    dr_values = [item.doubly_robust_return for item in episode_estimates]
    weight_total = sum(weights)
    ips_return = statistics.fmean(
        weight * value for weight, value in zip(weights, returns, strict=True)
    )
    snips_return = (
        sum(weight * value for weight, value in zip(weights, returns, strict=True))
        / weight_total
        if weight_total
        else 0.0
    )
    effective_sample_size = (
        weight_total**2 / sum(weight**2 for weight in weights)
        if any(weights)
        else 0.0
    )
    confidence_low, confidence_high = _bootstrap_interval(dr_values)
    scorer = ProcessRewardScorer.load(process_reward_artifact)
    supported_plan = VerifierGuidedMCTS(
        scorer,
        SearchPolicy(iterations=48, max_nodes=64),
        planning_policy=policy,
    ).plan(
        SearchRequest(
            request_id="offline-rl-supported-plan",
            route="rag",
            evidence_count=1,
            confidence=0.55,
            token_budget=1000,
        )
    )
    ood_plan = VerifierGuidedMCTS(
        scorer,
        SearchPolicy(iterations=48, max_nodes=64),
        planning_policy=policy,
    ).plan(
        SearchRequest(
            request_id="offline-rl-ood-plan",
            route="unseen-route",
            evidence_count=1,
            confidence=0.55,
            token_budget=1000,
        )
    )
    behavior_return = statistics.fmean(returns)
    doubly_robust_return = statistics.fmean(dr_values)
    action_accuracy = positive_action_matches / max(1, positive_actions)
    unsafe_rate = unsafe_target_actions / max(1, target_actions)
    supported_success = (
        supported_plan.terminal_action == "answer"
        and supported_plan.tree_policy_strategy == "offline_rl_puct"
        and not supported_plan.offline_rl_ood
    )
    ood_fallback = (
        ood_plan.tree_policy_strategy == "offline_rl_puct"
        and ood_plan.offline_rl_ood
        and ood_plan.terminal_action == "answer"
    )
    reasons = []
    if doubly_robust_return <= behavior_return:
        reasons.append("doubly robust target return did not improve behavior policy")
    if confidence_low <= behavior_return:
        reasons.append("target policy confidence bound did not clear behavior return")
    if effective_sample_size < 2:
        reasons.append("effective sample size is below the promotion floor")
    if support_violations:
        reasons.append("target policy selects actions without matching behavior support")
    if action_accuracy < 0.8:
        reasons.append("held-out positive-action accuracy is below 80%")
    if unsafe_rate > 0:
        reasons.append("target policy selected an action outside the safety mask")
    if not supported_success:
        reasons.append("PUCT integration failed the supported planning scenario")
    if not ood_fallback:
        reasons.append("unknown-route policy did not fall back to uniform PUCT")
    report = OfflineRLEvaluationReport(
        dataset_fingerprint=_hash([item.model_dump(mode="json") for item in test]),
        policy_fingerprint=artifact.artifact_fingerprint,
        test_episodes=len(episode_estimates),
        test_transitions=len(test),
        behavior_return=behavior_return,
        ips_return=ips_return,
        snips_return=snips_return,
        doubly_robust_return=doubly_robust_return,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        effective_sample_size=effective_sample_size,
        support_violations=support_violations,
        unsupported_decisions=unsupported_decisions,
        heldout_action_accuracy=action_accuracy,
        unsafe_target_action_rate=unsafe_rate,
        supported_planning_success=supported_success,
        ood_uniform_fallback=ood_fallback,
        policy_integrity=artifact.verify(),
        promoted=not reasons and artifact.verify(),
        reasons=reasons,
        episodes=episode_estimates,
    )
    report.report_fingerprint = _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )
    return report


def verify_report(report: OfflineRLEvaluationReport) -> bool:
    return report.report_fingerprint == _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a conservative offline-RL planning policy."
    )
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/evaluations/offline-rl/offline-rl-policy.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluations/offline-rl/report.json"),
    )
    parser.add_argument(
        "--process-reward-artifact",
        type=Path,
        default=Path("evals/experiments/process_reward_model.json"),
    )
    parser.add_argument("--require-promotion", action="store_true")
    args = parser.parse_args()
    transitions = load_planning_replay(args.dataset)
    artifact = train_offline_rl_policy(transitions)
    artifact.save(args.artifact)
    report = evaluate_offline_rl(
        artifact,
        transitions,
        process_reward_artifact=args.process_reward_artifact,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "behavior_return": report.behavior_return,
                "doubly_robust_return": report.doubly_robust_return,
                "confidence_interval": [report.confidence_low, report.confidence_high],
                "effective_sample_size": report.effective_sample_size,
                "support_violations": report.support_violations,
                "unsupported_decisions": report.unsupported_decisions,
                "heldout_action_accuracy": report.heldout_action_accuracy,
                "unsafe_target_action_rate": report.unsafe_target_action_rate,
                "supported_planning_success": report.supported_planning_success,
                "ood_uniform_fallback": report.ood_uniform_fallback,
                "promoted": report.promoted,
                "reasons": report.reasons,
            },
            indent=2,
        )
    )
    if args.require_promotion and (not report.promoted or not verify_report(report)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
