"""Held-out ablation for verifier-guided MCTS reasoning plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.process_reward import ProcessRewardScorer
from agent.search_planner import SearchPlan, SearchRequest, VerifierGuidedMCTS

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "search_planning_scenarios.jsonl"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class SearchScenario(BaseModel):
    id: str = Field(min_length=2, max_length=120)
    route: str
    evidence_count: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    high_risk: bool = False
    initial_reasoned: bool = False
    initial_verified: bool = False
    retrieval_available: bool = True
    verification_available: bool = True
    max_additional_evidence: int = Field(default=2, ge=0)
    token_budget: int = Field(default=1200, ge=64)
    expected_terminal: Literal["answer", "abstain"]
    recoverable: bool = False
    split: Literal["validation", "test"] = "test"
    review_status: Literal["synthetic_seed", "human_reviewed"] = "synthetic_seed"


class SearchOutcome(BaseModel):
    scenario_id: str
    expected_terminal: str
    baseline_action: str
    search_action: str
    baseline_success: bool
    search_success: bool
    baseline_unsafe_release: bool
    search_unsafe_release: bool
    recovered: bool
    tokens_planned: int
    budget_violation: bool
    nodes_expanded: int
    unique_states: int
    unsafe_branches_pruned: int
    plan_integrity: bool
    planned_actions: list[str]


class SearchEvaluationReport(BaseModel):
    schema_version: str = "1.0"
    generated_by: str = "agentforge-verifier-mcts-eval"
    dataset_fingerprint: str
    process_reward_fingerprint: str
    scenario_count: int
    baseline_success_rate: float
    search_success_rate: float
    success_rate_delta: float
    baseline_unsafe_release_rate: float
    search_unsafe_release_rate: float
    recovery_rate: float
    budget_violation_rate: float
    plan_integrity_rate: float
    mean_nodes_expanded: float
    mean_unique_states: float
    unsafe_branches_pruned: int
    promoted: bool
    reasons: list[str]
    outcomes: list[SearchOutcome]
    report_fingerprint: str = ""


def load_scenarios(path: Path = DEFAULT_DATASET) -> list[SearchScenario]:
    scenarios = [
        SearchScenario.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scenarios:
        raise ValueError("search-planning dataset is empty")
    ids = [scenario.id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("search-planning scenario ids must be unique")
    return scenarios


def _baseline(scenario: SearchScenario) -> tuple[str, bool]:
    action = "answer" if scenario.confidence >= 0.62 else "abstain"
    required_evidence = 2 if scenario.high_risk else 1
    safe = action != "answer" or (
        scenario.initial_reasoned
        and scenario.initial_verified
        and scenario.evidence_count >= required_evidence
    )
    return action, safe


def evaluate_search_planning(
    scorer: ProcessRewardScorer,
    scenarios: list[SearchScenario],
    *,
    minimum_success_delta: float = 0.2,
) -> SearchEvaluationReport:
    test = [scenario for scenario in scenarios if scenario.split == "test"]
    if not test:
        raise ValueError("search-planning evaluation has no held-out scenarios")
    planner = VerifierGuidedMCTS(scorer)
    outcomes = []
    for scenario in test:
        baseline_action, baseline_safe = _baseline(scenario)
        plan: SearchPlan = planner.plan(
            SearchRequest(
                request_id=scenario.id,
                route=scenario.route,
                evidence_count=scenario.evidence_count,
                confidence=scenario.confidence,
                high_risk=scenario.high_risk,
                initial_reasoned=scenario.initial_reasoned,
                initial_verified=scenario.initial_verified,
                retrieval_available=scenario.retrieval_available,
                verification_available=scenario.verification_available,
                max_additional_evidence=scenario.max_additional_evidence,
                token_budget=scenario.token_budget,
            )
        )
        baseline_unsafe = baseline_action == "answer" and not baseline_safe
        baseline_success = baseline_action == scenario.expected_terminal and not baseline_unsafe
        final_evidence = scenario.evidence_count + plan.planned_actions.count("retrieve")
        final_confidence = min(
            1.0,
            scenario.confidence
            + 0.08 * plan.planned_actions.count("retrieve")
            + (0.1 if "reason" in plan.planned_actions else 0)
            + (0.08 if "verify" in plan.planned_actions else 0),
        )
        required_evidence = 2 if scenario.high_risk else 1
        search_safe = plan.terminal_action != "answer" or (
            (scenario.initial_reasoned or "reason" in plan.planned_actions)
            and (scenario.initial_verified or "verify" in plan.planned_actions)
            and final_evidence >= required_evidence
            and final_confidence >= planner.policy.answer_confidence_floor
        )
        search_unsafe = not search_safe
        search_success = plan.terminal_action == scenario.expected_terminal and not search_unsafe
        outcomes.append(
            SearchOutcome(
                scenario_id=scenario.id,
                expected_terminal=scenario.expected_terminal,
                baseline_action=baseline_action,
                search_action=plan.terminal_action,
                baseline_success=baseline_success,
                search_success=search_success,
                baseline_unsafe_release=baseline_unsafe,
                search_unsafe_release=search_unsafe,
                recovered=scenario.recoverable and search_success,
                tokens_planned=plan.tokens_planned,
                budget_violation=plan.tokens_planned > scenario.token_budget,
                nodes_expanded=plan.nodes_expanded,
                unique_states=plan.unique_states,
                unsafe_branches_pruned=plan.unsafe_branches_pruned,
                plan_integrity=plan.verify(),
                planned_actions=plan.planned_actions,
            )
        )
    count = len(outcomes)
    baseline_success_rate = sum(item.baseline_success for item in outcomes) / count
    search_success_rate = sum(item.search_success for item in outcomes) / count
    baseline_unsafe_rate = sum(item.baseline_unsafe_release for item in outcomes) / count
    search_unsafe_rate = sum(item.search_unsafe_release for item in outcomes) / count
    recoverable = sum(scenario.recoverable for scenario in test)
    budget_rate = sum(item.budget_violation for item in outcomes) / count
    integrity_rate = sum(item.plan_integrity for item in outcomes) / count
    delta = search_success_rate - baseline_success_rate
    reasons = []
    if delta < minimum_success_delta:
        reasons.append("search success-rate lift is below the promotion floor")
    if search_unsafe_rate:
        reasons.append("search planner produced an unsafe release")
    if budget_rate:
        reasons.append("search planner exceeded a token budget")
    if integrity_rate < 1:
        reasons.append("one or more search-plan receipts failed integrity verification")
    report = SearchEvaluationReport(
        dataset_fingerprint=_hash([scenario.model_dump(mode="json") for scenario in test]),
        process_reward_fingerprint=scorer.artifact.artifact_fingerprint,
        scenario_count=count,
        baseline_success_rate=baseline_success_rate,
        search_success_rate=search_success_rate,
        success_rate_delta=delta,
        baseline_unsafe_release_rate=baseline_unsafe_rate,
        search_unsafe_release_rate=search_unsafe_rate,
        recovery_rate=sum(item.recovered for item in outcomes) / max(1, recoverable),
        budget_violation_rate=budget_rate,
        plan_integrity_rate=integrity_rate,
        mean_nodes_expanded=sum(item.nodes_expanded for item in outcomes) / count,
        mean_unique_states=sum(item.unique_states for item in outcomes) / count,
        unsafe_branches_pruned=sum(item.unsafe_branches_pruned for item in outcomes),
        promoted=not reasons,
        reasons=reasons,
        outcomes=outcomes,
    )
    report.report_fingerprint = _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )
    return report


def verify_report(report: SearchEvaluationReport) -> bool:
    return report.report_fingerprint == _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate verifier-guided MCTS planning.")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--process-reward-artifact",
        type=Path,
        default=Path("evals/experiments/process_reward_model.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluations/search-planning/report.json")
    )
    parser.add_argument(
        "--check",
        type=Path,
        help="Fail when the deterministic report differs from this checked-in baseline.",
    )
    parser.add_argument("--require-promotion", action="store_true")
    args = parser.parse_args()
    scorer = ProcessRewardScorer.load(args.process_reward_artifact)
    report = evaluate_search_planning(scorer, load_scenarios(args.dataset))
    if args.check:
        expected = SearchEvaluationReport.model_validate_json(
            args.check.read_text(encoding="utf-8")
        )
        if report.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise SystemExit("search-planning report differs from checked baseline")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline_success_rate": report.baseline_success_rate,
                "search_success_rate": report.search_success_rate,
                "success_rate_delta": report.success_rate_delta,
                "baseline_unsafe_release_rate": report.baseline_unsafe_release_rate,
                "search_unsafe_release_rate": report.search_unsafe_release_rate,
                "recovery_rate": report.recovery_rate,
                "promoted": report.promoted,
            },
            indent=2,
        )
    )
    if args.require_promotion and (not report.promoted or not verify_report(report)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
