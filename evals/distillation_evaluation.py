"""Search teacher generation, policy distillation, and held-out compute curves."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from pydantic import BaseModel

from agent.distilled_policy import (
    DistillationContext,
    DistilledPlanningPolicy,
    DistilledPolicyArtifact,
    SearchTeacherExample,
    fingerprint,
    train_distilled_policy,
)
from agent.process_reward import ProcessRewardScorer
from agent.search_planner import (
    ACTION_TOKEN_COST,
    SearchPlan,
    SearchPolicy,
    SearchRequest,
    SearchState,
    VerifierGuidedMCTS,
)
from agent.world_model import ROUTES
from evals.search_planning_evaluation import SearchScenario, load_scenarios


def generate_teacher_examples(scorer: ProcessRewardScorer) -> list[SearchTeacherExample]:
    """Authored training grid; does not read the frozen test scenario file."""
    teacher = VerifierGuidedMCTS(scorer, SearchPolicy(iterations=96, max_nodes=128))
    examples = []
    for split, confidence_offset, budget in (("train", 0.0, 1000), ("validation", 0.03, 950)):
        for route in ROUTES:
            for phase in range(7):
                request = SearchRequest(
                    request_id=f"teacher-{split}-{route}-{phase}",
                    route=route,
                    evidence_count=0 if phase in (0, 5, 6) else 2 if phase in (3, 4) else 1,
                    confidence=(0.38 if phase in (0, 5, 6) else 0.8 if phase in (3, 4) else 0.55)
                    + confidence_offset,
                    high_risk=phase == 2,
                    initial_reasoned=phase in (3, 4),
                    initial_verified=phase == 3,
                    retrieval_available=phase != 5,
                    verification_available=phase != 4,
                    token_budget=250 if phase == 6 else budget,
                )
                plan = teacher.plan(request)
                if not plan.verify():
                    raise ValueError("teacher plan integrity check failed")
                state = SearchState(
                    evidence_count=request.evidence_count,
                    confidence=request.confidence,
                    high_risk=request.high_risk,
                    reasoned=request.initial_reasoned,
                    verified=request.initial_verified,
                )
                actions = teacher._valid_actions(state, request)
                visits = {stat.action: stat.visits for stat in plan.root_action_stats}
                total = sum(visits.get(action, 0) for action in actions)
                if not total:
                    raise ValueError("teacher has no root visits")
                examples.append(
                    SearchTeacherExample(
                        example_id=request.request_id,
                        split=split,
                        context=DistillationContext(
                            route=route,
                            confidence=request.confidence,
                            evidence_count=request.evidence_count,
                            reasoned=request.initial_reasoned,
                            verified=request.initial_verified,
                            high_risk=request.high_risk,
                            retrieval_available=request.retrieval_available,
                            verification_available=request.verification_available,
                            remaining_tokens=request.token_budget,
                            remaining_retrievals=request.max_additional_evidence,
                            legal_actions=actions,
                        ),
                        target_probabilities={action: visits.get(action, 0) / total for action in actions},
                        teacher_plan_fingerprint=plan.plan_fingerprint,
                        teacher_model_fingerprint=scorer.artifact.artifact_fingerprint,
                    )
                )
    return examples


class ComputePoint(BaseModel):
    strategy: str
    iterations_budget: int
    scenario_count: int
    success_rate: float
    unsafe_releases: int
    budget_violations: int
    mean_expanded_nodes: float
    mean_reward_evaluations: float
    mean_planned_tokens: float
    fallback_scenarios: int
    all_receipts_valid: bool


class DistillationReport(BaseModel):
    generated_by: str = "agentforge-search-distillation-eval"
    training_fingerprint: str
    test_fingerprint: str
    policy_fingerprint: str
    validation_kl: float
    validation_uniform_kl: float
    teacher_success_rate: float
    minimum_matching_uniform_iterations: int | None
    minimum_matching_student_iterations: int | None
    iteration_savings_vs_uniform: float | None
    pareto_points: list[ComputePoint]
    points: list[ComputePoint]
    promoted: bool
    reasons: list[str]
    report_fingerprint: str = ""


def _request(scenario: SearchScenario) -> SearchRequest:
    values = scenario.model_dump(exclude={"id", "expected_terminal", "recoverable", "split", "review_status"})
    return SearchRequest(request_id=scenario.id, **values)


def audit_plan(plan: SearchPlan, request: SearchRequest) -> tuple[bool, bool]:
    """Replay the heuristic action contract independently of planner state flags."""
    evidence = request.evidence_count
    confidence = request.confidence
    reasoned, verified = request.initial_reasoned, request.initial_verified
    retrieved = 0
    tokens = 0
    violation = False
    terminal = False
    for action in plan.planned_actions:
        if terminal:
            violation = True
        tokens += ACTION_TOKEN_COST[action]
        if action == "retrieve":
            violation |= (
                not request.retrieval_available or reasoned or retrieved >= request.max_additional_evidence
            )
            retrieved += 1
            evidence += 1
            confidence = min(1.0, confidence + 0.08)
        elif action == "reason":
            violation |= reasoned or evidence == 0
            reasoned = True
            confidence = min(1.0, confidence + 0.1)
        elif action == "verify":
            violation |= not request.verification_available or not reasoned or verified
            verified = True
            confidence = min(1.0, confidence + 0.08)
        elif action == "answer":
            violation |= not (
                reasoned and verified and evidence >= (2 if request.high_risk else 1) and confidence >= 0.62
            )
            terminal = True
        else:
            terminal = True
    budget_violation = tokens > request.token_budget or tokens != plan.tokens_planned
    violation |= not terminal or not plan.planned_actions or plan.planned_actions[-1] != plan.terminal_action
    return bool(violation), budget_violation


def evaluate_distillation(
    artifact: DistilledPolicyArtifact,
    scorer: ProcessRewardScorer,
    scenarios: list[SearchScenario],
    *,
    budgets: tuple[int, ...] = (8, 16, 32, 96),
) -> DistillationReport:
    if not budgets or any(budget < 8 or budget > 2048 for budget in budgets):
        raise ValueError("search budgets must be between 8 and 2048 iterations")
    test = [scenario for scenario in scenarios if scenario.split == "test"]
    if not test:
        raise ValueError("distillation evaluation requires frozen test scenarios")
    if scorer.artifact.artifact_fingerprint not in artifact.teacher_model_fingerprints:
        raise ValueError("distilled-policy teacher scorer mismatch")
    student = DistilledPlanningPolicy(artifact)
    teacher = VerifierGuidedMCTS(scorer)
    teacher_success = sum(
        teacher.plan(_request(scenario)).terminal_action == scenario.expected_terminal for scenario in test
    ) / len(test)
    points = []
    for budget in sorted(set(budgets)):
        for strategy in ("uniform_uct", "distilled_puct"):
            planner = VerifierGuidedMCTS(
                scorer,
                SearchPolicy(iterations=budget, max_nodes=budget + 2),
                distilled_policy=student if strategy == "distilled_puct" else None,
            )
            plans = [planner.plan(_request(scenario)) for scenario in test]
            audits = [
                audit_plan(plan, _request(scenario)) for plan, scenario in zip(plans, test, strict=True)
            ]
            points.append(
                ComputePoint(
                    strategy=strategy,
                    iterations_budget=budget,
                    scenario_count=len(test),
                    success_rate=sum(
                        plan.terminal_action == scenario.expected_terminal and not unsafe and not over
                        for plan, scenario, (unsafe, over) in zip(plans, test, audits, strict=True)
                    )
                    / len(test),
                    unsafe_releases=sum(unsafe for unsafe, _ in audits),
                    budget_violations=sum(over for _, over in audits),
                    mean_expanded_nodes=statistics.fmean(plan.nodes_expanded for plan in plans),
                    mean_reward_evaluations=statistics.fmean(plan.reward_evaluations for plan in plans),
                    mean_planned_tokens=statistics.fmean(plan.tokens_planned for plan in plans),
                    fallback_scenarios=sum(plan.distilled_policy_fallbacks > 0 for plan in plans),
                    all_receipts_valid=all(plan.verify() for plan in plans),
                )
            )
    reasons = []
    if artifact.validation_kl >= artifact.validation_uniform_kl:
        reasons.append("student does not improve validation KL over uniform priors")
    for point in points:
        if point.unsafe_releases or point.budget_violations or not point.all_receipts_valid:
            reasons.append(f"{point.strategy}/{point.iterations_budget} failed safety, budget, or integrity")
    for budget in sorted(set(budgets)):
        baseline, candidate = [point for point in points if point.iterations_budget == budget]
        if candidate.success_rate < baseline.success_rate:
            reasons.append(f"student regresses baseline success at {budget} iterations")
    if max(point.success_rate for point in points if point.strategy == "distilled_puct") < teacher_success:
        reasons.append("student fails to match teacher planning quality")
    matching = {
        strategy: min(
            (
                point.iterations_budget
                for point in points
                if point.strategy == strategy
                and point.success_rate >= teacher_success
                and not point.unsafe_releases
                and not point.budget_violations
            ),
            default=None,
        )
        for strategy in ("uniform_uct", "distilled_puct")
    }
    uniform_min, student_min = matching["uniform_uct"], matching["distilled_puct"]
    # Quality and actual distinct scorer invocations define this front; latency is not inferred.
    pareto = [
        point
        for point in points
        if not any(
            other.success_rate >= point.success_rate
            and other.mean_reward_evaluations <= point.mean_reward_evaluations
            and (
                other.success_rate > point.success_rate
                or other.mean_reward_evaluations < point.mean_reward_evaluations
            )
            for other in points
        )
    ]
    report = DistillationReport(
        training_fingerprint=artifact.training_fingerprint,
        test_fingerprint=fingerprint([scenario.model_dump(mode="json") for scenario in test]),
        policy_fingerprint=artifact.artifact_fingerprint,
        validation_kl=artifact.validation_kl,
        validation_uniform_kl=artifact.validation_uniform_kl,
        teacher_success_rate=teacher_success,
        minimum_matching_uniform_iterations=uniform_min,
        minimum_matching_student_iterations=student_min,
        iteration_savings_vs_uniform=(1 - student_min / uniform_min if uniform_min and student_min else None),
        pareto_points=pareto,
        points=points,
        promoted=not reasons,
        reasons=reasons,
    )
    report.report_fingerprint = fingerprint(report.model_dump(mode="json", exclude={"report_fingerprint"}))
    return report


def verify_report(report: DistillationReport) -> bool:
    return bool(report.report_fingerprint) and report.report_fingerprint == fingerprint(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Distill MCTS search visits and evaluate compute curves.")
    parser.add_argument("--scorer", type=Path, default=Path("evals/experiments/process_reward_model.json"))
    parser.add_argument(
        "--scenarios", type=Path, default=Path("evals/datasets/search_planning_scenarios.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluations/distillation"))
    parser.add_argument("--require-promotion", action="store_true")
    args = parser.parse_args()
    scorer = ProcessRewardScorer.load(args.scorer)
    examples = generate_teacher_examples(scorer)
    artifact = train_distilled_policy(
        examples, teacher_policy_fingerprint=fingerprint(SearchPolicy().model_dump(mode="json"))
    )
    report = evaluate_distillation(artifact, scorer, load_scenarios(args.scenarios))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact.save(args.output_dir / "policy.json")
    (args.output_dir / "teacher-examples.jsonl").write_text(
        "\n".join(item.model_dump_json() for item in examples) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(report.model_dump_json(indent=2))
    return 2 if args.require_promotion and (not report.promoted or not verify_report(report)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
