"""Held-out evaluation for learned action-conditioned planning dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from pydantic import BaseModel, Field

from agent.process_reward import ProcessRewardScorer
from agent.search_planner import SearchRequest, VerifierGuidedMCTS
from agent.world_model import (
    AgentWorldModel,
    TransitionExample,
    WorldModelArtifact,
    load_transitions,
    train_world_model,
)

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "world_model_transitions.jsonl"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class TransitionOutcome(BaseModel):
    transition_id: str
    action: str
    actual_confidence_delta: float
    heuristic_confidence_delta: float
    learned_confidence_delta: float
    actual_evidence_delta: float
    heuristic_evidence_delta: float
    learned_evidence_delta: float
    actual_success: bool
    learned_success_probability: float = Field(ge=0, le=1)
    out_of_distribution: bool


class WorldModelEvaluationReport(BaseModel):
    schema_version: str = "1.0"
    generated_by: str = "agentforge-world-model-eval"
    dataset_fingerprint: str
    artifact_fingerprint: str
    transition_count: int
    heuristic_confidence_mae: float
    learned_confidence_mae: float
    confidence_mae_reduction: float
    heuristic_evidence_mae: float
    learned_evidence_mae: float
    heuristic_success_brier: float
    learned_success_brier: float
    success_temperature: float
    ood_detection_rate: float
    clean_planning_success: bool
    ood_planning_abstained: bool
    shifted_planning_abstained: bool
    artifact_integrity: bool
    promoted: bool
    reasons: list[str]
    outcomes: list[TransitionOutcome]
    report_fingerprint: str = ""


def _heuristic(action: str) -> tuple[float, float, float]:
    return (
        {"retrieve": 0.08, "reason": 0.1, "verify": 0.08}.get(action, 0.0),
        1.0 if action == "retrieve" else 0.0,
        1.0,
    )


def corrupt_one_world_model_member(
    artifact: WorldModelArtifact,
) -> WorldModelArtifact:
    """Create an integrity-valid but behaviorally shifted member for an OOD drill."""
    shifted = artifact.model_copy(deep=True)
    shifted.members[-1].confidence_weights[0] += 0.5
    shifted.members[-1].seal()
    shifted.seal()
    return shifted


def evaluate_world_model(
    artifact: WorldModelArtifact,
    examples: list[TransitionExample],
    *,
    process_reward_artifact: Path = Path("evals/experiments/process_reward_model.json"),
) -> WorldModelEvaluationReport:
    model = AgentWorldModel(artifact)
    test = [item for item in examples if item.split == "test"]
    if not test:
        raise ValueError("world-model evaluation has no held-out transitions")
    outcomes = []
    for item in test:
        prediction = model.predict_example(item)
        heuristic_confidence, heuristic_evidence, _ = _heuristic(item.action)
        actual_confidence = item.after_confidence - item.before_confidence
        actual_evidence = float(item.after_evidence - item.before_evidence)
        outcomes.append(
            TransitionOutcome(
                transition_id=item.transition_id,
                action=item.action,
                actual_confidence_delta=actual_confidence,
                heuristic_confidence_delta=heuristic_confidence,
                learned_confidence_delta=prediction.confidence_delta_mean,
                actual_evidence_delta=actual_evidence,
                heuristic_evidence_delta=heuristic_evidence,
                learned_evidence_delta=prediction.evidence_delta_mean,
                actual_success=item.transition_succeeded,
                learned_success_probability=prediction.success_probability,
                out_of_distribution=prediction.out_of_distribution,
            )
        )
    heuristic_confidence_mae = statistics.fmean(
        abs(item.actual_confidence_delta - item.heuristic_confidence_delta)
        for item in outcomes
    )
    learned_confidence_mae = statistics.fmean(
        abs(item.actual_confidence_delta - item.learned_confidence_delta)
        for item in outcomes
    )
    heuristic_evidence_mae = statistics.fmean(
        abs(item.actual_evidence_delta - item.heuristic_evidence_delta)
        for item in outcomes
    )
    learned_evidence_mae = statistics.fmean(
        abs(item.actual_evidence_delta - item.learned_evidence_delta)
        for item in outcomes
    )
    heuristic_success_brier = statistics.fmean(
        (1.0 - float(item.actual_success)) ** 2 for item in outcomes
    )
    learned_success_brier = statistics.fmean(
        (item.learned_success_probability - float(item.actual_success)) ** 2
        for item in outcomes
    )
    ood_probes = [
        model.predict(
            route=route,
            action="reason",
            high_risk=False,
            evidence_count=1,
            confidence=0.5,
            reasoned=False,
            verified=False,
        )
        for route in ("medical", "finance", "unseen-route")
    ]
    shifted_model = AgentWorldModel(corrupt_one_world_model_member(artifact))
    ood_probes.append(
        shifted_model.predict(
            route="rag",
            action="reason",
            high_risk=False,
            evidence_count=1,
            confidence=0.55,
            reasoned=False,
            verified=False,
        )
    )
    scorer = ProcessRewardScorer.load(process_reward_artifact)
    planner = VerifierGuidedMCTS(scorer, world_model=model)
    clean_plan = planner.plan(
        SearchRequest(
            request_id="world-model-clean-plan",
            route="rag",
            evidence_count=1,
            confidence=0.55,
            token_budget=1000,
        )
    )
    ood_plan = planner.plan(
        SearchRequest(
            request_id="world-model-ood-plan",
            route="unseen-route",
            evidence_count=1,
            confidence=0.55,
            token_budget=1000,
        )
    )
    shifted_plan = VerifierGuidedMCTS(scorer, world_model=shifted_model).plan(
        SearchRequest(
            request_id="world-model-member-shift-plan",
            route="rag",
            evidence_count=1,
            confidence=0.55,
            token_budget=1000,
        )
    )
    ood_detection_rate = sum(item.out_of_distribution for item in ood_probes) / len(
        ood_probes
    )
    reasons = []
    if learned_confidence_mae >= heuristic_confidence_mae:
        reasons.append("learned confidence dynamics did not improve held-out MAE")
    if learned_evidence_mae > heuristic_evidence_mae + 0.05:
        reasons.append("learned evidence dynamics regressed held-out MAE")
    if learned_success_brier >= heuristic_success_brier:
        reasons.append("learned transition success did not improve Brier score")
    if ood_detection_rate < 1:
        reasons.append("world model missed an unseen-route OOD probe")
    clean_success = clean_plan.terminal_action == "answer" and not clean_plan.world_model_ood
    ood_abstained = ood_plan.terminal_action == "abstain" and ood_plan.world_model_ood
    shifted_abstained = (
        shifted_plan.terminal_action == "abstain" and shifted_plan.world_model_ood
    )
    if not clean_success:
        reasons.append("model-predictive search failed the supported clean plan")
    if not ood_abstained:
        reasons.append("model-predictive search failed to abstain on OOD dynamics")
    if not shifted_abstained:
        reasons.append("model-predictive search missed ensemble-member shift")
    report = WorldModelEvaluationReport(
        dataset_fingerprint=_hash([item.model_dump(mode="json") for item in test]),
        artifact_fingerprint=artifact.artifact_fingerprint,
        transition_count=len(outcomes),
        heuristic_confidence_mae=heuristic_confidence_mae,
        learned_confidence_mae=learned_confidence_mae,
        confidence_mae_reduction=heuristic_confidence_mae - learned_confidence_mae,
        heuristic_evidence_mae=heuristic_evidence_mae,
        learned_evidence_mae=learned_evidence_mae,
        heuristic_success_brier=heuristic_success_brier,
        learned_success_brier=learned_success_brier,
        success_temperature=artifact.success_temperature,
        ood_detection_rate=ood_detection_rate,
        clean_planning_success=clean_success,
        ood_planning_abstained=ood_abstained,
        shifted_planning_abstained=shifted_abstained,
        artifact_integrity=artifact.verify(),
        promoted=not reasons and artifact.verify(),
        reasons=reasons,
        outcomes=outcomes,
    )
    report.report_fingerprint = _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )
    return report


def verify_report(report: WorldModelEvaluationReport) -> bool:
    return report.report_fingerprint == _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate learned planning dynamics.")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--artifact", type=Path, default=Path("data/evaluations/world-model/world-model.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluations/world-model/report.json")
    )
    parser.add_argument(
        "--process-reward-artifact",
        type=Path,
        default=Path("evals/experiments/process_reward_model.json"),
    )
    parser.add_argument("--check-artifact", type=Path)
    parser.add_argument("--check-report", type=Path)
    parser.add_argument("--require-promotion", action="store_true")
    args = parser.parse_args()
    examples = load_transitions(args.dataset)
    artifact = train_world_model(examples)
    if args.check_artifact:
        expected = WorldModelArtifact.load(args.check_artifact)
        if artifact.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise SystemExit("world-model artifact differs from checked baseline")
    artifact.save(args.artifact)
    report = evaluate_world_model(
        artifact,
        examples,
        process_reward_artifact=args.process_reward_artifact,
    )
    if args.check_report:
        expected_report = WorldModelEvaluationReport.model_validate_json(
            args.check_report.read_text(encoding="utf-8")
        )
        if report.model_dump(mode="json") != expected_report.model_dump(mode="json"):
            raise SystemExit("world-model report differs from checked baseline")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "heuristic_confidence_mae": report.heuristic_confidence_mae,
                "learned_confidence_mae": report.learned_confidence_mae,
                "heuristic_success_brier": report.heuristic_success_brier,
                "learned_success_brier": report.learned_success_brier,
                "success_temperature": report.success_temperature,
                "ood_detection_rate": report.ood_detection_rate,
                "clean_planning_success": report.clean_planning_success,
                "ood_planning_abstained": report.ood_planning_abstained,
                "shifted_planning_abstained": report.shifted_planning_abstained,
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
