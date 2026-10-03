"""Explicit process-label retraining, calibration and leakage-resistant controls."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path

from agent.adaptive_compute import (
    ComputePolicy,
    ComputeSignals,
    candidate_assessment,
    plan_compute,
    select_candidate,
)
from agent.execution_replay import ExecutionReplayStore, digest
from agent.process_reward import (
    ProcessRewardArtifact,
    ProcessRewardScorer,
    ProcessStep,
    train_process_reward_model,
)
from agent.process_supervision import ProcessCohort, ProcessSupervisionStore, ReviewedProcessCandidate
from agent.prospective_validation import HoldoutLedger
from evals.preference_evaluation import _paired_lower, write_once


def _temperature(probability: float, temperature: float) -> float:
    probability = min(1 - 1e-9, max(1e-9, probability))
    logit = math.log(probability / (1 - probability)) / temperature
    return 1 / (1 + math.exp(-max(-30, min(30, logit))))


def _rows(traces, scorer, split):
    result = []
    for trace in traces:
        if trace.split != split:
            continue
        predictions = scorer.step_probabilities(trace.steps, trace.high_risk)
        for step, probability in zip(trace.steps, predictions, strict=True):
            result.append((trace.group_id, step.step_label, probability))
    return result


def _ece(rows):
    labelled = [(target, probability) for _, target, probability in rows if target is not None]
    if not labelled:
        return None
    result = 0.0
    for index in range(10):
        bucket = [
            (target, probability)
            for target, probability in labelled
            if min(9, int(probability * 10)) == index
        ]
        if bucket:
            result += abs(sum(target - probability for target, probability in bucket)) / len(labelled)
    return result


def evaluate_process(
    cohort: ProcessCohort,
    store: ProcessSupervisionStore,
    tenant: str,
    artifact_key: bytes,
    ledger: HoldoutLedger,
) -> tuple[dict, ReviewedProcessCandidate]:
    if len(artifact_key) < 32 or artifact_key == store.replay.key or ledger.key != store.replay.key:
        raise ValueError("process learning requires independent model key and shared replay ledger key")
    store.validate_lineage(tenant, cohort)
    policy = {
        "version": "explicit-workflow-supervision-v1",
        "epochs": 300,
        "explicit_steps_only": True,
        "minimum_families": {"train": 20, "validation": 10, "test": 20},
        "minimum_labels_per_class": 5,
        "minimum_test_review_coverage": 0.8,
        "maximum_step_brier": 0.2,
        "maximum_step_ece": 0.2,
        "minimum_scope_test_families": 10,
        "minimum_paired_brier_improvement_lcb": 0,
        "temperature_grid": [0.25, 4, 0.05],
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 17,
        "missingness": "worst-case binary-target Brier and paired delta endpoints",
    }
    model_tenant = digest(artifact_key, "process-model-tenant", tenant)
    key = store.replay.key
    study = digest(key, "reviewed-process-study", [cohort.fingerprint, policy, model_tenant])
    families = sorted(
        {member.snapshot.family.task_family for member in cohort.members if member.split == "test"}
    )
    cached = ledger.reserve_families(study, cohort.fingerprint, families)
    if cached is not None:
        if cached.get("fingerprint") != digest(
            key, "process-training-cache", {k: v for k, v in cached.items() if k != "fingerprint"}
        ):
            raise ValueError("process training cache integrity failed")
        candidate = ReviewedProcessCandidate.model_validate(cached["candidate"])
        candidate.verify(artifact_key)
        if (
            candidate.tenant != model_tenant
            or candidate.cohort_fingerprint != cohort.fingerprint
            or candidate.report_fingerprint != cached["report"]["fingerprint"]
            or candidate.simulation != cohort.simulation
        ):
            raise ValueError("cached process candidate identity mismatch")
        return cached["report"], candidate
    traces = cohort.traces()
    training = [trace for trace in traces if trace.split == "train"]
    raw = train_process_reward_model(training, epochs=policy["epochs"], explicit_steps_only=True)
    raw_scorer = ProcessRewardScorer(raw)
    validation = _rows(traces, raw_scorer, "validation")
    scored_validation = [(target, probability) for _, target, probability in validation if target is not None]
    if not scored_validation:
        raise ValueError("step calibration requires independent validation annotations")
    temperature = min(
        (index / 20 for index in range(5, 81)),
        key=lambda value: (
            sum(
                (_temperature(probability, value) - target) ** 2 for target, probability in scored_validation
            ),
            abs(value - 1),
        ),
    )
    artifact = ProcessRewardArtifact.model_validate(
        {**raw.model_dump(), "weights": [weight / temperature for weight in raw.weights]}
    )
    artifact.seal()
    scorer = ProcessRewardScorer(artifact)
    targets = [step.step_label for trace in training for step in trace.steps if step.step_label is not None]
    constant = sum(targets) / len(targets)
    test_rows = _rows(traces, scorer, "test")
    group_errors = {}
    worst_groups = {}
    for family, target, probability in test_rows:
        group_errors.setdefault(family, [])
        worst_groups.setdefault(family, [])
        if target is not None:
            group_errors[family].append(((probability - target) ** 2, (constant - target) ** 2))
            worst_groups[family].append(
                ((probability - target) ** 2, (constant - target) ** 2 - (probability - target) ** 2)
            )
        else:
            worst_groups[family].append(
                (
                    max(probability**2, (1 - probability) ** 2),
                    min((constant - y) ** 2 - (probability - y) ** 2 for y in (0, 1)),
                )
            )
    family_means = {
        family: (sum(a for a, _ in rows) / len(rows), sum(b for _, b in rows) / len(rows))
        for family, rows in group_errors.items()
        if rows
    }
    brier = sum(a for a, _ in family_means.values()) / len(family_means) if family_means else None
    baseline_brier = sum(b for _, b in family_means.values()) / len(family_means) if family_means else None
    lower = _paired_lower([b - a for a, b in family_means.values()]) if family_means else None
    worst_brier = (
        sum(sum(a for a, _ in rows) / len(rows) for rows in worst_groups.values()) / len(worst_groups)
        if worst_groups
        else None
    )
    worst_lower = (
        _paired_lower([sum(b for _, b in rows) / len(rows) for rows in worst_groups.values()])
        if worst_groups
        else None
    )
    labelled = sum(target is not None for _, target, _ in test_rows)
    coverage = labelled / len(test_rows) if test_rows else 0
    ece = _ece(test_rows)
    failures, support = [], {}
    for split in ("train", "validation", "test"):
        rows = _rows(traces, scorer, split)
        positive = sum(target == 1 for _, target, _ in rows)
        negative = sum(target == 0 for _, target, _ in rows)
        split_families = len({trace.group_id for trace in traces if trace.split == split})
        reviewed_families = len({family for family, target, _ in rows if target is not None})
        support[split] = {
            "families": split_families,
            "reviewed_families": reviewed_families,
            "positive_steps": positive,
            "negative_steps": negative,
        }
        if reviewed_families < policy["minimum_families"][split] or min(positive, negative) < 5:
            failures.append("insufficient_split_family_or_label_support")
    if coverage < 0.8:
        failures.append("insufficient_test_step_review_coverage")
    if (
        brier is None
        or brier > 0.2
        or ece is None
        or ece > 0.2
        or lower is None
        or lower < 0
        or worst_brier is None
        or worst_brier > 0.2
        or worst_lower is None
        or worst_lower < 0
    ):
        failures.append("heldout_step_quality_hold")
    scopes = {}
    for member in cohort.members:
        if member.split == "test":
            source = member.snapshot.source
            scope = f"{source.route}:{int(source.high_risk)}"
            scopes.setdefault(scope, set()).add(member.snapshot.family.task_family)
    if any(len(values) < 10 for values in scopes.values()):
        failures.append("insufficient_route_risk_scope_support")
    # Outcome credit is an explicitly weak comparator, not claimed step truth.
    weak_traces = [
        trace.model_copy(
            update={"steps": [step.model_copy(update={"step_label": None}) for step in trace.steps]}
        )
        for trace in training
    ]
    weak = ProcessRewardScorer(train_process_reward_model(weak_traces, epochs=policy["epochs"]))
    weak_rows = _rows(traces, weak, "test")
    weak_groups = {}
    for family, target, probability in weak_rows:
        if target is not None:
            weak_groups.setdefault(family, []).append((probability - target) ** 2)
    weak_brier = (
        sum(sum(values) / len(values) for values in weak_groups.values()) / len(weak_groups)
        if weak_groups
        else None
    )
    report = {
        "evidence_kind": "synthetic_process_controls"
        if cohort.simulation
        else "reviewed_runtime_workflow_steps",
        "claim_scope": "predictive quality of typed workflow proxies; not chain-of-thought supervision or final-answer lift",
        "cohort_fingerprint": cohort.fingerprint,
        "policy": policy,
        "split_support": support,
        "test_review_coverage": coverage,
        "step_brier": brier,
        "step_ece": ece,
        "constant_baseline_brier": baseline_brier,
        "weak_outcome_credit_brier": weak_brier,
        "paired_brier_improvement_lower_95": lower,
        "worst_case_step_brier": worst_brier,
        "worst_case_paired_brier_improvement_lower_95": worst_lower,
        "calibration_temperature": temperature,
        "scope_test_families": {scope: len(values) for scope, values in scopes.items()},
        "explicit_training_steps": len(targets),
        "human_labeled_steps": artifact.human_labeled_steps,
        "gate_passed": not failures,
        "candidate_ready_for_review": not failures and not cohort.simulation,
        "production_activation": False,
        "failure_reasons": sorted(set(failures)),
        "exclusions": cohort.exclusions,
    }
    report["fingerprint"] = digest(key, "process-supervision-report", report)
    candidate = ReviewedProcessCandidate(
        tenant=model_tenant,
        cohort_fingerprint=cohort.fingerprint,
        report_fingerprint=report["fingerprint"],
        simulation=cohort.simulation,
        calibration_temperature=temperature,
        explicit_training_steps=len(targets),
        artifact=artifact,
    ).seal(artifact_key)
    cached = {"report": report, "candidate": candidate.model_dump(mode="json")}
    cached["fingerprint"] = digest(key, "process-training-cache", cached)
    ledger.finish(study, cached)
    return report, candidate


def seed_process(
    store: ProcessSupervisionStore,
    tenant: str,
    *,
    review_fraction: float = 1,
    shifted: bool = False,
    origin: str = "synthetic",
    varied_confidence: bool = False,
) -> ProcessCohort:
    clock = [0.0]
    store.replay.clock = lambda: clock[0]
    for split, timestamp, count in (("train", 100, 20), ("validation", 300, 10), ("test", 500, 20)):
        for index in range(count):
            clock[0] = float(timestamp + index)
            good_confidence = round(0.75 + 0.01 * (index % 20), 6) if varied_confidence else 0.9
            bad_confidence = round(min(0.99, 0.55 + 0.025 * (index % 20)), 6) if varied_confidence else 0.6
            policy = ComputePolicy()
            plan = plan_compute(
                ComputeSignals(
                    route="rag",
                    grounding_action="repair",
                    grounding_confidence=0.4,
                    uncertainty_decision="abstain",
                    evidence_count=2,
                ),
                policy,
            )
            candidates = [
                candidate_assessment(
                    candidate_id=name,
                    answer="PRIVATE synthetic workflow answer",
                    confidence=confidence,
                    grounded=good,
                    conformal_decision="release" if good else "abstain",
                    claim_keys=["private evidence"],
                    token_count=100,
                    latency_ms=10,
                )
                for name, confidence, good in (
                    ("good", good_confidence, True),
                    ("bad", bad_confidence, False),
                )
            ]
            receipt = select_candidate(plan, candidates, policy)
            ids = store.replay.capture(
                plan,
                receipt,
                tenant=tenant,
                request_id=f"process-{split}-{index}",
                task_family=f"process-family-{split}-{index}",
                consent=True,
                origin=origin,
            )
            captured = []
            for event_id, good in zip(ids, (True, False), strict=True):
                confidence = good_confidence if good else bad_confidence
                steps = [
                    ProcessStep(
                        step_id="PRIVATE step identifier", kind="retrieve", has_evidence=True, confidence=1
                    ),
                    ProcessStep(step_id="reason", kind="reason", has_evidence=True, confidence=confidence),
                    ProcessStep(
                        step_id="verify",
                        kind="verify",
                        has_evidence=True,
                        citation_valid=good,
                        error=not good,
                        confidence=confidence,
                    ),
                    ProcessStep(
                        step_id="answer",
                        kind="answer",
                        has_evidence=True,
                        citation_valid=good,
                        confidence=confidence,
                    ),
                ]
                store.capture(tenant, event_id, steps, consent=True)
                captured.append((event_id, good))
            for event_id, good in captured:
                if split != "test" or index < int(count * review_fraction):
                    for step_index in range(4):
                        correct = good or step_index < 2
                        if shifted and split == "test":
                            correct = not correct
                        store.review(
                            tenant,
                            event_id,
                            step_index,
                            "correct" if correct else "incorrect",
                            "synthetic-reviewer",
                        )
                store.replay.review(
                    tenant,
                    event_id,
                    verdict="correct" if good else "incorrect",
                    unsafe=not good,
                    reviewer="synthetic-reviewer",
                )
    clock[0] = 600.0
    return store.freeze(
        tenant, train_cutoff=150, validation_cutoff=350, embargo_seconds=50, simulation=origin == "synthetic"
    )


def run_drill() -> dict:
    reports = {}
    with tempfile.TemporaryDirectory(prefix="agentforge-process-supervision-") as root:
        for name, fraction, shifted in (
            ("clean", 1, False),
            ("missing_review", 0.5, False),
            ("shifted_labels", 1, True),
        ):
            key, model_key = (
                b"synthetic-process-replay-key-not-secret",
                b"synthetic-process-model-key-not-secret",
            )
            replay = ExecutionReplayStore(Path(root) / f"{name}.sqlite3", key)
            store = ProcessSupervisionStore(replay)
            cohort = seed_process(store, "demo", review_fraction=fraction, shifted=shifted)
            reports[name], _ = evaluate_process(
                cohort, store, "demo", model_key, HoldoutLedger(Path(root) / f"{name}-ledger.sqlite3", key)
            )
    return {
        "evidence_kind": "synthetic_process_supervision_ablation",
        "reports": reports,
        "gate_passed": reports["clean"]["gate_passed"]
        and not reports["missing_review"]["gate_passed"]
        and not reports["shifted_labels"]["gate_passed"],
        "production_activation": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", default=os.getenv("EXECUTION_REPLAY_PATH", "data/execution-replay/replay.sqlite3")
    )
    parser.add_argument("--ledger", default="data/execution-replay/holdout-ledger.sqlite3")
    parser.add_argument("--tenant", default="")
    commands = parser.add_subparsers(dest="command", required=True)
    queue = commands.add_parser("queue")
    queue.add_argument("--limit", type=int, default=100)
    review = commands.add_parser("review")
    review.add_argument("event_id")
    review.add_argument("--step-index", type=int, required=True)
    review.add_argument("--verdict", choices=["correct", "incorrect", "ambiguous"], required=True)
    review.add_argument("--reviewer", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--train-cutoff", type=float, required=True)
    freeze.add_argument("--validation-cutoff", type=float, required=True)
    freeze.add_argument("--embargo-seconds", type=float, default=3600)
    freeze.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("cohort")
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--require-gate", action="store_true")
    drill = commands.add_parser("drill")
    drill.add_argument("--output", default="data/evaluations/process-supervision/drill.json")
    drill.add_argument("--require-gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        protected = {
            Path(path).resolve()
            for path in (
                args.store,
                args.ledger,
                os.getenv("PROCESS_REWARD_MODEL_PATH", "data/evaluations/process-reward/model.json"),
                os.getenv("VERIFIER_ENSEMBLE_PATH", "data/evaluations/verifier-uncertainty/ensemble.json"),
            )
        }
        protected |= {
            Path(str(path) + suffix) for path in tuple(protected) for suffix in ("-wal", "-shm", "-journal")
        }
        if args.command == "evaluate":
            protected.add(Path(args.cohort).resolve())
        outputs = [Path(args.output).resolve()] if hasattr(args, "output") else []
        if args.command == "evaluate":
            outputs.append(Path(args.candidate).resolve())
        if len(set(outputs)) != len(outputs) or any(path in protected for path in outputs):
            raise ValueError("output aliases input, SQLite storage, or active verifier")
        if args.command == "drill":
            report = run_drill()
            write_once(args.output, report)
        else:
            if not args.tenant.strip():
                raise ValueError("process supervision requires tenant")
            replay = ExecutionReplayStore(args.store, os.getenv("EXECUTION_REPLAY_KEY", "").encode())
            store = ProcessSupervisionStore(replay)
            if args.command == "queue":
                print(json.dumps(store.queue(args.tenant, limit=args.limit), indent=2))
                return 0
            if args.command == "review":
                print(
                    store.review(
                        args.tenant, args.event_id, args.step_index, args.verdict, args.reviewer
                    ).model_dump_json()
                )
                return 0
            if args.command == "freeze":
                cohort = store.freeze(
                    args.tenant,
                    train_cutoff=args.train_cutoff,
                    validation_cutoff=args.validation_cutoff,
                    embargo_seconds=args.embargo_seconds,
                )
                write_once(args.output, cohort.model_dump(mode="json"))
                print(json.dumps({"candidates": len(cohort.members), "fingerprint": cohort.fingerprint}))
                return 0
            cohort = ProcessCohort.model_validate_json(Path(args.cohort).read_text(encoding="utf-8"))
            if cohort.simulation:
                raise ValueError("use the drill for synthetic cohorts; no production candidate export")
            report, candidate = evaluate_process(
                cohort,
                store,
                args.tenant,
                os.getenv("PROCESS_SUPERVISION_MODEL_KEY", "").encode(),
                HoldoutLedger(args.ledger, replay.key),
            )
            write_once(args.output, report)
            if report["candidate_ready_for_review"]:
                write_once(args.candidate, candidate.model_dump(mode="json"))
        print(json.dumps(report, indent=2))
        return 0 if not args.require_gate or report["gate_passed"] else 1
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"status": "held", "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
