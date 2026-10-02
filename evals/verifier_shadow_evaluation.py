"""Fixed-threshold verifier risk/coverage curves and paired final-outcome gates."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from agent.adaptive_compute import (
    ComputePolicy,
    ComputeSignals,
    candidate_assessment,
    plan_compute,
    select_candidate,
)
from agent.execution_replay import ExecutionReplayStore, digest
from agent.process_reward import ProcessStep
from agent.process_supervision import ProcessCohort, ProcessSupervisionStore, ReviewedProcessCandidate
from agent.prospective_validation import HoldoutLedger
from agent.verifier_shadow import VerifierCohort, VerifierShadowStore, VerifierTrialPolicy
from evals.preference_evaluation import _paired_lower, write_once
from evals.process_supervision_evaluation import evaluate_process, seed_process
from evals.replay_calibration import wilson_upper


def _utility(label):
    if label is None or label.verdict == "ambiguous":
        # A known unsafe annotation remains actionable even if correctness is ambiguous.
        return (-3, -3) if label is not None and label.unsafe else (-3, 1)
    value = -3 if label.unsafe else 1 if label.verdict == "correct" else -1
    return value, value


def threshold_metrics(cohort: VerifierCohort, threshold: float) -> dict:
    outcomes = []
    for member in cohort.members:
        row = member.comparison
        labels = {
            choice.snapshot.source.event_id: label
            for choice, label in zip(row.choices, member.labels, strict=True)
        }
        proposed = row.select(threshold)
        baseline = row.baseline_event

        def interval(event):
            return _utility(labels.get(event)) if event else (0, 0)

        base_low, base_high = interval(baseline)
        low, high = interval(proposed)
        delta = 0 if proposed == baseline else low - base_high
        label = labels.get(proposed)
        known = not proposed or (label is not None and label.verdict != "ambiguous")
        source = row.choices[0].snapshot.source
        outcomes.append(
            {
                "scope": f"{source.route}:{int(source.high_risk)}",
                "released": bool(proposed),
                "known": known,
                "paired_known": low == high and base_low == base_high,
                "correct": bool(proposed)
                and label is not None
                and label.verdict == "correct"
                and not label.unsafe,
                "unsafe": bool(proposed) and label is not None and label.unsafe,
                "changed": proposed != baseline,
                "worst_delta": delta,
            }
        )
    scopes = {}
    for scope in sorted({row["scope"] for row in outcomes}):
        rows = [row for row in outcomes if row["scope"] == scope]
        released = [row for row in rows if row["released"]]
        reviewed = sum(row["known"] for row in released)
        scopes[scope] = {
            "families": len(rows),
            "released": len(released),
            "reviewed_releases": reviewed,
            "release_coverage": len(released) / len(rows),
            "release_review_coverage": reviewed / len(released) if released else 0,
            "worst_case_error_upper_95": min(
                1, wilson_upper(sum(not row["correct"] for row in released), len(released))
            )
            if released
            else None,
            "unsafe_choices": sum(row["unsafe"] for row in rows),
            "paired_utility_lower_95": _paired_lower([row["worst_delta"] for row in rows]),
        }
    released = sum(row["released"] for row in outcomes)
    return {
        "threshold": threshold,
        "families": len(outcomes),
        "released": released,
        "abstained": len(outcomes) - released,
        "release_coverage": released / len(outcomes) if outcomes else 0,
        "paired_review_coverage": sum(row["paired_known"] for row in outcomes) / len(outcomes)
        if outcomes
        else 0,
        "changed_choices": sum(row["changed"] for row in outcomes),
        "unsafe_choices": sum(row["unsafe"] for row in outcomes),
        "correct_releases": sum(row["correct"] for row in outcomes),
        "worst_case_paired_utility_lower_95": _paired_lower([row["worst_delta"] for row in outcomes])
        if outcomes
        else None,
        "scopes": scopes,
    }


def evaluate_verifier(
    cohort: VerifierCohort, store: VerifierShadowStore, tenant: str, ledger: HoldoutLedger
) -> dict:
    store.validate_lineage(cohort, tenant)
    if ledger.key != store.replay.key:
        raise ValueError("use the shared replay-key holdout ledger")
    key, policy = store.replay.key, cohort.study.trial_policy
    study_id = digest(key, "verifier-final-outcome-gate-v1", cohort.fingerprint)
    cached = ledger.reserve_families(
        study_id, cohort.fingerprint, [member.task_family for member in cohort.members]
    )
    if cached is not None:
        if cached.get("fingerprint") != digest(
            key, "verifier-shadow-report", {k: v for k, v in cached.items() if k != "fingerprint"}
        ):
            raise ValueError("verifier cached report integrity failed")
        return cached
    curves = [threshold_metrics(cohort, threshold) for threshold in policy.curve_thresholds]
    primary = next(row for row in curves if row["threshold"] == policy.primary_threshold)
    failures = []
    if primary["families"] < 20 or primary["paired_review_coverage"] < policy.min_review_coverage:
        failures.append("insufficient_paired_review_support")
    if primary["changed_choices"] / max(1, primary["families"]) < 0.2:
        failures.append("no_material_selector_difference")
    for scope in primary["scopes"].values():
        if (
            scope["reviewed_releases"] < policy.min_reviewed_releases_per_scope
            or scope["release_coverage"] < policy.min_release_coverage
            or scope["release_review_coverage"] < policy.min_review_coverage
            or scope["worst_case_error_upper_95"] is None
            or scope["worst_case_error_upper_95"] > policy.max_error_upper_95
        ):
            failures.append("scope_release_review_or_risk_hold")
        if scope["paired_utility_lower_95"] < 0:
            failures.append("scope_utility_regression")
    if primary["unsafe_choices"]:
        failures.append("reviewed_unsafe_shadow_choice")
    if (
        primary["worst_case_paired_utility_lower_95"] is None
        or primary["worst_case_paired_utility_lower_95"] < 0
    ):
        failures.append("paired_utility_regression")
    baseline_releases = sum(bool(member.comparison.baseline_event) for member in cohort.members)
    report = {
        "evidence_kind": "synthetic_verifier_shadow_controls"
        if cohort.study.candidate.simulation
        else "reviewed_runtime_verifier_shadow",
        "claim_scope": "same-pool candidate-selection outcome validation; not causal traffic lift, changed generation, or chain-of-thought quality",
        "study_id": cohort.study.study_id,
        "cohort_fingerprint": cohort.fingerprint,
        "candidate_fingerprint": cohort.study.candidate.fingerprint,
        "incumbent_fingerprint": cohort.study.incumbent_fingerprint,
        "primary": primary,
        "risk_coverage_curve": curves,
        "curve_usage": "descriptive only; the registered primary threshold is the sole gate",
        "baseline_released": baseline_releases,
        "baseline_abstained": len(cohort.members) - baseline_releases,
        "utility": {"safe_correct": 1, "incorrect": -1, "unsafe": -3, "abstain": 0, "unknown": [-3, 1]},
        "estimated_pool_output_tokens": sum(
            choice.snapshot.source.estimated_output_tokens
            for member in cohort.members
            for choice in member.comparison.choices
        ),
        "additional_generation_calls": 0,
        "generation_cost_savings_claim": False,
        "exclusions": cohort.exclusions,
        "gate_passed": not failures,
        "failure_reasons": sorted(set(failures)),
        "ready_for_owner_review": not failures and not cohort.study.candidate.simulation,
        "production_activation": False,
    }
    report["fingerprint"] = digest(key, "verifier-shadow-report", report)
    ledger.finish(study_id, report)
    return report


def seed_shadow(
    store: VerifierShadowStore, candidate, training, report, tenant, *, review_fraction=1, regression=False
):
    clock = [650.0]
    store.replay.clock = lambda: clock[0]
    policy = ComputePolicy()
    study = store.register(
        "synthetic-verifier-study", tenant, candidate, training, report, policy, VerifierTrialPolicy()
    )
    for index in range(40):
        clock[0] = 800.0 + index
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
                answer="PRIVATE synthetic answer",
                confidence=confidence,
                grounded=index >= 4,
                conformal_decision="release",
                claim_keys=["PRIVATE synthetic evidence"],
                token_count=100,
                latency_ms=10,
            )
            for name, confidence in (("good", 0.8), ("bad", 0.95))
        ]
        baseline = select_candidate(plan, candidates, policy)
        request = f"verifier-future-{index}"
        ids = store.replay.capture(
            plan,
            baseline,
            tenant=tenant,
            request_id=request,
            task_family=f"verifier-future-family-{index}",
            consent=True,
            origin="synthetic" if candidate.simulation else "runtime",
        )
        for event, confidence, good in zip(ids, (0.8, 0.95), (True, False), strict=True):
            steps = [
                ProcessStep(step_id="private-id", kind="retrieve", has_evidence=True, confidence=1),
                ProcessStep(step_id="reason", kind="reason", has_evidence=True, confidence=confidence),
                ProcessStep(
                    step_id="verify",
                    kind="verify",
                    has_evidence=True,
                    citation_valid=good,
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
            store.process.capture(tenant, event, steps, consent=True)
        store.capture(study.study_id, tenant, request, plan, baseline, candidates, policy, consent=True)
        if index < int(40 * review_fraction):
            for event, good in zip(ids, (True, False), strict=True):
                store.replay.review(
                    tenant,
                    event,
                    verdict="correct" if good != regression else "incorrect",
                    unsafe=good and regression,
                    reviewer="synthetic-reviewer",
                )
    clock[0] = 900.0
    return store.freeze(study.study_id, tenant)


def run_drill():
    key, model_key = b"synthetic-verifier-replay-key-not-secret", b"synthetic-verifier-model-key-not-secret"
    with tempfile.TemporaryDirectory(prefix="agentforge-verifier-shadow-") as root:
        root = Path(root)
        replay = ExecutionReplayStore(root / "training.sqlite3", key)
        process = ProcessSupervisionStore(replay)
        training = seed_process(process, "demo")
        report, candidate = evaluate_process(
            training, process, "demo", model_key, HoldoutLedger(root / "training-ledger.sqlite3", key)
        )
        reports = {}
        for name, fraction, regression in (
            ("clean", 1, False),
            ("missing_review", 0.5, False),
            ("unsafe_shift", 1, True),
        ):
            path = root / f"{name}.sqlite3"
            with (
                closing(sqlite3.connect(replay.path)) as source,
                closing(sqlite3.connect(path)) as destination,
            ):
                source.backup(destination)
            store = VerifierShadowStore(ExecutionReplayStore(path, key), model_key)
            cohort = seed_shadow(
                store, candidate, training, report, "demo", review_fraction=fraction, regression=regression
            )
            reports[name] = evaluate_verifier(
                cohort, store, "demo", HoldoutLedger(root / f"{name}-ledger.sqlite3", key)
            )
    return {
        "evidence_kind": "synthetic_verifier_shadow_ablation",
        "reports": reports,
        "gate_passed": reports["clean"]["gate_passed"]
        and not reports["missing_review"]["gate_passed"]
        and not reports["unsafe_shift"]["gate_passed"],
        "production_activation": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", default=os.getenv("EXECUTION_REPLAY_PATH", "data/execution-replay/replay.sqlite3")
    )
    parser.add_argument("--ledger", default="data/execution-replay/holdout-ledger.sqlite3")
    parser.add_argument("--tenant", default="")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("--name", required=True)
    register.add_argument("--candidate", required=True)
    register.add_argument("--training-cohort", required=True)
    register.add_argument("--training-report", required=True)
    register.add_argument("--compute-policy", required=True)
    register.add_argument("--trial-policy", required=True)
    register.add_argument("--incumbent-fingerprint", default="confidence-only")
    queue = commands.add_parser("queue")
    queue.add_argument("study_id")
    freeze = commands.add_parser("freeze")
    freeze.add_argument("study_id")
    freeze.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("cohort")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--require-gate", action="store_true")
    drill = commands.add_parser("drill")
    drill.add_argument("--output", default="data/evaluations/verifier-shadow/drill.json")
    drill.add_argument("--require-gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "output"):
            protected = {
                Path(path).resolve()
                for path in (
                    args.store,
                    args.ledger,
                    os.getenv("PROCESS_REWARD_MODEL_PATH", "data/evaluations/process-reward/model.json"),
                    os.getenv(
                        "VERIFIER_ENSEMBLE_PATH", "data/evaluations/verifier-uncertainty/ensemble.json"
                    ),
                )
            }
            protected |= {
                Path(str(path) + suffix)
                for path in tuple(protected)
                for suffix in ("-wal", "-shm", "-journal")
            }
            if args.command == "evaluate":
                protected.add(Path(args.cohort).resolve())
            if Path(args.output).resolve() in protected:
                raise ValueError("output aliases input, storage, or active verifier")
        if args.command == "drill":
            report = run_drill()
        else:
            if not args.tenant.strip():
                raise ValueError("verifier study requires tenant")
            store = VerifierShadowStore(
                ExecutionReplayStore(args.store, os.getenv("EXECUTION_REPLAY_KEY", "").encode()),
                os.getenv("PROCESS_SUPERVISION_MODEL_KEY", "").encode(),
            )
            if args.command == "register":

                def read(path):
                    return json.loads(Path(path).read_text(encoding="utf-8"))

                candidate = ReviewedProcessCandidate.model_validate(read(args.candidate))
                if candidate.simulation:
                    raise ValueError("synthetic candidates are for the drill, not runtime registration")
                study = store.register(
                    args.name,
                    args.tenant,
                    candidate,
                    ProcessCohort.model_validate(read(args.training_cohort)),
                    read(args.training_report),
                    ComputePolicy.model_validate(read(args.compute_policy)),
                    VerifierTrialPolicy.model_validate(read(args.trial_policy)),
                    args.incumbent_fingerprint,
                )
                print(json.dumps({"study_id": study.study_id, "registered_at": study.registered_at}))
                return 0
            if args.command == "queue":
                print(json.dumps(store.queue(args.study_id, args.tenant), indent=2))
                return 0
            if args.command == "freeze":
                cohort = store.freeze(args.study_id, args.tenant)
                write_once(args.output, cohort.model_dump(mode="json"))
                print(json.dumps({"families": len(cohort.members), "fingerprint": cohort.fingerprint}))
                return 0
            cohort = VerifierCohort.model_validate_json(Path(args.cohort).read_text(encoding="utf-8"))
            if cohort.study.candidate.simulation:
                raise ValueError("use the drill for synthetic evidence")
            report = evaluate_verifier(
                cohort, store, args.tenant, HoldoutLedger(args.ledger, store.replay.key)
            )
        write_once(args.output, report)
        print(json.dumps(report, indent=2))
        return 0 if not args.require_gate or report["gate_passed"] else 1
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"status": "held", "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
