"""Fresh-family evaluation of independently reviewed step uncertainty."""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import statistics
import tempfile
from pathlib import Path

from agent.execution_replay import ExecutionReplayStore, digest
from agent.process_reward import ProcessRewardScorer, train_process_reward_model
from agent.process_supervision import ProcessCohort, ProcessSupervisionStore
from agent.prospective_validation import HoldoutLedger
from agent.reviewed_verifier_uncertainty import (
    ReviewedStepEnsemble,
    ReviewedStepScorer,
    UncertaintyPolicy,
    _temperature,
    fit_ensemble,
)
from evals.preference_evaluation import _paired_lower, write_once
from evals.process_supervision_evaluation import _ece, seed_process


def family_mean(rows, value):
    groups = {}
    for row in rows:
        result = value(row)
        if result is not None:
            groups.setdefault(row["family"], []).append(result)
    return {family: statistics.fmean(values) for family, values in groups.items()}


def upper_95(values):
    """Family-resampled macro error bound; NOT independent-step Wilson risk."""
    if not values:
        return None
    rng = random.Random(17)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(1000))
    return means[949]


def risk_curve(rows, threshold, maximum_spread, *, guard=True):
    accepted = [
        row
        for row in rows
        if (
            (not guard or row["supported"])
            and row["spread"] <= maximum_spread
            and row["confidence"] >= threshold
        )
    ]
    known = [row for row in accepted if row["target"] is not None]
    worst_errors = family_mean(
        accepted, lambda row: 1.0 if row["target"] is None else float(row["prediction"] != row["target"])
    )
    reviewed_errors = family_mean(known, lambda row: float(row["prediction"] != row["target"]))
    return {
        "threshold": threshold,
        "support_guard": guard,
        "steps": len(rows),
        "accepted_steps": len(accepted),
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "accepted_families": len(worst_errors),
        "reviewed_accepted_families": len(reviewed_errors),
        "review_coverage": len(known) / len(accepted) if accepted else 0.0,
        "reviewed_macro_error": statistics.fmean(reviewed_errors.values()) if reviewed_errors else None,
        "worst_case_macro_error": statistics.fmean(worst_errors.values()) if worst_errors else None,
        "worst_case_macro_error_upper_95": upper_95(list(worst_errors.values())),
    }


def evaluate_uncertainty(cohort, store, tenant, model_key, ledger, policy=None):
    policy = UncertaintyPolicy.model_validate((policy or UncertaintyPolicy()).model_dump())
    key = store.replay.key
    if len(model_key) < 32 or model_key == key or ledger.key != key:
        raise ValueError("reviewed uncertainty requires independent model key and shared replay ledger key")
    store.validate_lineage(tenant, cohort)
    model_tenant = digest(model_key, "reviewed-step-tenant", tenant)
    study = digest(key, "reviewed-step-study", [cohort.fingerprint, model_tenant, policy.model_dump()])
    families = sorted({m.snapshot.family.task_family for m in cohort.members if m.split == "test"})
    # Reserve before any held-out scores, curves, or label summaries are inspected.
    cached = ledger.reserve_families(study, cohort.fingerprint, families)
    if cached is not None:
        if cached.get("fingerprint") != digest(
            key, "reviewed-step-cache", {k: v for k, v in cached.items() if k != "fingerprint"}
        ):
            raise ValueError("reviewed uncertainty cache integrity failed")
        candidate = ReviewedStepEnsemble.model_validate(cached["candidate"])
        candidate.verify(model_key)
        report = cached["report"]
        if (
            candidate.tenant != model_tenant
            or candidate.cohort_fingerprint != cohort.fingerprint
            or candidate.policy != policy
            or candidate.simulation != cohort.simulation
            or candidate.report_fingerprint != report.get("fingerprint")
            or report.get("fingerprint")
            != digest(key, "reviewed-step-report", {k: v for k, v in report.items() if k != "fingerprint"})
        ):
            raise ValueError("reviewed uncertainty cache provenance mismatch")
        return report, candidate
    traces = cohort.traces()
    fit = fit_ensemble(traces, policy, model_key)
    # Provisional signature is internal only. Final candidate binds the completed report.
    candidate = ReviewedStepEnsemble(
        tenant=model_tenant,
        cohort_fingerprint=cohort.fingerprint,
        report_fingerprint="0" * 64,
        simulation=cohort.simulation,
        policy=policy,
        **fit,
    ).seal(model_key)
    scorer = ReviewedStepScorer(candidate, model_key)
    train = [t for t in traces if t.split == "train"]
    single = ProcessRewardScorer(
        train_process_reward_model(train, epochs=policy.epochs, explicit_steps_only=True)
    )
    validation = [
        (t.group_id, step.step_label, p)
        for t in traces
        if t.split == "validation"
        for step, p in zip(t.steps, single.step_probabilities(t.steps, t.high_risk), strict=True)
        if step.step_label is not None
    ]
    single_temperature = min(
        (i / 20 for i in range(5, 81)),
        key=lambda temp: (
            statistics.fmean(
                family_mean(
                    [{"family": f, "target": y, "probability": p} for f, y, p in validation],
                    lambda row: (_temperature(row["probability"], temp) - row["target"]) ** 2,
                ).values()
            ),
            abs(temp - 1),
        ),
    )
    targets = [s.step_label for t in train for s in t.steps if s.step_label is not None]
    constant = statistics.fmean(targets)
    sources = {m.snapshot.source.event_id: m.snapshot.source for m in cohort.members}
    rows = []
    for trace in traces:
        if trace.split != "test":
            continue
        estimates = scorer.estimates(trace.steps, trace.high_risk)
        single_predictions = single.step_probabilities(trace.steps, trace.high_risk)
        source = sources[trace.trace_id]
        for step, estimate, single_prediction in zip(trace.steps, estimates, single_predictions, strict=True):
            rows.append(
                {
                    "family": trace.group_id,
                    "scope": f"{source.route}:{int(source.high_risk)}",
                    "kind": step.kind,
                    "target": step.step_label,
                    "probability": estimate.mean,
                    "spread": estimate.spread,
                    "confidence": estimate.conservative_confidence,
                    "supported": estimate.feature_supported,
                    "prediction": float(estimate.predicted_correct),
                    "single": _temperature(single_prediction, single_temperature),
                }
            )

    def brier(name, worst=False):
        values = family_mean(
            rows,
            lambda row: (
                (max(row[name] ** 2, (1 - row[name]) ** 2) if worst else None)
                if row["target"] is None
                else (row[name] - row["target"]) ** 2
            ),
        )
        return statistics.fmean(values.values()) if values else None

    for row in rows:
        row["constant"] = constant
    primary = risk_curve(rows, policy.primary_confidence, policy.maximum_spread)
    scopes = {
        name: risk_curve(
            [r for r in rows if r["scope"] == name], policy.primary_confidence, policy.maximum_spread
        )
        for name in sorted({r["scope"] for r in rows})
    }
    kinds = {
        name: risk_curve(
            [r for r in rows if r["kind"] == name], policy.primary_confidence, policy.maximum_spread
        )
        for name in sorted({r["kind"] for r in rows})
    }
    support, failures = {}, []
    for split, minimum in (("train", 20), ("validation", 10), ("test", 20)):
        fold = [t for t in traces if t.split == split]
        labels = [s.step_label for t in fold for s in t.steps if s.step_label is not None]
        reviewed = {t.group_id for t in fold if any(s.step_label is not None for s in t.steps)}
        support[split] = {
            "families": len({t.group_id for t in fold}),
            "reviewed_families": len(reviewed),
            "positive_steps": labels.count(1),
            "negative_steps": labels.count(0),
        }
        if len(reviewed) < minimum or min(labels.count(1), labels.count(0)) < 5:
            failures.append("insufficient_split_family_or_class_support")
    review_coverage = sum(r["target"] is not None for r in rows) / len(rows) if rows else 0.0
    ece = _ece([(r["family"], r["target"], r["probability"]) for r in rows])
    worst_brier = brier("probability", True)
    worst_improvements = family_mean(
        rows,
        lambda row: (
            min((constant - y) ** 2 - (row["probability"] - y) ** 2 for y in (0, 1))
            if row["target"] is None
            else (constant - row["target"]) ** 2 - (row["probability"] - row["target"]) ** 2
        ),
    )
    improvement_lower = _paired_lower(list(worst_improvements.values())) if worst_improvements else None
    if review_coverage < 0.8:
        failures.append("insufficient_test_review_coverage")
    if (
        worst_brier is None
        or worst_brier > 0.2
        or ece is None
        or ece > 0.2
        or improvement_lower is None
        or improvement_lower < 0
    ):
        failures.append("heldout_calibration_quality_hold")
    for name, curve, minimum in [("overall", primary, 20)] + [
        (name, value, 10) for name, value in scopes.items()
    ]:
        if (
            curve["reviewed_accepted_families"] < minimum
            or curve["coverage"] < 0.5
            or curve["review_coverage"] < 0.8
            or curve["worst_case_macro_error_upper_95"] is None
            or curve["worst_case_macro_error_upper_95"] > 0.2
        ):
            failures.append(f"primary_selective_risk_or_coverage_hold:{name}")
    report = {
        "evidence_kind": "synthetic_reviewed_step_controls"
        if cohort.simulation
        else "reviewed_runtime_step_uncertainty",
        "cohort_fingerprint": cohort.fingerprint,
        "policy": policy.model_dump(),
        "claim_scope": "workflow-step predictions only; no final-answer lift, conformal guarantee or production OOD guarantee",
        "gate_rules": {
            "minimum_reviewed_families": {"train": 20, "validation": 10, "test": 20},
            "minimum_labels_per_class": 5,
            "minimum_review_coverage": 0.8,
            "minimum_primary_coverage": 0.5,
            "minimum_scope_reviewed_accepted_families": 10,
            "maximum_worst_brier_and_ece": 0.2,
            "minimum_worst_case_paired_brier_improvement_lower_95": 0.0,
            "maximum_primary_family_macro_error_upper_95": 0.2,
            "bootstrap_replicates": 1000,
            "bootstrap_seed": 17,
        },
        "split_support": support,
        "test_review_coverage": review_coverage,
        "family_macro_brier": brier("probability"),
        "worst_case_family_macro_brier": worst_brier,
        "step_weighted_ece": ece,
        "uncertainty_limitations": "ensemble spread and conservative confidence are heuristics; the family bootstrap percentile is descriptive, can be zero with no observed errors, and is not a finite-sample population risk guarantee",
        "single_explicit_model_brier": brier("single"),
        "worst_case_paired_constant_brier_improvement_lower_95": improvement_lower,
        "train_constant_brier": brier("constant"),
        "calibration_temperature": fit["calibration_temperature"],
        "primary": primary,
        "route_risk_slices": scopes,
        "step_kind_slices": kinds,
        "risk_coverage_curves": [
            risk_curve(rows, threshold, policy.maximum_spread) for threshold in policy.curve_confidences
        ],
        "unguarded_primary_ablation": risk_curve(
            rows, policy.primary_confidence, policy.maximum_spread, guard=False
        ),
        "unsupported_step_fraction": sum(not r["supported"] for r in rows) / len(rows) if rows else 0.0,
        "explicit_training_steps": fit["explicit_training_steps"],
        "gate_passed": not failures,
        "candidate_ready_for_review": not failures and not cohort.simulation,
        "production_activation": False,
        "failure_reasons": sorted(set(failures)),
        "exclusions": cohort.exclusions,
    }
    report["fingerprint"] = digest(key, "reviewed-step-report", report)
    candidate = candidate.model_copy(update={"report_fingerprint": report["fingerprint"]}).seal(model_key)
    cached = {"report": report, "candidate": candidate.model_dump(mode="json")}
    cached["fingerprint"] = digest(key, "reviewed-step-cache", cached)
    ledger.finish(study, cached)
    return report, candidate


def run_drill():
    reports = {}
    with tempfile.TemporaryDirectory(prefix="agentforge-reviewed-step-") as root:
        for name, fraction, shifted in (
            ("clean", 1, False),
            ("missing_review", 0.5, False),
            ("shifted_labels", 1, True),
        ):
            key, model_key = b"synthetic-reviewed-step-replay-key", b"synthetic-reviewed-step-model-key!"
            store = ProcessSupervisionStore(ExecutionReplayStore(Path(root) / f"{name}.sqlite3", key))
            cohort = seed_process(store, "demo", review_fraction=fraction, shifted=shifted)
            reports[name], candidate = evaluate_uncertainty(
                cohort, store, "demo", model_key, HoldoutLedger(Path(root) / f"{name}-ledger.sqlite3", key)
            )
        # Low ensemble spread can be confidently wrong outside observed features.
        # This probe changes no labels, scores no holdout anew and is descriptive.
        steps = cohort.traces()[0].steps
        scorer = ReviewedStepScorer(candidate, model_key)
        estimates = scorer.estimates(steps, high_risk=True)
        unseen = {
            "unsupported_steps": sum(not e.feature_supported for e in estimates),
            "accepted_steps": sum(e.accepted for e in estimates),
            "unguarded_accepted_steps": sum(
                e.spread <= candidate.policy.maximum_spread
                and e.conservative_confidence >= candidate.policy.primary_confidence
                for e in estimates
            ),
            "probe": "unseen high-risk workflow: identical features otherwise",
            "steps": len(estimates),
        }
    return {
        "evidence_kind": "synthetic_reviewed_step_uncertainty_drill",
        "reports": reports,
        "unseen_feature_probe": unseen,
        "gate_passed": reports["clean"]["gate_passed"]
        and not reports["missing_review"]["gate_passed"]
        and not reports["shifted_labels"]["gate_passed"]
        and unseen["accepted_steps"] == 0
        and unseen["unguarded_accepted_steps"] > 0,
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
    drill = commands.add_parser("drill")
    drill.add_argument("--output", default="data/evaluations/reviewed-verifier-uncertainty/drill.json")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("cohort")
    evaluate.add_argument("--policy", help="Freeze this typed policy before exposing fresh test families")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--candidate", required=True)
    for command in (drill, evaluate):
        command.add_argument("--require-gate", action="store_true")
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
        outputs = [Path(args.output).resolve()]
        if args.command == "evaluate":
            protected.add(Path(args.cohort).resolve())
            if args.policy:
                protected.add(Path(args.policy).resolve())
            outputs.append(Path(args.candidate).resolve())
        if len(set(outputs)) != len(outputs) or any(path in protected for path in outputs):
            raise ValueError("output aliases input, storage or active verifier")
        if args.command == "drill":
            report = run_drill()
        else:
            if not args.tenant.strip():
                raise ValueError("reviewed step evaluation requires tenant")
            cohort = ProcessCohort.model_validate_json(Path(args.cohort).read_text(encoding="utf-8"))
            if cohort.simulation:
                raise ValueError("use drill for synthetic cohorts; production export refused")
            policy = (
                UncertaintyPolicy.model_validate_json(Path(args.policy).read_text(encoding="utf-8"))
                if args.policy
                else UncertaintyPolicy()
            )
            replay = ExecutionReplayStore(args.store, os.getenv("EXECUTION_REPLAY_KEY", "").encode())
            report, candidate = evaluate_uncertainty(
                cohort,
                ProcessSupervisionStore(replay),
                args.tenant,
                os.getenv("PROCESS_SUPERVISION_MODEL_KEY", "").encode(),
                HoldoutLedger(args.ledger, replay.key),
                policy,
            )
            if report["candidate_ready_for_review"]:
                write_once(args.candidate, candidate.model_dump(mode="json"))
        write_once(args.output, report)
        print(json.dumps(report, indent=2))
        return 0 if not args.require_gate or report["gate_passed"] else 1
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"status": "held", "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
