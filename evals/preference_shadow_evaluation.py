"""Evaluate actual baseline-versus-shadow selections, not an offline proxy."""

from __future__ import annotations

import argparse
import json
import os
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
from agent.preference_ranking import PreferenceRanker, train_preferences
from agent.preference_shadow import PreferenceShadowStore, ShadowApproval, ShadowCohort
from agent.prospective_validation import HoldoutLedger
from evals.preference_evaluation import _paired_lower, seed_drill, write_once
from evals.replay_calibration import wilson_upper


def evaluate_shadow(
    cohort: ShadowCohort,
    shadow: PreferenceShadowStore,
    tenant: str,
    ranker: PreferenceRanker,
    policy: ComputePolicy,
    artifact_key: bytes,
    ledger: HoldoutLedger,
) -> tuple[dict, ShadowApproval | None]:
    key = shadow.replay.key
    shadow.validate_lineage(cohort, tenant)
    ranker.artifact.verify(artifact_key)
    if (
        ledger.key != key
        or ranker.tenant != tenant
        or ranker.artifact.fingerprint != cohort.study.artifact_fingerprint
        or digest(key, "shadow-compute-policy", policy.model_dump())
        != cohort.study.compute_policy_fingerprint
    ):
        raise ValueError("shadow evaluation identity, model, policy, or ledger mismatch")
    gate = {
        "version": "same-pool-shadow-gate-v1",
        "min_reviewed_families_per_scope": 20,
        "minimum_review_coverage": 0.8,
        "maximum_error_upper_95": 0.2,
        "minimum_disagreement_fraction": 0.2,
        "minimum_utility_lower_95": 0,
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 17,
        "risk_missingness": "unreviewed-outcomes-count-as-errors",
        "utility_missingness": "unknown-disagreement-minus-five; same-choice-zero",
        "maximum_known_unsafe_choices": 0,
        "maximum_evidence_age_seconds": 7 * 86400,
    }
    study_id = digest(key, "shadow-evaluation-v1", [cohort.fingerprint, gate])
    report = ledger.reserve_families(
        study_id, cohort.fingerprint, [member.task_family for member in cohort.members]
    )
    if report is None:
        outcomes = []
        for member in cohort.members:
            labels = (member.baseline_label, member.shadow_label)
            reviewed = all(label is not None and label.verdict != "ambiguous" for label in labels)
            baseline_correct = reviewed and labels[0].verdict == "correct" and not labels[0].unsafe
            shadow_correct = reviewed and labels[1].verdict == "correct" and not labels[1].unsafe
            row = member.comparison
            outcomes.append(
                {
                    "scope": f"{row.route}:{int(row.high_risk)}",
                    "reviewed": reviewed,
                    "baseline_correct": baseline_correct,
                    "shadow_correct": shadow_correct,
                    "unsafe": labels[1] is not None and labels[1].unsafe,
                    "changed": row.baseline_event != row.shadow_event,
                    "utility_delta": 5 * (int(shadow_correct) - int(baseline_correct)),
                }
            )
        reviewed_rows = [row for row in outcomes if row["reviewed"]]
        reviewed_count = len(reviewed_rows)
        coverage = reviewed_count / len(outcomes) if outcomes else 0
        changed = sum(row["changed"] for row in outcomes)
        unsafe = sum(row["unsafe"] for row in outcomes)
        lower = _paired_lower([row["utility_delta"] for row in reviewed_rows]) if reviewed_rows else None
        # Sensitivity bound: an unreviewed disagreement may be -5 utility;
        # identical choices share the same outcome, so their paired delta is zero.
        worst_lower = (
            _paired_lower(
                [
                    row["utility_delta"] if row["reviewed"] else (-5 if row["changed"] else 0)
                    for row in outcomes
                ]
            )
            if outcomes
            else None
        )
        failures, scopes = [], {}
        for scope in sorted({row["scope"] for row in outcomes}):
            rows = [row for row in outcomes if row["scope"] == scope]
            scored = [row for row in rows if row["reviewed"]]
            error_upper = min(1, wilson_upper(sum(not row["shadow_correct"] for row in rows), len(rows)))
            scopes[scope] = {
                "families": len(rows),
                "reviewed": len(scored),
                "review_coverage": len(scored) / len(rows),
                "error_upper_95": error_upper,
            }
            if len(scored) < 20 or len(scored) / len(rows) < 0.8 or error_upper > 0.2:
                failures.append("scope_support_review_or_risk_hold")
        if reviewed_count < 20 or coverage < 0.8:
            failures.append("insufficient_reviewed_coverage")
        if not outcomes or changed / len(outcomes) < 0.2:
            failures.append("no_material_selector_difference")
        if unsafe:
            failures.append("unsafe_shadow_selection")
        if lower is None or lower < 0 or worst_lower is None or worst_lower < 0:
            failures.append("paired_utility_regression")
        if (
            cohort.members
            and cohort.frozen_at - min(member.comparison.observed_at for member in cohort.members)
            >= 7 * 86400
        ):
            failures.append("stale_shadow_evidence")
        report = {
            "schema_version": "1.0",
            "evidence_kind": "synthetic_shadow_control"
            if cohort.study.simulation
            else "reviewed_runtime_shadow",
            "claim_scope": "paired selection quality on fresh baseline-released deliberation families; no causal traffic or factuality guarantee",
            "study_id": study_id,
            "registered_study_id": cohort.study.study_id,
            "cohort_fingerprint": cohort.fingerprint,
            "artifact_fingerprint": cohort.study.artifact_fingerprint,
            "compute_policy_fingerprint": cohort.study.compute_policy_fingerprint,
            "families": len(outcomes),
            "reviewed_families": reviewed_count,
            "review_coverage": coverage,
            "changed_choices": changed,
            "baseline_correct": sum(row["baseline_correct"] for row in reviewed_rows),
            "shadow_correct": sum(row["shadow_correct"] for row in reviewed_rows),
            "unsafe_shadow_choices": unsafe,
            "utility_lower_95": lower,
            "worst_case_utility_lower_95": worst_lower,
            "scopes": scopes,
            "policy": gate,
            "exclusions": cohort.exclusions,
            "gate_passed": not failures,
            "failure_reasons": sorted(set(failures)),
            "production_approval": not failures and not cohort.study.simulation,
        }
        report["fingerprint"] = digest(key, "shadow-evaluation-report", report)
        ledger.finish(study_id, report)
    elif report.get("fingerprint") != digest(
        key, "shadow-evaluation-report", {k: v for k, v in report.items() if k != "fingerprint"}
    ):
        raise ValueError("cached shadow report integrity failed")
    approval = None
    if report["production_approval"]:
        approval = ShadowApproval(
            tenant=ranker.artifact.tenant,
            artifact_fingerprint=ranker.artifact.fingerprint,
            compute_policy_fingerprint=digest(artifact_key, "approved-compute-policy", policy.model_dump()),
            study_id=cohort.study.study_id,
            cohort_fingerprint=cohort.fingerprint,
            report_fingerprint=report["fingerprint"],
            scopes=sorted(report["scopes"]),
            reviewed_families=report["reviewed_families"],
            review_coverage=report["review_coverage"],
            error_upper_95=max(scope["error_upper_95"] for scope in report["scopes"].values()),
            utility_lower_95=report["worst_case_utility_lower_95"],
            issued_at=cohort.frozen_at,
            expires_at=min(member.comparison.observed_at for member in cohort.members) + 7 * 86400,
        ).seal(artifact_key)
        approval.verify(artifact_key)
    return report, approval


def seed_shadow(
    shadow: PreferenceShadowStore,
    ranker: PreferenceRanker,
    tenant: str,
    *,
    families: int = 40,
    review_fraction: float = 1,
    regression: bool = False,
) -> tuple[ShadowCohort, ComputePolicy]:
    clock = [600.0]
    shadow.replay.clock = lambda: clock[0]
    policy = ComputePolicy()
    study = shadow.register("synthetic-shadow-study", tenant, ranker, policy)
    for index in range(families):
        clock[0] = 800.0 + index
        prefer_short = index % 2 == 0
        good_conf, bad_conf = (0.8, 0.9) if prefer_short else (0.98, 0.6)
        tokens = (100, 500) if prefer_short else (500, 100)
        candidates = [
            candidate_assessment(
                candidate_id=f"candidate-{j}",
                answer="PRIVATE synthetic content",
                confidence=confidence,
                grounded=True,
                conformal_decision="release",
                claim_keys=["evidence"],
                token_count=length,
                latency_ms=10,
                process_reward=0.9 if length == 500 else 0.7,
            )
            for j, confidence, length in ((1, good_conf, tokens[0]), (2, bad_conf, tokens[1]))
        ]
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
        baseline = select_candidate(plan, candidates, policy)
        alternate = select_candidate(plan, candidates, policy, preference_ranker=ranker)
        request = f"shadow-{index}"
        ids = shadow.replay.capture(
            plan,
            baseline,
            tenant=tenant,
            request_id=request,
            consent=True,
            origin="synthetic" if ranker.artifact.simulation else "runtime",
            task_family=f"shadow-family-{index}",
        )
        shadow.capture(study.study_id, tenant, request, plan, baseline, alternate, policy, consent=True)
        clock[0] += 0.5
        if index < int(families * review_fraction):
            for event, correct in zip(ids, (not regression, regression), strict=True):
                shadow.replay.review(
                    tenant,
                    event,
                    verdict="correct" if correct else "incorrect",
                    unsafe=regression and not correct,
                    reviewer="synthetic-reviewer",
                )
    clock[0] = 1000
    return shadow.freeze(study.study_id, tenant, embargo_seconds=50), policy


def run_drill() -> dict:
    replay_key, artifact_key = (
        b"synthetic-shadow-replay-key-not-secret",
        b"synthetic-shadow-artifact-key-not-secret",
    )
    reports = {}
    with tempfile.TemporaryDirectory(prefix="agentforge-shadow-") as root:
        for name, fraction, regression in (
            ("clean", 1, False),
            ("missing_review", 0.5, False),
            ("regression", 1, True),
        ):
            replay = ExecutionReplayStore(Path(root) / f"{name}.sqlite3", replay_key)
            training = seed_drill(replay, "demo")
            artifact = train_preferences(training, replay_key, artifact_key, "demo")
            ranker = PreferenceRanker(artifact, artifact_key, "demo", allow_simulation=True)
            shadow = PreferenceShadowStore(replay)
            cohort, policy = seed_shadow(
                shadow, ranker, "demo", review_fraction=fraction, regression=regression
            )
            report, approval = evaluate_shadow(
                cohort,
                shadow,
                "demo",
                ranker,
                policy,
                artifact_key,
                HoldoutLedger(Path(root) / f"{name}-ledger.sqlite3", replay_key),
            )
            reports[name] = report
            assert approval is None
    reports["gate_passed"] = (
        reports["clean"]["gate_passed"]
        and not reports["missing_review"]["gate_passed"]
        and not reports["regression"]["gate_passed"]
    )
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", default=os.getenv("EXECUTION_REPLAY_PATH", "data/execution-replay/replay.sqlite3")
    )
    parser.add_argument("--ledger", default="data/execution-replay/holdout-ledger.sqlite3")
    parser.add_argument("--tenant")
    parser.add_argument(
        "--artifact", default=os.getenv("PREFERENCE_RANKING_PATH", "data/evaluations/preferences/active.json")
    )
    parser.add_argument(
        "--compute-policy", help="JSON ComputePolicy, required for non-default runtime policy"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("--name", required=True)
    queue = commands.add_parser("queue")
    queue.add_argument("study_id")
    queue.add_argument("--audit-percent", type=int, default=10)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("study_id")
    freeze.add_argument("--embargo-seconds", type=float, default=3600)
    freeze.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("cohort")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--approval", required=True)
    evaluate.add_argument("--require-gate", action="store_true")
    drill = commands.add_parser("drill")
    drill.add_argument("--output", default="data/evaluations/preference-shadow/drill.json")
    drill.add_argument("--require-gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        protected = {
            Path(path).resolve()
            for path in (
                args.store,
                args.ledger,
                args.artifact,
                os.getenv("PREFERENCE_RANKING_PATH", "data/evaluations/preferences/active.json"),
                os.getenv(
                    "PREFERENCE_SHADOW_APPROVAL_PATH", "data/evaluations/preference-shadow/approval.json"
                ),
            )
        }
        protected |= {
            Path(str(path) + suffix) for path in tuple(protected) for suffix in ("-wal", "-shm", "-journal")
        }
        if args.compute_policy:
            protected.add(Path(args.compute_policy).resolve())
        if args.command == "evaluate":
            protected.add(Path(args.cohort).resolve())
        outputs = [Path(args.output).resolve()] if hasattr(args, "output") else []
        if args.command == "evaluate":
            outputs.append(Path(args.approval).resolve())
        if len(set(outputs)) != len(outputs) or any(path in protected for path in outputs):
            raise ValueError("output aliases a protected input, active approval, or SQLite store")
        if args.command == "drill":
            report = run_drill()
            write_once(args.output, report)
        else:
            if not args.tenant:
                raise ValueError("tenant is required")
            replay = ExecutionReplayStore(args.store, os.getenv("EXECUTION_REPLAY_KEY", "").encode())
            shadow = PreferenceShadowStore(replay)
            if args.command == "queue":
                print(
                    json.dumps(
                        shadow.review_queue(args.study_id, args.tenant, audit_percent=args.audit_percent),
                        indent=2,
                    )
                )
                return 0
            artifact_key = os.getenv("PREFERENCE_RANKING_KEY", "").encode()
            ranker = PreferenceRanker.load(args.artifact, artifact_key, args.tenant)
            policy = (
                ComputePolicy.model_validate_json(Path(args.compute_policy).read_text(encoding="utf-8"))
                if args.compute_policy
                else ComputePolicy()
            )
            if args.command == "register":
                study = shadow.register(args.name, args.tenant, ranker, policy)
                print(json.dumps({"study_id": study.study_id, "registered_at": study.registered_at}))
                return 0
            if args.command == "freeze":
                cohort = shadow.freeze(args.study_id, args.tenant, embargo_seconds=args.embargo_seconds)
                write_once(args.output, cohort.model_dump(mode="json"))
                print(json.dumps({"families": len(cohort.members), "exclusions": cohort.exclusions}))
                return 0
            cohort = ShadowCohort.model_validate_json(Path(args.cohort).read_text(encoding="utf-8"))
            report, approval = evaluate_shadow(
                cohort,
                shadow,
                args.tenant,
                ranker,
                policy,
                artifact_key,
                HoldoutLedger(args.ledger, replay.key),
            )
            write_once(args.output, report)
            if approval is not None:
                write_once(args.approval, approval.model_dump(mode="json"))
        print(json.dumps(report, indent=2))
        return 0 if not args.require_gate or report["gate_passed"] else 1
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "held", "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
