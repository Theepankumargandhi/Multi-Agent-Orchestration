"""Forward-time model validation, drift diagnostics and paired uncertainty."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent.adaptive_compute import ComputeSignals, candidate_assessment, plan_compute, select_candidate
from agent.execution_replay import ExecutionReplayStore, digest
from agent.prospective_validation import (
    FrozenCohort,
    HoldoutLedger,
    freeze_cohort,
    save_cohort,
    validate_live_lineage,
)
from agent.uncertainty import (
    detect_confidence_drift,
    fit_calibrator,
    load_calibrator,
    save_calibrator,
    selective_decision,
    verify_calibrator,
)
from evals.replay_calibration import recalibrate


class ValidationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    minimum_groups: int = Field(default=20, ge=20)
    minimum_coverage: float = Field(default=0.2, gt=0, le=1)
    target_error_rate: float = Field(default=0.2, gt=0, lt=1)
    confidence_js_limit: float = Field(default=0.15, gt=0, le=1)
    route_tv_limit: float = Field(default=0.25, ge=0, le=1)
    bootstrap_replicates: int = Field(default=400, ge=200, le=10000)
    bootstrap_seed: int = Field(default=17, ge=0, lt=2**32)
    noninferiority_margin: float = Field(default=0.02, ge=0, le=0.2)


class _CohortDataset:
    def __init__(self, cohort, key):
        self.cohort, self.key = cohort, key

    def dataset(self, tenant, *, allow_synthetic=False):
        self.cohort.verify(self.key)
        if self.cohort.tenant != digest(self.key, "tenant", tenant):
            raise ValueError("cohort tenant mismatch")
        if self.cohort.simulation and not allow_synthetic:
            raise ValueError("synthetic cohort requires explicit simulation mode")
        members = self.cohort.members
        manifest = {
            "split_policy": "chronological-embargo-family-separated-v1",
            "cohort_fingerprint": self.cohort.fingerprint,
            "simulation": self.cohort.simulation,
            "request_groups": len(members) + sum(self.cohort.excluded_groups.values()),
            "exported_groups": len(members),
            "excluded_groups": self.cohort.excluded_groups,
            "lineage": [
                {
                    "group": member.request_group,
                    "task_family": member.task_family,
                    "event_id": member.example.id,
                    "event": member.observation_fingerprint,
                    "label": member.label_fingerprint,
                    "unsafe": member.unsafe,
                    "split": member.example.split,
                }
                for member in members
            ],
        }
        manifest["fingerprint"] = digest(self.key, "prospective-dataset", manifest)
        return [member.example for member in members], manifest


def paired_bootstrap(
    candidate: list[tuple[bool, bool]], incumbent: list[tuple[bool, bool]], policy: ValidationPolicy
) -> dict:
    """Paired resampling: one row is one pre-selected, distinct task family.

    Utility is +1 for a correct release, -4 for an incorrect release, and zero
    for abstention. This is an explicit evaluation score, not business value.
    """
    if len(candidate) != len(incumbent) or not candidate:
        raise ValueError("paired outcomes require equal, nonempty family sets")
    coverage = [int(left[0]) - int(right[0]) for left, right in zip(candidate, incumbent, strict=True)]

    def utility(outcome):
        release, correct = outcome
        return (1 if correct else -4) if release else 0

    scores = [utility(left) - utility(right) for left, right in zip(candidate, incumbent, strict=True)]
    rng = random.Random(policy.bootstrap_seed)
    coverage_draws, utility_draws = [], []
    for _ in range(policy.bootstrap_replicates):
        indices = [rng.randrange(len(scores)) for _ in scores]
        coverage_draws.append(sum(coverage[index] for index in indices) / len(indices))
        utility_draws.append(sum(scores[index] for index in indices) / len(indices))

    def interval(values, differences):
        ordered = sorted(values)
        return {
            "delta": sum(differences) / len(differences),
            "lower_95": ordered[int(0.025 * (len(ordered) - 1))],
            "upper_95": ordered[int(0.975 * (len(ordered) - 1))],
        }

    return {
        "unit": "distinct-preselected-task-family",
        "families": len(scores),
        "replicates": policy.bootstrap_replicates,
        "seed": policy.bootstrap_seed,
        "utility_definition": "+1 correct release; -4 incorrect release; 0 abstention",
        "coverage": interval(coverage_draws, coverage),
        "utility": interval(utility_draws, scores),
    }


def _outcomes(members, artifact, key):
    if not verify_calibrator(artifact, key):
        raise ValueError("calibrator integrity verification failed")
    result = []
    for member in members:
        example = member.example
        receipt = selective_decision(
            route=example.route, confidence=example.confidence, artifact=artifact, integrity_key=key
        )
        result.append((receipt.decision == "release" and not receipt.out_of_distribution, example.correct))
    return result


def evaluate_cohort(
    cohort: FrozenCohort,
    store: ExecutionReplayStore,
    tenant: str,
    ledger: HoldoutLedger,
    *,
    policy: ValidationPolicy | None = None,
    calibrator_key: bytes | None = None,
    incumbent=None,
):
    policy = policy or ValidationPolicy()
    if store.key != ledger.key:
        raise ValueError("cohort and holdout ledger keys must match")
    validate_live_lineage(cohort, store, tenant)
    calibration = [member.example for member in cohort.members if member.example.split == "calibration"]
    # Fit identity from calibration data ONLY, before reserving/reading test outcomes.
    fitted = (
        fit_calibrator(calibration, target_error_rate=policy.target_error_rate, integrity_key=calibrator_key)
        if len(calibration) >= 10
        else None
    )
    if incumbent is not None and not verify_calibrator(incumbent, calibrator_key):
        raise ValueError("incumbent integrity verification failed")
    study_id = digest(
        store.key,
        "prospective-study",
        [
            cohort.fingerprint,
            fitted.artifact_fingerprint if fitted else None,
            incumbent.artifact_fingerprint if incumbent else None,
            policy.model_dump(),
        ],
    )
    cached = ledger.reserve(study_id, cohort)
    if cached is not None:
        expected = digest(
            store.key,
            "prospective-report",
            {name: value for name, value in cached.items() if name != "fingerprint"},
        )
        if (
            not isinstance(cached.get("fingerprint"), str)
            or not hmac.compare_digest(expected, cached["fingerprint"])
            or cached.get("study_id") != study_id
            or cached.get("cohort_fingerprint") != cohort.fingerprint
        ):
            raise ValueError("cached prospective report integrity verification failed")
        return fitted if cached["artifact_fingerprint"] else None, cached
    artifact, report = recalibrate(
        _CohortDataset(cohort, store.key),
        tenant,
        allow_synthetic=cohort.simulation,
        target_error_rate=policy.target_error_rate,
        minimum_groups=policy.minimum_groups,
        minimum_coverage=policy.minimum_coverage,
        calibrator_key=calibrator_key,
        incumbent=incumbent,
    )
    test = [member for member in cohort.members if member.example.split == "test"]
    drift, paired = None, None
    if artifact is not None:
        confidence = detect_confidence_drift(
            [member.example.confidence for member in test],
            artifact,
            threshold=policy.confidence_js_limit,
            integrity_key=calibrator_key,
        )
        train_routes = Counter(example.route for example in calibration)
        test_routes = Counter(member.example.route for member in test)
        route_tv = (
            sum(
                abs(train_routes[route] / len(calibration) - test_routes[route] / len(test))
                for route in train_routes.keys() | test_routes.keys()
            )
            / 2
        )
        drift = {"confidence": confidence.model_dump(), "route_total_variation": route_tv}
        if confidence.drift_detected:
            report["reasons"].append("confidence_distribution_shift")
        if route_tv > policy.route_tv_limit:
            report["reasons"].append("route_distribution_shift")
        if incumbent is not None:
            paired = paired_bootstrap(
                _outcomes(test, artifact, calibrator_key), _outcomes(test, incumbent, calibrator_key), policy
            )
            if any(
                paired[name]["lower_95"] < -policy.noninferiority_margin for name in ["coverage", "utility"]
            ):
                report["reasons"].append("paired_noninferiority_not_established")
    report.update(
        {
            "evaluation_kind": "frozen-forward-time-family-separated",
            "cohort_fingerprint": cohort.fingerprint,
            "study_id": study_id,
            "holdout_reserved_before_scoring": True,
            "drift": drift,
            "paired": paired,
            "validation_policy": policy.model_dump(),
            "gate_passed": not report["reasons"],
            "production_candidate_eligible": not report["reasons"] and not cohort.simulation,
            "automatic_deployment": False,
            "validated_scope": {"tenant": cohort.tenant, "routes": sorted(report["routes"])},
        }
    )
    report.pop("fingerprint")
    report["fingerprint"] = digest(store.key, "prospective-report", report)
    ledger.finish(study_id, report)
    return artifact, report


def _fixture(directory: Path, key: bytes, *, shifted=False):
    clock = [1000.0]
    store = ExecutionReplayStore(directory / "replay.sqlite3", key, clock=lambda: clock[0])
    plan = plan_compute(
        ComputeSignals(
            route="rag",
            grounding_action="repair",
            grounding_confidence=0.4,
            uncertainty_decision="abstain",
            evidence_count=2,
        )
    )
    for split in ["calibration", "test"]:
        for index in range(80):
            clock[0] = (1000 if split == "calibration" else 3000) + index
            correct = index % 4 != 0
            confidence = 0.3 if shifted and split == "test" and not correct else 0.9 if correct else 0.2
            candidate = candidate_assessment(
                candidate_id="candidate-1",
                answer="synthetic fixture",
                confidence=confidence,
                grounded=True,
                conformal_decision="not_evaluated",
                claim_keys=["fixture-evidence"],
                token_count=40,
                latency_ms=5,
            )
            ids = store.capture(
                plan,
                select_candidate(plan, [candidate]),
                tenant="fixture-tenant",
                request_id=f"{split}-{index}",
                task_family=f"{split}-family-{index}",
                consent=True,
                origin="synthetic",
            )
            clock[0] += 1
            store.review(
                "fixture-tenant",
                ids[0],
                verdict="correct" if correct else "incorrect",
                unsafe=False,
                reviewer="fixture-reviewer",
            )
    cohort = freeze_cohort(
        store,
        "fixture-tenant",
        calibration_cutoff=2000,
        embargo_seconds=500,
        frozen_at=4000,
        allow_synthetic=True,
    )
    return store, cohort


def run_drill() -> dict:
    key = b"synthetic-prospective-control-key-not-production"
    with tempfile.TemporaryDirectory(prefix="agentforge-prospective-") as directory:
        root = Path(directory)
        store, cohort = _fixture(root / "clean", key)
        incumbent = fit_calibrator(
            [member.example for member in cohort.members if member.example.split == "calibration"]
        )
        ledger = HoldoutLedger(root / "ledger.sqlite3", key)
        _, clean = evaluate_cohort(cohort, store, "fixture-tenant", ledger, incumbent=incumbent)
        _, retry = evaluate_cohort(cohort, store, "fixture-tenant", ledger, incumbent=incumbent)
        try:
            evaluate_cohort(
                cohort,
                store,
                "fixture-tenant",
                ledger,
                policy=ValidationPolicy(bootstrap_seed=99),
                incumbent=incumbent,
            )
            reuse_rejected = False
        except ValueError:
            reuse_rejected = True
        shifted_store, shifted_cohort = _fixture(root / "shifted", key, shifted=True)
        _, shifted = evaluate_cohort(
            shifted_cohort,
            shifted_store,
            "fixture-tenant",
            HoldoutLedger(root / "shifted-ledger.sqlite3", key),
        )
        return {
            "simulation": True,
            "clean": clean,
            "shifted": shifted,
            "exact_retry_cached": retry == clean,
            "changed_study_reuse_rejected": reuse_rejected,
            "passed": clean["gate_passed"]
            and not clean["production_candidate_eligible"]
            and retry == clean
            and reuse_rejected
            and "confidence_distribution_shift" in shifted["reasons"]
            and not shifted["gate_passed"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/execution-replay/replay.sqlite3")
    parser.add_argument("--ledger", default="data/execution-replay/holdout-ledger.sqlite3")
    parser.add_argument("--tenant")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument(
        "--cutoff", type=float, required=True, help="Calibration label-availability cutoff, Unix UTC seconds"
    )
    freeze.add_argument("--embargo-seconds", type=float, default=3600)
    freeze.add_argument("--snapshot-time", type=float, default=None)
    freeze.add_argument("--output", default="data/execution-replay/cohort.json")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("cohort")
    evaluate.add_argument("--incumbent")
    evaluate.add_argument("--artifact", default="data/execution-replay/prospective-candidate.json")
    evaluate.add_argument("--output", default="data/evaluations/prospective/report.json")
    evaluate.add_argument("--require-gate", action="store_true")
    drill = commands.add_parser("drill")
    drill.add_argument("--output", default="data/evaluations/prospective/drill.json")
    drill.add_argument("--require-gate", action="store_true")
    args = parser.parse_args()
    protected = {
        Path(args.store).resolve(),
        Path(args.ledger).resolve(),
        Path(
            os.getenv("UNCERTAINTY_CALIBRATOR_PATH", "data/evaluations/uncertainty/calibrator.json")
        ).resolve(),
    }
    for database in [args.store, args.ledger]:
        protected.update(Path(database + suffix).resolve() for suffix in ["-wal", "-shm", "-journal"])
    if args.command == "evaluate":
        protected.add(Path(args.cohort).resolve())
        if args.incumbent:
            protected.add(Path(args.incumbent).resolve())
        if (
            Path(args.artifact).resolve() in protected
            or Path(args.artifact).resolve() == Path(args.output).resolve()
        ):
            parser.error("candidate path must not overwrite protected inputs or the report")
    if Path(args.output).resolve() in protected:
        parser.error("output must not overwrite a protected input or active artifact")
    target = Path(args.output)
    if args.command == "drill":
        report = run_drill()
        passed = report["passed"]
    else:
        if not args.tenant:
            parser.error("--tenant is required")
        key = os.getenv("EXECUTION_REPLAY_KEY", "").encode()
        store = ExecutionReplayStore(args.store, key)
        if args.command == "freeze":
            cohort = freeze_cohort(
                store,
                args.tenant,
                calibration_cutoff=args.cutoff,
                embargo_seconds=args.embargo_seconds,
                frozen_at=args.snapshot_time if args.snapshot_time is not None else time.time(),
            )
            try:
                save_cohort(target, cohort, key)
            except ValueError as exc:
                parser.error(str(exc))
            print(
                json.dumps(
                    {"cohort": str(target), "members": len(cohort.members), "fingerprint": cohort.fingerprint}
                )
            )
            return
        cohort = FrozenCohort.model_validate_json(Path(args.cohort).read_text(encoding="utf-8"))
        calibrator_key = os.getenv("UNCERTAINTY_INTEGRITY_KEY", "").encode() or None
        incumbent = load_calibrator(args.incumbent, calibrator_key) if args.incumbent else None
        artifact, report = evaluate_cohort(
            cohort,
            store,
            args.tenant,
            HoldoutLedger(args.ledger, key),
            calibrator_key=calibrator_key,
            incumbent=incumbent,
        )
        passed = report["production_candidate_eligible"]
        if passed:
            save_calibrator(args.artifact, artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "simulation": report["simulation"], "report": str(target)}))
    if args.require_gate and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
