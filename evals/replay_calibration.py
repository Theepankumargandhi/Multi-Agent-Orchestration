"""Request-grouped correctness recalibration with an explicit held-out gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

from agent.adaptive_compute import ComputeSignals, candidate_assessment, plan_compute, select_candidate
from agent.execution_replay import ExecutionReplayStore, digest
from agent.uncertainty import (
    ConformalCalibrator,
    fit_calibrator,
    load_calibrator,
    save_calibrator,
    selective_decision,
    verify_calibrator,
)


def wilson_upper(errors: int, total: int) -> float:
    """Two-sided 95% Wilson interval's upper endpoint; no releases means unknown."""
    if not total:
        return 1.0
    z = 1.959963984540054
    rate = errors / total
    return (
        rate + z * z / (2 * total) + z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    ) / (1 + z * z / total)


def _metrics(examples, artifact, key, unsafe_ids):
    if not verify_calibrator(artifact, key):
        raise ValueError("calibrator integrity verification failed")
    released = []
    for item in examples:
        decision = selective_decision(
            route=item.route, confidence=item.confidence, artifact=artifact, integrity_key=key
        )
        if decision.decision == "release" and not decision.out_of_distribution:
            released.append(item)
    errors = sum(not item.correct for item in released)
    return {
        "request_groups": len(examples),
        "released": len(released),
        "errors": errors,
        "unsafe_releases": sum(item.id in unsafe_ids for item in released),
        "coverage": len(released) / max(1, len(examples)),
        "empirical_error": errors / len(released) if released else None,
        "error_wilson_upper_95": wilson_upper(errors, len(released)),
    }


def recalibrate(
    store: ExecutionReplayStore,
    tenant: str,
    *,
    allow_synthetic: bool = False,
    target_error_rate: float = 0.2,
    minimum_groups: int = 20,
    minimum_coverage: float = 0.2,
    calibrator_key: bytes | None = None,
    incumbent: ConformalCalibrator | None = None,
) -> tuple[ConformalCalibrator | None, dict]:
    if minimum_groups < 20 or not 0 < minimum_coverage <= 1 or not 0 < target_error_rate < 1:
        raise ValueError("gate requires >=20 groups per split and valid coverage/error limits")
    examples, manifest = store.dataset(tenant, allow_synthetic=allow_synthetic)
    # Use the same verified snapshot as fitting, not a second concurrent DB read.
    unsafe_ids = {row["event_id"] for row in manifest["lineage"] if row["unsafe"]}
    calibration = [item for item in examples if item.split == "calibration"]
    test = [item for item in examples if item.split == "test"]
    reasons = []
    if len(calibration) < minimum_groups:
        reasons.append("insufficient_calibration_request_groups")
    if len(test) < minimum_groups:
        reasons.append("insufficient_test_request_groups")
    artifact = None
    metrics = None
    route_metrics = {}
    baseline = None
    if not reasons:
        artifact = fit_calibrator(examples, target_error_rate=target_error_rate, integrity_key=calibrator_key)
        metrics = _metrics(test, artifact, calibrator_key, unsafe_ids)
        if metrics["unsafe_releases"]:
            reasons.append("unsafe_heldout_release")
        if metrics["coverage"] < minimum_coverage:
            reasons.append("insufficient_heldout_coverage")
        if metrics["error_wilson_upper_95"] > target_error_rate:
            reasons.append("heldout_error_upper_bound_exceeds_target")
        for route in sorted({item.route for item in test}):
            subset = [item for item in test if item.route == route]
            route_metrics[route] = _metrics(subset, artifact, calibrator_key, unsafe_ids)
            if len(subset) < minimum_groups or artifact.route_counts.get(route, 0) < 5:
                reasons.append(f"insufficient_route_support:{route}")
            elif (
                route_metrics[route]["coverage"] < minimum_coverage
                or route_metrics[route]["error_wilson_upper_95"] > target_error_rate
            ):
                reasons.append(f"route_gate_failed:{route}")
        if incumbent is not None:
            # load_calibrator/ selective_decision validate the incumbent seal.
            baseline = _metrics(test, incumbent, calibrator_key, unsafe_ids)
            if (
                metrics["coverage"] < baseline["coverage"]
                or metrics["error_wilson_upper_95"] > baseline["error_wilson_upper_95"]
            ):
                reasons.append("incumbent_regression")
    report = {
        "schema_version": "1.0",
        "simulation": allow_synthetic,
        "gate_passed": not reasons,
        "production_candidate_eligible": not reasons and not allow_synthetic,
        "auto_deployed": False,
        "reasons": reasons,
        "calibration_groups": len(calibration),
        "test_groups": len(test),
        "heldout": metrics,
        "routes": route_metrics,
        "incumbent": baseline,
        "thresholds": {
            "maximum_unsafe_releases": 0,
            "minimum_groups_per_split_and_route": minimum_groups,
            "minimum_coverage": minimum_coverage,
            "maximum_error_upper_95": target_error_rate,
        },
        "artifact_fingerprint": artifact.artifact_fingerprint if artifact else None,
        "dataset_manifest": manifest,
    }
    report["fingerprint"] = digest(store.key, "calibration-gate", report)
    return artifact, report


def run_drill() -> dict:
    """Synthetic integration checks, never attributed to users or live providers."""
    key = b"synthetic-replay-drill-key-not-for-production"
    with tempfile.TemporaryDirectory(prefix="agentforge-replay-") as directory:
        store = ExecutionReplayStore(Path(directory) / "replay.sqlite3", key)
        plan = plan_compute(
            ComputeSignals(
                route="rag",
                grounding_action="repair",
                grounding_confidence=0.4,
                uncertainty_decision="abstain",
                evidence_count=2,
            )
        )
        for index in range(240):
            correct = index % 4 != 0
            candidate = candidate_assessment(
                candidate_id="candidate-1",
                answer="synthetic fixture",
                confidence=0.9 if correct else 0.2,
                grounded=True,
                conformal_decision="not_evaluated",
                claim_keys=["synthetic-evidence"],
                token_count=40,
                latency_ms=5,
            )
            receipt = select_candidate(plan, [candidate])
            ids = store.capture(
                plan,
                receipt,
                tenant="synthetic-tenant",
                request_id=f"fixture-{index}",
                consent=True,
                origin="synthetic",
            )
            store.review(
                "synthetic-tenant",
                ids[0],
                verdict="correct" if correct else "incorrect",
                unsafe=False,
                reviewer="synthetic-reviewer",
            )
        _, clean = recalibrate(store, "synthetic-tenant", allow_synthetic=True)
        _, production = recalibrate(store, "synthetic-tenant")
        isolation = not store.records("other-tenant")
        tamper_id = store.records("synthetic-tenant")[0][0].event_id
        with store._db() as db:
            row = db.execute("SELECT payload FROM observations WHERE event_id=?", (tamper_id,)).fetchone()
            payload = json.loads(row[0])
            payload["confidence"] = 0.51
            db.execute("UPDATE observations SET payload=? WHERE event_id=?", (json.dumps(payload), tamper_id))
        try:
            store.dataset("synthetic-tenant", allow_synthetic=True)
            tamper_rejected = False
        except ValueError:
            tamper_rejected = True
        return {
            "simulation": True,
            "clean_gate": clean,
            "synthetic_excluded_from_production": not production["production_candidate_eligible"]
            and production["calibration_groups"] == 0,
            "cross_tenant_isolation": isolation,
            "tamper_rejected": tamper_rejected,
            "passed": clean["gate_passed"]
            and not clean["production_candidate_eligible"]
            and not production["gate_passed"]
            and isolation
            and tamper_rejected,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drill", action="store_true", help="Synthetic control tests; cannot produce a deployable candidate"
    )
    parser.add_argument("--store", default="data/execution-replay/replay.sqlite3")
    parser.add_argument("--tenant")
    parser.add_argument("--output", default="data/evaluations/execution-replay/report.json")
    parser.add_argument("--artifact", default="data/execution-replay/candidate-calibrator.json")
    parser.add_argument("--incumbent", help="Optional existing calibrator for coverage/risk non-regression")
    parser.add_argument("--require-gate", action="store_true")
    args = parser.parse_args()
    if args.drill:
        report = run_drill()
        passed = report["passed"]
    else:
        if not args.tenant:
            parser.error("--tenant is required outside the synthetic drill")
        active = Path(
            os.getenv("UNCERTAINTY_CALIBRATOR_PATH", "data/evaluations/uncertainty/calibrator.json")
        ).resolve()
        protected = {active, Path(args.store).resolve()}
        if args.incumbent:
            protected.add(Path(args.incumbent).resolve())
        outputs = {Path(args.artifact).resolve(), Path(args.output).resolve()}
        if len(outputs) != 2 or outputs & protected:
            parser.error(
                "candidate and report paths must be distinct and must not overwrite the store, incumbent or active calibrator"
            )
        key = os.getenv("EXECUTION_REPLAY_KEY", "").encode()
        calibrator_key = os.getenv("UNCERTAINTY_INTEGRITY_KEY", "").encode() or None
        store = ExecutionReplayStore(args.store, key)
        incumbent = load_calibrator(args.incumbent, calibrator_key) if args.incumbent else None
        artifact, report = recalibrate(store, args.tenant, calibrator_key=calibrator_key, incumbent=incumbent)
        passed = report["production_candidate_eligible"]
        if passed:
            # Candidate only. Never overwrite the active runtime artifact.
            save_calibrator(args.artifact, artifact)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "simulation": args.drill, "report": str(target)}))
    if args.require_gate and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
