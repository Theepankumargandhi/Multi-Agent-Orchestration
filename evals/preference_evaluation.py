"""Forward-time preference evaluation, shared holdout reservations, no auto-deploy."""

from __future__ import annotations

import argparse
import json
import os
import random
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
from agent.preference_ranking import (
    PreferenceArtifact,
    PreferenceCohort,
    PreferenceRanker,
    artifact_identity,
    freeze_preferences,
    train_preferences,
    validate_preference_lineage,
)
from agent.prospective_validation import HoldoutLedger
from evals.replay_calibration import wilson_upper


def _paired_lower(differences: list[float]) -> float:
    rng = random.Random(17)
    samples = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(1000)
    )
    return samples[24]


def evaluate_preferences(
    cohort: PreferenceCohort,
    store: ExecutionReplayStore,
    tenant: str,
    artifact_key: bytes,
    ledger: HoldoutLedger,
) -> tuple[dict, PreferenceArtifact]:
    if ledger.key != store.key:
        raise ValueError("preference evaluations must share the replay holdout key")
    validate_preference_lineage(cohort, store, tenant)
    artifact = train_preferences(cohort, store.key, artifact_key, tenant)
    policy = {
        "min_families_per_scope": 20,
        "max_error_upper": 0.2,
        "min_supported_fraction": 0.8,
        "min_reranked_fraction": 0.2,
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 17,
        "utility": "+1 safe correct; -4 otherwise",
    }
    study = digest(
        store.key, "preference-study-v1", [cohort.fingerprint, artifact_identity(artifact), policy]
    )
    # Reserve before accessing held-out outcomes or computing test metrics.
    cached = ledger.reserve_families(
        study, cohort.fingerprint, [pool.task_family for pool in cohort.pools if pool.split == "test"]
    )
    if cached is not None:
        fingerprint = cached.get("fingerprint")
        if fingerprint != digest(
            store.key, "preference-report", {k: v for k, v in cached.items() if k != "fingerprint"}
        ):
            raise ValueError("preference report integrity failed")
        return cached, artifact
    ranker = PreferenceRanker(artifact, artifact_key, tenant, allow_simulation=cohort.simulation)
    results = []
    for pool in cohort.pools:
        if pool.split != "test":
            continue
        views = [row.ranking_view() for row in pool.candidates]
        baseline = max(views, key=lambda row: (row.confidence, -row.token_count, row.candidate_id))
        decision = ranker.rank(views, baseline.candidate_id, route=pool.route, high_risk=pool.high_risk)
        rows = {row.observation.candidate: row for row in pool.candidates}
        selected = rows[decision.selected_candidate_id]
        before = rows[baseline.candidate_id]
        results.append(
            {
                "scope": f"{pool.route}:{int(pool.high_risk)}",
                "correct": selected.preferred,
                "baseline_correct": before.preferred,
                "unsafe": selected.label.unsafe,
                "supported": decision.reason != "unsupported_scope_or_feature_ood",
                "reranked": decision.status == "reranked",
                "utility_delta": 5 * (int(selected.preferred) - int(before.preferred)),
            }
        )
    count = len(results)
    supported = sum(row["supported"] for row in results)
    reranked = sum(row["reranked"] for row in results)
    correct = sum(row["correct"] for row in results)
    baseline_correct = sum(row["baseline_correct"] for row in results)
    unsafe = sum(row["unsafe"] for row in results)
    lower = _paired_lower([row["utility_delta"] for row in results]) if results else None
    slices = {}
    failures = []
    for scope in sorted({row["scope"] for row in results}):
        rows = [row for row in results if row["scope"] == scope]
        successes = sum(row["correct"] for row in rows)
        upper = wilson_upper(len(rows) - successes, len(rows))
        slices[scope] = {"families": len(rows), "correct": successes, "error_upper_95": upper}
        if len(rows) < 20 or upper > 0.2:
            failures.append("insufficient_scope_support_or_error_bound")
    if count < 20:
        failures.append("insufficient_test_families")
    if count == 0 or supported / count < 0.8:
        failures.append("feature_distribution_shift")
    if count == 0 or reranked / count < 0.2:
        failures.append("no_material_reranking")
    if unsafe:
        failures.append("unsafe_selection")
    if lower is None or lower < 0:
        failures.append("paired_utility_regression")
    report = {
        "schema_version": "1.0",
        "evidence_kind": "synthetic_control_drill"
        if cohort.simulation
        else "human_reviewed_metadata_preferences",
        "claim_scope": "fully reviewed grounded candidate pools; not end-to-end release quality or causal traffic lift",
        "study_id": study,
        "cohort_fingerprint": cohort.fingerprint,
        "artifact_fingerprint": artifact.fingerprint,
        "tenant": cohort.tenant,
        "training_families": artifact.training_families,
        "training_pairs": artifact.pair_count,
        "test_families": count,
        "supported_families": supported,
        "reranked_families": reranked,
        "correct_selections": correct,
        "confidence_baseline_correct": baseline_correct,
        "unsafe_selections": unsafe,
        "paired_utility_delta_lower_95": lower,
        "slices": slices,
        "policy": policy,
        "exclusions": cohort.exclusions,
        "gate_passed": not failures,
        "failure_reasons": sorted(set(failures)),
        "production_candidate": not failures and not cohort.simulation,
    }
    report["fingerprint"] = digest(store.key, "preference-report", report)
    ledger.finish(study, report)
    return report, artifact


def write_once(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) == payload:
            return
        raise ValueError("output exists with different evidence; use a fresh path")
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def seed_drill(
    store: ExecutionReplayStore,
    tenant: str,
    *,
    shifted: bool = False,
    origin: str = "synthetic",
    families: int = 40,
) -> PreferenceCohort:
    """Authored metadata correlations only; no factuality or real cost claim."""
    clock = [0.0]
    store.clock = lambda: clock[0]
    for split, timestamp in (("train", 100.0), ("test", 300.0)):
        for index in range(families):
            clock[0] = timestamp + index
            plan = plan_compute(
                ComputeSignals(
                    route="rag",
                    grounding_action="repair",
                    grounding_confidence=0.4,
                    uncertainty_decision="abstain",
                    evidence_count=2,
                ),
                ComputePolicy(max_extra_tokens=3000),
            )
            # Counterbalance: preferred answers are sometimes long, sometimes short.
            # Preference depends on the confidence/length interaction, not length alone.
            prefer_short = index % 2 == 0
            good_conf = 0.8 if prefer_short else 0.98
            bad_conf = 0.9 if prefer_short else 0.6
            tokens = (100, 500) if prefer_short else (500, 100)
            if shifted and split == "test":
                tokens = (900, 1000)
            candidates = [
                candidate_assessment(
                    candidate_id=f"candidate-{j}",
                    answer="synthetic control",
                    confidence=confidence,
                    grounded=True,
                    conformal_decision="not_evaluated",
                    claim_keys=["same-evidence"],
                    token_count=length,
                    latency_ms=10,
                )
                for j, confidence, length in ((1, good_conf, tokens[0]), (2, bad_conf, tokens[1]))
            ]
            receipt = select_candidate(plan, candidates)
            ids = store.capture(
                plan,
                receipt,
                tenant=tenant,
                request_id=f"{split}-{index}",
                consent=True,
                origin=origin,
                task_family=f"{split}-family-{index}",
            )
            clock[0] += 0.5
            for event_id, correct in zip(ids, (True, False), strict=True):
                store.review(
                    tenant,
                    event_id,
                    verdict="correct" if correct else "incorrect",
                    unsafe=False,
                    reviewer="drill-reviewer",
                )
    clock[0] = 500
    return freeze_preferences(store, tenant, cutoff=200, embargo_seconds=50, simulation=origin == "synthetic")


def run_drill() -> dict:
    replay_key, artifact_key = (
        b"synthetic-replay-drill-not-a-secret-key",
        b"synthetic-ranker-drill-not-a-secret-key",
    )
    reports = {}
    with tempfile.TemporaryDirectory(prefix="agentforge-preferences-") as directory:
        for shifted in (False, True):
            store = ExecutionReplayStore(Path(directory) / f"replay-{shifted}.sqlite3", replay_key)
            cohort = seed_drill(store, "drill-tenant", shifted=shifted)
            ledger = HoldoutLedger(Path(directory) / f"ledger-{shifted}.sqlite3", replay_key)
            report, _ = evaluate_preferences(cohort, store, "drill-tenant", artifact_key, ledger)
            reports["shifted" if shifted else "clean"] = report
    reports["gate_passed"] = reports["clean"]["gate_passed"] and not reports["shifted"]["gate_passed"]
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", default=os.getenv("EXECUTION_REPLAY_PATH", "data/execution-replay/replay.sqlite3")
    )
    parser.add_argument("--ledger", default="data/execution-replay/holdout-ledger.sqlite3")
    parser.add_argument("--tenant")
    commands = parser.add_subparsers(dest="command", required=True)
    drill = commands.add_parser("drill")
    drill.add_argument("--output", default="data/evaluations/preferences/drill.json")
    drill.add_argument("--require-gate", action="store_true")
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--cutoff", type=float, required=True)
    freeze.add_argument("--embargo-seconds", type=float, default=3600)
    freeze.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("cohort")
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--require-gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        # Reject input/store/active artifact aliasing BEFORE creating any outputs.
        protected = {
            Path(args.store).resolve(),
            Path(args.ledger).resolve(),
            Path(os.getenv("PREFERENCE_RANKING_PATH", "data/evaluations/preferences/active.json")).resolve(),
        }
        protected |= {
            Path(str(path) + suffix) for path in tuple(protected) for suffix in ("-wal", "-shm", "-journal")
        }
        if args.command == "evaluate":
            protected.add(Path(args.cohort).resolve())
        outputs = [Path(args.output).resolve()]
        if args.command == "evaluate":
            outputs.append(Path(args.candidate).resolve())
        if len(set(outputs)) != len(outputs) or any(path in protected for path in outputs):
            raise ValueError("output aliases protected store, cohort, or active artifact")
        if args.command == "drill":
            report = run_drill()
            write_once(args.output, report)
        else:
            if not args.tenant or not args.tenant.strip():
                raise ValueError("tenant is required")
            store = ExecutionReplayStore(args.store, os.getenv("EXECUTION_REPLAY_KEY", "").encode())
            if args.command == "freeze":
                cohort = freeze_preferences(
                    store, args.tenant, cutoff=args.cutoff, embargo_seconds=args.embargo_seconds
                )
                write_once(args.output, cohort.model_dump(mode="json"))
                print(
                    json.dumps(
                        {
                            "cohort_fingerprint": cohort.fingerprint,
                            "pools": len(cohort.pools),
                            "exclusions": cohort.exclusions,
                        }
                    )
                )
                return 0
            cohort = PreferenceCohort.model_validate_json(Path(args.cohort).read_text(encoding="utf-8"))
            report, artifact = evaluate_preferences(
                cohort,
                store,
                args.tenant,
                os.getenv("PREFERENCE_RANKING_KEY", "").encode(),
                HoldoutLedger(args.ledger, store.key),
            )
            write_once(args.output, report)
            if report["production_candidate"]:
                write_once(args.candidate, artifact.model_dump(mode="json"))
        print(json.dumps(report, indent=2))
        return 0 if not args.require_gate or report["gate_passed"] else 1
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "held", "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
