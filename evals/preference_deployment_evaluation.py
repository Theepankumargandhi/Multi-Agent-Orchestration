"""Owner CLI and credential-free deployment-sentinel failure controls."""

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
from agent.execution_replay import ExecutionReplayStore
from agent.preference_deployment import PreferenceDeploymentStore, SentinelPolicy
from agent.preference_ranking import PreferenceRanker, train_preferences
from agent.preference_shadow import PreferenceShadowStore, ShadowApproval, ShadowCohort
from agent.prospective_validation import HoldoutLedger
from evals.preference_evaluation import seed_drill, write_once
from evals.preference_shadow_evaluation import evaluate_shadow, seed_shadow


def seed_control(root: Path):
    """Authored runtime-schema fixtures, never live evidence or exportable release."""
    key = b"synthetic-deployment-replay-key-not-secret"
    artifact_key = b"synthetic-deployment-artifact-key-not-secret"
    tenant = "synthetic-deployment-control"
    replay = ExecutionReplayStore(root / "replay.sqlite3", key)
    training = seed_drill(replay, tenant, origin="runtime")
    ranker = PreferenceRanker(train_preferences(training, key, artifact_key, tenant), artifact_key, tenant)
    shadow = PreferenceShadowStore(replay)
    cohort, policy = seed_shadow(shadow, ranker, tenant)
    _, approval = evaluate_shadow(
        cohort, shadow, tenant, ranker, policy, artifact_key, HoldoutLedger(root / "ledger.sqlite3", key)
    )
    clock = [1100.0]
    replay.clock = lambda: clock[0]
    store = PreferenceDeploymentStore(replay, artifact_key)
    state = store.activate(
        tenant,
        ranker,
        cohort,
        approval,
        policy,
        owner="fixture-owner",
        expected_revision=0,
        sentinel=SentinelPolicy(review_deadline_seconds=60.0),
    )
    return store, state, ranker, policy, tenant, clock


def serve_control(store, state, ranker, tenant, index, *, family=None, consent=True):
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
            answer="PRIVATE fixture answer",
            confidence=confidence,
            grounded=True,
            conformal_decision="release",
            claim_keys=["PRIVATE fixture evidence"],
            token_count=tokens,
            latency_ms=10,
            process_reward=reward,
        )
        for name, confidence, tokens, reward in (("long", 0.9, 500, 0.9), ("short", 0.8, 100, 0.7))
    ]
    receipt = select_candidate(plan, candidates, policy, preference_ranker=ranker)
    ids = store.capture(
        state,
        plan,
        receipt,
        tenant=tenant,
        request_id=f"serve-{index}",
        consent=consent,
        task_family=family if family is not None else f"serve-family-{index}",
    )
    selected = next(
        obs.event_id for obs, _ in store.replay.records(tenant) if obs.event_id in ids and obs.selected
    )
    return plan, receipt, ids, selected


def run_drill() -> dict:
    reports = {}
    with tempfile.TemporaryDirectory(prefix="agentforge-deployment-control-") as root:
        base = Path(root)
        store, state, ranker, policy, tenant, clock = seed_control(base)
        # Snapshot before exposures, so each authored failure starts at the same release.
        with (
            closing(sqlite3.connect(store.replay.path)) as source,
            closing(sqlite3.connect(base / "baseline.sqlite3")) as target,
        ):
            source.backup(target)
        for scenario in (
            "healthy",
            "quality_regression",
            "missing_reviews",
            "unsafe",
            "source_deleted",
            "owner_revoked",
        ):
            path = base / f"{scenario}.sqlite3"
            with (
                closing(sqlite3.connect(base / "baseline.sqlite3")) as source,
                closing(sqlite3.connect(path)) as target,
            ):
                source.backup(target)
            replay = ExecutionReplayStore(path, store.replay.key, clock=lambda: clock[0])
            current = PreferenceDeploymentStore(replay, store.artifact_key)
            clock[0] = 1110.0
            if scenario == "source_deleted":
                cohort = current.deployment(state).cohort
                event = cohort.members[0].comparison.baseline_event
                with replay._db() as db:
                    db.execute("DELETE FROM observations WHERE event_id=?", (event,))
            elif scenario == "owner_revoked":
                current.revoke(tenant, expected_revision=state.revision)
            else:
                for index in range(20):
                    clock[0] = 1110.0 + index
                    _, _, _, selected = serve_control(current, state, ranker, tenant, index)
                    if scenario == "unsafe" and index == 0:
                        unsafe_event = selected
                    elif scenario != "missing_reviews":
                        replay.review(
                            tenant,
                            selected,
                            verdict="incorrect" if scenario == "quality_regression" else "correct",
                            unsafe=False,
                            reviewer="fixture-reviewer",
                        )
                if scenario == "unsafe":
                    replay.review(
                        tenant, unsafe_event, verdict="correct", unsafe=True, reviewer="fixture-reviewer"
                    )
            clock[0] = 1200.0
            try:
                _, active, report = current.admit(tenant, policy, route="rag", high_risk=False)
                reports[scenario] = {
                    "admitted": True,
                    "status": active.status,
                    "reason": report["reason"],
                    "scopes": report["scopes"],
                }
            except ValueError:
                active = current.state(tenant)
                reports[scenario] = {"admitted": False, "status": active.status, "reason": active.reason}
    return {
        "evidence_kind": "synthetic_runtime_schema_failure_controls",
        "scenarios": reports,
        "gate_passed": reports["healthy"]["admitted"]
        and all(not row["admitted"] for name, row in reports.items() if name != "healthy"),
        "production_evidence": False,
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
    activate = commands.add_parser("activate")
    activate.add_argument("--artifact", required=True)
    activate.add_argument("--cohort", required=True)
    activate.add_argument("--approval", required=True)
    activate.add_argument("--compute-policy", help="JSON ComputePolicy for non-default policy")
    activate.add_argument("--sentinel-policy", help="JSON SentinelPolicy fixed before serving")
    activate.add_argument("--owner", required=True)
    activate.add_argument("--expected-revision", type=int, required=True)
    commands.add_parser("status")
    check = commands.add_parser("check")
    check.add_argument(
        "--route", choices=["web", "hybrid", "rag", "kg", "math", "general", "code"], required=True
    )
    check.add_argument("--high-risk", action="store_true")
    revoke = commands.add_parser("revoke")
    revoke.add_argument("--expected-revision", type=int, required=True)
    drill = commands.add_parser("drill")
    drill.add_argument("--output", default="data/evaluations/preference-deployment/drill.json")
    drill.add_argument("--require-gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "drill":
            protected = {
                Path(args.store).resolve(),
                Path(args.ledger).resolve(),
                Path(
                    os.getenv("PREFERENCE_RANKING_PATH", "data/evaluations/preferences/active.json")
                ).resolve(),
                Path(
                    os.getenv(
                        "PREFERENCE_SHADOW_APPROVAL_PATH", "data/evaluations/preference-shadow/approval.json"
                    )
                ).resolve(),
            }
            protected |= {
                Path(str(path) + suffix)
                for path in tuple(protected)
                for suffix in ("-wal", "-shm", "-journal")
            }
            if Path(args.output).resolve() in protected:
                raise ValueError("output aliases protected storage or active artifacts")
            report = run_drill()
            write_once(args.output, report)
            print(json.dumps(report, indent=2))
            return 0 if not args.require_gate or report["gate_passed"] else 1
        replay = ExecutionReplayStore(args.store, os.getenv("EXECUTION_REPLAY_KEY", "").encode())
        artifact_key = os.getenv("PREFERENCE_RANKING_KEY", "").encode()
        store = PreferenceDeploymentStore(replay, artifact_key)
        if args.command == "activate":
            cohort = ShadowCohort.model_validate_json(Path(args.cohort).read_text(encoding="utf-8"))
            approval = ShadowApproval.model_validate_json(Path(args.approval).read_text(encoding="utf-8"))
            ranker = PreferenceRanker.load(args.artifact, artifact_key, args.tenant)
            policy = (
                ComputePolicy.model_validate_json(Path(args.compute_policy).read_text(encoding="utf-8"))
                if args.compute_policy
                else ComputePolicy()
            )
            sentinel = (
                SentinelPolicy.model_validate_json(Path(args.sentinel_policy).read_text(encoding="utf-8"))
                if args.sentinel_policy
                else SentinelPolicy()
            )
            _, expected = evaluate_shadow(
                cohort,
                PreferenceShadowStore(replay),
                args.tenant,
                ranker,
                policy,
                artifact_key,
                HoldoutLedger(args.ledger, replay.key),
            )
            if expected is None or expected != approval:
                raise ValueError("approval differs from the lineage-checked shared-ledger report")
            result = store.activate(
                args.tenant,
                ranker,
                cohort,
                approval,
                policy,
                owner=args.owner,
                expected_revision=args.expected_revision,
                sentinel=sentinel,
            ).model_dump(mode="json")
        elif args.command == "revoke":
            result = store.revoke(args.tenant, expected_revision=args.expected_revision).model_dump(
                mode="json"
            )
        elif args.command == "check":
            state = store.state(args.tenant)
            if state is None:
                raise ValueError("no deployment registered")
            deployment = store.deployment(state)
            _, current, report = store.admit(
                args.tenant, deployment.compute_policy, route=args.route, high_risk=args.high_risk
            )
            result = {"state": current.model_dump(mode="json"), "monitor": report}
        else:
            state = store.state(args.tenant)
            result = {"state": state.model_dump(mode="json") if state else None}
            if state:
                result["monitor"] = store.monitor(args.tenant, store.deployment(state))
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"status": "baseline_fallback", "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
