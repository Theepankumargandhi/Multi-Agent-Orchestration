"""Prospective final-answer validation for reviewed step ensembles; no activation."""

from __future__ import annotations

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
from agent.ensemble_outcome_shadow import (
    EnsembleOutcomeCohort,
    EnsembleOutcomeShadowStore,
    StepToAnswerPolicy,
)
from agent.execution_replay import ExecutionReplayStore
from agent.process_reward import ProcessStep
from agent.process_supervision import ProcessSupervisionStore
from agent.prospective_validation import HoldoutLedger
from agent.reviewed_verifier_uncertainty import ReviewedStepEnsemble
from agent.verifier_shadow import VerifierTrialPolicy
from evals.process_supervision_evaluation import seed_process
from evals.reviewed_verifier_uncertainty_evaluation import evaluate_uncertainty
from evals.verifier_shadow_evaluation import evaluate_verifier
from evals.verifier_shadow_evaluation import main as shadow_main


def seed_outcomes(
    store, candidate, training, report, tenant, *, review_fraction=1.0, unsafe_shift=False, unseen_risk=False
):
    clock = [650.0]
    store.replay.clock = lambda: clock[0]
    policy = ComputePolicy()
    study = store.register(
        "synthetic-ensemble-outcomes",
        tenant,
        candidate,
        training,
        report,
        policy,
        VerifierTrialPolicy(),
        aggregation=StepToAnswerPolicy(),
    )
    for index in range(40):
        clock[0] = 800.0 + index
        high_risk = unseen_risk and index >= 4
        plan = plan_compute(
            ComputeSignals(
                route="rag",
                high_risk=high_risk,
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
                claim_keys=["PRIVATE evidence"],
                token_count=100,
                latency_ms=10,
            )
            for name, confidence in (("good", 0.9), ("bad", 0.95))
        ]
        baseline = select_candidate(plan, candidates, policy)
        request = f"ensemble-future-{index}"
        ids = store.replay.capture(
            plan,
            baseline,
            tenant=tenant,
            request_id=request,
            task_family=f"ensemble-future-family-{index}",
            consent=True,
            origin="synthetic" if candidate.simulation else "runtime",
        )
        for event, confidence, good in zip(ids, (0.9, 0.95), (True, False), strict=True):
            steps = [
                ProcessStep(step_id="private-id", kind="retrieve", has_evidence=True, confidence=1),
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
            store.process.capture(tenant, event, steps, consent=True)
        store.capture(study.study_id, tenant, request, plan, baseline, candidates, policy, consent=True)
        if index < int(40 * review_fraction):
            for event, good in zip(ids, (True, False), strict=True):
                store.replay.review(
                    tenant,
                    event,
                    verdict="correct" if good != unsafe_shift else "incorrect",
                    unsafe=good and unsafe_shift,
                    reviewer="synthetic-reviewer",
                )
    clock[0] = 900.0
    return store.freeze(study.study_id, tenant)


def run_drill():
    key, model_key = b"synthetic-ensemble-replay-key-not-secret", b"synthetic-ensemble-model-key-not-secret"
    with tempfile.TemporaryDirectory(prefix="agentforge-ensemble-outcomes-") as root:
        root = Path(root)
        process = ProcessSupervisionStore(ExecutionReplayStore(root / "training.sqlite3", key))
        training = seed_process(process, "demo", varied_confidence=True)
        training_report, candidate = evaluate_uncertainty(
            training,
            process,
            "demo",
            model_key,
            HoldoutLedger(root / "training-ledger.sqlite3", key),
        )
        reports = {}
        for name, fraction, unsafe, unseen in (
            ("clean", 1.0, False, False),
            ("missing_review", 0.5, False, False),
            ("unsafe_outcome_shift", 1.0, True, False),
            ("unseen_high_risk", 1.0, False, True),
        ):
            path = root / f"{name}.sqlite3"
            with (
                closing(sqlite3.connect(process.replay.path)) as source,
                closing(sqlite3.connect(path)) as destination,
            ):
                source.backup(destination)
            store = EnsembleOutcomeShadowStore(ExecutionReplayStore(path, key), model_key)
            cohort = seed_outcomes(
                store,
                candidate,
                training,
                training_report,
                "demo",
                review_fraction=fraction,
                unsafe_shift=unsafe,
                unseen_risk=unseen,
            )
            reports[name] = evaluate_verifier(
                cohort, store, "demo", HoldoutLedger(root / f"{name}-ledger.sqlite3", key)
            )
    return {
        "evidence_kind": "synthetic_step_to_answer_outcome_controls",
        "training_report": training_report,
        "reports": reports,
        "gate_passed": training_report["gate_passed"]
        and reports["clean"]["gate_passed"]
        and all(not reports[name]["gate_passed"] for name in reports if name != "clean"),
        "production_activation": False,
    }


def main(argv=None):
    return shadow_main(
        argv,
        store_class=EnsembleOutcomeShadowStore,
        cohort_class=EnsembleOutcomeCohort,
        candidate_class=ReviewedStepEnsemble,
        drill_runner=run_drill,
        default_output="data/evaluations/ensemble-outcomes/drill.json",
        aggregation_class=StepToAnswerPolicy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
