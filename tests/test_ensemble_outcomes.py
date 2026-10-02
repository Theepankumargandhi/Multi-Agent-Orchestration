"""A reviewed step model needs independent, future final-answer evidence."""

import importlib
import json
import sqlite3
from contextlib import closing
from unittest.mock import AsyncMock

import pytest

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
    EnsembleOutcomeStudy,
    StepToAnswerPolicy,
    aggregate_steps,
)
from agent.execution_replay import ExecutionReplayStore
from agent.process_reward import ProcessStep
from agent.process_supervision import ProcessSupervisionStore
from agent.prospective_validation import HoldoutLedger
from agent.reviewed_verifier_uncertainty import StepEstimate, UncertaintyPolicy
from agent.verifier_shadow import VerifierShadowStore, VerifierStudy
from evals.ensemble_outcome_evaluation import main, run_drill, seed_outcomes
from evals.process_supervision_evaluation import seed_process
from evals.reviewed_verifier_uncertainty_evaluation import evaluate_uncertainty
from evals.verifier_shadow_evaluation import evaluate_verifier

KEY, MODEL_KEY = b"unit-ensemble-outcome-replay-key-long", b"unit-ensemble-outcome-model-key-long!"
TENANT = "private-owner"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("ensemble-outcomes")
    process = ProcessSupervisionStore(ExecutionReplayStore(root / "replay.sqlite3", KEY))
    training = seed_process(process, TENANT, varied_confidence=True, origin="runtime")
    report, candidate = evaluate_uncertainty(
        training, process, TENANT, MODEL_KEY, HoldoutLedger(root / "training-ledger.sqlite3", KEY)
    )
    store = EnsembleOutcomeShadowStore(process.replay, MODEL_KEY)
    cohort = seed_outcomes(store, candidate, training, report, TENANT)
    store.replay.clock = lambda: 1000.0
    return store, cohort


@pytest.fixture
def live(prepared, tmp_path):
    original, cohort = prepared
    path = tmp_path / "replay.sqlite3"
    with closing(sqlite3.connect(original.replay.path)) as source, closing(sqlite3.connect(path)) as target:
        source.backup(target)
    store = EnsembleOutcomeShadowStore(ExecutionReplayStore(path, KEY, clock=lambda: 1000.0), MODEL_KEY)
    return store, cohort


def pool(
    store,
    *,
    request="new",
    family="new-family",
    high_risk=False,
    grounded=True,
    conformal="release",
    timestamp=1100.0,
):
    store.replay.clock = lambda: timestamp
    policy = ComputePolicy()
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
            answer="PRIVATE original answer",
            confidence=confidence,
            grounded=grounded,
            conformal_decision=conformal,
            claim_keys=["PRIVATE evidence"],
            token_count=100,
            latency_ms=10,
        )
        for name, confidence in (("good", 0.9), ("bad", 0.95))
    ]
    baseline = select_candidate(plan, candidates, policy)
    events = store.replay.capture(
        plan, baseline, tenant=TENANT, request_id=request, task_family=family, consent=True
    )
    for event, confidence, good in zip(events, (0.9, 0.95), (True, False), strict=True):
        store.process.capture(
            TENANT,
            event,
            [
                ProcessStep(step_id="PRIVATE-id", kind="retrieve", has_evidence=True, confidence=1),
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
            ],
            consent=True,
        )
    return plan, baseline, candidates, policy, events


def capture(store, cohort, **kwargs):
    plan, baseline, candidates, policy, events = pool(store, **kwargs)
    row = store.capture(
        cohort.study.study_id,
        TENANT,
        kwargs.get("request", "new"),
        plan,
        baseline,
        candidates,
        policy,
        consent=True,
    )
    return row, events


def test_control_drill_never_treats_step_accuracy_as_outcome_proof():
    result = run_drill()
    assert result["gate_passed"] and not result["production_activation"]
    clean = result["reports"]["clean"]
    assert clean["primary"]["families"] == 40 and clean["baseline_abstained"] == 4
    assert clean["primary"]["correct_releases"] == clean["primary"]["released"] == 36
    assert not clean["ready_for_owner_review"] and clean["additional_generation_calls"] == 0
    for name in ("missing_review", "unsafe_outcome_shift", "unseen_high_risk"):
        assert not result["reports"][name]["gate_passed"]
    shifted = result["reports"]["unsafe_outcome_shift"]
    assert shifted["primary"]["unsafe_choices"] == 36
    unseen = result["reports"]["unseen_high_risk"]
    assert unseen["primary"]["released"] == 0
    assert unseen["unguarded_primary_ablation"]["released"] > 0


def test_confidently_incorrect_is_not_high_quality():
    steps = [ProcessStep(step_id="v", kind="verify"), ProcessStep(step_id="a", kind="answer")]
    estimates = [
        StepEstimate(
            mean=0.01,
            spread=0.0,
            conservative_confidence=0.99,
            predicted_correct=False,
            feature_supported=True,
            accepted=True,
        )
    ] * 2
    result = aggregate_steps(steps, estimates, UncertaintyPolicy(), StepToAnswerPolicy())
    assert not result.allowed and result.quality_score == 0.01
    assert "predicted_incorrect_step" in result.reasons
    with pytest.raises(ValueError, match="aligned"):
        aggregate_steps([], estimates, UncertaintyPolicy(), StepToAnswerPolicy())


@pytest.mark.parametrize("missing", ["verify", "answer"])
def test_required_verification_and_final_answer_cannot_be_skipped(missing):
    step = ProcessStep(step_id="step", kind="answer" if missing == "verify" else "verify")
    estimate = StepEstimate(
        mean=0.99,
        spread=0.0,
        conservative_confidence=0.99,
        predicted_correct=True,
        feature_supported=True,
        accepted=True,
    )
    result = aggregate_steps([step], [estimate], UncertaintyPolicy(), StepToAnswerPolicy())
    assert not result.allowed and "missing_required_verify_or_answer" in result.reasons


def test_registration_roundtrip_provenance_and_policy_are_frozen(live):
    store, cohort = live
    study = EnsembleOutcomeStudy.model_validate_json(cohort.study.model_dump_json())
    study.verify(KEY)
    assert store.study(study.study_id, TENANT) == study
    assert EnsembleOutcomeCohort.model_validate_json(cohort.model_dump_json()) == cohort
    with pytest.raises(ValueError):
        VerifierStudy.model_validate(study.model_dump())
    with pytest.raises(ValueError, match="immutable"):
        store.register(
            "synthetic-ensemble-outcomes",
            TENANT,
            study.candidate,
            study.training_cohort,
            study.training_report,
            study.compute_policy,
            study.trial_policy,
            aggregation=StepToAnswerPolicy(minimum_correct_probability=0.7),
        )
    with pytest.raises(ValueError, match="tenant"):
        store.study(study.study_id, "other")
    forged = dict(study.training_report, gate_passed=False)
    with pytest.raises(ValueError, match="provenance"):
        store.register(
            "forged",
            TENANT,
            study.candidate,
            study.training_cohort,
            forged,
            study.compute_policy,
            study.trial_policy,
            aggregation=study.aggregation,
        )


def test_comparison_is_content_free_and_exact_retry_is_safe(live):
    store, cohort = live
    plan, baseline, candidates, policy, events = pool(store)
    assert (
        store.capture(cohort.study.study_id, TENANT, "new", plan, baseline, candidates, policy, consent=False)
        is None
    )
    row = store.capture(
        cohort.study.study_id, TENANT, "new", plan, baseline, candidates, policy, consent=True
    )
    assert row.baseline_event == events[1] and row.proposed_event == events[0]
    assert "PRIVATE" not in row.model_dump_json() and TENANT not in row.model_dump_json()
    assert (
        store.capture(cohort.study.study_id, TENANT, "new", plan, baseline, candidates, policy, consent=True)
        == row
    )
    assert len(row.choices[0].step_decision.estimates) == 4


@pytest.mark.parametrize("grounded,conformal", [(False, "release"), (True, "abstain")])
def test_ensemble_cannot_reopen_grounding_or_conformal_abstention(live, grounded, conformal):
    store, cohort = live
    row, _ = capture(store, cohort, grounded=grounded, conformal=conformal)
    assert row.baseline_event == row.proposed_event == row.select(0) == row.select_unguarded(0) == ""


def test_unfamiliar_pool_is_held_even_at_zero_threshold_and_ablation_is_reviewed(live):
    store, cohort = live
    row, events = capture(store, cohort, high_risk=True)
    assert row.baseline_event == events[1] and row.proposed_event == row.select(0) == ""
    assert row.select_unguarded(0.6) == events[0]
    assert all(not choice.step_decision.allowed for choice in row.choices)
    queue = store.queue(cohort.study.study_id, TENANT)
    assert set(queue[-1]["event_ids"]) == set(events)


@pytest.mark.parametrize("review", ["step", "terminal"])
def test_comparison_must_precede_all_reviews(live, review):
    store, cohort = live
    plan, baseline, candidates, policy, events = pool(store)
    if review == "step":
        store.process.review(TENANT, events[0], 0, "correct", "reviewer")
    else:
        store.replay.review(TENANT, events[0], verdict="correct", unsafe=False, reviewer="reviewer")
    with pytest.raises(ValueError, match="precede all reviews"):
        store.capture(cohort.study.study_id, TENANT, "new", plan, baseline, candidates, policy, consent=True)


def test_cached_outcome_report_and_shared_exposure(live, tmp_path):
    store, cohort = live
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    report = evaluate_verifier(cohort, store, TENANT, ledger)
    assert report == evaluate_verifier(cohort, store, TENANT, ledger)
    assert report["ready_for_owner_review"] and not report["production_activation"]
    assert report["pipeline"] == "reviewed-step-ensemble-to-terminal-outcomes-v1"
    with pytest.raises(ValueError, match="already exposed"):
        ledger.reserve_families("another-model", cohort.fingerprint, [m.task_family for m in cohort.members])


@pytest.mark.parametrize("source", ["training", "workflow", "observation", "decision"])
def test_live_source_and_score_revocation_blocks_cached_reports(live, tmp_path, source):
    store, cohort = live
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    evaluate_verifier(cohort, store, TENANT, ledger)
    event = cohort.members[0].comparison.choices[0].snapshot.source.event_id
    with store.replay._db() as db:
        if source == "training":
            event = cohort.study.training_cohort.members[0].snapshot.source.event_id
            db.execute("DELETE FROM observations WHERE event_id=?", (event,))
        elif source == "observation":
            db.execute("DELETE FROM observations WHERE event_id=?", (event,))
        elif source == "workflow":
            db.execute(
                "UPDATE process_snapshots SET payload=json_set(payload, '$.steps[0].confidence', 0.1) WHERE event_id=?",
                (event,),
            )
        else:
            row = cohort.members[0].comparison.model_copy(deep=True)
            row.choices[0].step_decision.estimates[0].mean = 0.1
            row.seal(KEY)  # even a re-signed row must reproduce the registered model
            db.execute(
                "UPDATE ensemble_outcome_comparisons SET payload=? WHERE request_group=?",
                (row.model_dump_json(), row.request_group),
            )
    with pytest.raises(ValueError):
        evaluate_verifier(cohort, store, TENANT, ledger)


def test_unguarded_proposal_must_match_frozen_selection(live):
    _, cohort = live
    row = cohort.members[-1].comparison.model_copy(deep=True)
    row.unguarded_event = row.baseline_event
    row.seal(KEY)
    with pytest.raises(ValueError, match="unguarded"):
        row.verify(KEY)


def test_old_study_namespace_cannot_load_a_step_ensemble_study(live):
    store, cohort = live
    legacy = VerifierShadowStore(store.replay, MODEL_KEY)
    with pytest.raises(ValueError, match="register"):
        legacy.study(cohort.study.study_id, TENANT)
    assert store.study(cohort.study.study_id, TENANT) == cohort.study


def test_embargo_and_repeated_training_family_do_not_become_new_outcomes(live):
    store, cohort = live
    capture(store, cohort, request="embargo", family="embargo", timestamp=680.0)
    capture(store, cohort, request="repeat", family="process-family-train-0")
    store.replay.clock = lambda: 1200.0
    frozen = store.freeze(cohort.study.study_id, TENANT)
    assert len(frozen.members) == 40 and frozen.exclusions == {"preexisting_family_or_embargo": 2}


def test_late_terminal_reviews_cannot_improve_frozen_cached_report(live, tmp_path):
    store, cohort = live
    _, events = capture(store, cohort)
    store.replay.clock = lambda: 1101.0
    frozen = store.freeze(cohort.study.study_id, TENANT)
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    first = evaluate_verifier(frozen, store, TENANT, ledger)
    store.replay.clock = lambda: 1102.0
    for event in events:
        store.replay.review(TENANT, event, verdict="correct", unsafe=False, reviewer="late-reviewer")
    assert evaluate_verifier(frozen, store, TENANT, ledger) == first


def test_tenant_deletion_removes_new_sidecars_not_exposure(live, tmp_path):
    store, cohort = live
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    evaluate_verifier(cohort, store, TENANT, ledger)
    store.replay.delete_tenant(TENANT)
    with store.replay._db() as db:
        assert db.execute("SELECT count(*) FROM ensemble_outcome_studies").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM ensemble_outcome_comparisons").fetchone()[0] == 0
    with pytest.raises(ValueError):
        ledger.reserve_families("retry", cohort.fingerprint, [m.task_family for m in cohort.members])


def test_cli_roundtrip_registration_freeze_evaluate_and_path_protection(live, tmp_path, monkeypatch):
    store, cohort = live
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", KEY.decode())
    monkeypatch.setenv("PROCESS_SUPERVISION_MODEL_KEY", MODEL_KEY.decode())
    shared = [
        "--store",
        str(store.replay.path),
        "--ledger",
        str(tmp_path / "ledger.sqlite3"),
        "--tenant",
        TENANT,
    ]
    source, output = tmp_path / "cohort.json", tmp_path / "report.json"
    assert main([*shared, "queue", cohort.study.study_id]) == 0
    assert main([*shared, "freeze", cohort.study.study_id, "--output", str(source)]) == 0
    args = [*shared, "evaluate", str(source), "--output", str(output), "--require-gate"]
    assert main(args) == main(args) == 0
    assert json.loads(output.read_text())["ready_for_owner_review"]
    assert main([*shared, "evaluate", str(source), "--output", str(source)]) == 1
    assert main([*shared, "drill", "--output", str(store.replay.path) + "-wal"]) == 1
    active = tmp_path / "active.json"
    monkeypatch.setenv("VERIFIER_ENSEMBLE_PATH", str(active))
    assert main(["drill", "--output", str(active)]) == 1 and not active.exists()
    payloads = {
        "candidate": cohort.study.candidate.model_dump(),
        "training-cohort": cohort.study.training_cohort.model_dump(),
        "training-report": cohort.study.training_report,
        "compute-policy": cohort.study.compute_policy.model_dump(),
        "trial-policy": cohort.study.trial_policy.model_dump(),
        "aggregation-policy": cohort.study.aggregation.model_dump(),
    }
    args = [*shared, "register", "--name", "cli-study"]
    for flag, payload in payloads.items():
        path = tmp_path / (flag + ".json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        args += ["--" + flag, str(path)]
    assert main(args) == main(args) == 0


@pytest.mark.asyncio
async def test_runtime_observer_does_not_change_answer_or_model_calls(live, monkeypatch):
    store, cohort = live
    research = importlib.import_module("agent.research_assistant")
    for name, value in {
        "ENSEMBLE_OUTCOME_SHADOW_ENABLED": True,
        "ENSEMBLE_OUTCOME_SHADOW_STUDY_ID": cohort.study.study_id,
        "VERIFIER_SHADOW_ENABLED": False,
        "PROCESS_SUPERVISION_ENABLED": True,
        "PROCESS_SUPERVISION_MODEL_KEY": MODEL_KEY,
        "PREFERENCE_RANKING_ENABLED": False,
        "PREFERENCE_SHADOW_ENABLED": False,
        "PREFERENCE_DEPLOYMENT_ENABLED": False,
        "PROCESS_REWARD_MODEL_ENABLED": False,
        "EXECUTION_REPLAY_ENABLED": True,
        "EXECUTION_REPLAY_PATH": store.replay.path,
        "EXECUTION_REPLAY_KEY": KEY,
        "ADAPTIVE_COMPUTE_INTEGRITY_KEY": None,
        "UNCERTAINTY_CALIBRATION_ENABLED": False,
    }.items():
        monkeypatch.setattr(research, name, value)
    answer = "AgentForge uses LangGraph for agent orchestration [source](https://example.com/architecture)."
    llm = AsyncMock(return_value=answer)
    monkeypatch.setattr(research, "_call_llm", llm)
    plan = plan_compute(
        ComputeSignals(
            route="web",
            grounding_action="repair",
            grounding_confidence=0.5,
            uncertainty_decision="not_evaluated",
            evidence_count=1,
        ),
        cohort.study.compute_policy,
    )
    state = {
        "route": "web",
        "query": "PRIVATE How is AgentForge orchestrated?",
        "web_notes": "Link: https://example.com/architecture\nSnippet: AgentForge uses LangGraph for agent orchestration.",
        "adaptive_compute_plan": plan.model_dump(mode="json"),
    }
    config = {
        "configurable": {
            "user_id": TENANT,
            "execution_replay_request_id": "runtime-future",
            "execution_replay_consent": True,
            "execution_replay_task_family": "runtime-future-family",
        }
    }
    result = await research.adaptive_deliberation_agent(state, config)
    assert result["final_response"] == answer and llm.await_count == 2
    assert result["ensemble_outcome_shadow_receipt"]["status"] == "captured"
    assert len(store.comparisons(cohort.study.study_id, TENANT)) == 41
    config["configurable"]["execution_replay_consent"] = False
    result = await research.adaptive_deliberation_agent(state, config)
    assert result["final_response"] == answer and result["ensemble_outcome_shadow_receipt"] == {}
    config["configurable"]["execution_replay_consent"] = True
    config["configurable"]["execution_replay_request_id"] = "unavailable"
    monkeypatch.setattr(research, "ENSEMBLE_OUTCOME_SHADOW_STUDY_ID", "0" * 64)
    result = await research.adaptive_deliberation_agent(state, config)
    assert (
        result["final_response"] == answer
        and result["ensemble_outcome_shadow_receipt"]["status"] == "unavailable"
    )
