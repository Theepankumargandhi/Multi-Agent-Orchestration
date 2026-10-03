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
from agent.execution_replay import ExecutionReplayStore
from agent.process_reward import ProcessStep
from agent.process_supervision import ProcessSupervisionStore
from agent.prospective_validation import HoldoutLedger
from agent.verifier_shadow import VerifierShadowStore, VerifierStudy, VerifierTrialPolicy
from evals.process_supervision_evaluation import evaluate_process, seed_process
from evals.verifier_shadow_evaluation import (
    evaluate_verifier,
    main,
    run_drill,
    seed_shadow,
    threshold_metrics,
)

KEY = b"unit-test-verifier-replay-key-32-bytes"
MODEL_KEY = b"unit-test-verifier-model-key-32-bytes"
TENANT = "private-verifier-tenant"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    path = tmp_path_factory.mktemp("verifier-training")
    replay = ExecutionReplayStore(path / "replay.sqlite3", KEY)
    process = ProcessSupervisionStore(replay)
    # Authored runtime-origin fixture tests the real-data export path; it is not traffic evidence.
    training = seed_process(process, TENANT, origin="runtime")
    report, candidate = evaluate_process(
        training, process, TENANT, MODEL_KEY, HoldoutLedger(path / "training-ledger.sqlite3", KEY)
    )
    store = VerifierShadowStore(replay, MODEL_KEY)
    cohort = seed_shadow(store, candidate, training, report, TENANT)
    return store, cohort


@pytest.fixture
def live(tmp_path, prepared):
    original, cohort = prepared
    path = tmp_path / "replay.sqlite3"
    with (
        closing(sqlite3.connect(original.replay.path)) as source,
        closing(sqlite3.connect(path)) as destination,
    ):
        source.backup(destination)
    store = VerifierShadowStore(ExecutionReplayStore(path, KEY, clock=lambda: 1000.0), MODEL_KEY)
    return store, cohort


def pool(
    store,
    *,
    request="new",
    family="new-family",
    timestamp=1100,
    keys=None,
    policy=None,
    grounded=True,
    conformal="release",
):
    store.replay.clock = lambda: float(timestamp)
    policy = policy or ComputePolicy()
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
    keys = keys or (["PRIVATE-evidence"], ["PRIVATE-evidence"])
    candidates = [
        candidate_assessment(
            candidate_id=name,
            answer="PRIVATE original answer",
            confidence=confidence,
            grounded=grounded,
            conformal_decision=conformal,
            claim_keys=claims,
            token_count=100,
            latency_ms=10,
        )
        for name, confidence, claims in zip(("good", "bad"), (0.8, 0.95), keys, strict=True)
    ]
    baseline = select_candidate(plan, candidates, policy)
    ids = store.replay.capture(
        plan, baseline, tenant=TENANT, request_id=request, task_family=family, consent=True
    )
    for event, confidence, good in zip(ids, (0.8, 0.95), (True, False), strict=True):
        store.process.capture(
            TENANT,
            event,
            [
                ProcessStep(step_id="PRIVATE-id", kind="retrieve", has_evidence=True),
                ProcessStep(step_id="answer", kind="answer", citation_valid=good, confidence=confidence),
            ],
            consent=True,
        )
    return plan, baseline, candidates, policy, ids


def test_drill_preserves_abstentions_and_holds_missing_or_unsafe_outcomes():
    result = run_drill()
    assert result["gate_passed"] and not result["production_activation"]
    clean = result["reports"]["clean"]
    assert clean["primary"]["families"] == 40 and clean["baseline_abstained"] == 4
    assert clean["primary"]["released"] == 36 and clean["primary"]["correct_releases"] == 36
    assert clean["primary"]["unsafe_choices"] == 0
    assert not clean["ready_for_owner_review"] and clean["additional_generation_calls"] == 0
    assert not result["reports"]["missing_review"]["gate_passed"]
    assert "reviewed_unsafe_shadow_choice" in result["reports"]["unsafe_shift"]["failure_reasons"]


def test_registration_roundtrips_binds_provenance_and_is_immutable(live):
    store, cohort = live
    study = cohort.study
    loaded = VerifierStudy.model_validate_json(study.model_dump_json())
    loaded.verify(KEY)
    assert (
        store.register(
            "synthetic-verifier-study",
            TENANT,
            study.candidate,
            study.training_cohort,
            study.training_report,
            study.compute_policy,
            study.trial_policy,
        )
        == study
    )
    with pytest.raises(ValueError, match="immutable"):
        store.register(
            "synthetic-verifier-study",
            TENANT,
            study.candidate,
            study.training_cohort,
            study.training_report,
            study.compute_policy,
            study.trial_policy.model_copy(update={"primary_threshold": 0.5}),
        )
    with pytest.raises(ValueError, match="tenant"):
        store.study(study.study_id, "other")
    with pytest.raises(ValueError):
        store.register(
            "forged",
            TENANT,
            study.candidate.model_copy(update={"explicit_training_steps": 1}),
            study.training_cohort,
            study.training_report,
            study.compute_policy,
            study.trial_policy,
        )


@pytest.mark.parametrize("thresholds", [[0.6, 0.5], [0.6, 0.6], [0.0, 1.0], [-1, 0.6], [0.6, float("nan")]])
def test_threshold_grid_is_fixed_and_validated(thresholds):
    with pytest.raises(ValueError):
        VerifierTrialPolicy(curve_thresholds=thresholds).check()


def test_shadow_reuses_actual_selector_and_freezes_only_metadata(live):
    store, cohort = live
    plan, baseline, candidates, policy, _ = pool(store)
    row = store.capture(
        cohort.study.study_id, TENANT, "new", plan, baseline, candidates, policy, consent=True
    )
    assert "PRIVATE" not in row.model_dump_json() and TENANT not in row.model_dump_json()
    assert row.baseline_event != row.proposed_event
    store.replay.clock = lambda: 1101.0
    assert row == store.capture(
        cohort.study.study_id, TENANT, "new", plan, baseline, candidates, policy, consent=True
    )
    assert (
        store.capture(cohort.study.study_id, TENANT, "new", plan, baseline, candidates, policy, consent=False)
        is None
    )
    with pytest.raises(ValueError, match="incumbent"):
        store.capture(
            cohort.study.study_id,
            TENANT,
            "new",
            plan,
            baseline,
            candidates,
            policy,
            consent=True,
            incumbent_fingerprint="0" * 64,
        )
    assert len(store.queue(cohort.study.study_id, TENANT)) == 1


@pytest.mark.parametrize("review", ["terminal", "step"])
def test_shadow_cannot_be_created_after_any_review(live, review):
    store, cohort = live
    plan, baseline, candidates, policy, ids = pool(store)
    if review == "terminal":
        store.replay.review(TENANT, ids[0], verdict="correct", unsafe=False, reviewer="reviewer")
    else:
        store.process.review(TENANT, ids[0], 0, "correct", "reviewer")
    with pytest.raises(ValueError, match="precede all reviews"):
        store.capture(cohort.study.study_id, TENANT, "new", plan, baseline, candidates, policy, consent=True)


@pytest.mark.parametrize("grounded,conformal", [(False, "release"), (True, "abstain")])
def test_better_scores_never_reopen_ineligible_or_abstained_pool(live, grounded, conformal):
    store, cohort = live
    plan, baseline, candidates, policy, _ = pool(store, grounded=grounded, conformal=conformal)
    row = store.capture(
        cohort.study.study_id, TENANT, "new", plan, baseline, candidates, policy, consent=True
    )
    assert row.baseline_event == row.proposed_event == row.select(0) == ""


def test_consensus_boundary_uses_unrounded_selector_value(live):
    store, cohort = live
    compute = ComputePolicy(min_consensus=0.6666668)
    study = store.register(
        "boundary",
        TENANT,
        cohort.study.candidate,
        cohort.study.training_cohort,
        cohort.study.training_report,
        compute,
        VerifierTrialPolicy(),
    )
    plan, baseline, candidates, policy, _ = pool(store, policy=compute, keys=(["a", "b"], ["a", "b", "c"]))
    assert baseline.status == "abstained" and baseline.candidate_summaries[0].consensus == 0.666667
    row = store.capture(study.study_id, TENANT, "new", plan, baseline, candidates, policy, consent=True)
    assert row.proposed_event == "" and row.choices[0].consensus < policy.min_consensus


def test_all_abstain_curve_is_not_a_success_claim_and_primary_is_not_tuned(live, tmp_path):
    store, cohort = live
    all_abstain = threshold_metrics(cohort, 1.0)
    assert all_abstain["release_coverage"] == 0 and all_abstain["abstained"] == 40
    assert all_abstain["scopes"]["rag:0"]["worst_case_error_upper_95"] is None
    report = evaluate_verifier(cohort, store, TENANT, HoldoutLedger(tmp_path / "ledger.sqlite3", KEY))
    assert report["primary"]["threshold"] == 0.6
    assert report["curve_usage"].startswith("descriptive only")


def test_heldout_cache_and_family_exposure_are_shared(live, tmp_path):
    store, cohort = live
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    report = evaluate_verifier(cohort, store, TENANT, ledger)
    assert report == evaluate_verifier(cohort, store, TENANT, ledger)
    with pytest.raises(ValueError):
        ledger.reserve_families(
            "different-experiment", cohort.fingerprint, [member.task_family for member in cohort.members]
        )
    assert report["ready_for_owner_review"] and not report["production_activation"]


@pytest.mark.parametrize("source", ["workflow", "observation", "training"])
def test_source_deletion_or_tampering_blocks_cached_evaluation(live, tmp_path, source):
    store, cohort = live
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    evaluate_verifier(cohort, store, TENANT, ledger)
    event = cohort.members[0].comparison.choices[0].snapshot.source.event_id
    with store.replay._db() as db:
        if source == "workflow":
            db.execute(
                "UPDATE process_snapshots SET payload=json_set(payload, '$.steps[0].confidence', 0.1) WHERE event_id=?",
                (event,),
            )
        else:
            if source == "training":
                event = cohort.study.training_cohort.members[0].snapshot.source.event_id
            db.execute("DELETE FROM observations WHERE event_id=?", (event,))
    with pytest.raises(ValueError):
        evaluate_verifier(cohort, store, TENANT, ledger)


def test_old_families_and_embargo_do_not_become_future_evidence(live):
    store, cohort = live
    for timestamp, family, request in (
        (680, "embargo", "boundary"),
        (1100, "process-family-train-0", "reused"),
    ):
        plan, baseline, candidates, policy, _ = pool(
            store, timestamp=timestamp, family=family, request=request
        )
        store.capture(
            cohort.study.study_id, TENANT, request, plan, baseline, candidates, policy, consent=True
        )
    store.replay.clock = lambda: 1200.0
    frozen = store.freeze(cohort.study.study_id, TENANT)
    assert len(frozen.members) == 40
    assert frozen.exclusions == {"preexisting_family_or_embargo": 2}


def test_tenant_deletion_removes_private_trials_not_exposure(live, tmp_path):
    store, cohort = live
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    evaluate_verifier(cohort, store, TENANT, ledger)
    store.replay.delete_tenant(TENANT)
    with store.replay._db() as db:
        assert db.execute("SELECT count(*) FROM verifier_studies").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM verifier_comparisons").fetchone()[0] == 0
    with pytest.raises(ValueError):
        ledger.reserve_families(
            "fresh-study", cohort.fingerprint, [member.task_family for member in cohort.members]
        )


def test_cli_freeze_evaluate_and_active_path_protection(live, tmp_path, monkeypatch):
    store, cohort = live
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", KEY.decode())
    monkeypatch.setenv("PROCESS_SUPERVISION_MODEL_KEY", MODEL_KEY.decode())
    active = tmp_path / "active.json"
    active.write_text('{"owner": "preserve"}', encoding="utf-8")
    monkeypatch.setenv("PROCESS_REWARD_MODEL_PATH", str(active))
    shared = [
        "--store",
        str(store.replay.path),
        "--tenant",
        TENANT,
        "--ledger",
        str(tmp_path / "ledger.sqlite3"),
    ]
    source, output = tmp_path / "cohort.json", tmp_path / "report.json"
    assert main([*shared, "queue", cohort.study.study_id]) == 0
    assert main([*shared, "freeze", cohort.study.study_id, "--output", str(source)]) == 0
    args = [*shared, "evaluate", str(source), "--output", str(output), "--require-gate"]
    assert main(args) == main(args) == 0
    assert json.loads(output.read_text())["ready_for_owner_review"]
    assert main([*shared, "evaluate", str(source), "--output", str(active)]) == 1
    assert main([*shared, "evaluate", str(source), "--output", str(source)]) == 1
    assert json.loads(active.read_text()) == {"owner": "preserve"}


def test_late_labels_cannot_improve_a_frozen_or_cached_result(live, tmp_path):
    store, cohort = live
    plan, baseline, candidates, policy, ids = pool(store)
    store.capture(cohort.study.study_id, TENANT, "new", plan, baseline, candidates, policy, consent=True)
    store.replay.clock = lambda: 1101.0
    frozen = store.freeze(cohort.study.study_id, TENANT)
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    initial = evaluate_verifier(frozen, store, TENANT, ledger)
    store.replay.clock = lambda: 1102.0
    for event in ids:
        store.replay.review(TENANT, event, verdict="correct", unsafe=False, reviewer="reviewer")
    assert evaluate_verifier(frozen, store, TENANT, ledger) == initial
    assert store.freeze(cohort.study.study_id, TENANT, frozen_at=1101) == frozen


def test_cli_registration_requires_original_passing_provenance(live, tmp_path, monkeypatch):
    store, cohort = live
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", KEY.decode())
    monkeypatch.setenv("PROCESS_SUPERVISION_MODEL_KEY", MODEL_KEY.decode())
    names = {
        "--candidate": cohort.study.candidate.model_dump(mode="json"),
        "--training-cohort": cohort.study.training_cohort.model_dump(mode="json"),
        "--training-report": cohort.study.training_report,
        "--compute-policy": cohort.study.compute_policy.model_dump(mode="json"),
        "--trial-policy": cohort.study.trial_policy.model_dump(mode="json"),
    }
    args = ["--store", str(store.replay.path), "--tenant", TENANT, "register", "--name", "cli-study"]
    for flag, payload in names.items():
        path = tmp_path / (flag[2:] + ".json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        args.extend([flag, str(path)])
    assert main(args) == main(args) == 0
    forged = dict(cohort.study.training_report, gate_passed=False)
    (tmp_path / "training-report.json").write_text(json.dumps(forged), encoding="utf-8")
    assert main(args) == 1


@pytest.mark.asyncio
async def test_runtime_shadow_keeps_answer_and_model_call_count(live, monkeypatch):
    store, cohort = live
    research = importlib.import_module("agent.research_assistant")
    for name, value in {
        "VERIFIER_SHADOW_ENABLED": True,
        "VERIFIER_SHADOW_STUDY_ID": cohort.study.study_id,
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
    answer = "AgentForge uses LangGraph for agent orchestration [source](https://example.com/architecture)."
    llm = AsyncMock(return_value=answer)
    monkeypatch.setattr(research, "_call_llm", llm)
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
    assert result["verifier_shadow_receipt"]["status"] == "captured"
    assert len(store.comparisons(cohort.study.study_id, TENANT)) == 41
    config["configurable"]["execution_replay_consent"] = False
    result = await research.adaptive_deliberation_agent(state, config)
    assert result["final_response"] == answer and result["verifier_shadow_receipt"] == {}
    monkeypatch.setattr(research, "VERIFIER_SHADOW_STUDY_ID", "0" * 64)
    config["configurable"]["execution_replay_consent"] = True
    config["configurable"]["execution_replay_request_id"] = "unavailable"
    result = await research.adaptive_deliberation_agent(state, config)
    assert result["final_response"] == answer and result["verifier_shadow_receipt"]["status"] == "unavailable"
    assert (
        await research._capture_verifier_shadow(
            plan, None, [], cohort.study.compute_policy, config, {"confidence-only", "0" * 64}
        )
    )["status"] == "unavailable"
