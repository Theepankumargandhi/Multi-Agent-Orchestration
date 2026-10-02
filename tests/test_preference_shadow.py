import importlib
import json
from unittest.mock import AsyncMock

import pytest

from agent.adaptive_compute import (
    ComputePolicy,
    ComputeSignals,
    _fingerprint,
    _receipt_payload,
    candidate_assessment,
    plan_compute,
    select_candidate,
)
from agent.execution_replay import ExecutionReplayStore, digest
from agent.preference_ranking import PreferenceRanker, train_preferences
from agent.preference_shadow import PreferenceShadowStore, validate_approval
from agent.prospective_validation import HoldoutLedger
from evals.preference_evaluation import seed_drill, write_once
from evals.preference_shadow_evaluation import evaluate_shadow, main, run_drill, seed_shadow

KEY = b"unit-test-shadow-replay-key-32-bytes"
ARTIFACT_KEY = b"unit-test-shadow-artifact-key-32-bytes"
TENANT = "private-shadow-tenant"


@pytest.fixture(scope="module")
def ranker(tmp_path_factory):
    replay = ExecutionReplayStore(tmp_path_factory.mktemp("shadow-training") / "replay.sqlite3", KEY)
    cohort = seed_drill(replay, TENANT, origin="runtime")
    artifact = train_preferences(cohort, KEY, ARTIFACT_KEY, TENANT)
    return PreferenceRanker(artifact, ARTIFACT_KEY, TENANT)


@pytest.fixture
def live(tmp_path, ranker):
    store = PreferenceShadowStore(ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY))
    cohort, policy = seed_shadow(store, ranker, TENANT)
    return store, cohort, policy


def new_pair(store, ranker, *, request="new", family="new-family", grounded=True, consent=True):
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
            answer="PRIVATE-answer sensitive@example.com",
            confidence=conf,
            grounded=grounded,
            conformal_decision="release",
            claim_keys=["PRIVATE-evidence"],
            token_count=tokens,
            latency_ms=10,
            process_reward=reward,
        )
        for name, conf, tokens, reward in (("short", 0.8, 100, 0.7), ("long", 0.9, 500, 0.9))
    ]
    baseline = select_candidate(plan, candidates, policy)
    alternate = select_candidate(plan, candidates, policy, preference_ranker=ranker)
    ids = store.replay.capture(
        plan, baseline, tenant=TENANT, request_id=request, consent=consent, task_family=family
    )
    return plan, baseline, alternate, policy, ids


def test_control_drill_compares_actual_prm_and_blocks_bad_evidence():
    result = run_drill()
    assert result["gate_passed"]
    assert result["clean"]["baseline_correct"] == 20
    assert result["clean"]["shadow_correct"] == 40
    assert not result["clean"]["production_approval"]
    assert "insufficient_reviewed_coverage" in result["missing_review"]["failure_reasons"]
    assert "unsafe_shadow_selection" in result["regression"]["failure_reasons"]


def test_signed_private_comparisons_preserve_prm_and_are_immutable(live, ranker):
    store, cohort, policy = live
    rows = store.comparisons(cohort.study.study_id, TENANT)
    assert len(rows) == 40
    assert sum(row.baseline_event != row.shadow_event for row in rows) == 20
    assert any(candidate.process_reward is not None for row in rows for candidate in row.candidates)
    payload = rows[0].model_dump_json()
    assert TENANT not in payload and "PRIVATE" not in payload and "candidate-1" not in payload
    assert cohort.study == store.register("synthetic-shadow-study", TENANT, ranker, policy)
    with pytest.raises(ValueError, match="immutable"):
        store.register(
            "synthetic-shadow-study", TENANT, ranker, policy.model_copy(update={"min_consensus": 0.7})
        )
    with pytest.raises(ValueError, match="tenant"):
        store.study(cohort.study.study_id, "other")
    with pytest.raises(ValueError, match="tenant"):
        store.register("cross-tenant", "other", ranker, policy)


def test_capture_is_before_labels_idempotent_and_consent_gated(live, ranker):
    store, cohort, _ = live
    store.replay.clock = lambda: 1100.0
    plan, baseline, alternate, policy, ids = new_pair(store, ranker)
    row = store.capture(cohort.study.study_id, TENANT, "new", plan, baseline, alternate, policy, consent=True)
    for event in ids:
        store.replay.review(TENANT, event, verdict="correct", unsafe=False, reviewer="reviewer")
    assert (
        store.capture(cohort.study.study_id, TENANT, "new", plan, baseline, alternate, policy, consent=True)
        == row
    )
    assert (
        store.capture(cohort.study.study_id, TENANT, "new", plan, baseline, alternate, policy, consent=False)
        is None
    )
    plan, baseline, alternate, policy, ids = new_pair(
        store, ranker, request="post-review", family="later-family"
    )
    for event in ids:
        store.replay.review(TENANT, event, verdict="correct", unsafe=False, reviewer="reviewer")
    with pytest.raises(ValueError, match="precede review"):
        store.capture(
            cohort.study.study_id, TENANT, "post-review", plan, baseline, alternate, policy, consent=True
        )


@pytest.mark.parametrize("mutation", ["fingerprint", "pool", "policy", "choice"])
def test_capture_rejects_tamper_mismatched_pools_and_illegal_choices(live, ranker, mutation):
    store, cohort, _ = live
    store.replay.clock = lambda: 1100.0
    plan, baseline, alternate, policy, _ = new_pair(store, ranker)
    if mutation == "fingerprint":
        alternate.receipt_fingerprint = "0" * 64
    elif mutation == "pool":
        alternate.candidate_summaries[0].process_reward = 0.1
        alternate.receipt_fingerprint = _fingerprint(_receipt_payload(alternate))
    elif mutation == "policy":
        policy.min_confidence_gain = 0.8
    else:
        alternate.selected_candidate_id = "outsider"
        alternate.receipt_fingerprint = _fingerprint(_receipt_payload(alternate))
    with pytest.raises(ValueError):
        store.capture(cohort.study.study_id, TENANT, "new", plan, baseline, alternate, policy, consent=True)


def test_late_registration_and_unobserved_capture_are_rejected(live, ranker):
    store, cohort, policy = live
    store.replay.clock = lambda: 1100.0
    plan, baseline, alternate, policy, _ = new_pair(store, ranker)
    store.replay.clock = lambda: 1200.0
    later = store.register("registered-too-late", TENANT, ranker, policy)
    with pytest.raises(ValueError, match="preceded study"):
        store.capture(later.study_id, TENANT, "new", plan, baseline, alternate, policy, consent=True)
    with pytest.raises(ValueError, match="source binding"):
        store.capture(
            cohort.study.study_id, TENANT, "never-observed", plan, baseline, alternate, policy, consent=True
        )


def test_review_queue_prioritizes_disagreement_with_an_audit_lane(tmp_path, ranker):
    store = PreferenceShadowStore(ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY))
    cohort, _ = seed_shadow(store, ranker, TENANT, review_fraction=0.5)
    queue = store.review_queue(cohort.study.study_id, TENANT, audit_percent=100)
    assert len(queue) == 20
    assert sum(row["priority"] == "disagreement" for row in queue) == 10
    assert queue[0]["priority"] == "disagreement" and queue[-1]["priority"] == "audit"
    disagreements = store.review_queue(cohort.study.study_id, TENANT, audit_percent=0)
    assert len(disagreements) == 10
    with pytest.raises(ValueError, match="audit percent"):
        store.review_queue(cohort.study.study_id, TENANT, audit_percent=101)


def test_freeze_does_not_substitute_repeat_families_or_baseline_abstentions(live, ranker):
    store, cohort, _ = live
    store.replay.clock = lambda: 1100.0
    plan, baseline, alternate, policy, _ = new_pair(store, ranker, family="shadow-family-0")
    store.capture(cohort.study.study_id, TENANT, "new", plan, baseline, alternate, policy, consent=True)
    plan, baseline, alternate, policy, _ = new_pair(store, ranker, request="abstain", grounded=False)
    store.capture(cohort.study.study_id, TENANT, "abstain", plan, baseline, alternate, policy, consent=True)
    frozen = store.freeze(cohort.study.study_id, TENANT, embargo_seconds=50)
    assert len(frozen.members) == 40
    assert frozen.exclusions["repeated_family_request"] == 1
    assert frozen.exclusions["baseline_abstention"] == 1
    assert not store.freeze(cohort.study.study_id, TENANT, embargo_seconds=500).members


def test_forward_gate_emits_bound_approval_without_auto_deployment(live, ranker, tmp_path):
    store, cohort, policy = live
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    report, approval = evaluate_shadow(cohort, store, TENANT, ranker, policy, ARTIFACT_KEY, ledger)
    assert report["gate_passed"] and report["production_approval"]
    assert approval is not None and approval.reviewed_families == 40
    path = tmp_path / "candidate-approval.json"
    write_once(path, approval.model_dump(mode="json"))
    assert (
        validate_approval(path, ranker, ARTIFACT_KEY, policy, route="rag", high_risk=False, now=1001)
        == approval
    )
    loaded_policy = ComputePolicy.model_validate_json(policy.model_dump_json())
    assert (
        validate_approval(path, ranker, ARTIFACT_KEY, loaded_policy, route="rag", high_risk=False, now=1001)
        == approval
    )
    assert evaluate_shadow(cohort, store, TENANT, ranker, policy, ARTIFACT_KEY, ledger) == (report, approval)
    with pytest.raises(ValueError, match="already exposed"):
        ledger.reserve_families("another-study", cohort.fingerprint, [cohort.members[0].task_family])


@pytest.mark.parametrize("mutation", ["expired", "route", "risk", "policy", "artifact", "signature"])
def test_approval_cannot_authorize_other_contexts(live, ranker, tmp_path, mutation):
    store, cohort, policy = live
    _, approval = evaluate_shadow(
        cohort, store, TENANT, ranker, policy, ARTIFACT_KEY, HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    )
    path = tmp_path / "approval.json"
    route, risk, now = "rag", False, 1001
    if mutation == "expired":
        now = approval.expires_at
    elif mutation == "route":
        route = "web"
    elif mutation == "risk":
        risk = True
    elif mutation == "policy":
        policy = policy.model_copy(update={"min_consensus": 0.7})
    elif mutation == "artifact":
        approval.artifact_fingerprint = "0" * 64
        approval.seal(ARTIFACT_KEY)
    else:
        approval.fingerprint = "0" * 64
    write_once(path, approval.model_dump(mode="json"))
    with pytest.raises(ValueError):
        validate_approval(path, ranker, ARTIFACT_KEY, policy, route=route, high_risk=risk, now=now)


def test_future_labels_do_not_change_frozen_evidence_or_permit_reuse(tmp_path, ranker):
    store = PreferenceShadowStore(ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY))
    cohort, policy = seed_shadow(store, ranker, TENANT, review_fraction=0.5)
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    report, approval = evaluate_shadow(cohort, store, TENANT, ranker, policy, ARTIFACT_KEY, ledger)
    assert not report["gate_passed"] and approval is None
    store.replay.clock = lambda: 1100.0
    for obs, label in store.replay.records(TENANT):
        if label is None:
            store.replay.review(
                TENANT, obs.event_id, verdict="correct", unsafe=False, reviewer="later-reviewer"
            )
    assert evaluate_shadow(cohort, store, TENANT, ranker, policy, ARTIFACT_KEY, ledger)[0] == report
    later = store.freeze(cohort.study.study_id, TENANT, embargo_seconds=50)
    with pytest.raises(ValueError, match="already exposed"):
        evaluate_shadow(later, store, TENANT, ranker, policy, ARTIFACT_KEY, ledger)


def test_deleted_sources_revoke_cached_evidence_and_cascade_private_shadows(live, ranker, tmp_path):
    store, cohort, policy = live
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    evaluate_shadow(cohort, store, TENANT, ranker, policy, ARTIFACT_KEY, ledger)
    assert store.replay.delete_tenant(TENANT) == 80
    with store.replay._db() as db:
        assert db.execute("SELECT COUNT(*) FROM preference_shadows").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM preference_shadow_studies").fetchone()[0] == 0
    with pytest.raises(ValueError):
        evaluate_shadow(cohort, store, TENANT, ranker, policy, ARTIFACT_KEY, ledger)


def test_cohort_tamper_and_old_evidence_cannot_pass(live, ranker, tmp_path):
    store, cohort, policy = live
    altered = cohort.model_copy(deep=True)
    altered.members[0].comparison.candidates[0].eligible = False
    altered.members[0].comparison.seal(KEY)
    altered.seal(KEY)
    with pytest.raises(ValueError, match="release gates"):
        altered.verify(KEY)
    store.replay.clock = lambda: 1100.0
    plan, baseline, alternate, policy, _ = new_pair(store, ranker, request="partially-reviewed")
    pair = store.capture(
        cohort.study.study_id, TENANT, "partially-reviewed", plan, baseline, alternate, policy, consent=True
    )
    store.replay.review(TENANT, pair.shadow_event, verdict="ambiguous", unsafe=True, reviewer="reviewer")
    partial = store.freeze(cohort.study.study_id, TENANT, embargo_seconds=50)
    report, approval = evaluate_shadow(
        partial,
        store,
        TENANT,
        ranker,
        policy,
        ARTIFACT_KEY,
        HoldoutLedger(tmp_path / "partial-ledger.sqlite3", KEY),
    )
    assert report["unsafe_shadow_choices"] == 1 and approval is None
    assert "unsafe_shadow_selection" in report["failure_reasons"]
    with pytest.raises(ValueError, match="outside observed history"):
        store.freeze(cohort.study.study_id, TENANT, frozen_at=1101)
    store.replay.clock = lambda: 1000.0 + 7 * 86400
    old = store.freeze(cohort.study.study_id, TENANT, embargo_seconds=50, frozen_at=1000 + 7 * 86400)
    report, approval = evaluate_shadow(
        old, store, TENANT, ranker, policy, ARTIFACT_KEY, HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    )
    assert approval is None and "stale_shadow_evidence" in report["failure_reasons"]


def test_cli_register_queue_and_evaluate_preserve_active_paths(live, ranker, tmp_path, monkeypatch):
    store, cohort, _ = live
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", KEY.decode())
    monkeypatch.setenv("PREFERENCE_RANKING_KEY", ARTIFACT_KEY.decode())
    active = tmp_path / "active-approval.json"
    write_once(active, {"owner": "do not change"})
    monkeypatch.setenv("PREFERENCE_SHADOW_APPROVAL_PATH", str(active))
    artifact, source = tmp_path / "artifact.json", tmp_path / "cohort.json"
    write_once(artifact, ranker.artifact.model_dump(mode="json"))
    assert (
        PreferenceRanker.load(artifact, ARTIFACT_KEY, TENANT).artifact.fingerprint
        == ranker.artifact.fingerprint
    )
    legacy = ranker.artifact.model_dump(mode="json")
    legacy["risk_multiplier"] = 2
    legacy["fingerprint"] = digest(
        ARTIFACT_KEY, "PreferenceArtifact", {k: v for k, v in legacy.items() if k != "fingerprint"}
    )
    legacy_path = tmp_path / "legacy-artifact.json"
    write_once(legacy_path, legacy)
    assert (
        PreferenceRanker.load(legacy_path, ARTIFACT_KEY, TENANT).artifact.fingerprint == legacy["fingerprint"]
    )
    write_once(source, cohort.model_dump(mode="json"))
    shared = [
        "--store",
        str(store.replay.path),
        "--tenant",
        TENANT,
        "--artifact",
        str(artifact),
        "--ledger",
        str(tmp_path / "ledger.sqlite3"),
    ]
    assert main([*shared, "register", "--name", "synthetic-shadow-study"]) == 0
    assert main([*shared, "queue", cohort.study.study_id]) == 0
    report, approval = tmp_path / "report.json", tmp_path / "candidate-approval.json"
    evaluate = [
        *shared,
        "evaluate",
        str(source),
        "--output",
        str(report),
        "--approval",
        str(approval),
        "--require-gate",
    ]
    assert main(evaluate) == main(evaluate) == 0
    assert approval.exists() and json.loads(active.read_text()) == {"owner": "do not change"}
    evaluate[evaluate.index("--approval") + 1] = str(active)
    assert main(evaluate) == 1
    assert main(["register", "--name", "missing-tenant"]) == 1


@pytest.mark.asyncio
async def test_runtime_shadow_never_changes_answer_or_adds_model_calls(tmp_path, ranker, monkeypatch):
    research = importlib.import_module("agent.research_assistant")
    replay = ExecutionReplayStore(tmp_path / "runtime.sqlite3", KEY)
    shadow = PreferenceShadowStore(replay)
    policy = research._adaptive_compute_policy()
    study = shadow.register("runtime-shadow", TENANT, ranker, policy)
    for name, value in {
        "PREFERENCE_SHADOW_ENABLED": True,
        "PREFERENCE_RANKING_ENABLED": False,
        "PREFERENCE_SHADOW_STUDY_ID": study.study_id,
        "EXECUTION_REPLAY_ENABLED": True,
        "EXECUTION_REPLAY_PATH": replay.path,
        "EXECUTION_REPLAY_KEY": KEY,
        "PREFERENCE_RANKING_KEY": ARTIFACT_KEY,
        "ADAPTIVE_COMPUTE_INTEGRITY_KEY": None,
        "UNCERTAINTY_CALIBRATION_ENABLED": False,
    }.items():
        monkeypatch.setattr(research, name, value)
    monkeypatch.setattr(research.PreferenceRanker, "load", lambda *args: ranker)
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
        policy,
    )
    state = {
        "route": "web",
        "query": "How is AgentForge orchestrated?",
        "web_notes": "Link: https://example.com/architecture\nSnippet: AgentForge uses LangGraph for agent orchestration.",
        "adaptive_compute_plan": plan.model_dump(mode="json"),
    }
    config = {
        "configurable": {
            "user_id": TENANT,
            "execution_replay_request_id": "runtime-request",
            "execution_replay_consent": True,
            "execution_replay_task_family": "runtime-family",
        }
    }
    result = await research.adaptive_deliberation_agent(state, config)
    assert llm.await_count == 2
    assert result["final_response"] == answer
    assert result["adaptive_compute_receipt"]["preference_ranking"] is None
    assert result["preference_shadow_receipt"]["status"] == "captured"
    assert len(shadow.comparisons(study.study_id, TENANT)) == 1
    config["configurable"]["execution_replay_consent"] = False
    config["configurable"]["execution_replay_request_id"] = "no-consent"
    result = await research.adaptive_deliberation_agent(state, config)
    assert result["preference_shadow_receipt"] == {}
    assert len(shadow.comparisons(study.study_id, TENANT)) == 1
    monkeypatch.setattr(research, "PREFERENCE_RANKING_ENABLED", True)
    monkeypatch.setattr(research, "PREFERENCE_RANKING_REQUIRE_SHADOW_APPROVAL", True)
    monkeypatch.setattr(research, "PREFERENCE_SHADOW_APPROVAL_PATH", tmp_path / "missing-approval.json")
    result = await research.adaptive_deliberation_agent(state, config)
    assert (
        result["adaptive_compute_receipt"]["preference_ranking"]["reason"] == "preference_ranker_unavailable"
    )
    assert result["final_response"] == answer
