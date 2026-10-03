import importlib
import json
import math
import sqlite3
from contextlib import closing
from unittest.mock import AsyncMock

import pytest

from agent.adaptive_compute import ComputeSignals, plan_compute
from agent.execution_replay import ExecutionReplayStore
from agent.preference_deployment import PreferenceDeploymentStore, SentinelPolicy, sequential_trace
from evals.preference_deployment_evaluation import main, run_drill, seed_control, serve_control


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp("deployment-control")
    return seed_control(root)


@pytest.fixture
def live(tmp_path, baseline):
    original, state, ranker, policy, tenant, _ = baseline
    path = tmp_path / "replay.sqlite3"
    with closing(sqlite3.connect(original.replay.path)) as source, closing(sqlite3.connect(path)) as target:
        source.backup(target)
    clock = [1110.0]
    store = PreferenceDeploymentStore(
        ExecutionReplayStore(path, original.replay.key, clock=lambda: clock[0]), original.artifact_key
    )
    return store, state, ranker, policy, tenant, clock


def test_e_process_preserves_early_crossings_and_allocates_scope_alpha():
    policy = SentinelPolicy()
    assert not sequential_trace([0] * 100, policy, 1)["crossed"]
    result = sequential_trace([1] * 7 + [0] * 100, policy, 1)
    assert result["crossed"] and result["log_e"] < 0
    assert sequential_trace([], policy, 14)["log_threshold"] == pytest.approx(math.log(14 / 0.05))
    # The conditional mean of each bet is <=1 under any error probability <=p.
    for probability in (0, 0.1, policy.maximum_error):
        expectation = probability * policy.alternative_error / policy.maximum_error + (1 - probability) * (
            1 - policy.alternative_error
        ) / (1 - policy.maximum_error)
        assert expectation <= 1 + 1e-12
    with pytest.raises(ValueError):
        sequential_trace([2], policy, 1)
    with pytest.raises(ValueError):
        SentinelPolicy(maximum_error=0.6)


def test_registry_is_tenant_bound_signed_and_owner_revision_guarded(live):
    store, state, ranker, policy, tenant, clock = live
    admitted, current, report = store.admit(tenant, policy, route="rag", high_risk=False)
    assert admitted.artifact == ranker.artifact and current == state and report["reason"] == "healthy"
    assert store.state("other-tenant") is None
    deployment = store.deployment(state)
    payload = deployment.model_dump_json()
    assert tenant not in payload and "fixture-owner" not in payload and "PRIVATE" not in payload
    with pytest.raises(ValueError, match="stale"):
        store.revoke(tenant, expected_revision=0)
    clock[0] = 1120.0
    revoked = store.revoke(tenant, expected_revision=1)
    assert revoked.revision == 2 and revoked.previous == state.fingerprint
    assert store.revoke(tenant, expected_revision=2) == revoked
    with pytest.raises(ValueError, match="already exposed"):
        store.activate(
            tenant,
            ranker,
            deployment.cohort,
            deployment.approval,
            policy,
            owner="owner",
            expected_revision=2,
            sentinel=SentinelPolicy(alpha=0.01),
        )


@pytest.mark.parametrize("mismatch", ["route", "risk", "policy"])
def test_request_scope_mismatch_falls_back_without_global_revocation(live, mismatch):
    store, state, _, policy, tenant, _ = live
    route, risk = ("web" if mismatch == "route" else "rag"), mismatch == "risk"
    if mismatch == "policy":
        policy = policy.model_copy(update={"min_consensus": 0.8})
    with pytest.raises(ValueError, match="support"):
        store.admit(tenant, policy, route=route, high_risk=risk)
    assert store.state(tenant) == state


@pytest.mark.parametrize("source", ["approval", "served"])
def test_deleted_evidence_revokes_instead_of_erasing_exposure(live, source):
    store, state, ranker, policy, tenant, _ = live
    if source == "approval":
        event = store.deployment(state).cohort.members[0].comparison.baseline_event
    else:
        _, _, _, event = serve_control(store, state, ranker, tenant, 0)
    with store.replay._db() as db:
        db.execute("DELETE FROM observations WHERE event_id=?", (event,))
    with pytest.raises(ValueError):
        store.admit(tenant, policy, route="rag", high_risk=False)
    assert store.state(tenant).reason == "source_or_lease_invalid"


def test_expired_lease_revokes(live):
    store, state, _, policy, tenant, clock = live
    clock[0] = store.deployment(state).approval.expires_at
    with pytest.raises(ValueError, match="expired"):
        store.admit(tenant, policy, route="rag", high_risk=False)
    assert store.state(tenant).status == "revoked"


@pytest.mark.parametrize("failure", ["consent", "family", "revoked"])
def test_failed_serving_binding_rolls_back_observations_and_family(live, failure):
    store, state, ranker, _, tenant, _ = live
    count = len(store.replay.records(tenant))
    if failure == "revoked":
        store.revoke(tenant, expected_revision=state.revision)
    with pytest.raises(ValueError):
        serve_control(
            store,
            state,
            ranker,
            tenant,
            0,
            family="" if failure == "family" else None,
            consent=failure != "consent",
        )
    assert len(store.replay.records(tenant)) == count
    with store.replay._db() as db:
        assert db.execute("SELECT COUNT(*) FROM preference_served").fetchone()[0] == 0


def test_serving_binding_is_immutable_and_retries_do_not_add_samples(live):
    store, state, ranker, _, tenant, clock = live
    _, _, ids, event = serve_control(store, state, ranker, tenant, 0)
    store.replay.review(tenant, event, verdict="correct", unsafe=False, reviewer="reviewer")
    clock[0] = 1120.0
    assert serve_control(store, state, ranker, tenant, 0)[2] == ids
    report = store.monitor(tenant, store.deployment(state))
    assert report["captured_choices"] == 1 and report["fresh_families"] == 1
    assert report["scopes"]["rag:0"]["samples"] == 0


def test_safe_complete_feedback_stays_active_after_maturity(live):
    store, state, ranker, policy, tenant, clock = live
    for index in range(20):
        clock[0] = 1110.0 + index
        _, _, _, event = serve_control(store, state, ranker, tenant, index)
        store.replay.review(tenant, event, verdict="correct", unsafe=False, reviewer="reviewer")
    clock[0] = 1200.0
    _, _, report = store.admit(tenant, policy, route="rag", high_risk=False)
    assert report["scopes"]["rag:0"]["samples"] == 20
    assert report["scopes"]["rag:0"]["errors"] == 0


def test_bad_feedback_rolls_back_without_repeated_look_reset(live):
    store, state, ranker, policy, tenant, clock = live
    for index in range(7):
        clock[0] = 1110.0 + index
        _, _, _, event = serve_control(store, state, ranker, tenant, index)
        store.replay.review(tenant, event, verdict="incorrect", unsafe=False, reviewer="reviewer")
    report = store.monitor(tenant, store.deployment(state))
    assert report["scopes"]["rag:0"]["samples"] == 0
    clock[0] = 1200.0
    with pytest.raises(ValueError, match="sequential_error"):
        store.admit(tenant, policy, route="rag", high_risk=False)
    assert store.state(tenant).reason == "sequential_error"
    with pytest.raises(ValueError, match="no active"):
        store.admit(tenant, policy, route="rag", high_risk=False)


def test_late_labels_cannot_repair_missing_review_errors(live):
    store, state, ranker, _, tenant, clock = live
    events = []
    for index in range(7):
        clock[0] = 1110.0 + index
        events.append(serve_control(store, state, ranker, tenant, index)[3])
    clock[0] = 1200.0
    before = store.monitor(tenant, store.deployment(state))
    for event in events:
        store.replay.review(tenant, event, verdict="correct", unsafe=False, reviewer="late-reviewer")
    after = store.monitor(tenant, store.deployment(state))
    assert before == after and after["reason"] == "sequential_error"


def test_repeat_family_does_not_replace_first_outcome_but_unsafe_always_blocks(live):
    store, state, ranker, policy, tenant, clock = live
    first = serve_control(store, state, ranker, tenant, 0)[3]
    store.replay.review(tenant, first, verdict="correct", unsafe=False, reviewer="reviewer")
    clock[0] = 1111.0
    repeat = serve_control(store, state, ranker, tenant, 1, family="serve-family-0")[3]
    store.replay.review(tenant, repeat, verdict="incorrect", unsafe=True, reviewer="reviewer")
    report = store.monitor(tenant, store.deployment(state))
    assert report["fresh_families"] == 1 and report["reason"] == "unsafe_outcome"
    with pytest.raises(ValueError, match="unsafe_outcome"):
        store.admit(tenant, policy, route="rag", high_risk=False)


def test_selected_candidate_timestamp_need_not_be_first_pool_timestamp(live):
    store, state, ranker, _, tenant, clock = live

    def advancing_clock():
        clock[0] += 0.01
        return clock[0]

    store.replay.clock = advancing_clock
    _, receipt, _, event = serve_control(store, state, ranker, tenant, 0)
    assert receipt.selected_candidate_id == "short"
    store.replay.review(tenant, event, verdict="correct", unsafe=False, reviewer="reviewer")
    clock[0] = 1200.0
    report = store.monitor(tenant, store.deployment(state))
    assert report["fresh_families"] == 1 and report["scopes"]["rag:0"]["samples"] == 1


@pytest.mark.parametrize("race", ["source", "feedback"])
def test_source_or_feedback_race_is_rechecked_inside_serving_transaction(live, race):
    store, state, ranker, policy, tenant, _ = live
    store.admit(tenant, policy, route="rag", high_risk=False)
    if race == "source":
        event = store.deployment(state).cohort.members[0].comparison.baseline_event
        with store.replay._db() as db:
            db.execute("DELETE FROM observations WHERE event_id=?", (event,))
    else:
        event = serve_control(store, state, ranker, tenant, 0)[3]
        store.replay.review(tenant, event, verdict="correct", unsafe=True, reviewer="reviewer")
    count = len(store.replay.records(tenant))
    with pytest.raises(ValueError):
        serve_control(store, state, ranker, tenant, 1)
    assert len(store.replay.records(tenant)) == count


def test_audit_and_serving_tampering_are_detected(live):
    store, state, ranker, policy, tenant, _ = live
    serve_control(store, state, ranker, tenant, 0)
    with store.replay._db() as db:
        event, payload = db.execute("SELECT event_id, payload FROM preference_served").fetchone()
        changed = json.loads(payload)
        changed["scope"] = "web:1"
        db.execute("UPDATE preference_served SET payload=? WHERE event_id=?", (json.dumps(changed), event))
    with pytest.raises(ValueError):
        store.admit(tenant, policy, route="rag", high_risk=False)
    with store.replay._db() as db:
        db.execute("DELETE FROM preference_deployment_audit WHERE revision=1")
    with pytest.raises(ValueError, match="integrity"):
        store.state(tenant)


def test_tenant_deletion_removes_private_registry_and_other_tenants_are_untouched(live):
    store, state, ranker, _, tenant, _ = live
    serve_control(store, state, ranker, tenant, 0)
    assert store.replay.delete_tenant("other") == 0
    assert store.state(tenant) == state
    assert store.replay.delete_tenant(tenant) > 0
    assert store.state(tenant) is None
    with store.replay._db() as db:
        for table in (
            "preference_deployments",
            "preference_deployment_states",
            "preference_deployment_audit",
            "preference_served",
        ):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_control_drill_and_cli_protect_private_stores(tmp_path):
    report = run_drill()
    assert report["gate_passed"] and not report["production_evidence"]
    assert report["scenarios"]["quality_regression"]["reason"] == "sequential_error"
    assert report["scenarios"]["unsafe"]["reason"] == "unsafe_outcome"
    path = tmp_path / "replay.sqlite3"
    assert main(["--store", str(path), "drill", "--output", str(path)]) == 1
    assert not path.exists()


def test_coverage_hold_is_separate_from_e_process(live):
    store, state, ranker, _, tenant, clock = live
    deployment = store.deployment(state)
    deployment.sentinel_policy = SentinelPolicy(
        maximum_error=0.8, alternative_error=0.9, review_deadline_seconds=60.0
    )
    deployment.seal(store.replay.key)
    # Exercise the pinned-policy monitor directly; a live policy is never editable.
    for index in range(20):
        clock[0] = 1110.0 + index
        event = serve_control(store, state, ranker, tenant, index)[3]
        if index % 3 != 0:
            store.replay.review(tenant, event, verdict="correct", unsafe=False, reviewer="reviewer")
    clock[0] = 1200.0
    report = store.monitor(tenant, deployment)
    assert report["reason"] == "review_coverage"
    assert not report["scopes"]["rag:0"]["crossed"]


def test_clock_regression_and_unsafe_ambiguous_label_cannot_hide(live):
    store, state, ranker, policy, tenant, clock = live
    event = serve_control(store, state, ranker, tenant, 0)[3]
    store.replay.review(tenant, event, verdict="ambiguous", unsafe=True, reviewer="reviewer")
    with pytest.raises(ValueError, match="unsafe_outcome"):
        store.admit(tenant, policy, route="rag", high_risk=False)
    clock[0] = 1000.0
    with pytest.raises(ValueError, match="backwards"):
        store._change(
            None, store.state(tenant), state.tenant, state.deployment_id, "active", "owner_activation"
        )


def test_cli_status_check_activation_validation_and_revocation(live, baseline, tmp_path, monkeypatch):
    store, state, ranker, policy, tenant, _ = live
    evaluation = importlib.import_module("evals.preference_deployment_evaluation")
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", store.replay.key.decode())
    monkeypatch.setenv("PREFERENCE_RANKING_KEY", store.artifact_key.decode())
    monkeypatch.setattr(evaluation, "ExecutionReplayStore", lambda *args, **kwargs: store.replay)
    shared = ["--store", str(store.replay.path), "--tenant", tenant]
    assert main([*shared, "status"]) == 0
    assert main([*shared, "check", "--route", "rag"]) == 0
    deployment = store.deployment(state)
    paths = {}
    for name, value in (
        ("artifact", ranker.artifact),
        ("cohort", deployment.cohort),
        ("approval", deployment.approval),
        ("compute-policy", policy),
        ("sentinel-policy", deployment.sentinel_policy),
    ):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(value.model_dump_json(), encoding="utf-8")
    activation = [
        *shared,
        "--ledger",
        str(baseline[0].replay.path.parent / "ledger.sqlite3"),
        "activate",
        "--owner",
        "fixture-owner",
        "--expected-revision",
        "0",
    ]
    for name, path in paths.items():
        activation.extend([f"--{name}", str(path)])
    assert main(activation) == 1  # valid evidence, stale revision cannot replace active release
    assert main([*shared, "revoke", "--expected-revision", "1"]) == 0
    assert main([*shared, "check", "--route", "rag"]) == 1


@pytest.mark.asyncio
async def test_runtime_registry_capture_racing_revocation_and_consent(live, monkeypatch):
    store, state, _, policy, tenant, _ = live
    research = importlib.import_module("agent.research_assistant")
    for name, value in {
        "PREFERENCE_RANKING_ENABLED": True,
        "PREFERENCE_DEPLOYMENT_ENABLED": True,
        "PREFERENCE_SHADOW_ENABLED": False,
        "EXECUTION_REPLAY_ENABLED": True,
        "EXECUTION_REPLAY_PATH": store.replay.path,
        "EXECUTION_REPLAY_KEY": store.replay.key,
        "PREFERENCE_RANKING_KEY": store.artifact_key,
        "ADAPTIVE_COMPUTE_INTEGRITY_KEY": None,
        "UNCERTAINTY_CALIBRATION_ENABLED": False,
    }.items():
        monkeypatch.setattr(research, name, value)
    monkeypatch.setattr(research, "ExecutionReplayStore", lambda *args, **kwargs: store.replay)
    monkeypatch.setattr(research, "_adaptive_compute_policy", lambda: policy)
    answer = "AgentForge uses LangGraph for agent orchestration [source](https://example.com/architecture)."
    llm = AsyncMock(return_value=answer)
    monkeypatch.setattr(research, "_call_llm", llm)
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
    input_state = {
        "route": "rag",
        "query": "How is AgentForge orchestrated?",
        "rag_notes": "Link: https://example.com/architecture\nSnippet: AgentForge uses LangGraph for agent orchestration.",
        "adaptive_compute_plan": plan.model_dump(mode="json"),
    }
    config = {
        "configurable": {
            "user_id": tenant,
            "execution_replay_request_id": "runtime-1",
            "execution_replay_consent": True,
            "execution_replay_task_family": "runtime-family-1",
        }
    }
    result = await research.adaptive_deliberation_agent(input_state, config)
    assert result["preference_deployment_receipt"]["status"] == "captured"
    assert result["final_response"] == answer and llm.await_count == 2
    assert store.monitor(tenant, store.deployment(state))["captured_choices"] == 1
    config["configurable"].update(execution_replay_request_id="no-consent", execution_replay_consent=False)
    result = await research.adaptive_deliberation_agent(input_state, config)
    assert result["preference_deployment_receipt"]["status"] == "baseline_fallback"
    assert result["execution_replay_event_ids"] == [] and llm.await_count == 4
    original_capture = PreferenceDeploymentStore.capture

    def raced_capture(current, *args, **kwargs):
        current.revoke(tenant, expected_revision=1)
        return original_capture(current, *args, **kwargs)

    monkeypatch.setattr(PreferenceDeploymentStore, "capture", raced_capture)
    config["configurable"].update(
        execution_replay_request_id="runtime-race",
        execution_replay_consent=True,
        execution_replay_task_family="runtime-family-race",
    )
    result = await research.adaptive_deliberation_agent(input_state, config)
    assert result["preference_deployment_receipt"]["status"] == "baseline_fallback"
    assert (
        result["adaptive_compute_receipt"]["preference_ranking"]["reason"] == "preference_ranker_unavailable"
    )
    assert result["final_response"] == answer and llm.await_count == 6
    assert store.monitor(tenant, store.deployment(state))["captured_choices"] == 1
