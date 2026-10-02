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
from agent.process_reward import ProcessRewardScorer, ProcessStep, step_features, train_process_reward_model
from agent.process_supervision import ProcessSupervisionStore
from agent.prospective_validation import HoldoutLedger
from evals.process_supervision_evaluation import evaluate_process, main, run_drill, seed_process

KEY = b"unit-test-process-replay-key-32-bytes"
MODEL_KEY = b"unit-test-process-model-key-32-bytes"
TENANT = "private-process-tenant"


@pytest.fixture(scope="module")
def seed(tmp_path_factory):
    replay = ExecutionReplayStore(tmp_path_factory.mktemp("process-seed") / "replay.sqlite3", KEY)
    store = ProcessSupervisionStore(replay)
    # Authored unit-test data marked runtime to exercise the export path; not real traffic evidence.
    return store, seed_process(store, TENANT, origin="runtime")


@pytest.fixture
def live(tmp_path, seed):
    original, cohort = seed
    path = tmp_path / "replay.sqlite3"
    with closing(sqlite3.connect(original.replay.path)) as source, closing(sqlite3.connect(path)) as target:
        source.backup(target)
    replay = ExecutionReplayStore(path, KEY, clock=lambda: 650.0)
    return ProcessSupervisionStore(replay), cohort


def new_pool(store, *, request="new", family="new-family", timestamp=700):
    store.replay.clock = lambda: float(timestamp)
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
            answer="PRIVATE original answer",
            confidence=conf,
            grounded=True,
            conformal_decision="release",
            claim_keys=["PRIVATE evidence"],
            token_count=100,
            latency_ms=10,
        )
        for name, conf in (("one", 0.9), ("two", 0.8))
    ]
    ids = store.replay.capture(
        plan,
        select_candidate(plan, candidates, policy),
        tenant=TENANT,
        request_id=request,
        task_family=family,
        consent=True,
    )
    steps = [
        [
            ProcessStep(step_id="PRIVATE raw identifier", kind="retrieve", has_evidence=True),
            ProcessStep(step_id="answer", kind="answer", confidence=conf),
        ]
        for conf in (0.9, 0.8)
    ]
    return ids, steps


def test_synthetic_ablation_holds_sparse_and_shifted_reviews():
    result = run_drill()
    assert result["gate_passed"] and not result["production_activation"]
    clean = result["reports"]["clean"]
    assert clean["step_brier"] < clean["constant_baseline_brier"]
    assert clean["step_brier"] < clean["weak_outcome_credit_brier"]
    assert clean["worst_case_step_brier"] == pytest.approx(clean["step_brier"])
    assert clean["human_labeled_steps"] == 0 and clean["explicit_training_steps"] == 160
    assert not clean["candidate_ready_for_review"]
    sparse = result["reports"]["missing_review"]
    assert sparse["test_review_coverage"] == 0.5
    assert sparse["worst_case_step_brier"] > sparse["step_brier"]
    assert not sparse["gate_passed"] and not result["reports"]["shifted_labels"]["gate_passed"]


def test_explicit_training_never_imputes_terminal_targets(seed):
    _, cohort = seed
    traces = cohort.traces()
    first = train_process_reward_model(traces, epochs=5, explicit_steps_only=True)
    changed = [
        trace.model_copy(update={"outcome_quality": 1 - trace.outcome_quality, "safe": not trace.safe})
        for trace in traces
    ]
    second = train_process_reward_model(changed, epochs=5, explicit_steps_only=True)
    assert first.weights == second.weights and first.stopping_threshold == second.stopping_threshold == 0.35
    missing = [
        trace.model_copy(
            update={"steps": [step.model_copy(update={"step_label": None}) for step in trace.steps]}
        )
        for trace in traces
    ]
    with pytest.raises(ValueError, match="labelled"):
        train_process_reward_model(missing, explicit_steps_only=True)
    assert train_process_reward_model(missing, epochs=5).training_examples == 160


def test_feature_and_scoring_invariance_to_targets(seed):
    _, cohort = seed
    traces = cohort.traces()
    trace = next(trace for trace in traces if trace.split == "train" and trace.steps[2].step_label == 0)
    assert trace.steps[0].step_label == 1  # Failed answer does not make retrieval incorrect.
    scorer = ProcessRewardScorer(train_process_reward_model(traces, epochs=5, explicit_steps_only=True))
    changed = [step.model_copy(update={"step_label": 1 - step.step_label}) for step in trace.steps]
    assert scorer.step_probabilities(trace.steps) == scorer.step_probabilities(changed)
    for index, step in enumerate(trace.steps):
        assert step_features(step, index, len(trace.steps)) == step_features(
            changed[index], index, len(changed)
        )


def test_capture_strips_content_requires_consent_and_is_immutable(live):
    store, _ = live
    ids, steps = new_pool(store)
    assert store.capture(TENANT, ids[0], steps[0], consent=False) is None
    with pytest.raises(ValueError, match="original replay"):
        store.capture("other", ids[0], steps[0], consent=True)
    with pytest.raises(ValueError, match="targets"):
        store.capture(TENANT, ids[0], [steps[0][0].model_copy(update={"step_label": 1})], consent=True)
    snap = store.capture(TENANT, ids[0], steps[0], consent=True)
    payload = snap.model_dump_json()
    assert "PRIVATE" not in payload and TENANT not in payload and "new-family" not in payload
    assert store.capture(TENANT, ids[0], steps[0], consent=True) == snap
    mutated = [steps[0][0].model_copy(update={"has_evidence": False}), steps[0][1]]
    with pytest.raises(ValueError, match="immutable"):
        store.capture(TENANT, ids[0], mutated, consent=True)
    with pytest.raises(ValueError, match="source binding"):
        store.capture(TENANT, ids[1], steps[0], consent=True)


@pytest.mark.parametrize("review_type", ["step", "terminal"])
def test_whole_pool_precedes_reviews_and_partial_pool_is_rejected(live, review_type):
    store, _ = live
    ids, steps = new_pool(store)
    store.capture(TENANT, ids[0], steps[0], consent=True)
    with pytest.raises(ValueError, match="incomplete"):
        store.freeze(TENANT, train_cutoff=150, validation_cutoff=350, embargo_seconds=50, simulation=True)
    if review_type == "step":
        store.review(TENANT, ids[0], 0, "correct", "reviewer")
    else:
        store.replay.review(TENANT, ids[0], verdict="correct", unsafe=False, reviewer="reviewer")
    with pytest.raises(ValueError, match="precede any review"):
        store.capture(TENANT, ids[1], steps[1], consent=True)


def test_reviews_are_immutable_tenant_scoped_and_index_bound(live):
    store, _ = live
    ids, steps = new_pool(store)
    for event, features in zip(ids, steps, strict=True):
        store.capture(TENANT, event, features, consent=True)
    assert len(store.queue(TENANT)) == 2 and store.queue("other") == []
    label = store.review(TENANT, ids[0], 0, "ambiguous", "private reviewer")
    assert "private reviewer" not in label.model_dump_json()
    assert store.review(TENANT, ids[0], 0, "ambiguous", "private reviewer") == label
    for verdict, reviewer in (("correct", "private reviewer"), ("ambiguous", "other")):
        with pytest.raises(ValueError, match="immutable"):
            store.review(TENANT, ids[0], 0, verdict, reviewer)
    with pytest.raises(ValueError, match="index"):
        store.review(TENANT, ids[0], 10, "correct", "reviewer")
    with pytest.raises(ValueError, match="identity"):
        store.review(TENANT, ids[0], 1, "correct", " ")
    assert next(row for row in store.queue(TENANT) if row["event_id"] == ids[0])["step_indexes"] == [1]
    store.replay.clock = lambda: 699.0
    with pytest.raises(ValueError, match="precedes"):
        store.review(TENANT, ids[0], 1, "correct", "reviewer")


@pytest.mark.parametrize(
    "timestamp,family,reason",
    [(175, "embargo-family", "embargo"), (700, "process-family-train-0", "repeat_family")],
)
def test_embargo_and_earliest_family_cannot_move_to_later_fold(live, timestamp, family, reason):
    store, cohort = live
    ids, steps = new_pool(store, family=family, timestamp=timestamp)
    for event, features in zip(ids, steps, strict=True):
        store.capture(TENANT, event, features, consent=True)
    store.replay.clock = lambda: 750.0
    frozen = store.freeze(
        TENANT, train_cutoff=150, validation_cutoff=350, embargo_seconds=50, simulation=True
    )
    assert frozen.exclusions[reason] == 1 and len(frozen.members) == len(cohort.members)


def test_late_and_ambiguous_labels_remain_unlabelled_in_frozen_cohort(live):
    store, _ = live
    ids, steps = new_pool(store, timestamp=140)
    for event, features in zip(ids, steps, strict=True):
        store.capture(TENANT, event, features, consent=True)
    store.review(TENANT, ids[0], 0, "ambiguous", "reviewer")
    store.replay.clock = lambda: 151.0
    store.review(TENANT, ids[0], 1, "incorrect", "reviewer")
    store.replay.clock = lambda: 650.0
    frozen = store.freeze(
        TENANT, train_cutoff=150, validation_cutoff=350, embargo_seconds=50, simulation=True
    )
    trace = next(trace for trace in frozen.traces() if trace.trace_id == ids[0])
    assert all(step.step_label is None for step in trace.steps)
    assert trace.outcome_quality == 0.5


def test_cached_retries_and_shared_family_exposure_are_enforced(live, tmp_path):
    store, cohort = live
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    first = evaluate_process(cohort, store, TENANT, MODEL_KEY, ledger)
    assert first == evaluate_process(cohort, store, TENANT, MODEL_KEY, ledger)
    first[1].verify(MODEL_KEY)
    with pytest.raises(ValueError):
        first[1].model_copy(update={"explicit_training_steps": 1}).verify(MODEL_KEY)
    with pytest.raises(ValueError):
        ledger.reserve_families(
            "another-study",
            cohort.fingerprint,
            [member.snapshot.family.task_family for member in cohort.members if member.split == "test"],
        )
    with pytest.raises(ValueError, match="independent"):
        evaluate_process(cohort, store, TENANT, KEY, ledger)


@pytest.mark.parametrize("mutation", ["delete", "tamper"])
def test_deleted_or_tampered_original_source_blocks_cached_reuse(live, tmp_path, mutation):
    store, cohort = live
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    evaluate_process(cohort, store, TENANT, MODEL_KEY, ledger)
    event = cohort.members[0].snapshot.source.event_id
    with store.replay._db() as db:
        if mutation == "delete":
            db.execute("DELETE FROM observations WHERE event_id=?", (event,))
        else:
            db.execute(
                "UPDATE observations SET payload=json_set(payload, '$.confidence', 0.1) WHERE event_id=?",
                (event,),
            )
    with pytest.raises(ValueError):
        evaluate_process(cohort, store, TENANT, MODEL_KEY, ledger)


def test_tenant_deletion_removes_process_sidecars_not_holdout_exposure(live, tmp_path):
    store, cohort = live
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    evaluate_process(cohort, store, TENANT, MODEL_KEY, ledger)
    assert store.replay.delete_tenant(TENANT) == 100
    with store.replay._db() as db:
        assert db.execute("SELECT count(*) FROM process_snapshots").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM process_step_reviews").fetchone()[0] == 0
    with pytest.raises(ValueError):
        ledger.reserve_families(
            "new-study",
            cohort.fingerprint,
            [member.snapshot.family.task_family for member in cohort.members if member.split == "test"],
        )


def test_cli_rejects_active_and_storage_output_aliases(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    monkeypatch.setenv("PROCESS_REWARD_MODEL_PATH", str(active))
    assert main(["drill", "--output", str(active)]) == 1 and not active.exists()
    database = tmp_path / "replay.sqlite3"
    assert main(["--store", str(database), "drill", "--output", str(database) + "-wal"]) == 1
    assert main(["queue"]) == 1


def test_cli_freeze_evaluate_and_cached_export_never_overwrite_active_model(live, tmp_path, monkeypatch):
    store, _ = live
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", KEY.decode())
    monkeypatch.setenv("PROCESS_SUPERVISION_MODEL_KEY", MODEL_KEY.decode())
    active = tmp_path / "active.json"
    active.write_text('{"owner": "preserve"}', encoding="utf-8")
    monkeypatch.setenv("PROCESS_REWARD_MODEL_PATH", str(active))
    shared = [
        "--store",
        str(store.replay.path),
        "--ledger",
        str(tmp_path / "ledger.sqlite3"),
        "--tenant",
        TENANT,
    ]
    source, report, candidate = [tmp_path / name for name in ("cohort.json", "report.json", "candidate.json")]
    assert main([*shared, "queue"]) == 0
    assert (
        main(
            [
                *shared,
                "freeze",
                "--train-cutoff",
                "150",
                "--validation-cutoff",
                "350",
                "--embargo-seconds",
                "50",
                "--output",
                str(source),
            ]
        )
        == 0
    )
    args = [
        *shared,
        "evaluate",
        str(source),
        "--candidate",
        str(candidate),
        "--output",
        str(report),
        "--require-gate",
    ]
    assert main(args) == main(args) == 0
    assert json.loads(report.read_text())["candidate_ready_for_review"]
    assert json.loads(candidate.read_text())["artifact"]["human_labeled_steps"] == 160
    assert json.loads(active.read_text()) == {"owner": "preserve"}
    args[args.index("--candidate") + 1] = str(active)
    assert main(args) == 1 and json.loads(active.read_text()) == {"owner": "preserve"}
    args[args.index("--candidate") + 1] = str(source)
    assert main(args) == 1


def test_partial_explicit_training_excludes_unknown_steps(seed):
    _, cohort = seed
    traces = cohort.traces()
    partial = [
        trace.model_copy(
            update={
                "steps": [
                    step.model_copy(update={"step_label": None}) if index < 2 else step
                    for index, step in enumerate(trace.steps)
                ]
            }
        )
        for trace in traces
    ]
    trained = train_process_reward_model(partial, epochs=5, explicit_steps_only=True)
    assert trained.training_examples == trained.human_labeled_steps == 80


def test_missing_family_and_future_freeze_are_rejected(live):
    store, _ = live
    ids, steps = new_pool(store, family=None)
    with pytest.raises(ValueError, match="source binding"):
        store.capture(TENANT, ids[0], steps[0], consent=True)
    with pytest.raises(ValueError, match="future"):
        store.freeze(
            TENANT,
            train_cutoff=150,
            validation_cutoff=350,
            embargo_seconds=50,
            frozen_at=800,
            simulation=True,
        )


def test_snapshot_and_annotation_tampering_fail_closed(live):
    store, cohort = live
    event = cohort.members[0].snapshot.source.event_id
    with store.replay._db() as db:
        db.execute(
            "UPDATE process_step_reviews SET payload=json_set(payload, '$.verdict', 'incorrect') WHERE event_id=? AND step_index=0",
            (event,),
        )
    with pytest.raises(ValueError, match="integrity"):
        store.rows(TENANT)


def test_snapshot_after_fold_cutoff_cannot_enter_training(live):
    store, cohort = live
    ids, steps = new_pool(store, timestamp=140)
    store.replay.clock = lambda: 151.0
    for event, features in zip(ids, steps, strict=True):
        store.capture(TENANT, event, features, consent=True)
    store.replay.clock = lambda: 650.0
    frozen = store.freeze(
        TENANT, train_cutoff=150, validation_cutoff=350, embargo_seconds=50, simulation=True
    )
    assert frozen.exclusions["snapshot_after_fold_cutoff"] == 1
    assert len(frozen.members) == len(cohort.members)


def test_unreviewed_training_families_cannot_inflate_support(live, tmp_path):
    store, cohort = live
    family = next(member.snapshot.family.task_family for member in cohort.members if member.split == "train")
    events = [
        member.snapshot.source.event_id
        for member in cohort.members
        if member.snapshot.family.task_family == family
    ]
    with store.replay._db() as db:
        db.executemany("DELETE FROM process_step_reviews WHERE event_id=?", [(event,) for event in events])
    frozen = store.freeze(TENANT, train_cutoff=150, validation_cutoff=350, embargo_seconds=50, frozen_at=600)
    report, _ = evaluate_process(
        frozen, store, TENANT, MODEL_KEY, HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    )
    assert report["split_support"]["train"]["families"] == 20
    assert report["split_support"]["train"]["reviewed_families"] == 19
    assert not report["gate_passed"]
    assert "insufficient_split_family_or_label_support" in report["failure_reasons"]


@pytest.mark.asyncio
async def test_runtime_capture_is_non_serving_and_has_no_additional_model_calls(tmp_path, monkeypatch):
    research = importlib.import_module("agent.research_assistant")
    replay = ExecutionReplayStore(tmp_path / "runtime.sqlite3", KEY)
    for name, value in {
        "PROCESS_SUPERVISION_ENABLED": True,
        "PREFERENCE_RANKING_ENABLED": False,
        "PREFERENCE_SHADOW_ENABLED": False,
        "PREFERENCE_DEPLOYMENT_ENABLED": False,
        "EXECUTION_REPLAY_ENABLED": True,
        "EXECUTION_REPLAY_PATH": replay.path,
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
        research._adaptive_compute_policy(),
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
            "execution_replay_request_id": "runtime-request",
            "execution_replay_consent": True,
            "execution_replay_task_family": "runtime-family",
        }
    }
    result = await research.adaptive_deliberation_agent(state, config)
    assert result["final_response"] == answer and llm.await_count == 2
    assert result["process_supervision_receipt"]["status"] == "captured"
    rows = ProcessSupervisionStore(replay).rows(TENANT)
    assert len(rows) == 2
    for snapshot, labels in rows.values():
        assert [step.kind for step in snapshot.steps] == ["retrieve", "reason", "verify", "answer"]
        assert snapshot.steps[-1].confidence == snapshot.source.confidence
        assert (
            not labels
            and "PRIVATE" not in snapshot.model_dump_json()
            and answer not in snapshot.model_dump_json()
        )
    config["configurable"]["execution_replay_consent"] = False
    config["configurable"]["execution_replay_request_id"] = "no-consent"
    result = await research.adaptive_deliberation_agent(state, config)
    assert result["final_response"] == answer and result["process_supervision_receipt"] == {}
    assert len(ProcessSupervisionStore(replay).rows(TENANT)) == 2
    config["configurable"]["execution_replay_consent"] = True
    monkeypatch.setattr(
        research.ProcessSupervisionStore,
        "capture",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("unavailable")),
    )
    result = await research.adaptive_deliberation_agent(state, config)
    assert (
        result["final_response"] == answer
        and result["process_supervision_receipt"]["status"] == "unavailable"
    )
