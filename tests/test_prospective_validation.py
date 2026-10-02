import json
import sys

import pytest

from agent.adaptive_compute import ComputeSignals, candidate_assessment, plan_compute, select_candidate
from agent.execution_replay import ExecutionReplayStore, digest
from agent.prospective_validation import FrozenCohort, HoldoutLedger, freeze_cohort, validate_live_lineage
from agent.uncertainty import fit_calibrator, load_calibrator
from evals.prospective_evaluation import ValidationPolicy, evaluate_cohort, paired_bootstrap, run_drill

KEY = b"prospective-validation-fixture-key-32-bytes"
TENANT = "fixture-tenant"


def capture(
    store,
    clock,
    *,
    request="request-1",
    family="family-1",
    observed=1000,
    reviewed=None,
    confidence=0.9,
    verdict="correct",
    unsafe=False,
    review=True,
    origin="runtime",
    route="rag",
    siblings=False,
):
    """Runtime-origin branches are unit stubs, not live-provider observations."""
    clock[0] = observed
    plan = plan_compute(
        ComputeSignals(
            route=route,
            grounding_action="repair",
            grounding_confidence=0.4,
            uncertainty_decision="abstain",
            evidence_count=2,
        )
    )
    candidates = [
        candidate_assessment(
            candidate_id="candidate-1",
            answer="PRIVATE ANSWER",
            confidence=confidence,
            grounded=True,
            conformal_decision="not_evaluated",
            claim_keys=["PRIVATE EVIDENCE"],
            token_count=40,
            latency_ms=10,
        )
    ]
    if siblings:
        candidates.append(candidates[0].model_copy(update={"candidate_id": "candidate-2", "confidence": 0.8}))
    receipt = select_candidate(plan, candidates)
    ids = store.capture(
        plan, receipt, tenant=TENANT, request_id=request, consent=True, task_family=family, origin=origin
    )
    if review:
        clock[0] = reviewed if reviewed is not None else observed + 1
        for event_id in ids:
            store.review(TENANT, event_id, verdict=verdict, unsafe=unsafe, reviewer="fixture-reviewer")
    return plan, receipt, ids


def empty_store(tmp_path):
    clock = [1000.0]
    return ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY, clock=lambda: clock[0]), clock


def freeze(store, **overrides):
    arguments = dict(calibration_cutoff=2000, embargo_seconds=500, frozen_at=4000)
    arguments.update(overrides)
    return freeze_cohort(store, TENANT, **arguments)


def populated(tmp_path, *, unsafe_test=False, shifted=False, origin="runtime", test_route="rag"):
    store, clock = empty_store(tmp_path)
    for split in ["calibration", "test"]:
        for index in range(40):
            correct = index % 4 != 0
            capture(
                store,
                clock,
                request=f"{split}-{index}",
                family=f"{split}-family-{index}",
                observed=(1000 if split == "calibration" else 3000) + index,
                confidence=0.3 if shifted and split == "test" and not correct else 0.9 if correct else 0.2,
                verdict="correct" if correct else "incorrect",
                unsafe=unsafe_test and split == "test" and index == 1,
                origin=origin,
                route=test_route if split == "test" else "rag",
            )
    return store, freeze(store, allow_synthetic=origin == "synthetic")


def test_family_assignment_is_private_immutable_and_cannot_be_added_after_outcomes(tmp_path):
    store, clock = empty_store(tmp_path)
    plan, receipt, ids = capture(store, clock, family="PRIVATE TASK FAMILY")
    records, assignments = store.snapshot(TENANT)
    assignment = assignments[records[0][0].request_group]
    assert "PRIVATE TASK FAMILY" not in assignment.model_dump_json()
    assert assignment.task_family == digest(KEY, "task-family", [TENANT, "PRIVATE TASK FAMILY"])
    assert (
        store.capture(
            plan,
            receipt,
            tenant=TENANT,
            request_id="request-1",
            consent=True,
            task_family="PRIVATE TASK FAMILY",
        )
        == ids
    )
    with pytest.raises(ValueError, match="immutable"):
        store.capture(
            plan, receipt, tenant=TENANT, request_id="request-1", consent=True, task_family="different"
        )
    plan, receipt, _ = capture(store, clock, request="missing-family", family=None)
    with pytest.raises(ValueError, match="immutable"):
        store.capture(
            plan, receipt, tenant=TENANT, request_id="missing-family", consent=True, task_family="retroactive"
        )
    assert freeze(store).excluded_groups == {"missing_pre_execution_family": 1}


def test_legacy_observations_still_verify_without_retroactive_family_assignment(tmp_path):
    store, clock = empty_store(tmp_path)
    plan, receipt, _ = capture(store, clock, family=None)
    with store._db() as db:
        db.execute("DELETE FROM request_families")
    assert len(store.records(TENANT)) == 1
    assert freeze(store).members == []
    with pytest.raises(ValueError, match="first capture"):
        store.capture(plan, receipt, tenant=TENANT, request_id="request-1", consent=True, task_family="late")


def test_family_payload_tampering_and_tenant_deletion_fail_closed(tmp_path):
    store, clock = empty_store(tmp_path)
    capture(store, clock)
    cohort = freeze(store)
    assert store.snapshot("other-tenant") == ([], {})
    with store._db() as db:
        payload = json.loads(db.execute("SELECT payload FROM request_families").fetchone()[0])
        payload["task_family"] = "a" * 64
        db.execute("UPDATE request_families SET payload=?", (json.dumps(payload),))
    with pytest.raises(ValueError, match="integrity"):
        freeze(store)
    store.delete_tenant(TENANT)
    assert store.snapshot(TENANT) == ([], {})
    with pytest.raises(ValueError, match="revoked"):
        validate_live_lineage(cohort, store, TENANT)


def test_chronological_family_split_purges_repeats_and_delayed_training_labels(tmp_path):
    store, clock = empty_store(tmp_path)
    capture(store, clock, request="train", family="old-family", observed=1000)
    capture(store, clock, request="old-family-test", family="old-family", observed=3000)
    capture(store, clock, request="test", family="new-family", observed=3000)
    capture(store, clock, request="late-label", family="late-family", observed=1100, reviewed=2100)
    capture(store, clock, request="embargo", family="embargo-family", observed=2200)
    capture(store, clock, request="future-label", family="future-family", observed=3100, reviewed=4100)
    cohort = freeze(store)
    assert len(cohort.members) == 2
    assert {member.example.split for member in cohort.members} == {"calibration", "test"}
    assert len({member.task_family for member in cohort.members}) == 2
    assert cohort.excluded_groups == {
        "repeated_task_family": 1,
        "delayed_training_label": 1,
        "family_seen_before_test_window": 1,
        "label_after_snapshot": 1,
    }


def test_representative_selection_cannot_substitute_a_later_reviewed_request(tmp_path):
    store, clock = empty_store(tmp_path)
    capture(store, clock, request="first", family="same-family", observed=1000, review=False, siblings=True)
    capture(store, clock, request="later", family="same-family", observed=1100, verdict="correct")
    cohort = freeze(store)
    assert cohort.members == []
    assert cohort.excluded_groups == {"repeated_task_family": 1, "unreviewed": 1}


def test_frozen_cohort_rejects_tampering_family_overlap_and_invalid_timeline(tmp_path):
    store, cohort = populated(tmp_path)
    payload = cohort.model_dump()
    payload["members"][0]["example"]["prompt"] = "PRIVATE TEXT"
    with pytest.raises(ValueError):
        FrozenCohort.model_validate(payload)
    changed = cohort.model_copy(deep=True)
    changed.members[0].example.confidence = 0.11
    with pytest.raises(ValueError, match="integrity"):
        changed.verify(KEY)
    changed.seal(KEY)
    with pytest.raises(ValueError, match="lineage changed"):
        validate_live_lineage(changed, store, TENANT)
    overlap = cohort.model_copy(deep=True)
    overlap.members[1].task_family = overlap.members[0].task_family
    overlap.seal(KEY)
    with pytest.raises(ValueError, match="representative"):
        overlap.verify(KEY)
    late = cohort.model_copy(deep=True)
    next(member for member in late.members if member.example.split == "calibration").reviewed_at = 2100
    late.seal(KEY)
    with pytest.raises(ValueError, match="future calibration"):
        late.verify(KEY)


def test_holdout_exact_retry_is_cached_but_changed_or_overlapping_studies_are_rejected(tmp_path):
    store, cohort = populated(tmp_path / "data")
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    _, report = evaluate_cohort(cohort, store, TENANT, ledger)
    assert report["production_candidate_eligible"] and report["holdout_reserved_before_scoring"]
    assert evaluate_cohort(cohort, store, TENANT, ledger)[1] == report
    with pytest.raises(ValueError, match="already exposed"):
        evaluate_cohort(cohort, store, TENANT, ledger, policy=ValidationPolicy(bootstrap_seed=18))
    subset = cohort.model_copy(deep=True)
    subset.members = [
        member
        for member in subset.members
        if member != next(member for member in subset.members if member.example.split == "test")
    ]
    subset.seal(KEY)
    with pytest.raises(ValueError, match="already exposed"):
        evaluate_cohort(subset, store, TENANT, ledger)


def test_pending_study_and_missing_exposure_index_cannot_quietly_reuse_holdout(tmp_path):
    _, cohort = populated(tmp_path / "data")
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    ledger.reserve("pending-study", cohort)
    with pytest.raises(ValueError, match="pending"):
        ledger.reserve("pending-study", cohort)
    with ledger._db() as db:
        db.execute(
            "DELETE FROM exposures WHERE family_id=?",
            (next(member.task_family for member in cohort.members if member.example.split == "test"),),
        )
    with pytest.raises(ValueError, match="index integrity"):
        ledger.reserve("new-study", cohort)


def test_evaluation_checks_live_lineage_even_for_cached_completed_evidence(tmp_path):
    store, cohort = populated(tmp_path / "data")
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    evaluate_cohort(cohort, store, TENANT, ledger)
    with pytest.raises(ValueError, match="tenant"):
        evaluate_cohort(cohort, store, "other-tenant", ledger)
    store.delete_tenant(TENANT)
    with pytest.raises(ValueError, match="revoked"):
        evaluate_cohort(cohort, store, TENANT, ledger)


def test_paired_bootstrap_is_deterministic_and_uses_matched_family_outcomes():
    policy = ValidationPolicy()
    identical = [(True, True)] * 30 + [(False, False)] * 10
    report = paired_bootstrap(identical, identical, policy)
    assert report["coverage"] == report["utility"] == {"delta": 0, "lower_95": 0, "upper_95": 0}
    worse = [(True, False)] * 30 + [(False, False)] * 10
    improvement = paired_bootstrap(identical, worse, policy)
    assert improvement == paired_bootstrap(identical, worse, policy)
    assert improvement["utility"]["lower_95"] > 0
    assert paired_bootstrap(worse, identical, policy)["utility"]["upper_95"] < 0
    with pytest.raises(ValueError):
        paired_bootstrap(identical, worse[:2], policy)


def test_future_unsafe_labels_change_gate_not_calibration_artifact(tmp_path):
    clean_store, clean_cohort = populated(tmp_path / "clean")
    bad_store, bad_cohort = populated(tmp_path / "bad", unsafe_test=True)
    clean_artifact, clean = evaluate_cohort(
        clean_cohort, clean_store, TENANT, HoldoutLedger(tmp_path / "clean-ledger.sqlite3", KEY)
    )
    bad_artifact, bad = evaluate_cohort(
        bad_cohort, bad_store, TENANT, HoldoutLedger(tmp_path / "bad-ledger.sqlite3", KEY)
    )
    assert clean_artifact == bad_artifact
    assert clean["gate_passed"] and not bad["gate_passed"]
    assert "unsafe_heldout_release" in bad["reasons"]


def test_drift_is_measured_and_blocks_candidate_while_identical_incumbent_passes(tmp_path):
    store, cohort = populated(tmp_path / "clean")
    incumbent = fit_calibrator(
        [member.example for member in cohort.members if member.example.split == "calibration"]
    )
    _, report = evaluate_cohort(
        cohort, store, TENANT, HoldoutLedger(tmp_path / "clean-ledger.sqlite3", KEY), incumbent=incumbent
    )
    assert report["gate_passed"] and report["drift"]["route_total_variation"] == 0
    assert report["paired"]["utility"]["lower_95"] == 0
    shifted, shifted_cohort = populated(tmp_path / "shifted", shifted=True)
    _, report = evaluate_cohort(
        shifted_cohort, shifted, TENANT, HoldoutLedger(tmp_path / "shift-ledger.sqlite3", KEY)
    )
    assert "confidence_distribution_shift" in report["reasons"]
    assert report["reasons"] == ["confidence_distribution_shift"]
    assert report["heldout"]["coverage"] == 0.75 and report["heldout"]["errors"] == 0
    assert not report["production_candidate_eligible"]


def test_synthetic_drill_proves_controls_not_live_quality():
    report = run_drill()
    assert report["passed"] and report["simulation"]
    assert not report["clean"]["production_candidate_eligible"]
    assert report["changed_study_reuse_rejected"] and report["exact_retry_cached"]


def test_route_shift_and_insufficient_future_support_cannot_pass(tmp_path):
    store, cohort = populated(tmp_path / "routes", test_route="web")
    _, report = evaluate_cohort(cohort, store, TENANT, HoldoutLedger(tmp_path / "route-ledger.sqlite3", KEY))
    assert report["drift"]["route_total_variation"] == 1
    assert "route_distribution_shift" in report["reasons"]
    assert "insufficient_route_support:web" in report["reasons"]
    assert not report["production_candidate_eligible"]


def test_sparse_cohort_returns_hold_without_drift_or_paired_claims(tmp_path):
    store, cohort = populated(tmp_path / "data")
    cohort.members = [member for member in cohort.members if member.example.split == "calibration"] + [
        member for member in cohort.members if member.example.split == "test"
    ][:2]
    cohort.seal(KEY)
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    artifact, report = evaluate_cohort(cohort, store, TENANT, ledger)
    assert artifact is None and not report["production_candidate_eligible"]
    assert report["reasons"] == ["insufficient_test_request_groups"]
    assert report["drift"] is None and report["paired"] is None
    assert evaluate_cohort(cohort, store, TENANT, ledger)[1] == report


def test_service_hashes_family_before_graph_configuration_and_requires_opt_in(monkeypatch):
    from schema.schema import UserInput
    from service.service import _parse_input

    monkeypatch.setenv("EXECUTION_REPLAY_KEY", KEY.decode())
    user = UserInput(
        message="hello", execution_replay_consent=True, execution_replay_task_family="PRIVATE FAMILY"
    )
    parsed, _ = _parse_input(user, user_id=TENANT)
    configurable = parsed["config"]["configurable"]
    assert "execution_replay_task_family" not in configurable
    assert configurable["execution_replay_task_family_fingerprint"] == digest(
        KEY, "task-family", [TENANT, "PRIVATE FAMILY"]
    )
    assert "PRIVATE FAMILY" not in json.dumps(configurable)
    parsed, _ = _parse_input(
        UserInput(message="hello", execution_replay_task_family="PRIVATE FAMILY"), user_id=TENANT
    )
    assert parsed["config"]["configurable"]["execution_replay_task_family_fingerprint"] is None
    with pytest.raises(ValueError):
        UserInput(message="hello", execution_replay_task_family="  ")


def test_freeze_and_evaluate_cli_write_candidate_only_and_keep_cohort_immutable(
    tmp_path, monkeypatch, capsys
):
    from evals.prospective_evaluation import main

    store, cohort = populated(tmp_path / "data")
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", KEY.decode())
    cohort_path = tmp_path / "cohort.json"
    report_path = tmp_path / "report.json"
    artifact_path = tmp_path / "candidate.json"
    active = tmp_path / "active.json"
    active.write_text("protected incumbent", encoding="utf-8")
    monkeypatch.setenv("UNCERTAINTY_CALIBRATOR_PATH", str(active))
    prefix = [
        "prospective-evaluation",
        "--store",
        str(store.path),
        "--ledger",
        str(tmp_path / "ledger.sqlite3"),
        "--tenant",
        TENANT,
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        prefix
        + [
            "freeze",
            "--cutoff",
            "2000",
            "--embargo-seconds",
            "500",
            "--snapshot-time",
            "4000",
            "--output",
            str(cohort_path),
        ],
    )
    main()
    assert FrozenCohort.model_validate_json(cohort_path.read_text(encoding="utf-8")) == cohort
    before = cohort_path.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        prefix
        + [
            "freeze",
            "--cutoff",
            "2000",
            "--embargo-seconds",
            "500",
            "--snapshot-time",
            "4001",
            "--output",
            str(cohort_path),
        ],
    )
    with pytest.raises(SystemExit):
        main()
    assert cohort_path.read_bytes() == before
    monkeypatch.setattr(
        sys,
        "argv",
        prefix
        + [
            "evaluate",
            str(cohort_path),
            "--artifact",
            str(artifact_path),
            "--output",
            str(report_path),
            "--require-gate",
        ],
    )
    monkeypatch.delenv("UNCERTAINTY_INTEGRITY_KEY", raising=False)
    main()
    assert load_calibrator(artifact_path).calibration_count == 40
    assert json.loads(report_path.read_text(encoding="utf-8"))["production_candidate_eligible"]
    assert active.read_text(encoding="utf-8") == "protected incumbent" and cohort_path.read_bytes() == before
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["passed"]


def test_cli_refuses_to_overwrite_active_artifact_even_in_synthetic_drill(tmp_path, monkeypatch):
    from evals.prospective_evaluation import main

    active = tmp_path / "active.json"
    active.write_text("protected", encoding="utf-8")
    monkeypatch.setenv("UNCERTAINTY_CALIBRATOR_PATH", str(active))
    monkeypatch.setattr(sys, "argv", ["prospective-evaluation", "drill", "--output", str(active)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2 and active.read_text(encoding="utf-8") == "protected"
