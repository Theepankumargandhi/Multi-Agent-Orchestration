import importlib
import json
import sys

import pytest
from pydantic import ValidationError

from agent.adaptive_compute import ComputeSignals, candidate_assessment, plan_compute, select_candidate
from agent.execution_replay import ExecutionReplayStore, Observation, digest
from agent.uncertainty import fit_calibrator
from evals.replay_calibration import recalibrate, run_drill, wilson_upper

KEY = b"unit-test-private-replay-key-32-bytes"


def observation(
    store,
    request="request-1",
    tenant="private-tenant",
    confidence=0.9,
    siblings=False,
    origin="runtime",
    route="rag",
):
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
            answer="PRIVATE-ANSWER secret@example.com",
            confidence=confidence,
            grounded=True,
            conformal_decision="not_evaluated",
            claim_keys=["PRIVATE-EVIDENCE"],
            token_count=45,
            latency_ms=10,
        )
    ]
    if siblings:
        candidates.append(candidates[0].model_copy(update={"candidate_id": "candidate-2", "confidence": 0.8}))
    receipt = select_candidate(plan, candidates)
    ids = store.capture(plan, receipt, tenant=tenant, request_id=request, consent=True, origin=origin)
    return plan, receipt, ids


def populate(store, *, poisoned=False, unsafe_limit=None):
    unsafe_count = 0
    for index in range(240):
        correct = index % 4 != 0
        _, _, ids = observation(store, request=f"fixture-{index}", confidence=0.9 if correct else 0.2)
        group = digest(KEY, "request", ["private-tenant", f"fixture-{index}"])
        is_test = int(digest(KEY, "split-v1", group)[:8], 16) % 4 == 0
        unsafe = poisoned and is_test and correct and (unsafe_limit is None or unsafe_count < unsafe_limit)
        unsafe_count += int(unsafe)
        store.review(
            "private-tenant",
            ids[0],
            verdict="correct" if correct else "incorrect",
            unsafe=unsafe,
            reviewer="trusted-reviewer",
        )


def test_capture_privacy_tenant_scope_and_retry(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    plan, receipt, ids = observation(store)
    assert store.capture(plan, receipt, tenant="private-tenant", request_id="request-1", consent=True) == ids
    assert len(store.records("private-tenant")) == 1
    assert store.records("other-tenant") == []
    content = store.records("private-tenant")[0][0].model_dump_json()
    for private in [
        "PRIVATE-ANSWER",
        "secret@example.com",
        "PRIVATE-EVIDENCE",
        "private-tenant",
        "request-1",
        "candidate-1",
        "answer_fingerprint",
        "planned_actions",
        "propensity",
    ]:
        assert private not in content
    assert store.records("private-tenant")[0][0].estimated_output_tokens == 45


def test_consent_identity_and_strong_key_required(tmp_path):
    with pytest.raises(ValueError, match="32 bytes"):
        ExecutionReplayStore(tmp_path / "missing.sqlite3", b"weak")
    assert not (tmp_path / "missing.sqlite3").exists()
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    plan, receipt, _ = observation(store)
    for overrides in [{"consent": False}, {"consent": "true"}, {"tenant": ""}, {"request_id": ""}]:
        arguments = dict(tenant="private-tenant", request_id="not-recorded", consent=True)
        arguments.update(overrides)
        assert store.capture(plan, receipt, **arguments) == []
    assert len(store.records("private-tenant")) == 1


def test_plan_receipt_tamper_and_conflicting_retry_are_rejected(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    plan, receipt, _ = observation(store)
    receipt.selected_confidence = 0.1
    with pytest.raises(ValueError, match="integrity"):
        store.capture(plan, receipt, tenant="private-tenant", request_id="new", consent=True)
    with pytest.raises(ValueError, match="conflicting"):
        observation(store, confidence=0.85)
    assert len(store.records("private-tenant")) == 1


def test_delayed_labels_are_tenant_bound_immutable_and_idempotent(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    _, _, ids = observation(store)
    with pytest.raises(ValueError, match="not found"):
        store.review("other-tenant", ids[0], verdict="correct", unsafe=False, reviewer="reviewer")
    assert store.dataset("private-tenant")[0] == []
    label = store.review("private-tenant", ids[0], verdict="correct", unsafe=False, reviewer="reviewer")
    assert label == store.review(
        "private-tenant", ids[0], verdict="correct", unsafe=False, reviewer="reviewer"
    )
    with pytest.raises(ValueError, match="immutable"):
        store.review("private-tenant", ids[0], verdict="incorrect", unsafe=False, reviewer="reviewer")
    assert len(store.dataset("private-tenant")[0]) == 1


def test_payload_and_label_tampering_fail_closed(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    _, _, ids = observation(store)
    store.review("private-tenant", ids[0], verdict="correct", unsafe=False, reviewer="reviewer")
    with store._db() as db:
        payload = json.loads(db.execute("SELECT payload FROM labels").fetchone()[0])
        payload["unsafe"] = True
        db.execute("UPDATE labels SET payload=?", (json.dumps(payload),))
    with pytest.raises(ValueError, match="integrity"):
        store.dataset("private-tenant")


def test_siblings_do_not_inflate_training_or_leak_across_splits(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    for index in range(24):
        _, _, ids = observation(store, request=f"sibling-{index}", siblings=True)
        for event_id in ids:
            store.review("private-tenant", event_id, verdict="correct", unsafe=False, reviewer="reviewer")
    examples, manifest = store.dataset("private-tenant")
    assert len(examples) == 24
    assert manifest["excluded_sibling_candidates"] == 24
    assert len({row["group"] for row in manifest["lineage"]}) == 24
    assert store.dataset("private-tenant")[1] == manifest
    assert {row["split"] for row in manifest["lineage"]} == {"calibration", "test"}


def test_candidate_choice_does_not_change_to_get_a_reviewed_label(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    _, receipt, ids = observation(store, siblings=True)
    rows = store.records("private-tenant")
    unselected = next(event for event, _ in rows if not event.selected)
    store.review("private-tenant", unselected.event_id, verdict="correct", unsafe=False, reviewer="reviewer")
    examples, manifest = store.dataset("private-tenant")
    assert len(ids) == 2 and receipt.status == "released"
    assert examples == [] and manifest["excluded_groups"] == {"unreviewed": 1}


def test_ambiguous_unsupported_and_synthetic_labels_are_excluded(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    for request, route, origin, verdict in [
        ("ambiguous", "rag", "runtime", "ambiguous"),
        ("general", "general", "runtime", "correct"),
        ("synthetic", "rag", "synthetic", "correct"),
    ]:
        _, _, ids = observation(store, request=request, route=route, origin=origin)
        store.review("private-tenant", ids[0], verdict=verdict, unsafe=False, reviewer="reviewer")
    examples, manifest = store.dataset("private-tenant")
    assert examples == []
    assert manifest["excluded_groups"] == {"ambiguous": 1, "unsupported_route": 1, "synthetic": 1}
    assert len(store.dataset("private-tenant", allow_synthetic=True)[0]) == 1


def test_tenant_deletion_cascades_labels_without_affecting_other_tenants(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    _, _, ids = observation(store)
    store.review("private-tenant", ids[0], verdict="correct", unsafe=False, reviewer="reviewer")
    observation(store, tenant="other-tenant")
    assert store.delete_tenant("private-tenant") == 1
    assert store.records("private-tenant") == []
    assert len(store.records("other-tenant")) == 1
    with store._db() as db:
        assert db.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0


def test_gate_uses_independent_groups_and_heldout_labels_do_not_fit_artifact(tmp_path):
    store = ExecutionReplayStore(tmp_path / "clean.sqlite3", KEY)
    poisoned = ExecutionReplayStore(tmp_path / "poisoned.sqlite3", KEY)
    populate(store)
    populate(poisoned, poisoned=True)
    clean_artifact, clean = recalibrate(store, "private-tenant")
    bad_artifact, bad = recalibrate(poisoned, "private-tenant")
    assert clean["production_candidate_eligible"] and not clean["auto_deployed"]
    assert not bad["gate_passed"] and "heldout_error_upper_bound_exceeds_target" in bad["reasons"]
    assert "unsafe_heldout_release" in bad["reasons"]
    assert clean_artifact == bad_artifact
    assert clean["fingerprint"] != bad["fingerprint"]
    examples, _ = store.dataset("private-tenant")
    altered = [
        item.model_copy(update={"correct": not item.correct}) if item.split == "test" else item
        for item in examples
    ]
    assert fit_calibrator(altered) == clean_artifact


def test_zero_release_low_sample_and_tampered_incumbent_cannot_pass(tmp_path):
    assert wilson_upper(0, 0) == 1
    assert wilson_upper(0, 5) > 0.4
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    _, report = recalibrate(store, "private-tenant")
    assert not report["gate_passed"] and report["heldout"] is None
    populate(store)
    artifact, _ = recalibrate(store, "private-tenant")
    artifact.global_quantile = 0.99
    with pytest.raises(ValueError, match="integrity"):
        recalibrate(store, "private-tenant", incumbent=artifact)


def test_single_unsafe_release_blocks_even_when_correctness_error_bound_passes(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    populate(store, poisoned=True, unsafe_limit=1)
    _, report = recalibrate(store, "private-tenant")
    assert report["heldout"]["unsafe_releases"] == 1
    assert report["heldout"]["error_wilson_upper_95"] < 0.2
    assert report["reasons"] == ["unsafe_heldout_release"]
    assert not report["production_candidate_eligible"]


def test_synthetic_end_to_end_drill_cannot_create_production_candidate():
    report = run_drill()
    assert report["passed"] and report["simulation"]
    assert report["clean_gate"]["heldout"]["errors"] == 0
    assert not report["clean_gate"]["production_candidate_eligible"]
    assert report["synthetic_excluded_from_production"] and report["tamper_rejected"]


def test_observation_schema_rejects_content_fields_and_nonfinite_scores(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    observation(store)
    payload = store.records("private-tenant")[0][0].model_dump()
    with pytest.raises(ValidationError):
        Observation.model_validate({**payload, "prompt": "private"})
    with pytest.raises(ValidationError):
        Observation.model_validate({**payload, "latency_ms": float("nan")})


@pytest.mark.asyncio
async def test_runtime_capture_is_consent_gated_and_storage_failure_is_nonfatal(tmp_path, monkeypatch):
    research = importlib.import_module("agent.research_assistant")

    store = ExecutionReplayStore(tmp_path / "source.sqlite3", KEY)
    plan, receipt, _ = observation(store)
    monkeypatch.setattr(research, "EXECUTION_REPLAY_ENABLED", True)
    monkeypatch.setattr(research, "EXECUTION_REPLAY_PATH", tmp_path / "runtime.sqlite3")
    monkeypatch.setattr(research, "EXECUTION_REPLAY_KEY", KEY)
    monkeypatch.setattr(research, "ADAPTIVE_COMPUTE_INTEGRITY_KEY", None)
    config = {"configurable": {"user_id": "private-tenant", "execution_replay_request_id": "runtime-1"}}
    assert await research._capture_execution_replay(plan, receipt, config) == []
    assert not (tmp_path / "runtime.sqlite3").exists()
    config["configurable"]["execution_replay_consent"] = True
    ids = await research._capture_execution_replay(plan, receipt, config)
    assert len(ids) == 1
    runtime = ExecutionReplayStore(tmp_path / "runtime.sqlite3", KEY)
    assert runtime.records("private-tenant")[0][0].origin == "runtime"
    monkeypatch.setattr(research, "EXECUTION_REPLAY_KEY", b"weak")
    assert await research._capture_execution_replay(plan, receipt, config) == []


@pytest.mark.asyncio
async def test_actual_deliberation_records_verified_candidates_not_planning_actions(tmp_path, monkeypatch):
    research = importlib.import_module("agent.research_assistant")
    monkeypatch.setattr(research, "EXECUTION_REPLAY_ENABLED", True)
    monkeypatch.setattr(research, "EXECUTION_REPLAY_PATH", tmp_path / "runtime.sqlite3")
    monkeypatch.setattr(research, "EXECUTION_REPLAY_KEY", KEY)
    monkeypatch.setattr(research, "ADAPTIVE_COMPUTE_INTEGRITY_KEY", None)
    monkeypatch.setattr(research, "UNCERTAINTY_CALIBRATION_ENABLED", False)
    calls = []

    async def candidate_llm(*args, **kwargs):
        calls.append(1)
        return "AgentForge uses LangGraph for agent orchestration [source](https://example.com/architecture)."

    monkeypatch.setattr(research, "_call_llm", candidate_llm)
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
        "query": "How is AgentForge orchestrated?",
        "web_notes": "Link: https://example.com/architecture\nSnippet: AgentForge uses LangGraph for agent orchestration.",
        "adaptive_compute_plan": plan.model_dump(mode="json"),
    }
    config = {
        "configurable": {
            "user_id": "private-tenant",
            "execution_replay_request_id": "runtime-turn",
            "execution_replay_consent": True,
        }
    }
    result = await research.adaptive_deliberation_agent(state, config)
    store = ExecutionReplayStore(tmp_path / "runtime.sqlite3", KEY)
    rows = store.records("private-tenant")
    assert result["grounding_action"] == "pass"
    assert len(calls) == len(rows) == len(result["execution_replay_event_ids"]) == 2
    assert sum(event.selected for event, _ in rows) == 1
    assert all(event.grounded and event.origin == "runtime" and label is None for event, label in rows)


def test_always_abstain_calibrator_fails_coverage_and_risk_gate(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    for index in range(120):
        _, _, ids = observation(store, request=f"uncertain-{index}", confidence=0.5)
        store.review("private-tenant", ids[0], verdict="correct", unsafe=False, reviewer="reviewer")
    _, report = recalibrate(store, "private-tenant")
    assert report["heldout"]["released"] == 0
    assert not report["gate_passed"]
    assert "insufficient_heldout_coverage" in report["reasons"]
    assert "heldout_error_upper_bound_exceeds_target" in report["reasons"]


def test_service_assigns_unique_request_group_and_strict_opt_in():
    from schema.schema import UserInput
    from service.service import _parse_input

    with pytest.raises(ValidationError):
        UserInput(message="hello", execution_replay_consent="true")
    first, run_id = _parse_input(
        UserInput(message="hello", execution_replay_consent=True, thread_id="thread-1"), user_id="tenant"
    )
    second, _ = _parse_input(UserInput(message="hello", thread_id="thread-1"), user_id="tenant")
    config = first["config"]["configurable"]
    assert config["execution_replay_consent"] is True
    assert config["execution_replay_request_id"] == str(run_id)
    assert second["config"]["configurable"]["execution_replay_request_id"] != str(run_id)
    assert second["config"]["configurable"]["execution_replay_consent"] is False


def test_review_export_cli_keeps_content_private_and_does_not_overwrite_store(tmp_path, monkeypatch, capsys):
    from agent.execution_replay import main

    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    _, _, ids = observation(store)
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", KEY.decode())
    prefix = ["execution-replay", "--store", str(store.path), "--tenant", "private-tenant"]
    monkeypatch.setattr(
        sys, "argv", prefix + ["review", ids[0], "--verdict", "correct", "--reviewer", "reviewer"]
    )
    main()
    assert json.loads(capsys.readouterr().out)["verdict"] == "correct"
    target = tmp_path / "export.json"
    monkeypatch.setattr(sys, "argv", prefix + ["export", "--output", str(target)])
    main()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert len(data["examples"]) == 1
    assert "PRIVATE-ANSWER" not in target.read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "argv", prefix + ["export", "--output", str(store.path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2 and len(store.records("private-tenant")) == 1


def test_recalibration_cli_writes_only_candidate_and_preserves_active_artifact(tmp_path, monkeypatch, capsys):
    from agent.uncertainty import load_calibrator, save_calibrator
    from evals.replay_calibration import main

    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", KEY)
    populate(store)
    artifact, _ = recalibrate(store, "private-tenant", calibrator_key=KEY)
    active = tmp_path / "active.json"
    candidate = tmp_path / "candidate.json"
    report = tmp_path / "report.json"
    save_calibrator(active, artifact)
    before = active.read_bytes()
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", KEY.decode())
    monkeypatch.setenv("UNCERTAINTY_INTEGRITY_KEY", KEY.decode())
    monkeypatch.setenv("UNCERTAINTY_CALIBRATOR_PATH", str(active))
    prefix = [
        "replay-calibration",
        "--store",
        str(store.path),
        "--tenant",
        "private-tenant",
        "--incumbent",
        str(active),
    ]
    monkeypatch.setattr(
        sys, "argv", prefix + ["--artifact", str(candidate), "--output", str(report), "--require-gate"]
    )
    main()
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert load_calibrator(candidate, KEY) == artifact
    assert active.read_bytes() == before
    monkeypatch.setattr(sys, "argv", prefix + ["--artifact", str(active), "--output", str(report)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2 and active.read_bytes() == before
    monkeypatch.setattr(sys, "argv", prefix + ["--artifact", str(candidate), "--output", str(active)])
    with pytest.raises(SystemExit):
        main()
    assert active.read_bytes() == before


def test_recalibration_cli_holds_on_insufficient_data_without_candidate(tmp_path, monkeypatch):
    from evals.replay_calibration import main

    candidate = tmp_path / "candidate.json"
    report = tmp_path / "report.json"
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", KEY.decode())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay-calibration",
            "--store",
            str(tmp_path / "replay.sqlite3"),
            "--tenant",
            "tenant",
            "--artifact",
            str(candidate),
            "--output",
            str(report),
            "--require-gate",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert not candidate.exists()
    assert not json.loads(report.read_text(encoding="utf-8"))["production_candidate_eligible"]
