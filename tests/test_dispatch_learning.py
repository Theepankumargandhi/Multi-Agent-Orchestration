from __future__ import annotations

import random
from datetime import timedelta
from types import SimpleNamespace

import pytest

from code_agent.dispatch_learning import (
    AdvisorThresholds,
    DispatchAdvisor,
    DispatchAdvisorArtifact,
    DispatchContext,
    ReviewedDecision,
    evaluate_advisor,
    fingerprint,
    fit_advisor,
    load_advisor,
    policy_fingerprint,
    snapshots_from_report,
    validate_protocol,
)
from code_agent.models import RepairTournamentPolicy
from code_agent.repair_tournament import (
    build_adaptive_supervisor,
    report_fingerprint,
    verify_tournament_report,
)
from evals.dispatch_learning_evaluation import authored_records
from tests.test_adaptive_supervisor import supervisor
from tests.test_repair_tournament import code_task, fixture_repository


def training():
    return [row for row in authored_records() if row.split == "train"]


def recommend(artifact, context=None, **kwargs):
    context = context or training()[0].snapshot.context
    return DispatchAdvisor(artifact).recommend(context, candidate_id="candidate-1", baseline_role="boundary",
        policy_sha256=kwargs.get("policy_sha256", artifact.policy_sha256),
        scope_sha256=kwargs.get("scope_sha256", artifact.scope_sha256))


def reidentify(record, tag):
    value = record.model_copy(deep=True)
    value.snapshot.source_receipt_sha256 = fingerprint(tag)
    value.snapshot.record_id = fingerprint([value.snapshot.source_receipt_sha256, value.snapshot.candidate_id])
    value.snapshot.seal()
    return value


def configure_shadow(solver, artifact, advisor=True):
    payload = solver.policy.model_dump()
    payload["adaptive_dispatch"]["advisor_shadow"] = {
        "artifact_fingerprint": artifact.fingerprint, "scope_sha256": artifact.scope_sha256}
    solver.policy = RepairTournamentPolicy.model_validate(payload)
    solver.dispatch_advisor = DispatchAdvisor(artifact) if advisor else None


def test_training_is_order_independent_and_family_balanced():
    rows = training()
    artifact = fit_advisor(rows)
    assert artifact == fit_advisor(list(reversed(rows))) and artifact.verify()
    repeated = reidentify(rows[0], "repeat-within-one-family")
    duplicated = fit_advisor(rows + [repeated])
    assert sum(next(iter(duplicated.cells.values())).votes.values()) == 6
    assert len(duplicated.training_families) == 6
    assert duplicated.training_fingerprint != artifact.training_fingerprint
    assert recommend(duplicated).supporting_families == 6
    assert recommend(duplicated).proposed_role == "defensive"


@pytest.mark.parametrize("tamper", ["duplicate", "conflict", "used", "unreviewed", "snapshot", "future_review"])
def test_invalid_training_reviews_are_rejected(tamper):
    rows = training()
    if tamper == "duplicate":
        rows.append(rows[0])
    elif tamper == "conflict":
        extra = reidentify(rows[0], "conflict")
        extra.preferred_role = "boundary"
        rows.append(extra)
    elif tamper == "used":
        rows[0].snapshot.context.used_roles = ["minimal", "defensive"]
        rows[0].snapshot.seal()
    elif tamper == "unreviewed":
        rows[0].independent_review = False
    elif tamper == "snapshot":
        rows[0].snapshot.family_id = "changed-without-sealing"
    else:
        rows[0].reviewed_at = rows[0].snapshot.observed_at - timedelta(seconds=1)
    with pytest.raises(ValueError):
        fit_advisor(rows)


@pytest.mark.parametrize("tamper", ["family", "chronology", "availability", "scope", "policy", "missing_split"])
def test_protocol_rejects_leakage_and_mixed_lineage(tamper):
    rows = authored_records()
    val = next(row for row in rows if row.split == "validation")
    if tamper == "family":
        val.snapshot.family_id = rows[0].snapshot.family_id
        val.snapshot.seal()
    elif tamper == "chronology":
        val.snapshot.observed_at = rows[0].snapshot.observed_at
        val.snapshot.seal()
    elif tamper == "availability":
        rows[0].reviewed_at = val.snapshot.observed_at
    elif tamper in {"scope", "policy"}:
        setattr(val.snapshot, f"{tamper}_sha256", "b" * 64)
        val.snapshot.seal()
    else:
        rows = [row for row in rows if row.split != "test"]
    with pytest.raises(ValueError):
        validate_protocol(rows)


@pytest.mark.parametrize("tamper", ["digest", "votes", "negative", "boolean", "mask", "key", "families"])
def test_artifact_tampering_is_rejected_even_with_rehashed_invalid_structure(tamper):
    artifact = fit_advisor(training())
    cell = next(iter(artifact.cells.values()))
    if tamper == "digest":
        artifact.scope_sha256 = "b" * 64
    elif tamper == "votes":
        cell.votes["defensive"] = 7
    elif tamper == "negative":
        cell.votes["boundary"] = -1
    elif tamper == "boolean":
        cell.votes["boundary"] = True
    elif tamper == "mask":
        cell.context.used_roles = ["minimal", "defensive"]
    elif tamper == "key":
        artifact.cells = {"b" * 64: cell}
    else:
        artifact.training_families.append(artifact.training_families[0])
    if tamper != "digest":
        artifact.seal()
    assert not artifact.verify()
    with pytest.raises(ValueError):
        DispatchAdvisor(artifact)


def test_pinned_regular_artifact_loading(tmp_path):
    artifact = fit_advisor(training())
    path = tmp_path / "advisor.json"
    path.write_text(artifact.model_dump_json(), encoding="utf-8")
    assert load_advisor(path, artifact.fingerprint).artifact == artifact
    with pytest.raises(ValueError, match="pin"):
        load_advisor(path, "b" * 64)
    path.write_text(" " * 2_000_001, encoding="utf-8")
    with pytest.raises(ValueError, match="bounded"):
        load_advisor(path, artifact.fingerprint)


def test_support_agreement_and_exact_context_abstention():
    artifact = fit_advisor(training())
    supported = recommend(artifact)
    assert supported.disposition == "supported" and supported.label_agreement == pytest.approx(7 / 9)
    assert supported.serving_changed is False
    context = training()[0].snapshot.context.model_copy(update={"patch_size": "large"})
    assert recommend(artifact, context).disposition == "unseen_context"
    assert recommend(fit_advisor(training()[:2])).disposition == "insufficient_support"
    assert recommend(artifact, scope_sha256="b" * 64).disposition == "lineage_mismatch"
    assert recommend(artifact, policy_sha256="b" * 64).proposed_role is None
    rows = training()
    rows[0].preferred_role = rows[1].preferred_role = "boundary"
    assert recommend(fit_advisor(rows)).disposition == "low_agreement"
    guarded = fit_advisor(training(), AdvisorThresholds(min_agreement_lower_bound=0.9))
    assert recommend(guarded).disposition == "low_agreement"


def test_available_role_masks_are_enforced():
    with pytest.raises(ValueError):
        DispatchContext(category="unknown", used_roles=["minimal", "minimal"], changed_files="none",
                        patch_size="empty", call_headroom="low")
    rows = training()
    for row in rows:
        row.snapshot.context.used_roles = ["minimal", "boundary", "compatibility"]
        row.snapshot.seal()
    artifact = fit_advisor(rows)
    assert recommend(artifact, rows[0].snapshot.context).proposed_role == "defensive"


def test_holdout_labels_never_train_the_artifact_and_paired_metrics_are_explicit():
    rows = authored_records()
    artifact = fit_advisor(training())
    before = evaluate_advisor(rows, artifact)
    for row in rows:
        if row.split != "train":
            row.preferred_role = "boundary"
    after = evaluate_advisor(rows, artifact)
    assert after["artifact_fingerprint"] == before["artifact_fingerprint"]
    assert before["metrics"]["test"]["paired_review_agreement_delta_on_accepted"] == 1
    assert before["metrics"]["test"]["paired_family_bootstrap_95_ci"] == [1, 1]
    assert after["metrics"]["test"]["accepted_review_agreement"] == 0
    assert "test:review_disagreement_risk" in after["blocking_reasons"]
    assert after["serving_activation"] is False
    with pytest.raises(ValueError, match="training-only"):
        fit_advisor(rows)


def test_evaluation_is_invariant_to_input_order_with_multi_decision_families():
    rows = authored_records()
    rng = random.Random(0)
    for original in [row for row in rows if row.split != "train"]:
        for number in range(rng.randrange(1, 5)):
            extra = reidentify(original, [original.snapshot.record_id, number])
            extra.preferred_role = rng.choice(["boundary", "compatibility", "defensive"])
            rows.append(extra)
    artifact = fit_advisor([row for row in rows if row.split == "train"])
    assert evaluate_advisor(rows, artifact) == evaluate_advisor(list(reversed(rows)), artifact)


def test_ready_for_review_is_not_serving_and_uncovered_risk_is_not_zero():
    rows = authored_records()
    for row in rows:
        row.label_source = "human"
    thresholds = AdvisorThresholds(max_review_disagreement_upper_bound=0.5)
    artifact = fit_advisor([row for row in rows if row.split == "train"], thresholds)
    ready = evaluate_advisor(rows, artifact)
    assert ready["decision"] == "ready_for_owner_review" and not ready["serving_activation"]
    for row in rows:
        if row.split != "train":
            row.snapshot.context.patch_size = "large"
            row.snapshot.seal()
    held = evaluate_advisor(rows, artifact)
    metric = held["metrics"]["test"]
    assert held["decision"] == "held" and metric["coverage"] == 0
    assert metric["accepted_review_agreement"] is None and metric["family_disagreement_upper_bound"] == 1


@pytest.mark.asyncio
async def test_shadow_disagrees_without_changing_dispatch_calls_or_source(tmp_path):
    root = fixture_repository(tmp_path)
    failures = {"candidate-1": "AssertionError: wrong output"}
    baseline, _ = supervisor(failures=failures)
    original = await baseline.solve(code_task(), root)
    solver, dispatched = supervisor(failures=failures)
    artifact = fit_advisor(training())
    configure_shadow(solver, artifact)
    result = await solver.solve(code_task(), root)
    report = result.tournament
    assert result.status == original.status == "completed" and result.model_calls == original.model_calls
    assert [value.perspective for value in report.dispatch_decisions] == ["minimal", "boundary"]
    assert len(dispatched) == 2 and report.winner_id == original.tournament.winner_id
    shadow = report.dispatch_shadows[0]
    assert shadow.baseline_role == "boundary" and shadow.proposed_role == "defensive" and not shadow.serving_changed
    assert verify_tournament_report(report)
    assert root.joinpath("app.py").read_text() == "VALUE = 1\n"
    pending = snapshots_from_report(report, family_id="private-family-1", scope_sha256=artifact.scope_sha256,
                                    observed_at="2026-10-01T12:00:00Z")
    assert len(pending) == 1 and pending[0].verify()
    payload = pending[0].model_dump_json()
    assert all(value not in payload for value in ("wrong output", "preferred_role", "VALUE =", "winner_id", "stderr"))
    assert pending[0].policy_sha256 == policy_fingerprint(baseline.policy)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "exception", "malformed", "wrong_pin"])
async def test_advisor_failure_only_affects_shadow(tmp_path, failure):
    solver, _ = supervisor(failures={"candidate-1": "AssertionError"})
    artifact = fit_advisor(training())
    configure_shadow(solver, artifact, advisor=False)
    if failure != "missing":
        class Broken:
            def recommend(self, *args, **kwargs):
                if failure == "exception":
                    raise ValueError("untrusted advisor failure")
                value = recommend(artifact)
                if failure == "wrong_pin":
                    value.artifact_fingerprint = "b" * 64
                else:
                    value.proposed_role = None
                return value
        solver.dispatch_advisor = Broken()
    result = await solver.solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "completed" and verify_tournament_report(result.tournament)
    assert result.tournament.dispatch_shadows[0].disposition == "artifact_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["verified", "budget", "terminal"])
async def test_no_shadow_or_snapshot_for_unfunded_or_terminal_successor(tmp_path, scenario):
    failures = {} if scenario == "verified" else {"candidate-1": "AssertionError"}
    solver, _ = supervisor(failures=failures, **({"max_model_calls": 2} if scenario == "budget" else {}))
    if scenario == "terminal":
        original_factory = solver.candidate_factory
        def unsafe_factory(*args):
            team = original_factory(*args)
            original_solve = team.solve
            async def unsafe_solve(task, sandbox):
                result = await original_solve(task, sandbox)
                gate = result.verification.rounds[-1].quality_gates[0]
                gate.name, gate.status = "secret_scan", "failed"
                return result
            team.solve = unsafe_solve
            return team
        solver.candidate_factory = unsafe_factory
    artifact = fit_advisor(training())
    configure_shadow(solver, artifact)
    result = await solver.solve(code_task(), fixture_repository(tmp_path))
    assert verify_tournament_report(result.tournament) and not result.tournament.dispatch_shadows
    assert snapshots_from_report(result.tournament, family_id="one", scope_sha256=artifact.scope_sha256,
                                 observed_at="2026-10-01T00:00:00Z") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["context_sha256", "artifact_fingerprint", "baseline_role", "proposed_role", "serving_changed"])
async def test_rehashed_shadow_semantic_tampering_is_rejected(tmp_path, field):
    solver, _ = supervisor(failures={"candidate-1": "AssertionError"})
    configure_shadow(solver, fit_advisor(training()))
    result = await solver.solve(code_task(), fixture_repository(tmp_path))
    report = result.tournament.model_copy(deep=True)
    setattr(report.dispatch_shadows[0], field, {
        "context_sha256": "b" * 64, "artifact_fingerprint": "b" * 64,
        "baseline_role": "compatibility", "proposed_role": "minimal", "serving_changed": True}[field])
    report.fingerprint = report_fingerprint(report)
    assert not verify_tournament_report(report)


def test_factory_loads_only_explicitly_pinned_shadow_artifact(tmp_path, monkeypatch):
    artifact = fit_advisor(training())
    path = tmp_path / "advisor.json"
    path.write_text(artifact.model_dump_json(), encoding="utf-8")
    solver, _ = supervisor()
    configure_shadow(solver, artifact, advisor=False)
    monkeypatch.setattr("code_agent.repair_tournament.build_repair_tournament", lambda *a, **kw: SimpleNamespace(dispatch_advisor=None))
    monkeypatch.setenv("CODE_ADAPTIVE_DISPATCH_ARTIFACT", str(path))
    loaded = build_adaptive_supervisor("unused", policy=solver.policy)
    assert loaded.dispatch_advisor.artifact.fingerprint == artifact.fingerprint
    path.write_text("{}", encoding="utf-8")
    assert build_adaptive_supervisor("unused", policy=solver.policy).dispatch_advisor is None
    ordinary, _ = supervisor()
    assert build_adaptive_supervisor("unused", policy=ordinary.policy).dispatch_advisor is None


def test_artifact_roundtrip_is_exact():
    artifact = fit_advisor(training())
    assert DispatchAdvisorArtifact.model_validate_json(artifact.model_dump_json()).verify()
    with pytest.raises(ValueError):
        ReviewedDecision.model_validate({**training()[0].model_dump(), "auto_generated_label": True})
