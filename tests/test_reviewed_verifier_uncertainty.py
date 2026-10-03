"""Reviewed step ensembles must not borrow labels, families, or serving authority."""

import json
import sqlite3

import pytest

from agent.execution_replay import ExecutionReplayStore
from agent.process_reward import train_process_reward_model
from agent.process_supervision import ProcessSupervisionStore
from agent.prospective_validation import HoldoutLedger
from agent.reviewed_verifier_uncertainty import (
    ReviewedStepEnsemble,
    ReviewedStepScorer,
    UncertaintyPolicy,
    fit_ensemble,
)
from evals.process_supervision_evaluation import evaluate_process, seed_process
from evals.reviewed_verifier_uncertainty_evaluation import (
    evaluate_uncertainty,
    main,
    risk_curve,
    run_drill,
)

KEY, MODEL_KEY = b"unit-reviewed-step-replay-key-long", b"unit-reviewed-step-model-key-long!"
TENANT = "private-tenant"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("reviewed-step")
    store = ProcessSupervisionStore(ExecutionReplayStore(root / "replay.sqlite3", KEY))
    cohort = seed_process(store, TENANT, origin="runtime")  # authored unit data, not field evidence
    ledger = HoldoutLedger(root / "ledger.sqlite3", KEY)
    report, candidate = evaluate_uncertainty(cohort, store, TENANT, MODEL_KEY, ledger)
    return store, cohort, ledger, report, candidate


def test_clean_sparse_shifted_and_unseen_controls():
    result = run_drill()
    assert result["gate_passed"] and not result["production_activation"]
    clean = result["reports"]["clean"]
    assert clean["gate_passed"] and not clean["candidate_ready_for_review"]
    assert clean["family_macro_brier"] < clean["train_constant_brier"]
    assert clean["primary"]["accepted_families"] == 20
    assert clean["primary"]["coverage"] >= 0.5
    assert clean["unsupported_step_fraction"] == 0
    assert clean["worst_case_family_macro_brier"] == pytest.approx(clean["family_macro_brier"])
    assert not result["reports"]["missing_review"]["gate_passed"]
    assert result["reports"]["missing_review"]["test_review_coverage"] == 0.5
    assert not result["reports"]["shifted_labels"]["gate_passed"]
    assert result["unseen_feature_probe"]["unsupported_steps"] == 4
    assert result["unseen_feature_probe"]["accepted_steps"] == 0
    assert result["unseen_feature_probe"]["unguarded_accepted_steps"] == 4


def test_signed_roundtrip_and_cached_retry(trained):
    store, cohort, ledger, report, candidate = trained
    loaded = ReviewedStepEnsemble.model_validate_json(candidate.model_dump_json())
    loaded.verify(MODEL_KEY)
    assert (report, loaded) == evaluate_uncertainty(cohort, store, TENANT, MODEL_KEY, ledger)
    assert candidate.explicit_training_steps == 160
    assert report["candidate_ready_for_review"] and not report["production_activation"]
    assert len(report["risk_coverage_curves"]) == 5
    assert set(report["step_kind_slices"]) == {"retrieve", "reason", "verify", "answer"}
    assert "private-tenant" not in candidate.model_dump_json()
    assert "PRIVATE" not in candidate.model_dump_json()
    with pytest.raises(ValueError):
        loaded.model_copy(update={"simulation": True}).verify(MODEL_KEY)
    with pytest.raises(ValueError):
        loaded.verify(KEY)


def test_bootstrap_keeps_whole_candidate_families(trained, monkeypatch):
    import agent.reviewed_verifier_uncertainty as module

    _, cohort, _, _, _ = trained
    traces = cohort.traces()
    observed = []

    def inspect(sample, **kwargs):
        observed.append(sample)
        assert kwargs["explicit_steps_only"] is True
        assert all(t.split == "train" for t in sample)
        for family in {t.group_id for t in sample}:
            ids = [t.trace_id for t in sample if t.group_id == family]
            originals = [t.trace_id for t in traces if t.group_id == family]
            assert len(originals) == 2
            assert ids.count(originals[0]) == ids.count(originals[1])
        return train_process_reward_model(sample, **kwargs)

    monkeypatch.setattr(module, "train_process_reward_model", inspect)
    result = fit_ensemble(traces, UncertaintyPolicy(epochs=2), MODEL_KEY)
    assert len(observed) == 5
    assert len(set(result["bootstrap_fingerprints"])) == 5


def test_test_and_terminal_labels_cannot_change_training_or_calibration(trained):
    _, cohort, _, _, _ = trained
    traces = cohort.traces()
    policy = UncertaintyPolicy(epochs=4)
    first = fit_ensemble(traces, policy, MODEL_KEY)
    changed = [
        t.model_copy(
            update={
                "safe": not t.safe,
                "outcome_quality": 1 - t.outcome_quality,
                "steps": [s.model_copy(update={"step_label": 1 - s.step_label}) for s in t.steps]
                if t.split == "test"
                else t.steps,
            }
        )
        for t in traces
    ]
    second = fit_ensemble(changed, policy, MODEL_KEY)
    assert [m.weights for m in first["members"]] == [m.weights for m in second["members"]]
    for field in ("support", "calibration_temperature", "bootstrap_fingerprints"):
        assert first[field] == second[field]


@pytest.mark.parametrize("split", ["train", "validation"])
def test_explicit_labels_required_no_terminal_imputation(trained, split):
    traces = [
        t.model_copy(update={"steps": [s.model_copy(update={"step_label": None}) for s in t.steps]})
        if t.split == split
        else t
        for t in trained[1].traces()
    ]
    with pytest.raises(ValueError, match="label"):
        fit_ensemble(traces, UncertaintyPolicy(epochs=2), MODEL_KEY)


@pytest.mark.parametrize("mutation", ["policy", "confidence", "risk", "kind", "progress"])
def test_feature_guard_rejects_unseen_inputs(trained, mutation):
    steps = trained[1].traces()[0].steps
    scorer = ReviewedStepScorer(trained[4], MODEL_KEY)
    high_risk = False
    if mutation == "policy":
        steps = [s.model_copy(update={"policy_allowed": False}) for s in steps]
    elif mutation == "confidence":
        steps = [s.model_copy(update={"confidence": 0.1}) for s in steps]
    elif mutation == "kind":
        steps = [s.model_copy(update={"kind": "tool"}) for s in steps]
    elif mutation == "risk":
        high_risk = True
    else:
        steps = steps[:1]  # retrieve progress 1 instead of 0.25
    estimates = scorer.estimates(steps, high_risk)
    assert all(not row.feature_supported and not row.accepted for row in estimates)
    with pytest.raises(ValueError, match="nonempty"):
        scorer.estimates([])


def test_scoring_never_reads_targets(trained):
    scorer = ReviewedStepScorer(trained[4], MODEL_KEY)
    steps = trained[1].traces()[0].steps
    assert scorer.estimates(steps) == scorer.estimates(
        [s.model_copy(update={"step_label": 0}) for s in steps]
    )


def test_heterogeneous_families_create_spread_and_conservative_confidence(trained):
    traces = trained[1].traces()
    family = next(t.group_id for t in traces if t.split == "train")
    traces = [
        t.model_copy(update={"steps": [s.model_copy(update={"step_label": 0.0}) for s in t.steps]})
        if t.group_id == family
        else t
        for t in traces
    ]
    policy = UncertaintyPolicy(epochs=10)
    fit = fit_ensemble(traces, policy, MODEL_KEY)
    assert len({tuple(member.weights) for member in fit["members"]}) > 1
    candidate = trained[4].model_copy(update={"policy": policy, **fit}).seal(MODEL_KEY)
    estimates = ReviewedStepScorer(candidate, MODEL_KEY).estimates(traces[0].steps)
    assert any(e.spread > 0 for e in estimates)
    for row in estimates:
        assert row.conservative_confidence == pytest.approx(
            max(0, max(row.mean, 1 - row.mean) - 1.5 * row.spread)
        )
    policy = policy.model_copy(update={"maximum_spread": 1e-12})
    candidate = candidate.model_copy(update={"policy": policy}).seal(MODEL_KEY)
    assert all(not e.accepted for e in ReviewedStepScorer(candidate, MODEL_KEY).estimates(traces[0].steps))


def test_step_candidate_cannot_be_loaded_as_serving_trajectory_model(trained):
    from agent.process_supervision import ReviewedProcessCandidate
    from agent.verifier_ensemble import ProcessRewardEnsembleArtifact

    for schema in (ReviewedProcessCandidate, ProcessRewardEnsembleArtifact):
        with pytest.raises(ValueError):
            schema.model_validate(trained[4].model_dump())


@pytest.mark.parametrize(
    "fields",
    [
        {"curve_confidences": []},
        {"curve_confidences": [0.8, 0.5]},
        {"curve_confidences": [0.8, 0.8]},
        {"curve_confidences": [0.2, 0.8]},
        {"primary_confidence": 0.85},
        {"version": "future"},
        {"support_margin": float("nan")},
    ],
)
def test_policy_rejects_invalid_or_posthoc_grids(fields):
    with pytest.raises(ValueError):
        UncertaintyPolicy(**fields)


def test_unknowns_count_as_errors_and_abstention_is_not_zero_risk():
    rows = [
        {
            "family": "family",
            "target": None,
            "prediction": 1.0,
            "supported": True,
            "spread": 0.0,
            "confidence": 0.95,
        }
    ]
    curve = risk_curve(rows, 0.8, 0.15)
    assert curve["worst_case_macro_error"] == curve["worst_case_macro_error_upper_95"] == 1
    assert curve["review_coverage"] == 0 and curve["reviewed_macro_error"] is None
    abstain = risk_curve(rows, 1.0, 0.15)
    assert abstain["coverage"] == 0 and abstain["worst_case_macro_error_upper_95"] is None
    rows[0]["supported"] = False
    assert risk_curve(rows, 0.8, 0.15)["coverage"] == 0
    assert risk_curve(rows, 0.8, 0.15, guard=False)["coverage"] == 1


def test_family_macro_not_step_weighted_risk():
    base = {"prediction": 1.0, "supported": True, "spread": 0.0, "confidence": 0.95}
    rows = [{**base, "family": "long", "target": 1.0} for _ in range(100)]
    rows.append({**base, "family": "short", "target": 0.0})
    assert risk_curve(rows, 0.8, 0.15)["reviewed_macro_error"] == 0.5


def test_shared_exposure_and_policy_changes_require_fresh_families(trained):
    store, cohort, ledger, _, _ = trained
    with pytest.raises(ValueError, match="already exposed"):
        evaluate_uncertainty(cohort, store, TENANT, MODEL_KEY, ledger, UncertaintyPolicy(seed=42))
    with pytest.raises(ValueError, match="already exposed"):
        evaluate_process(cohort, store, TENANT, MODEL_KEY, ledger)
    with pytest.raises(ValueError, match="lineage|tenant"):
        evaluate_uncertainty(cohort, store, "other", MODEL_KEY, ledger)
    with pytest.raises(ValueError, match="independent"):
        evaluate_uncertainty(cohort, store, TENANT, KEY, ledger)


@pytest.mark.parametrize("mutation", ["delete", "tamper", "review"])
def test_live_lineage_revokes_cached_evidence(trained, tmp_path, mutation):
    from contextlib import closing

    original, cohort, _, report, candidate = trained
    store = ProcessSupervisionStore(
        ExecutionReplayStore(tmp_path / "clone.sqlite3", KEY, clock=lambda: 600.0)
    )
    with closing(
        sqlite3.connect(original.replay.path)
    ) as source:  # backup API instead of copying an open SQLite file
        with closing(sqlite3.connect(store.replay.path)) as target:
            source.backup(target)
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", KEY)
    # Evaluate once on the clone, then mutate the authoritative source.
    assert evaluate_uncertainty(cohort, store, TENANT, MODEL_KEY, ledger) == (report, candidate)
    event = cohort.members[0].snapshot.source.event_id
    with store.replay._db() as db:
        if mutation == "delete":
            db.execute("DELETE FROM observations WHERE event_id=?", (event,))
        elif mutation == "review":
            db.execute("DELETE FROM process_step_reviews WHERE event_id=?", (event,))
        else:
            db.execute(
                "UPDATE observations SET payload=json_set(payload, '$.confidence', 0.1) WHERE event_id=?",
                (event,),
            )
    with pytest.raises(ValueError):
        evaluate_uncertainty(cohort, store, TENANT, MODEL_KEY, ledger)


def test_cli_protects_inputs_and_active_models(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    monkeypatch.setenv("VERIFIER_ENSEMBLE_PATH", str(active))
    assert main(["drill", "--output", str(active)]) == 1 and not active.exists()
    db = tmp_path / "data.sqlite3"
    assert main(["--store", str(db), "drill", "--output", str(db) + "-wal"]) == 1
    source = tmp_path / "source.json"
    assert main(["evaluate", str(source), "--output", str(source), "--candidate", str(active)]) == 1
    assert not source.exists()


def test_cli_export_is_immutable_and_never_activates(trained, tmp_path, monkeypatch):
    store, cohort, ledger, _, _ = trained
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", KEY.decode())
    monkeypatch.setenv("PROCESS_SUPERVISION_MODEL_KEY", MODEL_KEY.decode())
    source, report, model = [tmp_path / name for name in ("cohort.json", "report.json", "model.json")]
    source.write_text(cohort.model_dump_json(), encoding="utf-8")
    args = [
        "--store",
        str(store.replay.path),
        "--ledger",
        str(ledger.path),
        "--tenant",
        TENANT,
        "evaluate",
        str(source),
        "--output",
        str(report),
        "--candidate",
        str(model),
        "--require-gate",
    ]
    assert main(args) == main(args) == 0
    assert not json.loads(report.read_text())["production_activation"]
    ReviewedStepEnsemble.model_validate_json(model.read_text()).verify(MODEL_KEY)
    model.write_text('{"owner": "preserve"}', encoding="utf-8")
    assert main(args) == 1 and json.loads(model.read_text()) == {"owner": "preserve"}


def test_cli_synthetic_export_refused(trained, tmp_path):
    cohort = trained[1].model_copy(update={"simulation": True}).seal(KEY)
    source, report, model = [tmp_path / name for name in ("cohort.json", "report.json", "model.json")]
    source.write_text(cohort.model_dump_json(), encoding="utf-8")
    assert (
        main(
            ["--tenant", TENANT, "evaluate", str(source), "--output", str(report), "--candidate", str(model)]
        )
        == 1
    )
    assert not model.exists() and not report.exists()
