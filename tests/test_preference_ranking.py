import importlib
import json

import pytest
from pydantic import ValidationError

from agent.adaptive_compute import (
    ComputePolicy,
    ComputeSignals,
    DeliberationReceipt,
    _fingerprint,
    candidate_assessment,
    plan_compute,
    select_candidate,
    verify_receipt,
)
from agent.execution_replay import ExecutionReplayStore, digest
from agent.preference_ranking import (
    PreferenceArtifact,
    PreferenceRanker,
    RankingCandidate,
    features,
    freeze_preferences,
    train_preferences,
    validate_preference_lineage,
)
from agent.prospective_validation import HoldoutLedger
from evals.preference_evaluation import evaluate_preferences, main, run_drill, seed_drill, write_once

REPLAY_KEY = b"unit-test-preference-replay-key-32-bytes"
ARTIFACT_KEY = b"unit-test-preference-artifact-key-32-bytes"
TENANT = "private-tenant"


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    store = ExecutionReplayStore(tmp_path_factory.mktemp("preferences") / "replay.sqlite3", REPLAY_KEY)
    cohort = seed_drill(store, TENANT, origin="runtime")
    artifact = train_preferences(cohort, REPLAY_KEY, ARTIFACT_KEY, TENANT)
    return store, cohort, artifact


def build_candidates():
    return [
        candidate_assessment(
            candidate_id=name,
            answer="PRIVATE answer text",
            confidence=conf,
            grounded=True,
            conformal_decision="release",
            claim_keys=["evidence"],
            token_count=tokens,
            latency_ms=10,
        )
        for name, conf, tokens in (("short", 0.8, 100), ("long", 0.9, 500))
    ]


def build_plan(**updates):
    return plan_compute(
        ComputeSignals(
            route="rag",
            grounding_action="repair",
            grounding_confidence=0.4,
            uncertainty_decision="abstain",
            evidence_count=2,
            **updates,
        ),
        ComputePolicy(max_candidates=5),
    )


def test_control_drill_learns_preferences_and_holds_shift():
    report = run_drill()
    assert report["gate_passed"]
    assert report["clean"]["correct_selections"] == 40
    assert report["clean"]["confidence_baseline_correct"] == 20
    assert report["clean"]["reranked_families"] == 20
    assert not report["clean"]["production_candidate"]
    assert "feature_distribution_shift" in report["shifted"]["failure_reasons"]


def test_human_reviewed_training_is_tenant_signed_and_content_free(learned):
    _, cohort, artifact = learned
    artifact.verify(ARTIFACT_KEY)
    assert artifact.training_families == artifact.pair_count == 40
    assert len(artifact.members) == 7
    assert not artifact.simulation
    text = cohort.model_dump_json()
    assert TENANT not in text and "drill-reviewer" not in text and "synthetic control" not in text
    view = RankingCandidate(candidate_id="opaque", confidence=0.8, token_count=100, latency_ms=10)
    assert len(features(view)) == 4
    with pytest.raises(ValidationError):
        RankingCandidate(**view.model_dump(), answer="do not store")


def test_test_labels_cannot_change_training_weights_or_support(learned):
    _, cohort, artifact = learned
    altered = cohort.model_copy(deep=True)
    for pool in altered.pools:
        if pool.split == "test":
            for row in pool.candidates:
                row.label.verdict = "incorrect"
                row.label.fingerprint = digest(
                    REPLAY_KEY, "ReviewLabel", row.label.model_dump(exclude={"fingerprint"})
                )
    altered.seal(REPLAY_KEY)
    changed = train_preferences(altered, REPLAY_KEY, ARTIFACT_KEY, TENANT)
    assert changed.model_dump() == artifact.model_dump()


def test_activation_rejects_synthetic_wrong_tenant_weak_key_and_tamper(learned):
    _, _, artifact = learned
    synthetic = artifact.model_copy(update={"simulation": True}).seal(ARTIFACT_KEY)
    with pytest.raises(ValueError, match="synthetic"):
        PreferenceRanker(synthetic, ARTIFACT_KEY, TENANT)
    with pytest.raises(ValueError, match="tenant"):
        PreferenceRanker(artifact, ARTIFACT_KEY, "other-tenant")
    with pytest.raises(ValueError):
        PreferenceRanker(artifact, b"weak", TENANT)
    with pytest.raises(ValueError, match="identity/key"):
        train_preferences(learned[1], REPLAY_KEY, REPLAY_KEY, TENANT)
    tampered = artifact.model_copy(deep=True)
    tampered.members[0][0] += 0.01
    with pytest.raises(ValueError, match="integrity"):
        PreferenceRanker(tampered, ARTIFACT_KEY, TENANT)
    payload = artifact.model_dump()
    payload["members"][0][0] = float("nan")
    with pytest.raises(ValidationError):
        PreferenceArtifact.model_validate(payload)


@pytest.mark.parametrize(
    "scope,risk,confidence,tokens,latency",
    [
        ("web", False, 0.8, 100, 10),
        ("rag", True, 0.8, 100, 10),
        ("rag", False, 0.99, 100, 10),
        ("rag", False, 0.8, 900, 10),
        ("rag", False, 0.8, 100, 100),
    ],
)
def test_ood_scope_and_feature_shift_preserve_baseline(learned, scope, risk, confidence, tokens, latency):
    ranker = PreferenceRanker(learned[2], ARTIFACT_KEY, TENANT)
    pool = [
        RankingCandidate(candidate_id="a", confidence=confidence, token_count=tokens, latency_ms=latency),
        RankingCandidate(candidate_id="b", confidence=0.9, token_count=500, latency_ms=10),
    ]
    decision = ranker.rank(pool, "b", route=scope, high_risk=risk)
    assert decision.status == "fallback" and decision.selected_candidate_id == "b"
    assert decision.reason == "unsupported_scope_or_feature_ood"


def test_uncertain_preferences_do_not_override_baseline(learned):
    artifact = learned[2].model_copy(deep=True)
    artifact.members = [[0] * 4 for _ in range(7)]
    artifact.seal(ARTIFACT_KEY)
    ranker = PreferenceRanker(artifact, ARTIFACT_KEY, TENANT)
    views = [
        RankingCandidate(
            candidate_id=row.candidate_id,
            confidence=row.confidence,
            token_count=row.token_count,
            latency_ms=row.latency_ms,
        )
        for row in build_candidates()
    ]
    assert ranker.rank(views, "long", route="rag", high_risk=False).selected_candidate_id == "long"
    with pytest.raises(ValueError, match="pool"):
        ranker.rank(views + [views[0]], "long", route="rag", high_risk=False)


def test_live_selection_reranks_and_receipt_binds_decision(learned):
    plan, candidates = build_plan(), build_candidates()
    assert select_candidate(plan, candidates).selected_candidate_id == "long"
    receipt = select_candidate(
        plan, candidates, preference_ranker=PreferenceRanker(learned[2], ARTIFACT_KEY, TENANT)
    )
    assert receipt.selected_candidate_id == "short"
    assert receipt.preference_ranking["status"] == "reranked"
    assert receipt.preference_ranking["artifact_fingerprint"] == learned[2].fingerprint
    assert verify_receipt(receipt)
    receipt.preference_ranking["selected_candidate_id"] = "long"
    assert not verify_receipt(receipt)


@pytest.mark.parametrize("mutation", ["grounding", "conformal", "confidence", "consensus", "budget"])
def test_preference_model_cannot_bypass_release_checks(learned, mutation):
    candidates = build_candidates()
    plan = build_plan()
    if mutation == "grounding":
        candidates[0].grounded = False
    elif mutation == "conformal":
        candidates[0].conformal_decision = "abstain"
    elif mutation == "confidence":
        candidates[0].confidence = 0.1
    elif mutation == "consensus":
        candidates[0].claim_keys = ["unrelated"]
    else:
        candidates[0].token_count = plan.token_budget + 1
    receipt = select_candidate(
        plan, candidates, preference_ranker=PreferenceRanker(learned[2], ARTIFACT_KEY, TENANT)
    )
    assert receipt.selected_candidate_id != "short"
    assert receipt.status != "released"  # One remaining answer cannot establish consensus.


def test_legacy_receipts_and_duplicate_candidate_checks():
    receipt = select_candidate(build_plan(), build_candidates())
    old_payload = receipt.model_dump(mode="json", exclude={"preference_ranking", "receipt_fingerprint"})
    old_payload["receipt_fingerprint"] = _fingerprint(old_payload)
    assert verify_receipt(DeliberationReceipt.model_validate(old_payload))
    with pytest.raises(ValueError, match="duplicate"):
        select_candidate(build_plan(), [build_candidates()[0]] * 2)
    fallback = select_candidate(build_plan(), build_candidates(), preference_unavailable=True)
    assert fallback.selected_candidate_id == "long"
    assert fallback.preference_ranking["reason"] == "preference_ranker_unavailable"


def test_complete_review_pools_and_cutoff_are_required(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", REPLAY_KEY)
    cohort = seed_drill(store, TENANT, origin="runtime")
    first = next(pool for pool in cohort.pools if pool.split == "train")
    event = first.candidates[0].observation.event_id
    with store._db() as db:
        db.execute("DELETE FROM labels WHERE event_id=?", (event,))
    frozen = freeze_preferences(store, TENANT, cutoff=200, embargo_seconds=50)
    assert frozen.exclusions["incomplete_review_pool"] == 1
    assert first.task_family not in {pool.task_family for pool in frozen.pools}
    other = next(pool for pool in frozen.pools if pool.split == "train")
    label = other.candidates[0].label.model_copy(update={"reviewed_at": 250.0})
    label.fingerprint = digest(REPLAY_KEY, "ReviewLabel", label.model_dump(exclude={"fingerprint"}))
    with store._db() as db:
        db.execute("UPDATE labels SET payload=? WHERE event_id=?", (label.model_dump_json(), label.event_id))
    assert freeze_preferences(store, TENANT, cutoff=200, embargo_seconds=50).exclusions["late_label"] == 1
    with pytest.raises(ValueError, match="revoked"):
        validate_preference_lineage(cohort, store, TENANT)


def test_synthetic_pools_and_insufficient_training_are_held(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", REPLAY_KEY)
    cohort = seed_drill(store, TENANT, families=10)
    with pytest.raises(ValueError, match="20 contrasting"):
        train_preferences(cohort, REPLAY_KEY, ARTIFACT_KEY, TENANT)
    clean = freeze_preferences(store, TENANT, cutoff=200, embargo_seconds=50)
    assert not clean.pools and clean.exclusions["ungrounded_or_synthetic"] == 20
    with pytest.raises(ValueError, match="time window"):
        freeze_preferences(store, TENANT, cutoff=1000, embargo_seconds=50)


def test_shared_holdout_exact_retry_and_revocation(learned, tmp_path):
    store, cohort, _ = learned
    ledger = HoldoutLedger(tmp_path / "holdout.sqlite3", REPLAY_KEY)
    report, artifact = evaluate_preferences(cohort, store, TENANT, ARTIFACT_KEY, ledger)
    assert report["gate_passed"] and report["production_candidate"]
    assert evaluate_preferences(cohort, store, TENANT, ARTIFACT_KEY, ledger) == (report, artifact)
    test_families = [pool.task_family for pool in cohort.pools if pool.split == "test"]
    with pytest.raises(ValueError, match="already exposed"):
        ledger.reserve_families("another-study", cohort.fingerprint, test_families[:1])
    with pytest.raises(ValueError, match="tenant"):
        evaluate_preferences(cohort, store, "other", ARTIFACT_KEY, ledger)
    with pytest.raises(ValueError, match="share"):
        evaluate_preferences(
            cohort, store, TENANT, ARTIFACT_KEY, HoldoutLedger(tmp_path / "other.sqlite3", ARTIFACT_KEY)
        )


def test_source_deletion_blocks_cached_report(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", REPLAY_KEY)
    cohort = seed_drill(store, TENANT, origin="runtime")
    ledger = HoldoutLedger(tmp_path / "ledger.sqlite3", REPLAY_KEY)
    evaluate_preferences(cohort, store, TENANT, ARTIFACT_KEY, ledger)
    store.delete_tenant(TENANT)
    with pytest.raises(ValueError, match="revoked"):
        evaluate_preferences(cohort, store, TENANT, ARTIFACT_KEY, ledger)


def test_earliest_unreviewed_family_request_cannot_be_replaced(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", REPLAY_KEY)
    cohort = seed_drill(store, TENANT, origin="runtime")
    group = digest(REPLAY_KEY, "request", [TENANT, "train-0"])
    earliest = next(pool for pool in cohort.pools if pool.request_group == group)
    with store._db() as db:
        db.execute("DELETE FROM labels WHERE event_id=?", (earliest.candidates[0].observation.event_id,))
    store.clock = lambda: 160.0
    plan = build_plan()
    ids = store.capture(
        plan,
        select_candidate(plan, build_candidates()),
        tenant=TENANT,
        request_id="later-reviewed",
        consent=True,
        task_family="train-family-0",
    )
    for event in ids:
        store.review(TENANT, event, verdict="correct", unsafe=False, reviewer="reviewer")
    store.clock = lambda: 500.0
    frozen = freeze_preferences(store, TENANT, cutoff=200, embargo_seconds=50)
    assert earliest.task_family not in {pool.task_family for pool in frozen.pools}
    assert frozen.exclusions["repeated_family_requests"] == 1
    assert frozen.exclusions["incomplete_review_pool"] == 1


def test_unsafe_and_regressing_heldout_preferences_block_candidate(tmp_path):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", REPLAY_KEY)
    cohort = seed_drill(store, TENANT, origin="runtime")
    for pool in cohort.pools:
        if pool.split != "test":
            continue
        for row in pool.candidates:
            # Fixture-only inversion: simulate reviewed outcomes that do not generalize.
            label = row.label.model_copy(
                update={"verdict": "incorrect" if row.preferred else "correct", "unsafe": row.preferred}
            )
            label.fingerprint = digest(REPLAY_KEY, "ReviewLabel", label.model_dump(exclude={"fingerprint"}))
            with store._db() as db:
                db.execute(
                    "UPDATE labels SET payload=? WHERE event_id=?", (label.model_dump_json(), label.event_id)
                )
    changed = freeze_preferences(store, TENANT, cutoff=200, embargo_seconds=50)
    report, _ = evaluate_preferences(
        changed, store, TENANT, ARTIFACT_KEY, HoldoutLedger(tmp_path / "ledger.sqlite3", REPLAY_KEY)
    )
    assert not report["production_candidate"]
    assert report["unsafe_selections"] == 40
    assert "unsafe_selection" in report["failure_reasons"]
    assert "paired_utility_regression" in report["failure_reasons"]


def test_cli_freeze_and_failed_evaluation_do_not_create_candidate(tmp_path, monkeypatch):
    store = ExecutionReplayStore(tmp_path / "replay.sqlite3", REPLAY_KEY)
    seed_drill(store, TENANT, shifted=True, origin="runtime")
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", REPLAY_KEY.decode())
    monkeypatch.setenv("PREFERENCE_RANKING_KEY", ARTIFACT_KEY.decode())
    source, report, artifact = (tmp_path / name for name in ("cohort.json", "report.json", "candidate.json"))
    shared = ["--store", str(store.path), "--tenant", TENANT, "--ledger", str(tmp_path / "ledger.sqlite3")]
    assert (
        main([*shared, "freeze", "--cutoff", "200", "--embargo-seconds", "50", "--output", str(source)]) == 0
    )
    assert (
        main(
            [
                *shared,
                "evaluate",
                str(source),
                "--output",
                str(report),
                "--candidate",
                str(artifact),
                "--require-gate",
            ]
        )
        == 1
    )
    assert report.exists() and not artifact.exists()
    assert "feature_distribution_shift" in json.loads(report.read_text())["failure_reasons"]
    assert main(["freeze", "--cutoff", "200", "--output", str(tmp_path / "no-identity.json")]) == 1


def test_model_failure_or_illegal_winner_cannot_change_baseline(learned):
    ranker = PreferenceRanker(learned[2], ARTIFACT_KEY, TENANT)

    def illegal_winner(*args, **kwargs):
        from agent.preference_ranking import RankingDecision

        return RankingDecision(
            artifact_fingerprint=learned[2].fingerprint,
            status="reranked",
            reason="invalid stub",
            baseline_candidate_id="long",
            selected_candidate_id="outsider",
        )

    ranker.rank = illegal_winner
    receipt = select_candidate(build_plan(), build_candidates(), preference_ranker=ranker)
    assert receipt.selected_candidate_id == "long"
    assert receipt.preference_ranking["reason"] == "preference_ranker_unavailable"


def test_cohort_tamper_and_family_leakage_are_rejected(learned):
    cohort = learned[1].model_copy(deep=True)
    cohort.pools[1].task_family = cohort.pools[0].task_family
    cohort.seal(REPLAY_KEY)
    with pytest.raises(ValueError, match="leakage"):
        cohort.verify(REPLAY_KEY)
    cohort = learned[1].model_copy(deep=True)
    cohort.pools[0].candidates[0].observation.confidence = 0.1
    cohort.seal(REPLAY_KEY)
    with pytest.raises(ValueError, match="source integrity"):
        cohort.verify(REPLAY_KEY)


def test_cli_writes_only_gated_candidate_and_preserves_active(learned, tmp_path, monkeypatch):
    store, cohort, _ = learned
    monkeypatch.setenv("EXECUTION_REPLAY_KEY", REPLAY_KEY.decode())
    monkeypatch.setenv("PREFERENCE_RANKING_KEY", ARTIFACT_KEY.decode())
    active = tmp_path / "active.json"
    write_once(active, {"owner": "do not change"})
    monkeypatch.setenv("PREFERENCE_RANKING_PATH", str(active))
    source = tmp_path / "cohort.json"
    write_once(source, cohort.model_dump(mode="json"))
    report, candidate = tmp_path / "report.json", tmp_path / "candidate.json"
    command = [
        "--store",
        str(store.path),
        "--ledger",
        str(tmp_path / "ledger.sqlite3"),
        "--tenant",
        TENANT,
        "evaluate",
        str(source),
        "--output",
        str(report),
        "--candidate",
        str(candidate),
        "--require-gate",
    ]
    assert main(command) == 0
    assert main(command) == 0
    assert candidate.exists() and json.loads(active.read_text()) == {"owner": "do not change"}
    bad = command.copy()
    bad[bad.index("--candidate") + 1] = str(active)
    assert main(bad) == 1
    assert len(store.records(TENANT)) == 160
    with pytest.raises(ValueError, match="different evidence"):
        write_once(report, {"different": True})


@pytest.mark.asyncio
async def test_runtime_missing_model_falls_back_without_changing_safety(monkeypatch, tmp_path):
    research = importlib.import_module("agent.research_assistant")
    monkeypatch.setattr(research, "PREFERENCE_RANKING_ENABLED", True)
    monkeypatch.setattr(research, "PREFERENCE_RANKING_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(research, "PREFERENCE_RANKING_KEY", ARTIFACT_KEY)
    monkeypatch.setattr(research, "EXECUTION_REPLAY_ENABLED", False)
    monkeypatch.setattr(research, "UNCERTAINTY_CALIBRATION_ENABLED", False)
    monkeypatch.setattr(research, "ADAPTIVE_COMPUTE_INTEGRITY_KEY", None)

    async def answer(*args, **kwargs):
        return "AgentForge uses LangGraph for agent orchestration [source](https://example.com/architecture)."

    monkeypatch.setattr(research, "_call_llm", answer)
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
    result = await research.adaptive_deliberation_agent(state, {"configurable": {"user_id": TENANT}})
    receipt = DeliberationReceipt.model_validate(result["adaptive_compute_receipt"])
    assert receipt.status == "released"
    assert receipt.preference_ranking["reason"] == "preference_ranker_unavailable"
    assert verify_receipt(receipt)
