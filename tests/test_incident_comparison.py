"""Frozen family comparisons with simulated labels and no real model or human reviews."""
import pytest

from agent.incident_comparison import comparison_report, register_comparison
from agent.incident_investigation import run_investigation
from agent.incident_models import InvestigationPlan, TrialReview
from agent.incident_store import IncidentConflict
from tests.test_incident_investigation import LiveAdapterDouble, bundle, label_all, shadow_bundle, store


async def registered(vault):
    parent = vault.create("alice", bundle(origin="operator"), InvestigationPlan(), consent=True)
    await run_investigation(vault, parent, "alice", LiveAdapterDouble())
    label_all(vault, parent)
    children = [vault.create("alice", shadow_bundle(index), InvestigationPlan(), consent=True, parent_id=parent) for index in range(3)]
    report = register_comparison(vault, parent, "alice", children)
    assert report["decision"] == "hold" and report["families_registered"] == 3
    return parent, children


@pytest.mark.asyncio
async def test_frozen_paired_comparison_includes_negative_outcomes_usage_and_abstention(tmp_path):
    vault = store(tmp_path)
    parent, children = await registered(vault)
    for index, child in enumerate(children):
        await run_investigation(vault, child, "alice", LiveAdapterDouble())
        vault.grant(child, "alice", "reviewer")
        for trial in vault.get(child, "alice")["trials"]:
            correct = (trial["arm"] == "candidate") if index < 2 else (trial["arm"] == "baseline")
            vault.review(child, "reviewer", TrialReview(trial_id=trial["trial_id"], trial_fingerprint=trial["fingerprint"],
                verdict="correct" if correct else "incorrect", safe=True, abstained=not correct,
                rationale="Simulated protocol label, not an independent human evaluation."))
    report = comparison_report(vault, parent, "alice")
    assert report["decision"] == "reviewed_comparison_complete" and report["serving_changed"] is False
    assert report["metrics"]["baseline"]["correctness"] == pytest.approx(1 / 3)
    assert report["metrics"]["candidate"]["correctness"] == pytest.approx(2 / 3)
    assert report["paired_regression_rate"] == pytest.approx(1 / 3)
    assert report["metrics"]["candidate"]["abstention_rate"] == pytest.approx(1 / 3)
    assert report["family_bootstrap_95_interval"][0] < 0  # Retain uncertainty, not a positive-lift claim.
    assert report["usage"]["billed_cost_usd"] is None and not report["usage"]["complete"]
    assert report["metrics"]["candidate"]["generation_latency_ms"]["observations"] == 6
    assert report["metrics"]["baseline"]["usage"]["admitted_calls"] == 6
    vault.delete(children[-1], "alice")
    held = comparison_report(vault, parent, "alice")
    assert held["decision"] == "hold" and held["paired_correctness_delta"] is None
    assert held["families_registered"] == 3 and held["families_fully_reviewed"] == 2
    with pytest.raises(IncidentConflict):
        register_comparison(vault, parent, "alice", children)


@pytest.mark.asyncio
async def test_comparison_cannot_cherry_pick_execute_before_registration_or_cross_tenants(tmp_path):
    vault = store(tmp_path)
    parent, children = await registered(vault)
    with pytest.raises(KeyError):
        comparison_report(vault, parent, "outsider")
    vault.revoke(parent, "alice", "independent-reviewer")
    assert "diagnostic_reviews_or_candidate_unavailable" in comparison_report(vault, parent, "alice")["hold_reasons"]
    vault.grant(parent, "alice", "independent-reviewer")
    second = vault.create("alice", bundle(origin="operator", family="second-parent"), InvestigationPlan(), consent=True)
    await run_investigation(vault, second, "alice", LiveAdapterDouble())
    label_all(vault, second)
    more = [vault.create("alice", shadow_bundle(index, "second"), InvestigationPlan(), consent=True, parent_id=second) for index in range(3)]
    await run_investigation(vault, more[0], "alice", LiveAdapterDouble())
    with pytest.raises(IncidentConflict):
        register_comparison(vault, second, "alice", more)
    with pytest.raises(ValueError):
        register_comparison(vault, second, "alice", more[:2])


@pytest.mark.asyncio
async def test_review_revocation_and_unsafe_outcomes_hold_report(tmp_path):
    vault = store(tmp_path)
    parent, children = await registered(vault)
    for child in children:
        await run_investigation(vault, child, "alice", LiveAdapterDouble())
        label_all(vault, child, correct_arms=("candidate",), unsafe=("candidate",))
    report = comparison_report(vault, parent, "alice")
    assert "reviewed_unsafe_candidate" in report["hold_reasons"]
    vault.revoke(children[0], "alice", "independent-reviewer")
    report = comparison_report(vault, parent, "alice")
    assert report["metrics"]["candidate"]["correctness"] is None
    assert report["metrics"]["candidate"]["correctness_lower"] == pytest.approx(2 / 3)
    assert report["metrics"]["candidate"]["correctness_upper"] == 1
