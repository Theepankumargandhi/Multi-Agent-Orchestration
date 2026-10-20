"""Authored replay experiments and access-control tests; no live-provider accuracy claims."""
import asyncio
import hashlib
import json
import sqlite3

import pytest

from agent.incident_investigation import analyze, run_investigation
from agent.incident_models import (
    IncidentBundle,
    IncidentDocument,
    IncidentPipeline,
    InvestigationPlan,
    TrialReview,
    intervention,
)
from agent.incident_pipeline import ExtractiveControlReplay, prepare_context
from agent.incident_store import IncidentConflict, IncidentStore


def bundle(*, family="packing-fixture", origin="synthetic", pipeline=None):
    text = "Background information without the answer. " * 15 + "\nThe telescope request limit is 42.\nOnly for authenticated users."
    return IncidentBundle(origin=origin, family=family, query="What is the telescope request limit for authenticated users?",
        observed_answer="Background information.", documents=[IncidentDocument(document_id="manual", text=text,
        version_sha256=hashlib.sha256(text.encode()).hexdigest())], retrieved_ids=["manual"], ranked_ids=["manual"],
        pipeline=pipeline or IncidentPipeline(context_chars=180), model="groq:example-model", prompt="Answer from supplied documents.",
        application_revision="authored-fixture-v1")


def store(tmp_path, **kwargs):
    return IncidentStore(tmp_path / "private.db", b"a" * 32, **kwargs)


def shadow_bundle(index, prefix="fresh"):
    value = bundle(origin="operator", family=f"{prefix}-{index}").model_dump()
    product = ("satellite", "cache", "billing")[index]
    value["query"] = value["query"].replace("telescope", product)
    for document in value["documents"]:
        document["text"] = document["text"].replace("telescope", product)
        document["version_sha256"] = hashlib.sha256(document["text"].encode()).hexdigest()
    return IncidentBundle.model_validate(value)


def label_all(vault, identifier, *, correct_arms=("packing",), unsafe=()):
    vault.grant(identifier, "alice", "independent-reviewer")
    data = vault.get(identifier, "alice")
    for trial in data["trials"]:
        verdict = "correct" if trial["arm"] in correct_arms else "incorrect"
        vault.review(identifier, "independent-reviewer", TrialReview(trial_id=trial["trial_id"], trial_fingerprint=trial["fingerprint"],
            verdict=verdict, safe=trial["arm"] not in unsafe, rationale="Authored protocol control, not a genuine human judgement."))


class LiveAdapterDouble(ExtractiveControlReplay):
    """Trusted unit-test double for admission logic; makes zero API requests."""
    execution_kind = "live_provider"


def test_one_factor_arms_and_context_reproduction():
    original = bundle()
    before = original.model_dump()
    context, _ = prepare_context(original, original.pipeline)
    assert "limit is 42" not in context
    for arm in ("retrieval", "reranking", "packing", "generation"):
        alternative = intervention(original.pipeline, arm)
        assert [key for key, value in original.pipeline.model_dump().items() if getattr(alternative, key) != value] == [arm]
    context, ids = prepare_context(original, intervention(original.pipeline, "packing"))
    assert "limit is 42" in context and "authenticated users" in context and ids == ["manual"]
    assert original.model_dump() == before


@pytest.mark.parametrize("alter", ["hash", "duplicate", "unknown_rank", "credential", "large"])
def test_capture_rejects_invalid_or_credentialed_snapshots(alter):
    data = bundle().model_dump()
    if alter == "hash":
        data["documents"][0]["version_sha256"] = "0" * 64
    elif alter == "duplicate":
        data["documents"].append(data["documents"][0])
    elif alter == "unknown_rank":
        data["ranked_ids"] = ["absent"]
    elif alter == "credential":
        data["query"] = "Authorization: Bearer " + "x" * 30
    else:
        data["query"] = "x" * 2001
    with pytest.raises(ValueError):
        IncidentBundle.model_validate(data)


def test_encryption_tenant_isolation_expiry_and_deletion(tmp_path):
    clock = [1000.0]
    vault = store(tmp_path, clock=lambda: clock[0])
    with pytest.raises(ValueError):
        vault.create("alice", bundle(), InvestigationPlan(), consent=False)
    identifier = vault.create("alice", bundle(), InvestigationPlan(), consent=True, ttl_seconds=60)
    assert b"telescope" not in vault.path.read_bytes() and b"Background" not in vault.path.read_bytes()
    with pytest.raises(KeyError):
        vault.get(identifier, "bob")
    with pytest.raises(KeyError):
        vault.review_bundle(identifier, "bob")
    vault.grant(identifier, "alice", "bob")
    assert vault.review_bundle(identifier, "bob")["bundle"]["query"] == bundle().query
    clock[0] = 1060
    with pytest.raises(KeyError):
        vault.get(identifier, "alice")
    assert vault.purge_expired() == 1
    identifier = vault.create("alice", bundle(), InvestigationPlan(), consent=True)
    vault.delete(identifier, "alice")
    with pytest.raises(KeyError):
        vault.get(identifier, "alice")


@pytest.mark.parametrize("tamper", ["ciphertext", "expiry", "tenant"])
def test_aead_rejects_tampered_ciphertext_and_authenticated_metadata(tmp_path, tamper):
    vault = store(tmp_path)
    identifier = vault.create("alice", bundle(), InvestigationPlan(), consent=True)
    with sqlite3.connect(vault.path) as db:
        if tamper == "ciphertext":
            db.execute("UPDATE incidents SET body=? WHERE id=?", (b"bad" * 20, identifier))
        elif tamper == "expiry":
            db.execute("UPDATE incidents SET expires=expires+100 WHERE id=?", (identifier,))
        else:
            db.execute("UPDATE incidents SET tenant=? WHERE id=?", (vault.principal("mallory"), identifier))
    with pytest.raises(IncidentConflict):
        vault.get(identifier, "mallory" if tamper == "tenant" else "alice")


@pytest.mark.asyncio
async def test_real_context_operations_but_control_results_never_become_live_diagnosis(tmp_path):
    vault = store(tmp_path)
    identifier = vault.create("alice", bundle(), InvestigationPlan(), consent=True)
    report = await run_investigation(vault, identifier, "alice", ExtractiveControlReplay())
    assert report["calls_admitted"] == 10 and report["analysis"]["decision"] == "abstain"
    data = vault.get(identifier, "alice")
    assert all("limit is 42" in row["answer"] for row in data["trials"] if row["arm"] == "packing")
    assert all("limit is 42" not in row["answer"] for row in data["trials"] if row["arm"] != "packing")
    label_all(vault, identifier)
    report = vault.report(identifier, "alice")
    assert report["analysis"]["supported_component"] == "packing"
    assert report["analysis"]["reason"] == "control_only_not_a_live_diagnosis"
    assert report["analysis"]["decision"] == "abstain" and not report["usage_complete"]
    assert "telescope" not in json.dumps(report) and "Background" not in json.dumps(report)
    with pytest.raises(IncidentConflict):
        await run_investigation(vault, identifier, "alice", ExtractiveControlReplay())


@pytest.mark.asyncio
@pytest.mark.parametrize("arms,unsafe,reason", [
    (("packing", "generation"), (), "multiple_explanations"),
    ((), (), "no_supported_intervention"),
    (("packing",), ("packing",), "no_supported_intervention"),
    (("baseline", "packing"), (), "baseline_failure_not_reproduced"),
])
async def test_competing_explanations_failed_baselines_and_unsafe_fixes_abstain(tmp_path, arms, unsafe, reason):
    vault = store(tmp_path)
    identifier = vault.create("alice", bundle(), InvestigationPlan(), consent=True)
    await run_investigation(vault, identifier, "alice", ExtractiveControlReplay())
    label_all(vault, identifier, correct_arms=arms, unsafe=unsafe)
    assert vault.report(identifier, "alice")["analysis"]["reason"] == reason


@pytest.mark.asyncio
async def test_budget_hold_before_any_trial_and_terminal_failure_stops_all_branches(tmp_path):
    vault = store(tmp_path)
    identifier = vault.create("alice", bundle(), InvestigationPlan(max_calls=1), consent=True)
    report = await run_investigation(vault, identifier, "alice", ExtractiveControlReplay())
    assert report["status"] == "budget_hold" and vault.get(identifier, "alice")["trials"] == []
    class FailedReplay(ExtractiveControlReplay):
        async def answer(self, *_args):
            raise RuntimeError("PRIVATE_PROVIDER_DETAIL")
    identifier = vault.create("alice", bundle(), InvestigationPlan(), consent=True)
    report = await run_investigation(vault, identifier, "alice", FailedReplay())
    assert report["calls_admitted"] == 1 and report["state"] == "incomplete"
    assert "PRIVATE_PROVIDER_DETAIL" not in json.dumps(vault.get(identifier, "alice"))


@pytest.mark.asyncio
async def test_interrupted_admission_is_not_silently_retried(tmp_path):
    vault = store(tmp_path)
    identifier = vault.create("alice", bundle(), InvestigationPlan(), consent=True)
    class Interrupted(ExtractiveControlReplay):
        async def answer(self, *_args):
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await run_investigation(vault, identifier, "alice", Interrupted())
    assert vault.get(identifier, "alice")["trials"][0]["status"] == "started"
    with pytest.raises(IncidentConflict):
        await run_investigation(vault, identifier, "alice", ExtractiveControlReplay())


@pytest.mark.asyncio
async def test_independent_immutable_review_and_stale_bindings(tmp_path):
    vault = store(tmp_path)
    identifier = vault.create("alice", bundle(), InvestigationPlan(), consent=True)
    await run_investigation(vault, identifier, "alice", ExtractiveControlReplay())
    trial = vault.get(identifier, "alice")["trials"][0]
    review = TrialReview(trial_id=trial["trial_id"], trial_fingerprint=trial["fingerprint"], verdict="incorrect", safe=True, rationale="Independent review control")
    with pytest.raises(IncidentConflict):
        vault.review(identifier, "alice", review)
    vault.grant(identifier, "alice", "bob")
    with pytest.raises(IncidentConflict):
        vault.review(identifier, "bob", review.model_copy(update={"trial_fingerprint": "0" * 64}))
    vault.review(identifier, "bob", review)
    vault.review(identifier, "bob", review)
    with pytest.raises(IncidentConflict):
        vault.review(identifier, "bob", review.model_copy(update={"verdict": "correct"}))
    assert analyze(vault.get(identifier, "alice"))["decision"] == "abstain"
    vault.revoke(identifier, "alice", "bob")
    with pytest.raises(KeyError):
        vault.review_bundle(identifier, "bob")


@pytest.mark.asyncio
async def test_shadow_gate_requires_fresh_reviewed_families_and_retains_deleted_exposures(tmp_path):
    # An operator-origin/live-adapter double exercises gate mechanics only, not real model quality.
    vault = store(tmp_path)
    parent = vault.create("alice", bundle(origin="operator"), InvestigationPlan(), consent=True)
    await run_investigation(vault, parent, "alice", LiveAdapterDouble())
    label_all(vault, parent)
    proposal = vault.report(parent, "alice")["analysis"]
    assert proposal["decision"] == "supported_under_replay"
    assert vault.shadow_gate(parent, "alice")["decision"] == "hold"
    shadows = []
    for index in range(3):
        child = vault.create("alice", shadow_bundle(index), InvestigationPlan(), consent=True, parent_id=parent)
        shadows.append(child)
        await run_investigation(vault, child, "alice", LiveAdapterDouble())
        label_all(vault, child, correct_arms=("candidate",))
    gate = vault.shadow_gate(parent, "alice")
    assert gate["decision"] == "ready_for_owner_review" and not gate["serving_changed"]
    vault.revoke(parent, "alice", "independent-reviewer")
    assert vault.shadow_gate(parent, "alice")["decision"] == "hold"
    vault.grant(parent, "alice", "independent-reviewer")
    vault.delete(shadows[0], "alice")
    assert vault.shadow_gate(parent, "alice")["decision"] == "hold"
    with pytest.raises(IncidentConflict):
        vault.create("alice", bundle(origin="operator", family="fresh-0"), InvestigationPlan(), consent=True, parent_id=parent)
    with pytest.raises(IncidentConflict):
        vault.create("alice", bundle(origin="operator"), InvestigationPlan(), consent=True, parent_id=parent)
    with pytest.raises(IncidentConflict):
        vault.create("alice", bundle(origin="operator", family="renamed-original"), InvestigationPlan(), consent=True, parent_id=parent)


@pytest.mark.asyncio
async def test_capture_source_binding_and_expiring_run_window_hold_before_requests(tmp_path, monkeypatch):
    vault = store(tmp_path)
    identifier = vault.create("alice", bundle(), InvestigationPlan(), consent=True, ttl_seconds=60)
    with pytest.raises(IncidentConflict):
        await run_investigation(vault, identifier, "alice", ExtractiveControlReplay())
    identifier = vault.create("alice", bundle(), InvestigationPlan(), consent=True)
    monkeypatch.setattr("agent.incident_store.runtime_fingerprint", lambda: "different-source")
    with pytest.raises(IncidentConflict):
        await run_investigation(vault, identifier, "alice", ExtractiveControlReplay())
    assert vault.get(identifier, "alice")["trials"] == []


@pytest.mark.asyncio
async def test_live_adapter_uses_declared_budget_schema_and_citation_scope(monkeypatch):
    from agent.incident_models import AnswerDraft
    from agent.incident_pipeline import LiveIncidentReplay, prepare_replay
    seen = []
    class Model:
        def with_structured_output(self, schema):
            assert schema is AnswerDraft
            return self
        async def ainvoke(self, messages, config=None):
            assert messages and config["callbacks"]
            return AnswerDraft(answer="42", citations=["manual"])
    def build(identifier, **kwargs):
        seen.append((identifier, kwargs))
        return Model()
    monkeypatch.setattr("code_agent.providers.build_provider", build)
    source, plan = bundle(), InvestigationPlan()
    replay = LiveIncidentReplay(source.model, plan)
    result = await replay.answer(prepare_replay(source, source.pipeline, plan.max_completion_tokens), source.pipeline)
    assert result["input_tokens"] is None and seen == [(source.model, {"max_completion_tokens": 512, "timeout_seconds": 20})]


@pytest.mark.asyncio
async def test_unsafe_shadow_holds_repair(tmp_path):
    vault = store(tmp_path)
    parent = vault.create("alice", bundle(origin="operator"), InvestigationPlan(), consent=True)
    await run_investigation(vault, parent, "alice", LiveAdapterDouble())
    label_all(vault, parent)
    for index in range(3):
        child = vault.create("alice", shadow_bundle(index, "unsafe"), InvestigationPlan(), consent=True, parent_id=parent)
        await run_investigation(vault, child, "alice", LiveAdapterDouble())
        label_all(vault, child, correct_arms=("baseline", "candidate"), unsafe=("candidate",) if index == 0 else ())
    assert vault.shadow_gate(parent, "alice")["decision"] == "hold"
