"""Private pilot provenance and matched-arm admission. No live providers or application imports."""
import copy
import hashlib
import shutil

import pytest

from code_agent.upgrade_inputs import CounterexamplePolicy
from code_agent.upgrade_models import fingerprint
from code_agent.upgrade_study import StudyRequest, StudyTask, freeze_study, run_study
from evals.upgrade_counterexample_evaluation import control_spec
from evals.upgradepilot_evaluation import FIXTURES


def migration_report(spec, **overrides):
    """Controller-shaped receipt double; no fixture application executes on the host."""
    report = {"protocol": spec.protocol, "task_id": spec.task_id, "family": spec.family,
        "spec_fingerprint": fingerprint(spec.model_dump()), "image_id": spec.image,
        "status": "held", "reason": "authored-hold", "duration_ms": 1.0, "model_calls": 0,
        "live_provider_calls": 0, "usage_complete": False, "provider_input_tokens": None,
        "provider_output_tokens": None, "billed_cost_usd": None, "source_unchanged": True, "patch": ""}
    report.update(overrides)
    if report["patch"]:
        report["patch_sha256"] = hashlib.sha256(report["patch"].encode()).hexdigest()
    report["fingerprint"] = fingerprint(report)
    return report


def setup_study(tmp_path, monkeypatch):
    repository = tmp_path / "repositories/snapshot"
    shutil.copytree(FIXTURES / "union", repository)
    (repository / "LICENSE").write_text("Authored test license, not a third-party licensing assertion.")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("code_agent.upgrade_study.resolve_image", lambda _: "sha256:" + "a" * 64)
    spec = control_spec()
    task = StudyTask(task_id=spec.task_id, repository="repositories/snapshot", upstream_url="https://github.com/example/authored",
        upstream_commit="b" * 40, license_file="LICENSE", license_identifier="authored-test", owner_authorized=True, spec=spec)
    request = StudyRequest(study_id="authored", model="groq:not-a-real-call", policy=CounterexamplePolicy(), tasks=[task])
    return request, repository, freeze_study(request, tmp_path)


def test_study_freezes_source_license_image_and_campaign_without_inference(tmp_path, monkeypatch):
    request, _, frozen = setup_study(tmp_path, monkeypatch)
    assert frozen["fingerprint"] == fingerprint({key: value for key, value in frozen.items() if key != "fingerprint"})
    assert frozen["snapshots"][request.tasks[0].task_id]["source_fingerprint"]
    assert frozen["request"]["tasks"][0]["spec"]["image"] == "sha256:" + "a" * 64
    assert "owner_attested" in frozen["upstream_and_license_metadata"]


@pytest.mark.asyncio
async def test_study_requires_separate_live_authorization_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="authorization"):
        await run_study({}, tmp_path, tmp_path / "data/upgradepilot/unauthorized")
    assert not (tmp_path / "data").exists()


@pytest.mark.asyncio
async def test_tampering_or_source_change_blocks_all_model_admissions(tmp_path, monkeypatch):
    _, repository, frozen = setup_study(tmp_path, monkeypatch)
    tampered = copy.deepcopy(frozen)
    tampered["request"]["model"] = "openai:changed"
    with pytest.raises(ValueError, match="integrity"):
        await run_study(tampered, tmp_path, tmp_path / "data/upgradepilot/tampered", authorize_provider=True)
    (repository / "app/models.py").write_text("raise RuntimeError('must not execute on host')\n")
    with pytest.raises(ValueError, match="changed"):
        await run_study(frozen, tmp_path, tmp_path / "data/upgradepilot/changed", authorize_provider=True)
    assert not (tmp_path / "data").exists()


@pytest.mark.asyncio
async def test_three_arms_share_frozen_inputs_and_retain_holds_and_unknown_costs(tmp_path, monkeypatch):
    _, _, frozen = setup_study(tmp_path, monkeypatch)
    admissions = []

    async def fake_migration(repository, spec, **kwargs):
        admissions.append((repository, spec, kwargs))
        return migration_report(spec)

    monkeypatch.setattr("code_agent.upgrade_study.run_migration", fake_migration)
    summary = await run_study(frozen, tmp_path, tmp_path / "data/upgradepilot/arms", authorize_provider=True)
    assert summary["complete"] and summary["recorded_attempts"] == 3
    assert [item[2]["discovery_feedback"] for item in admissions] == [False, False, True]
    assert [item[2]["model"] for item in admissions] == [None, "groq:not-a-real-call", "groq:not-a-real-call"]
    assert len({fingerprint(item[1].model_dump()) for item in admissions}) == 1
    assert all(item[2]["expected_campaign_commitment"] == frozen["snapshots"][request_id]["campaign_commitment"]
        for item in admissions for request_id in frozen["snapshots"])
    assert all(row["status"] == "held" and row["billed_cost_usd"] is None for row in summary["outcomes"])
    assert len(list((tmp_path / "data/upgradepilot/arms").glob("attempts/*/report.json"))) == 3
    assert all(row["ready_for_owner_review_rate"] == 0 and row["total_billed_cost_usd"] is None
        for row in summary["arm_metrics"].values())


@pytest.mark.asyncio
async def test_incomplete_study_keeps_completed_attempts_and_marks_unknown_usage(tmp_path, monkeypatch):
    _, _, frozen = setup_study(tmp_path, monkeypatch)

    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("authored worker interruption")

    monkeypatch.setattr("code_agent.upgrade_study.run_migration", unavailable)
    output = tmp_path / "data/upgradepilot/interrupted"
    with pytest.raises(RuntimeError):
        await run_study(frozen, tmp_path, output, authorize_provider=True)
    import json
    summary = json.loads((output / "study.json").read_bytes())
    assert not summary["complete"] and summary["incomplete_attempt_usage"] == "unknown_not_zero"


def test_private_snapshot_and_owner_authorization_are_required():
    spec = control_spec()
    fields = dict(task_id=spec.task_id, repository="../outside", upstream_url="https://github.com/example/authored",
        upstream_commit="b" * 40, license_file="LICENSE", license_identifier="test", owner_authorized=True, spec=spec)
    with pytest.raises(ValueError):
        StudyTask(**fields)
    with pytest.raises(ValueError):
        StudyTask(**{**fields, "repository": "repositories/snapshot", "owner_authorized": False})


@pytest.mark.asyncio
async def test_schema_aware_freeze_binds_source_schema_and_revokes_changed_admissions(tmp_path, monkeypatch):
    from code_agent.upgrade_inputs import generate_campaign
    from code_agent.upgrade_schema import SchemaPolicy
    request, _, _ = setup_study(tmp_path, monkeypatch)
    profile = SchemaPolicy(targets=[{"seed_case_id": "integer", "model": "app.models:Request"}])
    task = request.tasks[0].model_copy(update={"schema_policy": profile})
    request = request.model_copy(update={"tasks": [task]})
    schema_identity = ["a" * 64]

    def prepared(_repository, spec, _profile, policy):
        return generate_campaign(spec, policy), {"fingerprint": schema_identity[0]}

    monkeypatch.setattr("code_agent.upgrade_study.prepare_schema_campaign", prepared)
    frozen = freeze_study(request, tmp_path)
    assert frozen["snapshots"][task.task_id]["source_schema_fingerprint"] == "a" * 64
    assert frozen["schema_generator"] == "schema-grounded-boundaries-v1"
    schema_identity[0] = "b" * 64
    with pytest.raises(ValueError, match="source schema changed"):
        await run_study(frozen, tmp_path, tmp_path / "data/upgradepilot/schema-revoked", authorize_provider=True)
    assert not (tmp_path / "data").exists()
