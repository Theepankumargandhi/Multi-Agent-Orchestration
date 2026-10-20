from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from code_agent.maintainer_pilot import (
    PilotRequest,
    PilotReview,
    execute_pilot,
    pilot_status,
    record_review,
    register_pilot,
    revoke_consent,
)
from code_agent.upgrade_models import fingerprint

ROOT = Path(__file__).parent.parent
IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def owner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "repositories/owned-control"
    shutil.copytree(ROOT / "tests/fixtures/upgradepilot/constraints", source)
    shutil.copy2(ROOT / "LICENSE", source / "LICENSE")
    spec = json.loads((source / "migration.json").read_text())
    spec["task_id"] = "owner-pilot"
    consent = {"granted": True, "maintainer_handle": "project-owner", "repository_url": "https://github.com/example/owned",
               "upstream_commit": "b" * 40, "evidence_reference": "Authored unit control, not genuine maintainer consent."}
    request = PilotRequest.model_validate({"pilot_id": "owner-pilot", "evidence_kind": "authored_control",
        "repository": "repositories/owned-control", "repository_url": consent["repository_url"],
        "upstream_commit": consent["upstream_commit"], "license_identifier": "MIT", "consent": consent, "spec": spec})
    return tmp_path, source, request, tmp_path / "data/maintainer-pilots/control"


def register(owner):
    root, source, request, directory = owner
    registration = register_pilot(request, root, directory, image_resolver=lambda _: IMAGE)
    return registration


def test_consent_scope_and_privacy_defaults(owner):
    root, source, request, directory = owner
    assert not request.consent.publish_content_free_summary and not request.consent.identity_authenticated
    for update in ({"granted": False}, {"upstream_commit": "c" * 40}, {"repository_url": "https://github.com/other/other"}):
        payload = request.model_dump()
        payload["consent"].update(update)
        with pytest.raises(ValueError):
            PilotRequest.model_validate(payload)
    register(owner)
    assert pilot_status(directory, root)["execution_status"] == "not_run"
    with pytest.raises(ValueError, match="sharing"):
        pilot_status(directory, root, public=True)
    with pytest.raises(FileExistsError):
        register(owner)


@pytest.mark.asyncio
async def test_pilot_execution_review_tamper_and_revocation(owner):
    root, source, request, directory = owner
    registration = register(owner)
    before = (source / "app/models.py").read_bytes()
    calls = []

    async def runner(repository, spec, **kwargs):
        calls.append(kwargs)
        assert not kwargs["authorize_provider"]
        patch = "authored fake receipt patch; unit test only"
        report = {"task_id": spec.task_id, "status": "ready_for_owner_review", "reason": "scripted_unit_control",
                  "spec_fingerprint": fingerprint(spec.model_dump()), "patch": patch, "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                  "source_unchanged": True, "live_provider_calls": 0, "duration_ms": 10}
        report["fingerprint"] = fingerprint(report)
        return report

    report = await execute_pilot(directory, root, runner=runner, image_resolver=lambda _: IMAGE)
    assert len(calls) == 1 and (source / "app/models.py").read_bytes() == before
    assert pilot_status(directory, root)["ready_for_owner_review"]
    with pytest.raises(ValueError, match="already attempted"):
        await execute_pilot(directory, root, runner=runner, image_resolver=lambda _: IMAGE)
    review = PilotReview(pilot_id=request.pilot_id, registration_sha256=registration["fingerprint"],
        report_sha256=report["fingerprint"], patch_sha256=report["patch_sha256"], maintainer_handle="project-owner",
        decision="accepted", evidence_url="https://github.com/example/owned/pull/1")
    record_review(directory, root, review)
    summary = pilot_status(directory, root)
    assert summary["maintainer_reported_decision"] == "accepted"
    assert not summary["real_maintainer_reported_acceptance"] and not summary["github_status_verified"]
    with pytest.raises(ValueError):
        record_review(directory, root, review)
    (directory / "execution/report.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        pilot_status(directory, root)


def test_revocation_blocks_even_when_source_disappears(owner):
    root, source, request, directory = owner
    register(owner)
    (source / "app/models.py").unlink()
    revoke_consent(directory)
    with pytest.raises(ValueError, match="revoked"):
        pilot_status(directory, root)


@pytest.mark.asyncio
async def test_interrupted_admission_never_retries_and_model_needs_consent(owner):
    root, source, request, directory = owner
    register(owner)

    async def interrupted(*args, **kwargs):
        raise RuntimeError("simulated crash after admission")

    with pytest.raises(ValueError, match="both explicit"):
        await execute_pilot(directory, root, model="groq:example", runner=interrupted)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await execute_pilot(directory, root, runner=interrupted, image_resolver=lambda _: IMAGE)
    with pytest.raises(ValueError, match="already attempted"):
        await execute_pilot(directory, root, runner=interrupted, image_resolver=lambda _: IMAGE)
    assert pilot_status(directory, root)["execution_status"] == "running"


def test_reported_merge_requires_correct_repository_pr_and_commit():
    values = {"pilot_id": "pilot", "registration_sha256": "a" * 64, "report_sha256": "b" * 64,
              "patch_sha256": "c" * 64, "maintainer_handle": "owner", "decision": "merged",
              "evidence_url": "https://github.com/example/owned/pull/1"}
    with pytest.raises(ValueError):
        PilotReview.model_validate(values)
    assert PilotReview.model_validate({**values, "merge_commit": "d" * 40}).identity_authenticated is False
