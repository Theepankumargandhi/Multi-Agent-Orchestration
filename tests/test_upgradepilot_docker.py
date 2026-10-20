"""Optional real-container migration boundaries, no live model or external repository writes."""

import pytest

from code_agent.sandbox import DockerSandbox
from code_agent.upgrade_models import DEFAULT_IMAGE, MigrationSpec
from code_agent.upgrade_runner import run_migration
from evals.upgradepilot_evaluation import FIXTURES, ScriptedSpecialist


@pytest.mark.asyncio
async def test_real_union_contract_hold_and_scripted_verified_repair():
    if not DockerSandbox.available() or not DockerSandbox.image_available(DEFAULT_IMAGE):
        pytest.skip("Dedicated pinned migration image required; no host execution fallback")
    root = FIXTURES / "union"
    spec = MigrationSpec.model_validate_json((root / "migration.json").read_text())
    models = tuple(ScriptedSpecialist(role, "union") for role in ("analysis", "actions", "review"))
    report = await run_migration(root, spec, scripted_models=models)
    assert report["status"] == "ready_for_owner_review" and report["source_unchanged"]
    assert report["baseline"]["tests_exit_code"] == 0 and report["baseline"]["contracts"]["regressions"] == 1
    assert not report["baseline"]["passed"]
    assert report["repair"]["passed"] and report["repair"]["contracts"]["regressions"] == 0
    assert report["specialist_stages"] == {"analysis": "completed", "implementation": "completed", "test_author": "completed", "review": "completed"}
    assert "union_mode" in report["patch"] and "pydantic==2.12.4" in report["patch"]
    assert report["model_calls"] > 0 and report["live_provider_calls"] == 0 and not report["external_repository_writes"]
    assert "owner_checks.py" not in report["patch"]


@pytest.mark.asyncio
async def test_unauthorized_live_model_never_reaches_model_factory(monkeypatch):
    from code_agent import providers
    called = []
    monkeypatch.setattr(providers, "build_provider", lambda *_args, **_kwargs: called.append(True))
    root = FIXTURES / "union"
    spec = MigrationSpec.model_validate_json((root / "migration.json").read_text())
    report = await run_migration(root, spec, model="groq:example", authorize_provider=False)
    assert report["status"] == "held" and report["model_calls"] == 0 and not called


@pytest.mark.asyncio
async def test_windows_contract_and_dependency_newlines_are_preserved(tmp_path):
    import shutil
    if not DockerSandbox.available() or not DockerSandbox.image_available(DEFAULT_IMAGE):
        pytest.skip("Dedicated pinned migration image required")
    root = tmp_path / "windows-source"
    shutil.copytree(FIXTURES / "constraints", root)
    for name in ("requirements.txt", "owner_checks.py"):
        path = root / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    spec = MigrationSpec.model_validate_json((root / "migration.json").read_text())
    report = await run_migration(root, spec)
    assert report["status"] == "ready_for_owner_review" and report["baseline"]["owner_files_unchanged"]
    assert (root / "requirements.txt").read_bytes() == b"pydantic==1.10.24\r\n"
    assert report["model_calls"] == 0 and report["source_unchanged"]
