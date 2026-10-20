from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from code_agent.patch_inputs import PatchVerifySpec
from code_agent.patch_verify import tree_fingerprint, verify_patch
from code_agent.sandbox import DockerSandbox

FIXTURE = Path(__file__).parent / "fixtures/patch_verify"
pytestmark = pytest.mark.integration


@pytest.fixture
def repository(tmp_path):
    if not DockerSandbox.available() or not DockerSandbox.image_available("agentforge-code-sandbox:local"):
        pytest.skip("requires the prebuilt local Docker sandbox")
    destination = tmp_path / "repository"
    shutil.copytree(FIXTURE, destination)
    return destination


@pytest.mark.asyncio
@pytest.mark.parametrize("name,expected", [("fee-regression", "regression_found"), ("fee-correct", "checks_passed")])
async def test_real_docker_repeated_behavior(repository, name, expected):
    before = tree_fingerprint(repository)
    spec = PatchVerifySpec.model_validate_json((FIXTURE / f"{name}.json").read_bytes())
    report = await verify_patch(repository, (FIXTURE / f"{name}.diff").read_text(), spec)
    assert report["status"] == expected, report["reason"]
    assert len(report["trials"]) == 4 and all(row["tests"]["passed"] for row in report["trials"])
    assert report["source_unchanged"] and before == tree_fingerprint(repository)
    assert report["live_provider_calls"] == 0
    if expected == "regression_found":
        assert any(row["args"] == [0] for row in report["counterexamples"])


@pytest.mark.asyncio
async def test_skipped_tests_cannot_be_passing_evidence(repository):
    (repository / "owner_checks.py").write_text("import pytest\n\n@pytest.mark.skip\ndef test_skipped():\n    assert True\n")
    spec = PatchVerifySpec.model_validate_json((FIXTURE / "fee-correct.json").read_bytes())
    report = await verify_patch(repository, (FIXTURE / "fee-correct.diff").read_text(), spec)
    assert report["status"] == "inconclusive" and report["reason"] == "original_tests_not_all_passed"


@pytest.mark.asyncio
async def test_import_errors_are_inconclusive_not_counterexamples(repository):
    spec = PatchVerifySpec.model_validate_json((FIXTURE / "fee-correct.json").read_bytes())
    value = spec.model_dump()
    value["contracts"][0]["entrypoint"] = "missing_module:fee"
    spec = PatchVerifySpec.model_validate(value)
    report = await verify_patch(repository, (FIXTURE / "fee-correct.diff").read_text(), spec)
    assert report["status"] == "inconclusive" and not report["counterexamples"]
