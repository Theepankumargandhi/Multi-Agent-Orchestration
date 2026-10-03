import json
from pathlib import Path

import pytest

from code_agent.models import RepairTournamentPolicy
from code_agent.oracle_calibration import reference_fingerprint
from code_agent.sandbox import DockerUnavailableError
from evals.oracle_calibration_evaluation import REFERENCE, main, markdown, run_controls
from tests.test_oracle_calibration import CalibrationSandbox


@pytest.mark.asyncio
async def test_strong_weak_and_wrong_oracle_controls_are_disclosed_and_held():
    report = await run_controls(sandbox_factory=CalibrationSandbox, execution_mode="authored_test_double")
    assert report["controls_passed"] and report["decision"] == "held" and not report["production_activation"]
    assert report["model_calls"] == 0
    assert [row["receipt"]["eligible"] for row in report["arms"]] == [True, False, False]
    assert [row["receipt"]["mutation_score"] for row in report["arms"]] == [1, 0.5, 0]
    assert "authored_test_double" in markdown(report)
    assert "Authored operator reference" not in json.dumps(report)
    with pytest.raises(ValueError, match="disclose"):
        await run_controls(sandbox_factory=CalibrationSandbox)


def test_cli_protects_source_outputs_and_strict_gate_stays_held(tmp_path, monkeypatch):
    async def authored():
        return await run_controls(sandbox_factory=CalibrationSandbox, execution_mode="authored_test_double")
    monkeypatch.setattr("evals.oracle_calibration_evaluation.run_controls", authored)
    target = tmp_path / "report.json"
    assert main(["--output", str(target)]) == 0
    before = target.read_bytes()
    assert main(["--output", str(target)]) == 2 and target.read_bytes() == before
    assert main(["--output", str(REFERENCE / "app.py")]) == 2
    assert main(["--output", str(tmp_path / "wrong.md")]) == 2
    assert main(["--output", str(tmp_path / "strict.json"), "--require-gate"]) == 1


def test_unavailable_docker_creates_no_fake_report(tmp_path, monkeypatch):
    async def unavailable():
        raise DockerUnavailableError("unavailable")
    monkeypatch.setattr("evals.oracle_calibration_evaluation.run_controls", unavailable)
    target = tmp_path / "report.json"
    assert main(["--output", str(target)]) == 2 and not target.exists()


@pytest.mark.asyncio
async def test_sandbox_unavailability_is_not_mislabelled_as_a_failed_control():
    class Unavailable(CalibrationSandbox):
        def start(self):
            raise DockerUnavailableError("unavailable")
    with pytest.raises(DockerUnavailableError):
        await run_controls(sandbox_factory=Unavailable, execution_mode="authored_test_double")


def test_frozen_policy_and_ci_never_run_live_generation():
    policy = RepairTournamentPolicy.model_validate_json(Path("evals/experiments/oracle_calibration_policy.json").read_text())
    assert policy.regression_challenges.oracle_calibration.reference_sha256 == reference_fingerprint(REFERENCE)
    workflow = Path(".github/workflows/ci.yml").read_text()
    assert "python -m evals.oracle_calibration_evaluation" in workflow
    assert "oracle_calibration_policy.json" not in workflow


def test_failed_controls_return_nonzero(tmp_path, monkeypatch):
    async def failed():
        report = await run_controls(sandbox_factory=CalibrationSandbox, execution_mode="authored_test_double")
        report["controls_passed"] = False
        return report
    monkeypatch.setattr("evals.oracle_calibration_evaluation.run_controls", failed)
    assert main(["--output", str(tmp_path / "failed.json")]) == 1
