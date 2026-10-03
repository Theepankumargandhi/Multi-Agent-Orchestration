import asyncio
import json
from pathlib import Path

import pytest

from code_agent.models import SandboxCommandResult
from code_agent.regression_challenges import PROBE_COMMAND, RECEIPT_PATH
from code_agent.sandbox import DockerUnavailableError
from evals.regression_challenge_evaluation import FIXTURE, authored_suite, main, markdown, run_controls
from tests.test_repair_tournament import ControlledSandbox


class AuthoredProbeSandbox(ControlledSandbox):
    """Explicit command-result oracle; no host execution and no actual Docker result."""

    def run(self, command):
        if command == PROBE_COMMAND:
            from code_agent.regression_challenges import fingerprint
            content = self.workspace.read_file("app.py")
            statuses = (["matched"] * 4 if "max(lower" in content else
                        ["matched", "mismatched", "matched", "mismatched"] if "value == -1" in content else
                        ["mismatched", "mismatched", "matched", "mismatched"])
            self.workspace.write_file(RECEIPT_PATH, json.dumps({"suite_sha256": fingerprint(authored_suite().model_dump(mode="json")),
                                                               "statuses": statuses}))
            return SandboxCommandResult(command=command, exit_code=0, duration_ms=0)
        return super().run(command)


@pytest.mark.asyncio
async def test_disclosed_ablation_checks_weak_patch_and_strong_fix():
    report = await run_controls(sandbox_factory=AuthoredProbeSandbox, execution_mode="authored_test_double")
    assert report["controls_passed"] and report["source_unchanged"]
    assert report["decision"] == "held" and not report["production_activation"] and report["model_calls"] == 0
    assert [row["probes_matched"] for row in report["arms"]] == [False, False, True]
    assert "authored_test_double" in markdown(report)
    with pytest.raises(ValueError, match="disclose"):
        await run_controls(sandbox_factory=AuthoredProbeSandbox)
    with pytest.raises(ValueError, match="execution mode"):
        await run_controls(execution_mode="unknown")


def test_cli_protects_sources_and_never_activates(tmp_path, monkeypatch):
    async def controls():
        return await run_controls(sandbox_factory=AuthoredProbeSandbox, execution_mode="authored_test_double")
    monkeypatch.setattr("evals.regression_challenge_evaluation.run_controls", controls)
    target = tmp_path / "report.json"
    assert main(["--output", str(target)]) == 0
    before = target.read_bytes()
    assert main(["--output", str(target)]) == 2 and target.read_bytes() == before
    assert main(["--output", str(FIXTURE / "app.py")]) == 2
    assert main(["--output", str(tmp_path / "wrong.md")]) == 2
    assert main(["--output", str(tmp_path / "strict.json"), "--require-gate"]) == 1


def test_unavailable_docker_produces_no_fake_execution_report(tmp_path, monkeypatch):
    async def unavailable():
        raise DockerUnavailableError("unavailable")
    monkeypatch.setattr("evals.regression_challenge_evaluation.run_controls", unavailable)
    target = tmp_path / "report.json"
    assert main(["--output", str(target)]) == 2 and not target.exists()


def test_failed_controls_are_not_swallowed(tmp_path, monkeypatch):
    async def failed():
        report = await run_controls(sandbox_factory=AuthoredProbeSandbox, execution_mode="authored_test_double")
        report["controls_passed"] = False
        return report
    monkeypatch.setattr("evals.regression_challenge_evaluation.run_controls", failed)
    assert main(["--output", str(tmp_path / "report.json")]) == 1


def test_ci_runs_docker_controls_not_live_generation():
    workflow = Path(".github/workflows/ci.yml").read_text()
    assert "python -m evals.regression_challenge_evaluation" in workflow
    assert "regression_challenge_smoke.jsonl" not in workflow
    policy = json.loads(Path("evals/experiments/regression_challenge_policy.json").read_text())
    assert policy["regression_challenges"]["allowed_targets"] == ["app.clamp"]


def test_source_fixture_is_not_modified_by_disclosed_controls():
    before = (FIXTURE / "app.py").read_bytes()
    asyncio.run(run_controls(sandbox_factory=AuthoredProbeSandbox, execution_mode="authored_test_double"))
    assert (FIXTURE / "app.py").read_bytes() == before
