"""Keep long-running evaluations independent of tests and publication checks."""

from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def jobs():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]


@pytest.mark.parametrize(
    "job_id,minimum,maximum",
    [("tests", 30, 45), ("ai-evaluations", 30, 45), ("lint", 5, 10), ("dependency-audit", 10, 20)],
)
def test_checks_have_independent_bounded_budgets(jobs, job_id, minimum, maximum):
    job = jobs[job_id]
    assert "needs" not in job and "if" not in job
    assert minimum <= job["timeout-minutes"] <= maximum
    assert not job.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in job["steps"])
    assert any(step.get("uses", "").startswith("actions/checkout@") for step in job["steps"])
    assert any(
        step.get("uses", "").startswith("actions/setup-python@")
        and step.get("with", {}).get("python-version") == "3.11"
        for step in job["steps"]
    )


@pytest.mark.parametrize("job_id", ["tests", "ai-evaluations"])
def test_execution_jobs_prepare_their_own_runtime(jobs, job_id):
    job = jobs[job_id]
    assert job["env"]["PYTHON_DOTENV_DISABLED"] == "1"
    commands = [step.get("run", "") for step in job["steps"]]
    script = "\n".join(commands)
    assert "python -m pip install -r requirements.txt" in script
    assert "python -m pip install --no-deps --editable ." in script
    assert "mkdir -p data/checkpoints data/store data/rag" in script
    assert "python scripts/setup_code_parsers.py" in script
    assert "docker build -f docker/Dockerfile.code-sandbox -t agentforge-code-sandbox:local ." in script
    assert not any("download-artifact" in step.get("uses", "") for step in job["steps"])


def test_pytest_retains_coverage_annotations_and_local_reports(jobs):
    steps = jobs["tests"]["steps"]
    command = next(step for step in steps if step.get("name") == "Run unit and integration tests")
    assert command["run"] == (
        "python -m pytest -q --cov --cov-report=term-missing --cov-report=xml --junitxml=test-results.xml"
    )
    assert command["env"]["ENABLE_USER_AUTH"] == "true"
    assert all("python -m evals." not in step.get("run", "") for step in steps)
    annotation = next(step for step in steps if step.get("name") == "Annotate pytest failures")
    assert annotation["if"] == "failure() && hashFiles('test-results.xml') != ''"
    reports = {
        step["with"]["name"]: (step["with"]["path"], step.get("if"))
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    }
    assert reports == {
        "coverage-xml": ("coverage.xml", "always()"),
        "pytest-junit": ("test-results.xml", "always()"),
    }


def test_evaluation_job_keeps_reproducibility_quality_and_post_training_gates(jobs):
    steps = jobs["ai-evaluations"]["steps"]
    script = "\n".join(step.get("run", "") for step in steps)
    assert "python -m pytest" not in script
    assert "pip-audit" not in script and "python -m ruff" not in script
    for gate in (
        "python -m evals.oracle_calibration_evaluation",
        "python -m code_agent.retrieval_learning train-fusion",
        "--check evals/experiments/code_context_fusion.json",
        "python -m evals.run_offline_evals --min-score 0.95",
        "python -m evals.run_experiments --no-store --min-score 0.95",
        "python -m evals.ensemble_outcome_evaluation drill --require-gate",
        "python -m post_training.cli smoke",
        "python -m post_training.cli register",
    ):
        assert gate in script
    gate_step = next(step for step in steps if step.get("name") == "Run offline AI evaluation gates")
    assert gate_step["env"]["ENABLE_USER_AUTH"] == "true"


def test_lint_and_audit_are_not_hidden_behind_execution_jobs(jobs):
    lint = next(step for step in jobs["lint"]["steps"] if step.get("name") == "Lint")
    assert lint["run"] == "python -m ruff check ."
    audit = next(step for step in jobs["dependency-audit"]["steps"] if step.get("name") == "Dependency audit")
    assert "pip-audit -r requirements-service.txt" in audit["run"]
    assert "if" not in audit and "||" not in audit["run"]
    # Keep the existing exceptions; splitting jobs must not add vulnerability waivers.
    waivers = [entry.split()[0] for entry in audit["run"].split("--ignore-vuln ")[1:]]
    assert set(waivers) == {
        "PYSEC-2026-311",
        "CVE-2026-45830",
        "CVE-2026-45831",
        "CVE-2026-45833",
    }
    assert len(waivers) == 4


def test_artifact_names_remain_unique_and_with_their_producing_job(jobs):
    owners = {}
    names = []
    for job_id, job in jobs.items():
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/upload-artifact@"):
                name = step["with"]["name"]
                names.append(name)
                owners[name] = job_id
    assert all(count == 1 for count in Counter(names).values())
    assert owners["coverage-xml"] == owners["pytest-junit"] == "tests"
    assert owners["experimental-retrieval-decision"] == "ai-evaluations"
    assert owners["ensemble-final-outcome-validation"] == "ai-evaluations"
    assert jobs["docker-build-check"]["needs"] == "tests"
