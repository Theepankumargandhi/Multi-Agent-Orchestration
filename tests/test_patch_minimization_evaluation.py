import json
from pathlib import Path

import pytest

from code_agent.benchmark_release import BenchmarkMatrix, BenchmarkVariant, DatasetLock
from code_agent.evaluation import dataset_fingerprint, load_code_cases
from evals.patch_minimization_evaluation import docker_drill, evaluate


def test_authored_controls_are_not_live_improvement_claims():
    result = evaluate()
    assert result["total"] == result["passed"] == 8
    assert not result["repository_execution_measured"] and not result["live_coding_resolution_measured"]
    assert result["provider_calls"] == 0


def test_frozen_matrix_changes_only_minimization_flag():
    path = Path("evals/experiments/patch_minimization_matrix.json")
    matrix = BenchmarkMatrix.model_validate(json.loads(path.read_text()))
    baseline, candidate = matrix.variants
    assert not baseline.patch_minimization_enabled and candidate.patch_minimization_enabled
    assert baseline.model_dump(exclude={"name", "patch_minimization_enabled"}) == candidate.model_dump(exclude={"name", "patch_minimization_enabled"})
    cases = load_code_cases((path.parent / matrix.dataset).resolve())[:matrix.max_cases]
    lock = DatasetLock.model_validate_json((path.parent / matrix.dataset_lock).read_text())
    assert len(cases) == 5 and dataset_fingerprint(cases) == lock.dataset_fingerprint
    assert [row.id for row in cases] == lock.case_ids
    with pytest.raises(ValueError, match="verified-PR"):
        BenchmarkVariant(name="invalid", model="scripted", workflow="single_agent", patch_minimization_enabled=True)


@pytest.mark.asyncio
async def test_docker_drill_abstains_without_host_fallback(monkeypatch):
    monkeypatch.setattr("code_agent.sandbox.DockerSandbox.available", staticmethod(lambda: False))
    result = await docker_drill()
    assert result["status"] == "unavailable" and result["reason"] == "docker_unavailable"
    assert not result["mutation_writes"] and not result["removed_paths"]
