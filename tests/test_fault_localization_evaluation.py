import json
from pathlib import Path

from code_agent.benchmark_release import BenchmarkMatrix, DatasetLock
from code_agent.evaluation import dataset_fingerprint, load_code_cases
from evals.fault_localization_evaluation import evaluate


def test_authored_controls_do_not_claim_live_coding_quality():
    result = evaluate()
    assert result["passed"] == result["total"] == 8
    assert result["provider_calls"] == result["docker_runs"] == 0
    assert result["coding_resolution_measured"] is False


def test_matrix_changes_only_localization_flag():
    path = Path("evals/experiments/fault_localization_matrix.json")
    matrix = BenchmarkMatrix.model_validate(json.loads(path.read_text()))
    first, second = matrix.variants
    assert not first.fault_localization_enabled and second.fault_localization_enabled
    assert first.model_dump(exclude={"name", "fault_localization_enabled"}) == second.model_dump(exclude={"name", "fault_localization_enabled"})
    cases = load_code_cases((path.parent / matrix.dataset).resolve())[:matrix.max_cases]
    lock = DatasetLock.model_validate_json((path.parent / matrix.dataset_lock).read_text())
    assert len(cases) == 5
    assert dataset_fingerprint(cases) == lock.dataset_fingerprint
    assert [row.id for row in cases] == lock.case_ids
