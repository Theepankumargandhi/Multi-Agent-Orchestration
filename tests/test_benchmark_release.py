import hashlib
import json
from pathlib import Path

import pytest

from code_agent.benchmark_release import (
    DatasetLock,
    benchmark_card,
    build_release,
    create_dataset_lock,
    load_matrix,
    paired_comparison,
    write_release,
)
from code_agent.evaluation import load_code_cases
from code_agent.evaluation_models import (
    CodeBenchmarkOutcome,
    CodeBenchmarkReport,
    CodeTaskScore,
)


def _score(resolved: bool, duration_ms: float = 1000) -> CodeTaskScore:
    return CodeTaskScore(
        resolved=resolved,
        patch_nonempty=resolved,
        tests_passed=resolved,
        baseline_passed=False,
        regression=False,
        command_timed_out=False,
        original_repository_unchanged=True,
        policy_violations=0,
        iterations=3,
        tool_calls=4,
        model_calls=3,
        duration_ms=duration_ms,
        changed_files=1 if resolved else 0,
        patch_chars=100 if resolved else 0,
        prompt_tokens=100,
        completion_tokens=20,
        estimated_cost_usd=0.1,
        verification_passed=resolved,
    )


def _report(
    name: str,
    resolutions: list[bool],
    *,
    cost: float,
    p95: float,
) -> CodeBenchmarkReport:
    outcomes = [
        CodeBenchmarkOutcome(
            case_id=f"case-{index}",
            repository="fixture",
            status="completed" if resolved else "failed",
            failure_category="resolved" if resolved else "test_failure",
            score=_score(resolved, p95),
        )
        for index, resolved in enumerate(resolutions, start=1)
    ]
    resolved = sum(resolutions)
    return CodeBenchmarkReport(
        run_id=f"run-{name}",
        generated_at="2026-09-12T00:00:00+00:00",
        dataset_name="frozen-benchmark",
        dataset_path="dataset.jsonl",
        dataset_fingerprint="same-dataset",
        config_fingerprint=f"config-{name}",
        model=name,
        workflow="verified_pr",
        context_strategy="hybrid_rerank",
        sandbox_policy={"network_enabled": False},
        total=len(outcomes),
        resolved=resolved,
        pass_at_1=resolved / len(outcomes),
        pass_at_1_confidence_interval=[0.0, 1.0],
        test_pass_rate=resolved / len(outcomes),
        regression_rate=0.0,
        timeout_rate=0.0,
        policy_violation_rate=0.0,
        verification_pass_rate=resolved / len(outcomes),
        p50_duration_ms=p95,
        p95_duration_ms=p95,
        average_iterations=3,
        average_tool_calls=4,
        average_changed_files=1,
        total_prompt_tokens=200,
        total_completion_tokens=40,
        total_estimated_cost_usd=cost,
        cost_per_resolved_usd=cost / resolved if resolved else None,
        failure_categories={"resolved": resolved, "test_failure": len(outcomes) - resolved},
        slice_metrics={},
        outcomes=outcomes,
    )


def _write_report(path: Path, report: CodeBenchmarkReport) -> None:
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def test_paired_release_reports_case_level_delta_significance_and_pareto(tmp_path: Path):
    baseline = _report("baseline", [True, False], cost=0.2, p95=900)
    candidate = _report("candidate", [True, True], cost=0.3, p95=1000)
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_report(baseline_path, baseline)
    _write_report(candidate_path, candidate)

    comparison = paired_comparison("baseline", baseline, "candidate", candidate)
    assert comparison.candidate_only == 1
    assert comparison.baseline_only == 0
    assert comparison.pass_at_1_delta == 0.5
    assert comparison.paired_bootstrap_95_ci == [0.0, 1.0]
    assert comparison.exact_mcnemar_p_value == 1.0

    release = build_release(
        [
            ("baseline", baseline_path, baseline),
            ("candidate", candidate_path, candidate),
        ],
        title="Frozen paired benchmark",
        baseline_name="baseline",
    )
    assert release.winner == "candidate"
    assert release.variants[0].name == "candidate"
    assert {item.name for item in release.variants if item.pareto_efficient} == {
        "baseline",
        "candidate",
    }
    assert "not an official SWE-bench leaderboard score" in benchmark_card(release)


def test_release_artifacts_are_integrity_bound(tmp_path: Path):
    baseline = _report("baseline", [False, False], cost=0.1, p95=800)
    candidate = _report("candidate", [True, False], cost=0.2, p95=900)
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_report(baseline_path, baseline)
    _write_report(candidate_path, candidate)
    release = build_release(
        [("baseline", baseline_path, baseline), ("candidate", candidate_path, candidate)],
        title="Integrity release",
    )
    destination = write_release(release, tmp_path / "releases")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    for name, evidence in manifest["artifacts"].items():
        payload = (destination / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == evidence["sha256"]
        assert len(payload) == evidence["bytes"]


def test_release_rejects_unpaired_or_different_datasets(tmp_path: Path):
    baseline = _report("baseline", [True, False], cost=0.1, p95=800)
    candidate = _report("candidate", [True], cost=0.1, p95=800)
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_report(left, baseline)
    _write_report(right, candidate)
    with pytest.raises(ValueError, match="identical case IDs"):
        build_release(
            [("baseline", left, baseline), ("candidate", right, candidate)],
            title="Invalid release",
        )
    candidate.dataset_fingerprint = "different"
    with pytest.raises(ValueError, match="different dataset fingerprints"):
        build_release(
            [("baseline", left, baseline), ("candidate", right, candidate)],
            title="Invalid release",
        )


def test_checked_in_matrix_matches_frozen_dataset_selection():
    matrix_path = Path("evals/experiments/code_agent_release_matrix.json")
    matrix = load_matrix(matrix_path)
    cases = load_code_cases(Path("evals/datasets/code_agent_smoke.jsonl"))
    lock = DatasetLock.model_validate_json(
        Path("evals/experiments/code_agent_smoke.lock.json").read_text(encoding="utf-8")
    )
    generated = create_dataset_lock(
        cases,
        dataset_name=lock.dataset_name,
        dataset_path=lock.dataset_path,
    )
    assert matrix.variants[0].name == "small-single-map"
    assert len(matrix.variants) == 4
    assert generated.dataset_fingerprint == lock.dataset_fingerprint
    assert generated.case_ids == lock.case_ids
