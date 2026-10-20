"""Portable data contracts shared by coding evaluation runners and dashboards."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from code_agent.provider_reliability import ProviderFailure


class CodeBenchmarkCase(BaseModel):
    id: str = Field(min_length=1, max_length=300)
    repository: str = Field(min_length=1, max_length=300)
    issue: str = Field(min_length=10, max_length=100_000)
    test_command: list[str] = Field(min_length=1, max_length=30)
    model: str | None = Field(default=None, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=30)
    source: Literal["local", "swe-bench"] = "local"
    split: Literal["validation", "test"] = "test"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeTaskScore(BaseModel):
    resolved: bool
    patch_nonempty: bool
    tests_passed: bool
    baseline_passed: bool
    regression: bool
    command_timed_out: bool
    original_repository_unchanged: bool
    policy_violations: int
    iterations: int
    tool_calls: int
    model_calls: int
    usage_accounting_complete: bool | None = None
    candidate_regressions: int = 0
    duration_ms: float
    changed_files: int
    patch_chars: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    verification_passed: bool = False
    repair_rounds: int = 0
    blocking_findings: int = 0
    telemetry_valid: bool = False
    telemetry_spans: int = 0


class CodeBenchmarkOutcome(BaseModel):
    mutation_testing_status_counts: dict[str, int] = Field(default_factory=dict)
    mutation_measured: bool = False
    mutation_killed: int = Field(default=0, ge=0)
    mutation_survived: int = Field(default=0, ge=0)
    mutation_invalid: int = Field(default=0, ge=0)
    mutation_completed_trials: int = Field(default=0, ge=0)
    loop_guard_status_counts: dict[str, int] = Field(default_factory=dict)
    patch_minimized: bool = False
    minimization_removed_file_changes: int = Field(default=0, ge=0)
    minimization_completed_trials: int = Field(default=0, ge=0)
    patch_minimization_status_counts: dict[str, int] = Field(default_factory=dict)
    fault_localized: bool = False
    fault_localization_status_counts: dict[str, int] = Field(default_factory=dict)
    provider_failures: list[ProviderFailure] = Field(default_factory=list)
    case_id: str
    repository: str = ""
    tags: list[str] = Field(default_factory=list)
    status: str = "failed"
    failure_category: str = "execution_error"
    score: CodeTaskScore | None = None
    trajectory_path: str = ""
    patch_path: str = ""
    patch_sha256: str = ""
    error: str = ""


class CodeBenchmarkReport(BaseModel):
    mutation_testing_status_counts: dict[str, int] = Field(default_factory=dict)
    mutation_measured_cases: int = Field(default=0, ge=0)
    mutation_killed: int = Field(default=0, ge=0)
    mutation_survived: int = Field(default=0, ge=0)
    mutation_invalid: int = Field(default=0, ge=0)
    mutation_completed_trials: int = Field(default=0, ge=0)
    loop_guard_status_counts: dict[str, int] = Field(default_factory=dict)
    patch_minimized_cases: int = Field(default=0, ge=0)
    minimization_removed_file_changes: int = Field(default=0, ge=0)
    minimization_completed_trials: int = Field(default=0, ge=0)
    patch_minimization_status_counts: dict[str, int] = Field(default_factory=dict)
    fault_localized_cases: int = Field(default=0, ge=0)
    fault_localization_status_counts: dict[str, int] = Field(default_factory=dict)
    execution_status: Literal["completed", "paused"] = "completed"
    quality_metrics_valid: bool = True
    pending_case_ids: list[str] = Field(default_factory=list)
    attempted_cases: int = 0
    retained_attempts: int = 0
    recovery_usage_unknown: bool = False
    schema_version: str = "2.0"
    run_id: str
    generated_at: str
    dataset_name: str
    dataset_path: str
    dataset_fingerprint: str
    config_fingerprint: str
    model: str
    workflow: str = "single_agent"
    context_strategy: str = "repository_map"
    sandbox_policy: dict[str, Any]
    total: int
    resolved: int
    pass_at_1: float
    pass_at_1_confidence_interval: list[float]
    test_pass_rate: float
    regression_rate: float
    timeout_rate: float
    policy_violation_rate: float
    verification_pass_rate: float = 0.0
    telemetry_coverage_rate: float = 0.0
    p50_duration_ms: float
    p95_duration_ms: float
    average_iterations: float
    average_tool_calls: float
    average_model_calls: float = 0.0
    total_model_calls: int = 0
    usage_accounting_complete: bool = False
    prices_configured: bool = False
    total_candidate_regressions: int = 0
    average_changed_files: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_estimated_cost_usd: float
    cost_per_resolved_usd: float | None
    failure_categories: dict[str, int]
    slice_metrics: dict[str, dict[str, float]]
    outcomes: list[CodeBenchmarkOutcome]
