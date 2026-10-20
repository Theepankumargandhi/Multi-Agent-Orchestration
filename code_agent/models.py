"""Validated contracts for sandboxed coding tasks and tool actions."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from code_agent.provider_reliability import ProviderFailure

ActionKind = Literal["list", "read", "read_lines", "search", "write", "edit", "delete", "test", "finish"]

SecuritySource = Literal[
    "operator",
    "user",
    "repository",
    "retrieval",
    "tool_output",
    "memory",
    "peer_agent",
]


class SandboxPolicy(BaseModel):
    allowed_images: set[str] = Field(default_factory=lambda: {"agentforge-code-sandbox:local"})
    image: str = "agentforge-code-sandbox:local"
    cpus: float = Field(default=1.0, gt=0, le=4)
    memory: str = "1g"
    pids_limit: int = Field(default=256, ge=32, le=1024)
    command_timeout_seconds: int = Field(default=120, ge=1, le=900)
    task_timeout_seconds: int = Field(default=900, ge=30, le=3600)
    max_output_chars: int = Field(default=20000, ge=1000, le=100000)
    max_repository_bytes: int = Field(default=50_000_000, ge=100_000, le=500_000_000)
    max_files: int = Field(default=5000, ge=10, le=50000)
    max_file_bytes: int = Field(default=1_000_000, ge=1000, le=10_000_000)
    max_patch_chars: int = Field(default=200_000, ge=1000, le=2_000_000)
    max_iterations: int = Field(default=20, ge=1, le=100)
    max_writes: int = Field(default=8, ge=1, le=50)
    max_changed_files: int = Field(default=20, ge=1, le=200)
    max_repair_rounds: int = Field(default=2, ge=0, le=5)
    max_workflow_writes: int = Field(default=16, ge=1, le=100)
    targeted_edits_only: bool = False
    delivery_verification_enabled: bool = False
    require_specialist_completion: bool = False
    protected_paths: list[str] = Field(default_factory=lambda: ["pytest.ini", "conftest.py", "*/conftest.py"], max_length=100)
    controller_managed_paths: list[str] = Field(default_factory=list, max_length=100)
    execution_retrieval_enabled: bool = False
    repair_memory_enabled: bool = False
    fault_localization_enabled: bool = False
    fault_localization_max_refreshes: int = Field(default=2, ge=1, le=4)
    fault_localization_timeout_seconds: int = Field(default=120, ge=30, le=240)
    fault_localization_max_tokens: int = Field(default=768, ge=128, le=1024)
    patch_minimization_enabled: bool = False
    patch_minimization_max_files: int = Field(default=3, ge=1, le=6)
    patch_minimization_timeout_seconds: int = Field(default=180, ge=30, le=300)
    loop_guard_mode: Literal["off", "shadow", "enforce"] = "off"
    loop_guard_repetition_limit: int = Field(default=3, ge=3, le=5)
    loop_guard_max_cycle_length: int = Field(default=4, ge=1, le=4)
    mutation_testing_mode: Literal["off", "report", "enforce"] = "off"
    mutation_testing_max_mutants: int = Field(default=3, ge=1, le=6)
    mutation_testing_min_mutants: int = Field(default=1, ge=1, le=6)
    mutation_testing_min_score: float = Field(default=1.0, ge=0, le=1)
    mutation_testing_timeout_seconds: int = Field(default=180, ge=30, le=300)
    counterfactual_verification_enabled: bool = False
    counterfactual_timeout_seconds: int = Field(default=180, ge=30, le=300)
    execution_retrieval_max_refreshes: int = Field(default=4, ge=1, le=12)
    execution_retrieval_max_tokens: int = Field(default=1024, ge=128, le=2048)
    min_changed_line_coverage: float = Field(default=0.8, ge=0, le=1)
    network_enabled: bool = False
    security_policy_enabled: bool = True
    security_policy_version: str = Field(default="agentforge-security-v1", min_length=1, max_length=100)
    allowed_test_executables: set[str] = Field(
        default_factory=lambda: {"python", "python3", "pytest", "ruff"}
    )

    @field_validator("image")
    @classmethod
    def image_must_be_allowlisted(cls, value: str, info) -> str:
        allowed = info.data.get("allowed_images") or {"agentforge-code-sandbox:local"}
        if value not in allowed:
            raise ValueError("sandbox image is not allowlisted")
        return value

    @field_validator("memory")
    @classmethod
    def valid_memory_limit(cls, value: str) -> str:
        if not re.fullmatch(r"[1-9][0-9]*(?:[kmg])", value.lower()):
            raise ValueError("memory must look like 512m or 1g")
        return value.lower()

    @model_validator(mode="after")
    def mutation_bounds(self):
        if self.mutation_testing_min_mutants > self.mutation_testing_max_mutants:
            raise ValueError("minimum mutation count cannot exceed the maximum")
        return self


class CodeTask(BaseModel):
    # Set by the authenticated server/worker, never accepted by CodeTaskRequest.
    repair_memory_namespace: str = Field(default="", max_length=200, exclude=True)
    benchmark_case_id: str = Field(default="", max_length=300, exclude=True)
    issue: str = Field(min_length=10, max_length=100_000)
    repository: str = Field(min_length=1, max_length=300)
    model: str = Field(default="gpt-4o-mini", min_length=1, max_length=100)
    test_command: list[str] = Field(
        default_factory=lambda: ["python", "-m", "pytest", "-q"], min_length=1, max_length=30
    )
    policy: SandboxPolicy = Field(default_factory=SandboxPolicy)

    @field_validator("repository")
    @classmethod
    def safe_repository_name(cls, value: str) -> str:
        clean = value.strip().replace("\\", "/")
        if clean.startswith("/") or ".." in clean.split("/") or not re.fullmatch(
            r"[A-Za-z0-9._/-]+", clean
        ):
            raise ValueError("repository must be a safe relative path")
        return clean

    @field_validator("test_command")
    @classmethod
    def bounded_command(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 500 or "\x00" in item for item in value):
            raise ValueError("test command contains an invalid argument")
        return value


class CodeAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: ActionKind
    path: str = Field(default="", max_length=500)
    pattern: str = Field(default="", max_length=1000)
    content: str = Field(default="", max_length=1_000_000)
    rationale: str = Field(default="", max_length=600)
    start_line: int = Field(default=1, ge=1, le=1_000_000)
    end_line: int = Field(default=100, ge=1, le=1_000_000)
    expected_sha256: str = Field(default="", pattern=r"^(?:[a-f0-9]{64})?$")
    old_text: str = Field(default="", max_length=16000)
    new_text: str = Field(default="", max_length=16000)

    @model_validator(mode="after")
    def edit_contract(self):
        if self.kind == "edit" and (not self.path or not self.expected_sha256 or not self.old_text):
            raise ValueError("edit requires path, expected_sha256 and nonempty old_text")
        if self.kind == "read_lines" and self.end_line < self.start_line:
            raise ValueError("read_lines requires an ordered line range")
        return self


class ToolObservation(BaseModel):
    iteration: int
    action: ActionKind
    ok: bool
    summary: str
    path: str = ""
    output: str = ""
    duration_ms: float = 0.0
    stage: str = "implementation"


class SecurityArtifact(BaseModel):
    """Content provenance passed to the policy engine without persisting raw content."""

    source: SecuritySource
    trusted: bool = False
    sha256: str
    taints: list[str] = Field(default_factory=list, max_length=20)


class SecurityPolicyEvent(BaseModel):
    decision_id: str
    policy_version: str
    action: ActionKind
    path: str = ""
    stage: str = "implementation"
    allowed: bool
    severity: Literal["none", "low", "medium", "high", "critical"] = "none"
    rule_ids: list[str] = Field(default_factory=list, max_length=20)
    categories: list[str] = Field(default_factory=list, max_length=20)
    source_taints: list[str] = Field(default_factory=list, max_length=20)
    requires_human_approval: bool = False
    latency_ms: float = Field(default=0.0, ge=0)


class SecuritySummary(BaseModel):
    policy_version: str
    evaluated_actions: int = Field(default=0, ge=0)
    blocked_actions: int = Field(default=0, ge=0)
    warned_actions: int = Field(default=0, ge=0)
    tainted_decisions: int = Field(default=0, ge=0)
    events: list[SecurityPolicyEvent] = Field(default_factory=list)
    fingerprint: str = ""


class TelemetrySpanRecord(BaseModel):
    span_id: str = Field(min_length=16, max_length=16)
    parent_span_id: str = Field(default="", max_length=16)
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["agent", "model", "retrieval", "tool", "security", "evaluation"]
    start_time_unix_nano: int = Field(ge=0)
    end_time_unix_nano: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    status: Literal["ok", "error"] = "ok"
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)


class TelemetryTrace(BaseModel):
    schema_version: str = "1.0"
    trace_id: str = Field(min_length=32, max_length=32)
    generated_at: str
    service_name: str = "agentforge-code-agent"
    content_capture_enabled: bool = False
    spans: list[TelemetrySpanRecord] = Field(default_factory=list)
    fingerprint: str


class SandboxCommandResult(BaseModel):
    command: list[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: float
    timed_out: bool = False


class AnalysisPlan(BaseModel):
    summary: str = Field(min_length=1, max_length=2000)
    relevant_paths: list[str] = Field(default_factory=list, max_length=20)
    risks: list[str] = Field(default_factory=list, max_length=20)
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=20)
    test_strategy: list[str] = Field(default_factory=list, max_length=20)


class ReviewFinding(BaseModel):
    severity: Literal["critical", "high", "medium", "low"]
    category: Literal["correctness", "security", "testing", "maintainability", "scope"]
    message: str = Field(min_length=1, max_length=1200)
    recommendation: str = Field(default="", max_length=1200)
    path: str = Field(default="", max_length=500)
    line: int | None = Field(default=None, ge=1)


class ReviewVerdict(BaseModel):
    approved: bool
    summary: str = Field(min_length=1, max_length=2000)
    findings: list[ReviewFinding] = Field(default_factory=list, max_length=40)


class QualityGateResult(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    status: Literal["passed", "failed", "skipped"]
    summary: str = Field(min_length=1, max_length=2000)
    command: list[str] = Field(default_factory=list, max_length=30)
    duration_ms: float = Field(default=0.0, ge=0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class CodeContextFile(BaseModel):
    path: str
    language: str
    rank: int = Field(ge=1)
    score: float = Field(ge=0)
    lexical_score: float = Field(ge=0)
    graph_score: float = Field(ge=0)
    semantic_score: float = Field(default=0.0, ge=0)
    rerank_score: float = Field(default=0.0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    snippet_sha256: str = ""
    sha256: str
    reasons: list[str] = Field(default_factory=list, max_length=10)


class CodeContextReceipt(BaseModel):
    query: str
    candidate_files: int = Field(ge=0)
    selected_files: list[CodeContextFile] = Field(default_factory=list)
    context_chars: int = Field(ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    original_context_chars: int = Field(default=0, ge=0)
    duplicate_lines_removed: int = Field(default=0, ge=0)
    query_plan: dict[str, Any] = Field(default_factory=dict)
    strategy: str = "lexical_graph"
    packing_policy: Literal["legacy", "balanced_v1"] = "legacy"
    source_line_ranges: dict[str, list[list[int]]] = Field(default_factory=dict)
    packing_diagnostics: dict[str, int] = Field(default_factory=dict)
    embedding_backend: str = "none"
    reranker_backend: str = "none"
    fusion_backend: str = "fixed-weight-v1"
    parser_backends: dict[str, int] = Field(default_factory=dict)
    fallbacks: list[str] = Field(default_factory=list, max_length=20)
    index_reused_files: int = Field(default=0, ge=0)
    index_parsed_files: int = Field(default=0, ge=0)
    index_incremental_files: int = Field(default=0, ge=0)
    fingerprint: str


class CounterfactualTrial(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    arm: Literal["patched", "original"]
    repeat: int = Field(ge=0, le=1)
    source_sha256: str = ""
    manifest_sha256: str = ""
    receipt_sha256: str = ""
    outcomes: dict[str, Literal["passed", "assertion", "error", "skipped"]] = Field(default_factory=dict, max_length=200)
    exit_code: int | None = None
    workspace_unchanged: bool = False
    duration_ms: float = Field(default=0, ge=0)


class CounterfactualReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: str = "1.0"
    status: Literal["passed", "failed", "unavailable", "budget_exhausted"] = "failed"
    reason: str = "not_run"
    test_file_sha256: dict[str, str] = Field(default_factory=dict, max_length=6)
    eligible_test_sha256: list[str] = Field(default_factory=list, max_length=40)
    discriminating_test_sha256: list[str] = Field(default_factory=list, max_length=200)
    original_sha256: str = ""
    candidate_sha256: str = ""
    sources_unchanged: bool = False
    trials: list[CounterfactualTrial] = Field(default_factory=list, max_length=4)
    fingerprint: str = ""


class PatchAblationTrial(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    phase: Literal["control", "ablation", "confirmation"]
    removed_paths: list[str] = Field(default_factory=list, max_length=6)
    repeat: int = Field(ge=0, le=1)
    source_sha256: str
    receipt_sha256: str
    outcomes: dict[str, Literal["passed", "failed", "skipped", "error"]] = Field(max_length=200)
    exit_code: int
    duration_ms: float = Field(ge=0)


class PatchMinimizationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    status: Literal["minimized", "unchanged", "unsupported", "unavailable", "budget_exhausted", "rejected", "error", "integrity_error"]
    reason: str
    original_sha256: str = ""
    candidate_before_sha256: str = ""
    candidate_after_sha256: str = ""
    command_sha256: str = ""
    restored_file_sha256: dict[str, str] = Field(default_factory=dict, max_length=6)
    eligible_paths: list[str] = Field(default_factory=list, max_length=6)
    removed_paths: list[str] = Field(default_factory=list, max_length=6)
    trials: list[PatchAblationTrial] = Field(default_factory=list, max_length=16)
    mutation_writes: int = Field(default=0, ge=0, le=12)
    patch_chars_before: int = Field(default=0, ge=0)
    patch_chars_after: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0, ge=0)
    fingerprint: str = ""


class MutationTestTrial(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    phase: Literal["control", "mutant"]
    mutant_id: str = ""
    repeat: int = Field(ge=0, le=1)
    source_sha256: str
    manifest_sha256: str
    receipt_sha256: str
    outcomes: dict[str, Literal["passed", "assertion", "error", "skipped"]] = Field(max_length=200)
    exit_code: int
    duration_ms: float = Field(ge=0)


class PatchMutationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mutant_id: str
    path: str
    line: int = Field(ge=1)
    column: int = Field(ge=0)
    operator: Literal["boundary_shift", "comparison_flip", "arithmetic_swap", "boolean_flip"]
    original_file_sha256: str
    mutated_file_sha256: str
    expected_source_sha256: str
    status: Literal["pending", "killed", "survived", "invalid"] = "pending"
    killing_test_sha256: list[str] = Field(default_factory=list, max_length=200)


class PatchMutationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: Literal["report", "enforce"]
    status: Literal["measured", "insufficient_evidence", "inconclusive", "unsupported", "unavailable", "budget_exhausted", "error", "integrity_error"] = "error"
    reason: str = "mutation_execution_invalid"
    min_mutants: int = Field(ge=1, le=6)
    min_score: float = Field(ge=0, le=1)
    original_sha256: str = ""
    candidate_sha256: str = ""
    command_sha256: str = ""
    test_file_sha256: dict[str, str] = Field(default_factory=dict, max_length=6)
    eligible_test_sha256: list[str] = Field(default_factory=list, max_length=40)
    mutants: list[PatchMutationEvidence] = Field(default_factory=list, max_length=6)
    trials: list[MutationTestTrial] = Field(default_factory=list, max_length=14)
    killed: int = Field(default=0, ge=0, le=6)
    survived: int = Field(default=0, ge=0, le=6)
    invalid: int = Field(default=0, ge=0, le=6)
    score: float = Field(default=0, ge=0, le=1)
    sources_unchanged: bool = False
    duration_ms: float = Field(default=0, ge=0)
    fingerprint: str = ""


class VerificationRound(BaseModel):
    mutation_testing: PatchMutationReport | None = None
    patch_minimization: PatchMinimizationReport | None = None
    round: int = Field(ge=0, le=5)
    verdict: ReviewVerdict
    quality_gates: list[QualityGateResult] = Field(default_factory=list)
    counterfactual: CounterfactualReport | None = None


class VerificationReport(BaseModel):
    workflow: str = "verified_pr_v1"
    analysis: AnalysisPlan
    stage_outcomes: dict[str, str] = Field(default_factory=dict)
    context: CodeContextReceipt | None = None
    test_files_changed: list[str] = Field(default_factory=list)
    rounds: list[VerificationRound] = Field(default_factory=list)
    repair_rounds: int = 0
    final_decision: Literal["verified", "blocked"] = "blocked"
    blocking_reasons: list[str] = Field(default_factory=list)


class OracleCalibrationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    reference_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    max_mutants: int = Field(default=4, ge=1, le=8)
    min_mutants: int = Field(default=2, ge=1, le=8)
    min_mutation_score: float = Field(default=1.0, ge=0, le=1)
    timeout_seconds: float = Field(default=180, gt=0, le=900)

    @model_validator(mode="after")
    def feasible_minimum(self):
        if self.min_mutants > self.max_mutants:
            raise ValueError("minimum mutants cannot exceed the cap")
        return self


class RegressionChallengePolicy(BaseModel):
    """Operator-owned Python call targets; never accepted from a task's model output."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    allowed_targets: list[str] = Field(min_length=1, max_length=8)
    max_probes: int = Field(default=8, ge=1, le=16)
    generation_timeout_seconds: float = Field(default=60, gt=0, le=300)
    oracle_calibration: OracleCalibrationPolicy | None = None

    @field_validator("allowed_targets")
    @classmethod
    def explicit_public_targets(cls, value):
        if len(value) != len(set(value)) or any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+", target)
            or len(target) > 200 for target in value
        ):
            raise ValueError("targets must be unique public module.function names")
        return value


class RegressionProbeEvidence(BaseModel):
    suite_sha256: str = ""
    execution_sha256: str = ""
    statuses: list[Literal["matched", "mismatched", "error"]] = Field(default_factory=list, max_length=16)
    repeated: bool = False
    stable: bool = False
    workspace_unchanged: bool = False


class MutationProbeEvidence(BaseModel):
    mutant_id: str
    path: str
    line: int = Field(ge=1)
    operator: Literal["comparison_flip", "guard_removal", "min_max_swap", "arithmetic_swap"]
    source_sha256: str
    status: Literal["killed", "survived", "invalid"] = "invalid"
    killing_probe_indices: list[int] = Field(default_factory=list)
    execution: RegressionProbeEvidence | None = None


class OracleCalibrationReport(BaseModel):
    schema_version: str = "1.0"
    policy: OracleCalibrationPolicy
    suite_sha256: str
    probe_count: int = Field(ge=1, le=16)
    reference_sha256: str = ""
    reference_unchanged: bool = False
    overlay_sha256: dict[str, str] = Field(default_factory=dict)
    reference_owner_tests_passed: bool = False
    reference_execution: RegressionProbeEvidence | None = None
    mutants: list[MutationProbeEvidence] = Field(default_factory=list)
    killed_mutants: int = 0
    mutation_score: float = 0.0
    minimal_cover_indices: list[int] = Field(default_factory=list)
    command_runs: int = 0
    eligible: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)
    fingerprint: str = ""


class DispatchAdvisorShadowPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    scope_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class DispatchShadowEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    candidate_id: str
    context_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    baseline_role: Literal["boundary", "compatibility", "defensive"]
    proposed_role: Literal["boundary", "compatibility", "defensive"] | None = None
    disposition: Literal["supported", "unseen_context", "insufficient_support", "low_agreement", "lineage_mismatch", "artifact_unavailable"]
    supporting_families: int = Field(default=0, ge=0)
    label_agreement: float = Field(default=0, ge=0, le=1)
    serving_changed: Literal[False] = False


class AdaptiveDispatchPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    verification_model_calls: int = Field(default=1, ge=1, le=10)
    verification_prompt_tokens: int = Field(default=4096, ge=64, le=100_000)
    verification_seconds: float = Field(default=30, gt=0, le=300)
    repeated_failure_limit: int = Field(default=2, ge=2, le=4)
    advisor_shadow: DispatchAdvisorShadowPolicy | None = None


class FailureDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal["verified", "wrong_localization", "incomplete_fix", "dependency_or_contract",
                      "regression", "safety_hold", "provider_error", "budget", "timeout", "unknown"]
    signals: list[str] = Field(default_factory=list, max_length=12)
    command_sha256: str = ""
    failure_signature: str = ""
    evidence_sha256: str = ""
    heuristic: bool = True


class DispatchDecision(BaseModel):
    candidate_id: str
    perspective: Literal["minimal", "boundary", "defensive", "compatibility"]
    trigger: str


class RepairTournamentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    candidates: int = Field(default=3, ge=1, le=4)
    max_parallel: int = Field(default=2, ge=1, le=2)
    max_model_calls: int = Field(default=64, ge=1, le=1000)
    max_estimated_prompt_tokens: int = Field(default=100_000, ge=64, le=1_000_000)
    candidate_timeout_seconds: float = Field(default=300, gt=0, le=3600)
    challenge_timeout_seconds: float = Field(default=60, gt=0, le=300)
    regression_challenges: RegressionChallengePolicy | None = None
    adaptive_dispatch: AdaptiveDispatchPolicy | None = None

    @model_validator(mode="after")
    def bounded_parallelism(self):
        if self.max_parallel > self.candidates:
            raise ValueError("parallelism cannot exceed candidate count")
        if self.adaptive_dispatch:
            if self.max_parallel != 1:
                raise ValueError("adaptive dispatch is sequential and requires max_parallel=1")
            if (self.max_model_calls <= self.adaptive_dispatch.verification_model_calls
                    or self.max_estimated_prompt_tokens <= self.adaptive_dispatch.verification_prompt_tokens):
                raise ValueError("adaptive budgets must leave room for both repair and verification")
            if self.adaptive_dispatch.verification_seconds >= self.candidate_timeout_seconds:
                raise ValueError("candidate deadline must leave time for repair before the verification reserve")
        return self


class RepairCandidateEvidence(BaseModel):
    candidate_id: str
    perspective: str
    status: Literal["completed", "failed", "timed_out", "errored", "skipped"] = "failed"
    eligible: bool = False
    reasons: list[str] = Field(default_factory=list)
    patch_sha256: str = ""
    verification_sha256: str = ""
    changed_files: int = 0
    changed_lines: int = 0
    patch_chars: int = 0
    allocated_writes: int = 0
    reported_writes: int = 0
    model_calls: int = 0
    challenge_sha256: str = ""
    challenge_approved: bool = False
    challenge_blocking_findings: int = 0
    final_gate_statuses: dict[str, str] = Field(default_factory=dict)
    regression_probes: RegressionProbeEvidence | None = None
    diagnosis: FailureDiagnosis | None = None


class RepairTournamentReport(BaseModel):
    schema_version: str = "1.0"
    workflow: str = "repair_tournament_v1"
    policy: RepairTournamentPolicy
    task_sha256: str
    source_sha256: str
    snapshot_sha256: str = ""
    source_unchanged: bool = False
    candidates: list[RepairCandidateEvidence] = Field(default_factory=list)
    winner_id: str = ""
    unique_eligible_patches: int = 0
    selection_rule: str = "fewest-files_then_changed-lines_then-patch-chars_then-candidate-id"
    model_calls: int = 0
    shared_model_calls: int = 0
    regression_suite_sha256: str = ""
    regression_suite: dict[str, Any] | None = None
    regression_baseline: RegressionProbeEvidence | None = None
    oracle_calibration: OracleCalibrationReport | None = None
    denied_model_calls: int = 0
    estimated_prompt_tokens_reserved: int = 0
    usage_accounting_complete: bool = True
    calls_with_usage: int = Field(default=0, ge=0)
    candidate_regressions: int = Field(default=0, ge=0)
    dispatch_decisions: list[DispatchDecision] = Field(default_factory=list)
    dispatch_shadows: list[DispatchShadowEvidence] = Field(default_factory=list)
    stop_reason: str = ""
    blocking_reasons: list[str] = Field(default_factory=list)
    fingerprint: str = ""


class ExecutionRetrievalReceipt(BaseModel):
    """Content-free per-stage retrieval evidence, not a diagnosis or quality score."""

    iteration: int = Field(ge=0)
    stage: str
    trigger: Literal["baseline_failure", "test_failure", "workspace_changed"]
    status: Literal["selected", "no_anchors", "budget_exhausted", "unsafe_context", "error"]
    failure_sha256: str
    query_sha256: str = ""
    context_sha256: str = ""
    index_sha256: str = ""
    selected_file_sha256: dict[str, str] = Field(default_factory=dict)
    anchor_paths: list[str] = Field(default_factory=list)
    estimated_tokens: int = Field(default=0, ge=0)
    parsed_files: int = Field(default=0, ge=0)
    reused_files: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)


class RepairMemoryReceipt(BaseModel):
    stage: str
    iteration: int = Field(ge=0)
    status: Literal["selected", "no_match", "unscoped", "unavailable"]
    scope_sha256: str = ""
    query_sha256: str = ""
    snapshot_sha256: str = ""
    context_sha256: str = ""
    lesson_scores: dict[str, float] = Field(default_factory=dict)
    approximate_tokens: int = Field(default=0, ge=0)
    invalid_records: int = Field(default=0, ge=0)


class SuspiciousLine(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    path: str
    line: int = Field(ge=1)
    source_sha256: str
    failed_cover: int = Field(ge=1, le=200)
    passed_cover: int = Field(ge=0, le=200)
    score: float = Field(ge=0, le=1)


class FaultLocalizationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    stage: str
    iteration: int = Field(ge=0)
    status: Literal["selected", "insufficient_evidence", "unsupported", "unavailable", "unstable", "error",
                    "budget_exhausted", "unsafe_context", "awaiting_owner_test", "source_changed"]
    reason: str
    source_sha256: str = ""
    command_sha256: str = ""
    spectrum_sha256: str = ""
    context_sha256: str = ""
    passing_tests: int = Field(default=0, ge=0, le=200)
    failing_tests: int = Field(default=0, ge=0, le=200)
    ranked_lines: list[SuspiciousLine] = Field(default_factory=list, max_length=12)
    trial_sha256: list[str] = Field(default_factory=list, max_length=2)
    approximate_tokens: int = Field(default=0, ge=0, le=1024)
    duration_ms: float = Field(default=0, ge=0)


class AgentLoopReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    candidate_id: str = Field(default="", max_length=100)
    stage: str = Field(max_length=100)
    iteration: int = Field(ge=1, le=100)
    mode: Literal["shadow", "enforce"]
    disposition: Literal["warning", "stopped", "shadow_limit", "unavailable"]
    reason: Literal["repeated_tool_source_cycle", "source_manifest_unavailable"]
    cycle_length: int = Field(default=0, ge=0, le=4)
    repetitions: int = Field(default=0, ge=0, le=5)
    repetition_limit: int = Field(default=3, ge=3, le=5)
    max_cycle_length: int = Field(default=4, ge=1, le=4)
    source_sha256: str = ""
    witness_sha256: list[str] = Field(default_factory=list, max_length=20)
    fingerprint: str = ""


class CodeAgentResult(BaseModel):
    loop_guard: list[AgentLoopReceipt] = Field(default_factory=list)
    fault_localization: list[FaultLocalizationReceipt] = Field(default_factory=list)
    repair_memory: list[RepairMemoryReceipt] = Field(default_factory=list)
    execution_retrieval: list[ExecutionRetrievalReceipt] = Field(default_factory=list)
    provider_failures: list[ProviderFailure] = Field(default_factory=list)
    status: Literal["completed", "failed", "budget_exhausted"]
    summary: str
    patch: str = ""
    changed_files: list[str] = Field(default_factory=list)
    baseline_test: SandboxCommandResult | None = None
    final_test: SandboxCommandResult | None = None
    iterations: int = 0
    writes: int = 0
    tool_calls: int = 0
    model_calls: int = 0
    prompt_tokens: int = 0
    usage_accounting_complete: bool | None = None
    completion_tokens: int = 0
    estimated_cost_usd: float = Field(default=0.0, ge=0)
    total_duration_ms: float = 0.0
    observations: list[ToolObservation] = Field(default_factory=list)
    model: str = ""
    sandbox: dict[str, Any] = Field(default_factory=dict)
    verification: VerificationReport | None = None
    security: SecuritySummary | None = None
    telemetry: TelemetryTrace | None = None
    tournament: RepairTournamentReport | None = None
