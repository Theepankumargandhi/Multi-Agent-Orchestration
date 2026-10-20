"""Reproducible outcome evaluation and trajectory artifacts for coding agents."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from code_agent.agent import CodingAgent, build_coding_model
from code_agent.benchmark_checkpoint import BenchmarkCheckpoint, checkpoint_scope, digest, image_identities
from code_agent.evaluation_models import (
    CodeBenchmarkCase,
    CodeBenchmarkOutcome,
    CodeBenchmarkReport,
    CodeTaskScore,
)
from code_agent.intelligence import STRATEGIES
from code_agent.models import CodeAgentResult, CodeTask, RepairTournamentPolicy, SandboxPolicy
from code_agent.provider_reliability import classify_provider_error
from code_agent.sandbox import DockerSandbox
from code_agent.verified_pr import build_verified_pr_agent


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_fingerprint(cases: list[CodeBenchmarkCase]) -> str:
    return _canonical_hash([case.model_dump(mode="json") for case in cases])


def load_code_cases(dataset: Path) -> list[CodeBenchmarkCase]:
    cases: list[CodeBenchmarkCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(dataset.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = CodeBenchmarkCase.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"invalid benchmark case at line {line_number}: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"duplicate coding benchmark case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("coding benchmark contains no cases")
    return cases


def score_code_result(
    result: CodeAgentResult,
    *,
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
) -> CodeTaskScore:
    baseline = result.baseline_test
    final = result.final_test
    baseline_passed = bool(baseline and baseline.exit_code == 0 and not baseline.timed_out)
    tests_passed = bool(final and final.exit_code == 0 and not final.timed_out)
    patch_nonempty = bool(result.patch.strip() and result.changed_files)
    original_unchanged = result.sandbox.get("original_repository_unchanged") is True
    policy_violations = sum(
        observation.summary.startswith("Tool policy rejected") for observation in result.observations
    )
    estimated_cost = (
        result.prompt_tokens * input_cost_per_million
        + result.completion_tokens * output_cost_per_million
    ) / 1_000_000
    verification = result.verification
    telemetry = result.telemetry
    telemetry_valid = bool(
        telemetry
        and telemetry.content_capture_enabled is False
        and telemetry.spans
        and telemetry.spans[0].attributes.get("gen_ai.operation.name") == "invoke_agent"
    )
    from code_agent.delivery import result_delivery_valid
    delivery_valid = result_delivery_valid(result)
    return CodeTaskScore(
        resolved=result.status == "completed" and tests_passed and patch_nonempty and original_unchanged
        and not any(failure.terminal for failure in result.provider_failures)
        and (not result.sandbox.get("delivery_required") or delivery_valid),
        patch_nonempty=patch_nonempty,
        tests_passed=tests_passed,
        baseline_passed=baseline_passed,
        regression=baseline_passed and not tests_passed,
        command_timed_out=bool(final and final.timed_out),
        original_repository_unchanged=original_unchanged,
        policy_violations=policy_violations,
        iterations=result.iterations,
        tool_calls=result.tool_calls,
        model_calls=result.model_calls,
        usage_accounting_complete=result.usage_accounting_complete,
        candidate_regressions=result.tournament.candidate_regressions if result.tournament else int(baseline_passed and not tests_passed),
        duration_ms=result.total_duration_ms,
        changed_files=len(result.changed_files),
        patch_chars=len(result.patch),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        estimated_cost_usd=estimated_cost,
        verification_passed=bool(verification and verification.final_decision == "verified"),
        repair_rounds=verification.repair_rounds if verification else 0,
        blocking_findings=len(verification.blocking_reasons) if verification else 0,
        telemetry_valid=telemetry_valid,
        telemetry_spans=len(telemetry.spans) if telemetry else 0,
    )


def _failure_category(result: CodeAgentResult, score: CodeTaskScore) -> str:
    if any(failure.terminal for failure in result.provider_failures):
        return "provider_" + next(failure.category for failure in result.provider_failures if failure.terminal)
    if score.resolved:
        return "resolved"
    if result.provider_failures:
        return "provider_" + result.provider_failures[-1].category
    if any(
        observation.summary.startswith("Action model failed")
        for observation in result.observations
    ):
        return "model_error"
    if score.command_timed_out:
        return "timeout"
    if score.policy_violations:
        return "sandbox_policy_rejection"
    if result.status == "budget_exhausted":
        return "budget_exhausted"
    if not score.original_repository_unchanged:
        return "source_integrity_failure"
    if not score.patch_nonempty:
        return "no_patch"
    if not score.tests_passed:
        return "test_failure"
    return "execution_error"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _bootstrap_pass_interval(values: list[float], samples: int = 2000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(20260907)
    means = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _artifact_name(case_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", case_id).strip("-.")[:80] or "case"
    return f"{slug}-{hashlib.sha256(case_id.encode()).hexdigest()[:10]}"


def _write_case_artifacts(
    run_dir: Path,
    run_id: str,
    case: CodeBenchmarkCase,
    result: CodeAgentResult,
    score: CodeTaskScore,
    dataset_hash: str,
    config_hash: str,
) -> tuple[str, str, str]:
    name = _artifact_name(case.id)
    trajectory_relative = Path("trajectories") / f"{name}.json"
    patch_relative = Path("patches") / f"{name}.diff"
    trajectory_path = run_dir / trajectory_relative
    patch_path = run_dir / patch_relative
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(result.patch, encoding="utf-8", newline="")
    patch_hash = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    trajectory = {
        "schema_version": "1.0",
        "run_id": run_id,
        "case": case.model_dump(mode="json"),
        "dataset_fingerprint": dataset_hash,
        "config_fingerprint": config_hash,
        "score": score.model_dump(mode="json"),
        "patch_sha256": patch_hash,
        "result": result.model_dump(mode="json", exclude={"patch"}),
    }
    trajectory_path.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return trajectory_relative.as_posix(), patch_relative.as_posix(), patch_hash


def _slice_metrics(outcomes: list[CodeBenchmarkOutcome]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[CodeBenchmarkOutcome]] = defaultdict(list)
    for outcome in outcomes:
        buckets[f"repository:{outcome.repository}"].append(outcome)
        for tag in outcome.tags:
            buckets[f"tag:{tag}"].append(outcome)
    metrics = {}
    for name, values in sorted(buckets.items()):
        scored = [item.score for item in values if item.score is not None]
        resolved = sum(score.resolved for score in scored)
        metrics[name] = {
            "total": float(len(values)),
            "resolved": float(resolved),
            "pass_at_1": resolved / len(values) if values else 0.0,
            "average_duration_ms": statistics.fmean(score.duration_ms for score in scored)
            if scored
            else 0.0,
        }
    return metrics


async def run_benchmark(cases, *, checkpoint_dir: Path | None = None, resume: bool = False,
                        retry_provider_hold: bool = False, retry_interrupted: bool = False, **kwargs):
    """Optional case-boundary recovery. No checkpoint means the original API."""
    from code_agent.memory_snapshot import prepare_memory
    if kwargs.get("policy"):
        if kwargs["policy"].counterfactual_verification_enabled and kwargs.get("workflow", "single_agent") == "single_agent":
            raise ValueError("counterfactual verification requires a verified-PR workflow")
        if kwargs["policy"].patch_minimization_enabled and kwargs.get("workflow", "single_agent") == "single_agent":
            raise ValueError("patch minimization requires a verified-PR workflow")
        if kwargs["policy"].mutation_testing_mode != "off" and kwargs.get("workflow", "single_agent") == "single_agent":
            raise ValueError("patch mutation testing requires a verified-PR workflow")
        kwargs["memory_snapshot"] = prepare_memory(cases, kwargs["policy"], kwargs.get("memory_snapshot"), kwargs.get("workflow", "single_agent"))
    if resume or retry_provider_hold or retry_interrupted:
        if checkpoint_dir is None or not resume:
            raise ValueError("recovery options require checkpoint_dir and resume")
    if checkpoint_dir is None:
        return await _run_benchmark(cases, **kwargs)
    with BenchmarkCheckpoint(checkpoint_dir, resume=resume) as checkpoint:
        return await _run_benchmark(cases, _checkpoint=checkpoint,
                                    _retry_provider_hold=retry_provider_hold,
                                    _retry_interrupted=retry_interrupted, **kwargs)


async def _run_benchmark(
    cases: list[CodeBenchmarkCase],
    *,
    repository_root: Path,
    default_model: str,
    policy: SandboxPolicy,
    dataset_name: str = "local-code-tasks",
    dataset_path: str = "",
    output_dir: Path | None = None,
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
    image_by_repository: dict[str, str] | None = None,
    workflow: str = "single_agent",
    context_strategy: str = "hybrid_rerank",
    tournament_policy: RepairTournamentPolicy | None = None,
    _checkpoint: BenchmarkCheckpoint | None = None,
    _retry_provider_hold: bool = False,
    _retry_interrupted: bool = False,
    memory_snapshot=None,
) -> CodeBenchmarkReport:
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("benchmark cases must be nonempty and have unique IDs")
    from code_agent.memory_snapshot import prepare_memory
    memory_snapshot = prepare_memory(cases, policy, memory_snapshot, workflow)
    if policy.counterfactual_verification_enabled and workflow == "single_agent":
        raise ValueError("counterfactual verification requires a verified-PR workflow")
    if policy.patch_minimization_enabled and workflow == "single_agent":
        raise ValueError("patch minimization requires a verified-PR workflow")
    if policy.mutation_testing_mode != "off" and workflow == "single_agent":
        raise ValueError("patch mutation testing requires a verified-PR workflow")
    if workflow not in {"single_agent", "verified_pr", "repair_tournament", "adaptive_supervisor"}:
        raise ValueError("workflow must be single_agent, verified_pr, repair_tournament, or adaptive_supervisor")
    if workflow in {"repair_tournament", "adaptive_supervisor"} and tournament_policy is None:
        from code_agent.repair_tournament import (
            adaptive_policy_from_environment,
            tournament_policy_from_environment,
        )

        tournament_policy = adaptive_policy_from_environment() if workflow == "adaptive_supervisor" else tournament_policy_from_environment()
    if bool(tournament_policy and tournament_policy.adaptive_dispatch) != (workflow == "adaptive_supervisor"):
        raise ValueError("adaptive_dispatch policy is required only for adaptive_supervisor")
    if context_strategy not in STRATEGIES:
        raise ValueError(f"context strategy must be one of: {', '.join(STRATEGIES)}")
    effective_context_strategy = (
        context_strategy if workflow != "single_agent" else "repository_map"
    )
    outcomes: list[CodeBenchmarkOutcome] = []
    predictions: list[dict[str, str]] = []
    root = repository_root.resolve()
    dataset_hash = dataset_fingerprint(cases)
    policy_payload = policy.model_dump(mode="json")
    repository_images = image_by_repository or {}
    untrusted_images = set(repository_images.values()) - policy.allowed_images
    if untrusted_images:
        raise ValueError(f"repository image map contains non-allowlisted images: {sorted(untrusted_images)}")
    config_hash = _canonical_hash(
        {
            "model": default_model,
            "policy": policy_payload,
            "input_cost_per_million": input_cost_per_million,
            "output_cost_per_million": output_cost_per_million,
            "image_by_repository": repository_images,
            "workflow": workflow,
            "context_strategy": effective_context_strategy,
            "tournament_policy": tournament_policy.model_dump(mode="json") if tournament_policy else None,
            **({"memory_snapshot_sha256": memory_snapshot.fingerprint,
                "memory_audit_sha256": memory_snapshot.audit_sha256} if memory_snapshot else {}),
        }
    )
    run_id = f"code-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = output_dir.resolve() / run_id if output_dir else None
    previous = {}
    if _checkpoint:
        identities = await asyncio.to_thread(image_identities, {policy.image, *repository_images.values()})
        scope = await asyncio.to_thread(checkpoint_scope, cases, root, policy, config_hash)
        run_id = _checkpoint.bind(digest([scope, dataset_hash, identities]), run_id)
        run_dir = _checkpoint.directory
        previous = {case.id: _checkpoint.previous(case.id, retry_provider_hold=_retry_provider_hold,
                                                 retry_interrupted=_retry_interrupted) for case in cases}
    elif run_dir:
        run_dir.mkdir(parents=True, exist_ok=False)

    paused = False
    for case in cases:
        if previous.get(case.id):
            outcomes.append(CodeBenchmarkOutcome.model_validate(previous[case.id]["outcome"]))
            predictions.append(previous[case.id]["prediction"])
            continue
        if paused:
            outcomes.append(CodeBenchmarkOutcome(case_id=case.id, repository=case.repository,
                                                tags=case.tags, status="not_run", failure_category="not_run"))
            continue
        artifact_dir = _checkpoint.start(case.id) if _checkpoint else run_dir
        sandbox: DockerSandbox | None = None
        held = False
        try:
            case_policy = SandboxPolicy.model_validate(
                {
                    **policy.model_dump(),
                    "image": repository_images.get(case.repository, policy.image),
                }
            )
            task = CodeTask(
                repository=case.repository,
                issue=case.issue,
                model=case.model or default_model,
                test_command=case.test_command,
                policy=case_policy,
                benchmark_case_id=case.id,
            )
            repository = (root / task.repository).resolve()
            repository.relative_to(root)
            if not repository.is_dir():
                raise ValueError("benchmark repository does not exist below repository root")
            if workflow in {"repair_tournament", "adaptive_supervisor"}:
                from code_agent.repair_tournament import build_adaptive_supervisor, build_repair_tournament

                builder = build_adaptive_supervisor if workflow == "adaptive_supervisor" else build_repair_tournament
                agent = builder(task.model, policy=tournament_policy,
                    context_strategy=effective_context_strategy, input_cost_per_million=input_cost_per_million,
                    output_cost_per_million=output_cost_per_million)
                result = await agent.solve(task, repository)
            else:
                sandbox = DockerSandbox(repository, case_policy)
                await asyncio.to_thread(sandbox.start)
                agent = (
                    build_verified_pr_agent(task.model, context_strategy=effective_context_strategy,
                        **({"repair_memory_snapshot": memory_snapshot} if memory_snapshot else {}))
                    if workflow == "verified_pr"
                    else CodingAgent(build_coding_model(task.model),
                        **({"repair_memory_snapshot": memory_snapshot} if memory_snapshot else {}))
                )
                result = await asyncio.wait_for(
                    agent.solve(task, sandbox),
                    timeout=policy.task_timeout_seconds + 30,
                )
            score = score_code_result(
                result,
                input_cost_per_million=input_cost_per_million,
                output_cost_per_million=output_cost_per_million,
            )
            trajectory_path = patch_path = patch_hash = ""
            valid_delivery = True
            if case_policy.delivery_verification_enabled and result.status == "completed":
                from code_agent.delivery import result_delivery_valid
                valid_delivery = result_delivery_valid(result, task.test_command)
                if not valid_delivery:
                    score = score.model_copy(update={"resolved": False})
            valid_memory = True
            if memory_snapshot:
                valid_memory = memory_snapshot.valid_receipts(task, result.repair_memory)
                if not valid_memory:
                    score = score.model_copy(update={"resolved": False})
            valid_counterfactual = True
            mutation_reports = [row.mutation_testing for row in result.verification.rounds if row.mutation_testing] if result.verification else []
            valid_mutation = True
            if policy.mutation_testing_mode != "off" and result.status == "completed":
                from code_agent.patch_mutation import mutation_gate
                latest_mutation = result.verification.rounds[-1].mutation_testing if result.verification and result.verification.rounds else None
                valid_mutation = bool(latest_mutation and latest_mutation.mode == policy.mutation_testing_mode
                    and latest_mutation.min_mutants == policy.mutation_testing_min_mutants
                    and latest_mutation.min_score == policy.mutation_testing_min_score
                    and len(latest_mutation.mutants) <= policy.mutation_testing_max_mutants
                    and mutation_gate(latest_mutation).status != "failed")
                if not valid_mutation:
                    score = score.model_copy(update={"resolved": False})
            minimization_reports = [row.patch_minimization for row in result.verification.rounds if row.patch_minimization] if result.verification else []
            valid_minimization = True
            if policy.patch_minimization_enabled and result.status == "completed":
                from code_agent.patch_minimization import minimization_gate
                latest = result.verification.rounds[-1].patch_minimization if result.verification and result.verification.rounds else None
                valid_minimization = bool(latest and minimization_gate(latest).status != "failed")
                if not valid_minimization:
                    score = score.model_copy(update={"resolved": False})
            if policy.counterfactual_verification_enabled and result.status == "completed":
                from code_agent.counterfactual import valid_report
                verification = result.verification
                valid_counterfactual = bool(verification and verification.rounds and valid_report(verification.rounds[-1].counterfactual))
                if not valid_counterfactual:
                    score = score.model_copy(update={"resolved": False})
            if artifact_dir:
                trajectory_path, patch_path, patch_hash = _write_case_artifacts(
                    artifact_dir, run_id, case, result, score, dataset_hash, config_hash
                )
                if _checkpoint:
                    prefix = artifact_dir.relative_to(run_dir).as_posix()
                    trajectory_path, patch_path = f"{prefix}/{trajectory_path}", f"{prefix}/{patch_path}"
            held = any(failure.terminal for failure in result.provider_failures)
            outcomes.append(
                CodeBenchmarkOutcome(
                    case_id=case.id,
                    repository=case.repository,
                    tags=case.tags,
                    status=result.status,
                    failure_category=("delivery_evidence_invalid" if not valid_delivery and not held
                                      else "counterfactual_evidence_invalid" if not valid_counterfactual and not held
                                      else "mutation_evidence_invalid" if not valid_mutation and not held
                                      else "minimization_evidence_invalid" if not valid_minimization and not held
                                      else "memory_evidence_invalid" if memory_snapshot and not valid_memory and not held
                                      else _failure_category(result, score)),
                    score=score,
                    trajectory_path=trajectory_path,
                    patch_path=patch_path,
                    patch_sha256=patch_hash,
                    provider_failures=result.provider_failures,
                    loop_guard_status_counts=dict(Counter(item.disposition for item in result.loop_guard)),
                    mutation_testing_status_counts=dict(Counter(item.status for item in mutation_reports)),
                    mutation_measured=any(item.status == "measured" for item in mutation_reports),
                    mutation_killed=sum(item.killed for item in mutation_reports),
                    mutation_survived=sum(item.survived for item in mutation_reports),
                    mutation_invalid=sum(item.invalid for item in mutation_reports),
                    mutation_completed_trials=sum(len(item.trials) for item in mutation_reports),
                    fault_localized=any(item.status == "selected" for item in result.fault_localization),
                    fault_localization_status_counts=dict(Counter(item.status for item in result.fault_localization)),
                    patch_minimized=any(item.status == "minimized" for item in minimization_reports),
                    minimization_removed_file_changes=sum(len(item.removed_paths) for item in minimization_reports if item.status == "minimized"),
                    minimization_completed_trials=sum(len(item.trials) for item in minimization_reports),
                    patch_minimization_status_counts=dict(Counter(item.status for item in minimization_reports)),
                )
            )
            predictions.append(
                {
                    "instance_id": str(case.metadata.get("instance_id") or case.id),
                    "model_name_or_path": task.model,
                    "model_patch": result.patch,
                }
            )
        except Exception as exc:
            failure = classify_provider_error(exc, stage="benchmark")
            held = failure.terminal
            outcomes.append(
                CodeBenchmarkOutcome(
                    case_id=case.id,
                    repository=case.repository,
                    tags=case.tags,
                    failure_category=("provider_" + failure.category if failure.category != "unknown"
                                      else "memory_evidence_invalid" if memory_snapshot else "execution_error"),
                    provider_failures=[failure] if failure.category != "unknown" else [],
                    error=type(exc).__name__,
                )
            )
            predictions.append(
                {
                    "instance_id": str(case.metadata.get("instance_id") or case.id),
                    "model_name_or_path": case.model or default_model,
                    "model_patch": "",
                }
            )
        finally:
            if sandbox is not None:
                await asyncio.to_thread(sandbox.close)
        if _checkpoint:
            _checkpoint.finish(case.id, outcomes[-1], predictions[-1], held=held)
        paused = paused or held

    scored = [outcome.score for outcome in outcomes if outcome.score is not None]
    accounting_scores = scored
    recovery_unknown = False
    retained_attempts = sum(outcome.status != "not_run" for outcome in outcomes)
    retried = False
    if _checkpoint:
        histories = _checkpoint.state["attempts"]
        all_attempts = [attempt for attempts in histories.values() for attempt in attempts]
        retained_attempts = len(all_attempts)
        retried = any(len(attempts) > 1 for attempts in histories.values())
        recovery_unknown = any(attempt["status"] == "running" or not attempt.get("outcome", {}).get("score") for attempt in all_attempts)
        accounting_scores = [CodeTaskScore.model_validate(attempt["outcome"]["score"])
                             for attempt in all_attempts if attempt.get("outcome", {}).get("score")]
    resolved_values = [float(bool(outcome.score and outcome.score.resolved)) for outcome in outcomes]
    resolved = int(sum(resolved_values))
    total = len(outcomes)
    total_cost = sum(score.estimated_cost_usd for score in accounting_scores)
    report = CodeBenchmarkReport(
        mutation_testing_status_counts=dict(sum((Counter(outcome.mutation_testing_status_counts) for outcome in outcomes), Counter())),
        mutation_measured_cases=sum(outcome.mutation_measured for outcome in outcomes),
        mutation_killed=sum(outcome.mutation_killed for outcome in outcomes),
        mutation_survived=sum(outcome.mutation_survived for outcome in outcomes),
        mutation_invalid=sum(outcome.mutation_invalid for outcome in outcomes),
        mutation_completed_trials=sum(outcome.mutation_completed_trials for outcome in outcomes),
        loop_guard_status_counts=dict(sum((Counter(outcome.loop_guard_status_counts) for outcome in outcomes), Counter())),
        patch_minimized_cases=sum(outcome.patch_minimized for outcome in outcomes),
        minimization_removed_file_changes=sum(outcome.minimization_removed_file_changes for outcome in outcomes),
        minimization_completed_trials=sum(outcome.minimization_completed_trials for outcome in outcomes),
        patch_minimization_status_counts=dict(sum((Counter(outcome.patch_minimization_status_counts) for outcome in outcomes), Counter())),
        fault_localized_cases=sum(outcome.fault_localized for outcome in outcomes),
        fault_localization_status_counts=dict(sum(
            (Counter(outcome.fault_localization_status_counts) for outcome in outcomes), Counter())),
        run_id=run_id,
        execution_status="paused" if paused else "completed",
        quality_metrics_valid=not paused and not retried and not any(outcome.failure_category in {"delivery_evidence_invalid", "memory_evidence_invalid", "counterfactual_evidence_invalid", "minimization_evidence_invalid", "mutation_evidence_invalid"} for outcome in outcomes),
        pending_case_ids=[outcome.case_id for outcome in outcomes if outcome.status == "not_run" or any(f.terminal for f in outcome.provider_failures)],
        attempted_cases=sum(outcome.status != "not_run" for outcome in outcomes),
        retained_attempts=retained_attempts,
        recovery_usage_unknown=recovery_unknown,
        generated_at=datetime.now(UTC).isoformat(),
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_hash,
        config_fingerprint=config_hash,
        model=default_model,
        workflow=workflow,
        context_strategy=effective_context_strategy,
        sandbox_policy={**policy_payload, "image_by_repository": repository_images,
                        "tournament_policy": tournament_policy.model_dump(mode="json") if tournament_policy else None,
                        **({"repair_memory_snapshot_sha256": memory_snapshot.fingerprint,
                            "repair_memory_audit_sha256": memory_snapshot.audit_sha256} if memory_snapshot else {})},
        total=total,
        resolved=resolved,
        pass_at_1=resolved / total if total else 0.0,
        pass_at_1_confidence_interval=_bootstrap_pass_interval(resolved_values),
        test_pass_rate=sum(score.tests_passed for score in scored) / total if total else 0.0,
        regression_rate=sum(score.regression for score in scored) / total if total else 0.0,
        timeout_rate=sum(score.command_timed_out for score in scored) / total if total else 0.0,
        policy_violation_rate=sum(score.policy_violations > 0 for score in scored) / total
        if total
        else 0.0,
        verification_pass_rate=sum(score.verification_passed for score in scored) / total
        if total
        else 0.0,
        telemetry_coverage_rate=sum(score.telemetry_valid for score in scored) / total
        if total
        else 0.0,
        p50_duration_ms=_percentile([score.duration_ms for score in scored], 0.5),
        p95_duration_ms=_percentile([score.duration_ms for score in scored], 0.95),
        average_iterations=statistics.fmean(score.iterations for score in scored) if scored else 0.0,
        average_tool_calls=statistics.fmean(score.tool_calls for score in scored) if scored else 0.0,
        average_model_calls=sum(score.model_calls for score in scored) / total if total else 0.0,
        total_model_calls=sum(score.model_calls for score in accounting_scores),
        usage_accounting_complete=(not paused and not recovery_unknown and len(scored) == total and total > 0
                                   and all(score.usage_accounting_complete is True for score in accounting_scores)),
        prices_configured=input_cost_per_million > 0 and output_cost_per_million > 0,
        total_candidate_regressions=sum(score.candidate_regressions for score in scored),
        average_changed_files=statistics.fmean(score.changed_files for score in scored)
        if scored
        else 0.0,
        total_prompt_tokens=sum(score.prompt_tokens for score in accounting_scores),
        total_completion_tokens=sum(score.completion_tokens for score in accounting_scores),
        total_estimated_cost_usd=total_cost,
        cost_per_resolved_usd=total_cost / resolved if resolved and not paused and not retried
        and not any(outcome.failure_category in {"delivery_evidence_invalid", "memory_evidence_invalid", "counterfactual_evidence_invalid", "minimization_evidence_invalid", "mutation_evidence_invalid"} for outcome in outcomes) else None,
        failure_categories=dict(Counter(outcome.failure_category for outcome in outcomes)),
        slice_metrics=_slice_metrics(outcomes),
        outcomes=outcomes,
    )
    if run_dir:
        report_path = run_dir / "report.json"
        report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        prediction_lines = [json.dumps(item, sort_keys=True) for item in predictions]
        predictions_path = run_dir / "predictions.jsonl"
        predictions_path.write_text(
            "\n".join(prediction_lines) + ("\n" if prediction_lines else ""), encoding="utf-8"
        )
        artifacts = {}
        for artifact in sorted(run_dir.rglob("*")):
            if artifact.is_file() and artifact.name not in {"manifest.json", ".checkpoint.lock", "checkpoint.json"}:
                relative = artifact.relative_to(run_dir).as_posix()
                artifacts[relative] = {
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "bytes": artifact.stat().st_size,
                }
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {"schema_version": "1.0", "run_id": run_id, "artifacts": artifacts},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return report


def _summary(report: CodeBenchmarkReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "model": report.model,
        "workflow": report.workflow,
        "context_strategy": report.context_strategy,
        "fault_localized_cases": report.fault_localized_cases,
        "patch_minimized_cases": report.patch_minimized_cases,
        "minimization_removed_file_changes": report.minimization_removed_file_changes,
        "minimization_completed_trials": report.minimization_completed_trials,
        "patch_minimization_status_counts": report.patch_minimization_status_counts,
        "loop_guard_status_counts": report.loop_guard_status_counts,
        "mutation_testing_status_counts": report.mutation_testing_status_counts,
        "mutation_measured_cases": report.mutation_measured_cases,
        "mutation_killed": report.mutation_killed,
        "mutation_survived": report.mutation_survived,
        "mutation_invalid": report.mutation_invalid,
        "mutation_completed_trials": report.mutation_completed_trials,
        "fault_localization_status_counts": report.fault_localization_status_counts,
        "dataset_fingerprint": report.dataset_fingerprint,
        "total": report.total,
        "execution_status": report.execution_status,
        "quality_metrics_valid": report.quality_metrics_valid,
        "pending_case_ids": report.pending_case_ids,
        "resolved": report.resolved,
        "pass_at_1": report.pass_at_1 if report.quality_metrics_valid else None,
        "verification_pass_rate": report.verification_pass_rate if report.quality_metrics_valid else None,
        "usage_accounting_complete": report.usage_accounting_complete,
        "prices_configured": report.prices_configured,
        "recovery_usage_unknown": report.recovery_usage_unknown,
        "known_estimated_cost_usd": report.total_estimated_cost_usd,
        "cost_usd": report.total_estimated_cost_usd
        if report.usage_accounting_complete and report.prices_configured else None,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run reproducible sandbox coding-agent evaluations")
    parser.add_argument("dataset", type=Path, help="AgentForge coding-task JSONL")
    parser.add_argument("--repository-root", type=Path, default=Path("repositories"))
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument(
        "--workflow",
        choices=["verified_pr", "single_agent", "repair_tournament", "adaptive_supervisor"],
        default="verified_pr",
        help="Agent workflow evaluated for each case",
    )
    parser.add_argument("--image", default="agentforge-code-sandbox:local")
    parser.add_argument("--tournament-policy", type=Path, help="Explicit repair-tournament policy JSON for preregistered runs")
    parser.add_argument(
        "--image-map",
        type=Path,
        help="Operator-controlled JSON mapping from repository names to prebuilt images",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluations/code-agent"))
    parser.add_argument("--checkpoint-dir", type=Path, help="Private durable directory; must not exist for a new run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-provider-hold", action="store_true")
    parser.add_argument("--retry-interrupted", action="store_true")
    parser.add_argument("--dataset-name", default="local-code-tasks")
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--mutation-testing", choices=["off", "report", "enforce"], default="off",
                        help="Patch-targeted changed-regression-test sensitivity in fresh Docker copies")
    parser.add_argument("--loop-guard", choices=["off", "shadow", "enforce"], default="off",
                        help="Stage-local repeated tool/source-cycle detection; shadow does not change planning")
    parser.add_argument("--patch-minimization", action="store_true", help="Opt-in file-level patch ablation before final independent review")
    parser.add_argument("--fault-localization", action="store_true",
                        help="Opt-in repeated Python/pytest spectra in fresh Docker copies")
    parser.add_argument("--execution-retrieval", action="store_true",
                        help="Enable bounded local failure-guided code retrieval (off by default)")
    parser.add_argument("--counterfactual-verification", action="store_true",
                        help="Require stable original-fail/patched-pass regression evidence (pytest workflows only)")
    parser.add_argument("--fail-under", type=float, default=0.0)
    parser.add_argument("--memory-snapshot", type=Path, help="Private frozen repair-memory snapshot (no mutable store)")
    parser.add_argument("--memory-snapshot-sha256", help="Expected canonical snapshot fingerprint")
    parser.add_argument(
        "--context-strategy",
        choices=["lexical", "lexical_graph", "hybrid", "hybrid_rerank"],
        default="hybrid_rerank",
        help="Code-context retrieval strategy for verified_pr runs",
    )
    parser.add_argument("--targeted-edits-only", action="store_true", help="Require exact hash-checked edits for existing files")
    parser.add_argument("--fresh-delivery", action="store_true", help="Require fresh-checkout Docker patch application and tests")
    parser.add_argument("--require-specialists", action="store_true", help="Require completed verified-PR specialist stages")
    args = parser.parse_args()
    if args.counterfactual_verification and args.workflow == "single_agent":
        parser.error("counterfactual verification requires verified_pr or a verified-PR-based tournament workflow")
    if args.patch_minimization and args.workflow == "single_agent":
        parser.error("patch minimization requires a verified-PR workflow")
    if args.mutation_testing != "off" and args.workflow == "single_agent":
        parser.error("patch mutation testing requires a verified-PR workflow")
    tournament_policy = None
    if args.tournament_policy:
        if args.workflow not in {"repair_tournament", "adaptive_supervisor"}:
            parser.error("--tournament-policy requires --workflow repair_tournament or adaptive_supervisor")
        tournament_policy = RepairTournamentPolicy.model_validate_json(args.tournament_policy.read_text(encoding="utf-8"))
    cases = load_code_cases(args.dataset)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    memory_snapshot = None
    if bool(args.memory_snapshot) != bool(args.memory_snapshot_sha256):
        parser.error("--memory-snapshot and --memory-snapshot-sha256 must be supplied together")
    if args.memory_snapshot:
        if args.workflow not in {"single_agent", "verified_pr"}:
            parser.error("memory snapshots support --workflow single_agent or verified_pr only")
        from code_agent.memory_snapshot import FrozenRepairMemory
        try:
            memory_snapshot = FrozenRepairMemory.load(args.memory_snapshot, args.memory_snapshot_sha256, cases)
        except (ValueError, OSError) as exc:
            parser.error(f"memory snapshot preflight failed ({type(exc).__name__}); review its binding and leakage audit")
    if args.validate_only:
        print(
            json.dumps(
                {"cases": len(cases), "dataset_fingerprint": dataset_fingerprint(cases)},
                sort_keys=True,
            )
        )
        return
    models = args.models or ["gpt-4o-mini"]
    if args.checkpoint_dir and len(models) != 1:
        parser.error("checkpoint runs require exactly one model")
    image_map = {}
    if args.image_map:
        image_map = json.loads(args.image_map.read_text(encoding="utf-8"))
        if not isinstance(image_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in image_map.items()
        ):
            raise ValueError("image map must be a JSON object of repository-to-image strings")
    policy = SandboxPolicy(
        image=args.image,
        allowed_images={args.image, *image_map.values()},
        network_enabled=False,
        execution_retrieval_enabled=args.execution_retrieval,
        fault_localization_enabled=args.fault_localization,
        patch_minimization_enabled=args.patch_minimization,
        loop_guard_mode=args.loop_guard,
        mutation_testing_mode=args.mutation_testing,
        repair_memory_enabled=memory_snapshot is not None,
        counterfactual_verification_enabled=args.counterfactual_verification,
        targeted_edits_only=args.targeted_edits_only,
        delivery_verification_enabled=args.fresh_delivery,
        require_specialist_completion=args.require_specialists,
    )
    reports = []
    for model in models:
        report = asyncio.run(
            run_benchmark(
                cases,
                repository_root=args.repository_root,
                default_model=model,
                policy=policy,
                memory_snapshot=memory_snapshot,
                dataset_name=args.dataset_name,
                dataset_path=str(args.dataset),
                output_dir=args.output_dir,
                input_cost_per_million=max(0.0, args.input_cost_per_million),
                output_cost_per_million=max(0.0, args.output_cost_per_million),
                image_by_repository=image_map,
                workflow=args.workflow,
                context_strategy=args.context_strategy,
                tournament_policy=tournament_policy,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
                retry_provider_hold=args.retry_provider_hold,
                retry_interrupted=args.retry_interrupted,
            )
        )
        reports.append(report)
        print(json.dumps(_summary(report), sort_keys=True))
        if report.execution_status == "paused":
            break
    if any(not report.quality_metrics_valid or report.pass_at_1 < args.fail_under for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
