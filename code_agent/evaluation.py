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
from code_agent.evaluation_models import (
    CodeBenchmarkCase,
    CodeBenchmarkOutcome,
    CodeBenchmarkReport,
    CodeTaskScore,
)
from code_agent.intelligence import STRATEGIES
from code_agent.models import CodeAgentResult, CodeTask, RepairTournamentPolicy, SandboxPolicy
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
    return CodeTaskScore(
        resolved=result.status == "completed" and tests_passed and patch_nonempty and original_unchanged,
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
    if score.resolved:
        return "resolved"
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


async def run_benchmark(
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
) -> CodeBenchmarkReport:
    if workflow not in {"single_agent", "verified_pr", "repair_tournament"}:
        raise ValueError("workflow must be single_agent, verified_pr, or repair_tournament")
    if workflow == "repair_tournament" and tournament_policy is None:
        from code_agent.repair_tournament import tournament_policy_from_environment

        tournament_policy = tournament_policy_from_environment()
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
        }
    )
    run_id = f"code-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = output_dir.resolve() / run_id if output_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=False)

    for case in cases:
        sandbox: DockerSandbox | None = None
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
            )
            repository = (root / task.repository).resolve()
            repository.relative_to(root)
            if not repository.is_dir():
                raise ValueError("benchmark repository does not exist below repository root")
            if workflow == "repair_tournament":
                from code_agent.repair_tournament import build_repair_tournament

                agent = build_repair_tournament(task.model, policy=tournament_policy,
                    context_strategy=effective_context_strategy, input_cost_per_million=input_cost_per_million,
                    output_cost_per_million=output_cost_per_million)
                result = await agent.solve(task, repository)
            else:
                sandbox = DockerSandbox(repository, case_policy)
                await asyncio.to_thread(sandbox.start)
                agent = (
                    build_verified_pr_agent(task.model, context_strategy=effective_context_strategy)
                    if workflow == "verified_pr"
                    else CodingAgent(build_coding_model(task.model))
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
            if run_dir:
                trajectory_path, patch_path, patch_hash = _write_case_artifacts(
                    run_dir, run_id, case, result, score, dataset_hash, config_hash
                )
            outcomes.append(
                CodeBenchmarkOutcome(
                    case_id=case.id,
                    repository=case.repository,
                    tags=case.tags,
                    status=result.status,
                    failure_category=_failure_category(result, score),
                    score=score,
                    trajectory_path=trajectory_path,
                    patch_path=patch_path,
                    patch_sha256=patch_hash,
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
            outcomes.append(
                CodeBenchmarkOutcome(
                    case_id=case.id,
                    repository=case.repository,
                    tags=case.tags,
                    error=f"{type(exc).__name__}: {exc}",
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

    scored = [outcome.score for outcome in outcomes if outcome.score is not None]
    resolved_values = [float(bool(outcome.score and outcome.score.resolved)) for outcome in outcomes]
    resolved = int(sum(resolved_values))
    total = len(outcomes)
    total_cost = sum(score.estimated_cost_usd for score in scored)
    report = CodeBenchmarkReport(
        run_id=run_id,
        generated_at=datetime.now(UTC).isoformat(),
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_hash,
        config_fingerprint=config_hash,
        model=default_model,
        workflow=workflow,
        context_strategy=effective_context_strategy,
        sandbox_policy={**policy_payload, "image_by_repository": repository_images,
                        "tournament_policy": tournament_policy.model_dump(mode="json") if tournament_policy else None},
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
        average_changed_files=statistics.fmean(score.changed_files for score in scored)
        if scored
        else 0.0,
        total_prompt_tokens=sum(score.prompt_tokens for score in scored),
        total_completion_tokens=sum(score.completion_tokens for score in scored),
        total_estimated_cost_usd=total_cost,
        cost_per_resolved_usd=total_cost / resolved if resolved else None,
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
            if artifact.is_file() and artifact.name != "manifest.json":
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
        "dataset_fingerprint": report.dataset_fingerprint,
        "total": report.total,
        "resolved": report.resolved,
        "pass_at_1": report.pass_at_1,
        "verification_pass_rate": report.verification_pass_rate,
        "cost_usd": report.total_estimated_cost_usd,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run reproducible sandbox coding-agent evaluations")
    parser.add_argument("dataset", type=Path, help="AgentForge coding-task JSONL")
    parser.add_argument("--repository-root", type=Path, default=Path("repositories"))
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument(
        "--workflow",
        choices=["verified_pr", "single_agent", "repair_tournament"],
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
    parser.add_argument("--dataset-name", default="local-code-tasks")
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fail-under", type=float, default=0.0)
    parser.add_argument(
        "--context-strategy",
        choices=["lexical", "lexical_graph", "hybrid", "hybrid_rerank"],
        default="hybrid_rerank",
        help="Code-context retrieval strategy for verified_pr runs",
    )
    args = parser.parse_args()
    tournament_policy = None
    if args.tournament_policy:
        if args.workflow != "repair_tournament":
            parser.error("--tournament-policy requires --workflow repair_tournament")
        tournament_policy = RepairTournamentPolicy.model_validate_json(args.tournament_policy.read_text(encoding="utf-8"))
    cases = load_code_cases(args.dataset)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if args.validate_only:
        print(
            json.dumps(
                {"cases": len(cases), "dataset_fingerprint": dataset_fingerprint(cases)},
                sort_keys=True,
            )
        )
        return
    models = args.models or ["gpt-4o-mini"]
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
    )
    reports = []
    for model in models:
        report = asyncio.run(
            run_benchmark(
                cases,
                repository_root=args.repository_root,
                default_model=model,
                policy=policy,
                dataset_name=args.dataset_name,
                dataset_path=str(args.dataset),
                output_dir=args.output_dir,
                input_cost_per_million=max(0.0, args.input_cost_per_million),
                output_cost_per_million=max(0.0, args.output_cost_per_million),
                image_by_repository=image_map,
                workflow=args.workflow,
                context_strategy=args.context_strategy,
                tournament_policy=tournament_policy,
            )
        )
        reports.append(report)
        print(json.dumps(_summary(report), sort_keys=True))
    if any(report.pass_at_1 < args.fail_under for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
