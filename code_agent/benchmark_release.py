"""Frozen experiment matrices and paired benchmark evidence releases."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from code_agent.evaluation import dataset_fingerprint, load_code_cases, run_benchmark
from code_agent.evaluation_models import CodeBenchmarkCase, CodeBenchmarkReport
from code_agent.models import SandboxPolicy

ContextStrategy = Literal["lexical", "lexical_graph", "hybrid", "hybrid_rerank"]


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


class DatasetLock(BaseModel):
    schema_version: str = "1.0"
    dataset_name: str
    dataset_path: str
    dataset_fingerprint: str
    case_ids: list[str]
    total: int
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    frozen_at: str


class BenchmarkVariant(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._-]+$")
    model: str = Field(min_length=1, max_length=100)
    workflow: Literal["single_agent", "verified_pr"] = "verified_pr"
    context_strategy: ContextStrategy = "hybrid_rerank"
    input_cost_per_million: float = Field(default=0.0, ge=0)
    output_cost_per_million: float = Field(default=0.0, ge=0)


class BenchmarkMatrix(BaseModel):
    schema_version: str = "1.0"
    name: str = Field(min_length=1, max_length=200)
    dataset: str
    dataset_name: str
    dataset_lock: str
    repository_root: str
    output_dir: str = "data/evaluations/benchmark-release"
    image: str = "agentforge-code-sandbox:local"
    image_map: str = ""
    max_cases: int = Field(default=50, ge=1, le=500)
    score_source: Literal["local_harness"] = "local_harness"
    variants: list[BenchmarkVariant] = Field(min_length=2, max_length=30)

    @model_validator(mode="after")
    def unique_variant_names(self) -> "BenchmarkMatrix":
        names = [item.name for item in self.variants]
        if len(names) != len(set(names)):
            raise ValueError("benchmark variant names must be unique")
        return self


class VariantEvidence(BaseModel):
    name: str
    rank: int = Field(ge=1)
    pareto_efficient: bool
    report_path: str
    report_sha256: str
    run_id: str
    model: str
    workflow: str
    context_strategy: str
    resolved: int
    total: int
    pass_at_1: float
    pass_at_1_confidence_interval: list[float]
    verification_pass_rate: float
    telemetry_coverage_rate: float
    total_tokens: int
    total_cost_usd: float
    cost_per_resolved_usd: float | None
    p50_duration_ms: float
    p95_duration_ms: float
    failure_categories: dict[str, int]


class PairedComparison(BaseModel):
    baseline: str
    candidate: str
    sample_size: int
    both_resolved: int
    neither_resolved: int
    baseline_only: int
    candidate_only: int
    pass_at_1_delta: float
    paired_bootstrap_95_ci: list[float]
    exact_mcnemar_p_value: float
    total_cost_delta_usd: float
    p95_duration_delta_ms: float


class BenchmarkRelease(BaseModel):
    schema_version: str = "1.0"
    release_id: str
    title: str
    generated_at: str
    score_source: Literal["local_harness"] = "local_harness"
    externally_verified: bool = False
    dataset_name: str
    dataset_fingerprint: str
    dataset_lock_sha256: str
    case_ids: list[str]
    total_cases: int
    baseline: str
    winner: str
    matrix_fingerprint: str
    variants: list[VariantEvidence]
    comparisons: list[PairedComparison]


def create_dataset_lock(
    cases: list[CodeBenchmarkCase],
    *,
    dataset_name: str,
    dataset_path: str,
) -> DatasetLock:
    split_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for case in cases:
        split_counts[case.split] = split_counts.get(case.split, 0) + 1
        source_counts[case.source] = source_counts.get(case.source, 0) + 1
    return DatasetLock(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint(cases),
        case_ids=[case.id for case in cases],
        total=len(cases),
        split_counts=split_counts,
        source_counts=source_counts,
        frozen_at=datetime.now(UTC).isoformat(),
    )


def load_matrix(path: Path) -> BenchmarkMatrix:
    return BenchmarkMatrix.model_validate_json(path.read_text(encoding="utf-8"))


def _resolved_by_case(report: CodeBenchmarkReport) -> dict[str, bool]:
    return {
        outcome.case_id: bool(outcome.score and outcome.score.resolved)
        for outcome in report.outcomes
    }


def _paired_bootstrap(deltas: list[float], seed: str, samples: int = 5000) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    if len(deltas) == 1:
        return [deltas[0], deltas[0]]
    rng = random.Random(int(seed[:16], 16))
    means = [statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(samples)]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _exact_mcnemar(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, candidate_only)
    probability = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * probability)


def paired_comparison(
    baseline_name: str,
    baseline: CodeBenchmarkReport,
    candidate_name: str,
    candidate: CodeBenchmarkReport,
) -> PairedComparison:
    left = _resolved_by_case(baseline)
    right = _resolved_by_case(candidate)
    if set(left) != set(right):
        raise ValueError("paired benchmark reports must contain identical case IDs")
    pairs = [(left[case_id], right[case_id]) for case_id in sorted(left)]
    both = sum(a and b for a, b in pairs)
    neither = sum(not a and not b for a, b in pairs)
    baseline_only = sum(a and not b for a, b in pairs)
    candidate_only = sum(not a and b for a, b in pairs)
    deltas = [float(b) - float(a) for a, b in pairs]
    seed = _canonical_hash(
        [baseline.config_fingerprint, candidate.config_fingerprint, sorted(left)]
    )
    return PairedComparison(
        baseline=baseline_name,
        candidate=candidate_name,
        sample_size=len(pairs),
        both_resolved=both,
        neither_resolved=neither,
        baseline_only=baseline_only,
        candidate_only=candidate_only,
        pass_at_1_delta=statistics.fmean(deltas) if deltas else 0.0,
        paired_bootstrap_95_ci=_paired_bootstrap(deltas, seed),
        exact_mcnemar_p_value=_exact_mcnemar(baseline_only, candidate_only),
        total_cost_delta_usd=(
            candidate.total_estimated_cost_usd - baseline.total_estimated_cost_usd
        ),
        p95_duration_delta_ms=candidate.p95_duration_ms - baseline.p95_duration_ms,
    )


def _pareto_names(named_reports: list[tuple[str, Path, CodeBenchmarkReport]]) -> set[str]:
    efficient: set[str] = set()
    for name, _, report in named_reports:
        dominated = False
        for other_name, _, other in named_reports:
            if other_name == name:
                continue
            no_worse = (
                other.pass_at_1 >= report.pass_at_1
                and other.total_estimated_cost_usd <= report.total_estimated_cost_usd
                and other.p95_duration_ms <= report.p95_duration_ms
            )
            strictly_better = (
                other.pass_at_1 > report.pass_at_1
                or other.total_estimated_cost_usd < report.total_estimated_cost_usd
                or other.p95_duration_ms < report.p95_duration_ms
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            efficient.add(name)
    return efficient


def build_release(
    named_reports: list[tuple[str, Path, CodeBenchmarkReport]],
    *,
    title: str,
    baseline_name: str | None = None,
    dataset_lock_sha256: str = "",
) -> BenchmarkRelease:
    if len(named_reports) < 2:
        raise ValueError("a benchmark release requires at least two reports")
    names = [name for name, _, _ in named_reports]
    if len(names) != len(set(names)):
        raise ValueError("benchmark report labels must be unique")
    baseline_name = baseline_name or names[0]
    if baseline_name not in names:
        raise ValueError("baseline label does not match a report")
    fingerprints = {report.dataset_fingerprint for _, _, report in named_reports}
    if len(fingerprints) != 1:
        raise ValueError("benchmark reports use different dataset fingerprints")
    case_sets = [set(_resolved_by_case(report)) for _, _, report in named_reports]
    if any(case_ids != case_sets[0] for case_ids in case_sets[1:]):
        raise ValueError("benchmark reports do not contain identical case IDs")
    totals = {report.total for _, _, report in named_reports}
    if totals != {len(case_sets[0])}:
        raise ValueError("benchmark report totals do not match their case outcomes")

    ranked = sorted(
        named_reports,
        key=lambda item: (
            -item[2].pass_at_1,
            item[2].total_estimated_cost_usd,
            item[2].p95_duration_ms,
            item[0],
        ),
    )
    ranks = {name: rank for rank, (name, _, _) in enumerate(ranked, start=1)}
    pareto = _pareto_names(named_reports)
    variants = [
        VariantEvidence(
            name=name,
            rank=ranks[name],
            pareto_efficient=name in pareto,
            report_path=path.as_posix(),
            report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            run_id=report.run_id,
            model=report.model,
            workflow=report.workflow,
            context_strategy=report.context_strategy,
            resolved=report.resolved,
            total=report.total,
            pass_at_1=report.pass_at_1,
            pass_at_1_confidence_interval=report.pass_at_1_confidence_interval,
            verification_pass_rate=report.verification_pass_rate,
            telemetry_coverage_rate=report.telemetry_coverage_rate,
            total_tokens=report.total_prompt_tokens + report.total_completion_tokens,
            total_cost_usd=report.total_estimated_cost_usd,
            cost_per_resolved_usd=report.cost_per_resolved_usd,
            p50_duration_ms=report.p50_duration_ms,
            p95_duration_ms=report.p95_duration_ms,
            failure_categories=report.failure_categories,
        )
        for name, path, report in named_reports
    ]
    report_by_name = {name: report for name, _, report in named_reports}
    baseline_report = report_by_name[baseline_name]
    comparisons = [
        paired_comparison(baseline_name, baseline_report, name, report)
        for name, _, report in named_reports
        if name != baseline_name
    ]
    matrix_payload = [
        {
            "name": name,
            "dataset": report.dataset_fingerprint,
            "config": report.config_fingerprint,
            "report_sha256": evidence.report_sha256,
        }
        for (name, _, report), evidence in zip(named_reports, variants, strict=True)
    ]
    generated = datetime.now(UTC)
    generated_at = generated.isoformat()
    return BenchmarkRelease(
        release_id=(
            f"benchmark-{generated.strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{_canonical_hash(matrix_payload)[:8]}"
        ),
        title=title,
        generated_at=generated_at,
        score_source="local_harness",
        externally_verified=False,
        dataset_name=named_reports[0][2].dataset_name,
        dataset_fingerprint=next(iter(fingerprints)),
        dataset_lock_sha256=dataset_lock_sha256,
        case_ids=sorted(case_sets[0]),
        total_cases=len(case_sets[0]),
        baseline=baseline_name,
        winner=ranked[0][0],
        matrix_fingerprint=_canonical_hash(matrix_payload),
        variants=sorted(variants, key=lambda item: item.rank),
        comparisons=comparisons,
    )


def benchmark_card(release: BenchmarkRelease) -> str:
    source_label = (
        "official SWE-bench evaluator" if release.externally_verified else "AgentForge local harness"
    )
    lines = [
        f"# {release.title}",
        "",
        f"- Release: `{release.release_id}`",
        f"- Score source: **{source_label}**",
        f"- Dataset: `{release.dataset_name}` ({release.total_cases} cases)",
        f"- Dataset fingerprint: `{release.dataset_fingerprint}`",
        f"- Matrix fingerprint: `{release.matrix_fingerprint}`",
        f"- Winner under the declared ordering: **{release.winner}**",
        "",
    ]
    if not release.externally_verified:
        lines.extend(
            [
                "> These are local harness results, not an official SWE-bench leaderboard score.",
                "",
            ]
        )
    lines.extend(
        [
            "## Variant results",
            "",
            "| Rank | Variant | Model | Workflow | Context | Resolved | Pass@1 | 95% CI | Cost | P95 | Pareto |",
            "|---:|---|---|---|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for item in release.variants:
        low, high = item.pass_at_1_confidence_interval
        lines.append(
            f"| {item.rank} | {item.name} | {item.model} | {item.workflow} | "
            f"{item.context_strategy} | {item.resolved}/{item.total} | {item.pass_at_1:.1%} | "
            f"{low:.1%}–{high:.1%} | ${item.total_cost_usd:.4f} | "
            f"{item.p95_duration_ms / 1000:.2f}s | {'yes' if item.pareto_efficient else 'no'} |"
        )
    lines.extend(["", "## Paired comparisons against baseline", ""])
    if not release.comparisons:
        lines.append("No candidate comparisons were available.")
    for item in release.comparisons:
        low, high = item.paired_bootstrap_95_ci
        lines.extend(
            [
                f"### {item.candidate}",
                "",
                f"- Pass@1 delta: {item.pass_at_1_delta:+.1%} (paired bootstrap 95% CI {low:+.1%} to {high:+.1%})",
                f"- Candidate-only wins: {item.candidate_only}; baseline-only wins: {item.baseline_only}",
                f"- Exact McNemar p-value: {item.exact_mcnemar_p_value:.4f}",
                f"- Cost delta: ${item.total_cost_delta_usd:+.4f}; P95 latency delta: {item.p95_duration_delta_ms / 1000:+.2f}s",
                "",
            ]
        )
    lines.extend(
        [
            "## Reproduction and reporting constraints",
            "",
            "Compare only runs with the recorded dataset fingerprint and exact case IDs. "
            "Infrastructure failures remain in the denominator. Pricing is supplied at run time. "
            "Use the official SWE-bench evaluator before claiming an official leaderboard result.",
            "",
        ]
    )
    return "\n".join(lines)


def write_release(release: BenchmarkRelease, output_root: Path) -> Path:
    destination = output_root.resolve() / release.release_id
    destination.mkdir(parents=True, exist_ok=False)
    release_path = destination / "release.json"
    card_path = destination / "benchmark-card.md"
    release_path.write_text(release.model_dump_json(indent=2) + "\n", encoding="utf-8")
    card_path.write_text(benchmark_card(release), encoding="utf-8")
    artifacts = {}
    for path in (release_path, card_path):
        artifacts[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": "1.0",
        "release_id": release.release_id,
        "matrix_fingerprint": release.matrix_fingerprint,
        "artifacts": artifacts,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


async def execute_matrix(path: Path) -> tuple[BenchmarkRelease, Path]:
    matrix = load_matrix(path)
    base = path.resolve().parent
    dataset_path = _resolve(base, matrix.dataset)
    lock_path = _resolve(base, matrix.dataset_lock)
    repository_root = _resolve(base, matrix.repository_root)
    output_root = _resolve(base, matrix.output_dir)
    cases = load_code_cases(dataset_path)[: matrix.max_cases]
    lock = DatasetLock.model_validate_json(lock_path.read_text(encoding="utf-8"))
    if dataset_fingerprint(cases) != lock.dataset_fingerprint:
        raise ValueError("dataset does not match the frozen dataset lock")
    if [case.id for case in cases] != lock.case_ids:
        raise ValueError("dataset case order or selection differs from the frozen lock")
    image_map: dict[str, str] = {}
    if matrix.image_map:
        loaded = json.loads(_resolve(base, matrix.image_map).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in loaded.items()
        ):
            raise ValueError("image map must contain string repository-to-image entries")
        image_map = loaded
    policy = SandboxPolicy(
        image=matrix.image,
        allowed_images={matrix.image, *image_map.values()},
        network_enabled=False,
    )
    run_root = output_root / "runs"
    named_reports: list[tuple[str, Path, CodeBenchmarkReport]] = []
    for variant in matrix.variants:
        report = await run_benchmark(
            cases,
            repository_root=repository_root,
            default_model=variant.model,
            policy=policy,
            dataset_name=matrix.dataset_name,
            dataset_path=dataset_path.as_posix(),
            output_dir=run_root,
            input_cost_per_million=variant.input_cost_per_million,
            output_cost_per_million=variant.output_cost_per_million,
            image_by_repository=image_map,
            workflow=variant.workflow,
            context_strategy=variant.context_strategy,
        )
        named_reports.append((variant.name, run_root / report.run_id / "report.json", report))
    release = build_release(
        named_reports,
        title=matrix.name,
        baseline_name=matrix.variants[0].name,
        dataset_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    )
    destination = write_release(release, output_root / "releases")
    return release, destination


def _load_named_reports(values: list[str]) -> list[tuple[str, Path, CodeBenchmarkReport]]:
    loaded = []
    for value in values:
        if "=" not in value:
            raise ValueError("reports must be supplied as LABEL=PATH")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        loaded.append(
            (label, path, CodeBenchmarkReport.model_validate_json(path.read_text(encoding="utf-8")))
        )
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Create statistically paired coding benchmark releases")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lock_parser = subparsers.add_parser("lock", help="Freeze an exact benchmark selection")
    lock_parser.add_argument("dataset", type=Path)
    lock_parser.add_argument("output", type=Path)
    lock_parser.add_argument("--name", required=True)
    lock_parser.add_argument("--max-cases", type=int, default=0)

    validate_parser = subparsers.add_parser("validate", help="Validate a frozen matrix")
    validate_parser.add_argument("matrix", type=Path)

    run_parser = subparsers.add_parser("run", help="Execute a frozen experiment matrix")
    run_parser.add_argument("matrix", type=Path)

    release_parser = subparsers.add_parser("release", help="Compare existing run reports")
    release_parser.add_argument("reports", nargs="+")
    release_parser.add_argument("--title", required=True)
    release_parser.add_argument("--baseline")
    release_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "lock":
        cases = load_code_cases(args.dataset)
        if args.max_cases > 0:
            cases = cases[: args.max_cases]
        lock = create_dataset_lock(
            cases,
            dataset_name=args.name,
            dataset_path=args.dataset.as_posix(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(lock.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"cases": lock.total, "dataset_fingerprint": lock.dataset_fingerprint}))
    elif args.command == "validate":
        matrix = load_matrix(args.matrix)
        base = args.matrix.resolve().parent
        cases = load_code_cases(_resolve(base, matrix.dataset))[: matrix.max_cases]
        lock_path = _resolve(base, matrix.dataset_lock)
        lock = DatasetLock.model_validate_json(lock_path.read_text(encoding="utf-8"))
        valid = (
            dataset_fingerprint(cases) == lock.dataset_fingerprint
            and [case.id for case in cases] == lock.case_ids
        )
        print(json.dumps({"matrix": matrix.name, "variants": len(matrix.variants), "cases": len(cases), "frozen": valid}))
        if not valid:
            raise SystemExit(1)
    elif args.command == "run":
        release, destination = asyncio.run(execute_matrix(args.matrix))
        print(json.dumps({"release_id": release.release_id, "winner": release.winner, "path": str(destination)}))
    else:
        reports = _load_named_reports(args.reports)
        release = build_release(
            reports,
            title=args.title,
            baseline_name=args.baseline,
        )
        destination = write_release(release, args.output_dir)
        print(json.dumps({"release_id": release.release_id, "winner": release.winner, "path": str(destination)}))


if __name__ == "__main__":
    main()
