"""CLI entry point for reproducible agent experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evals.platform import ExperimentStore, load_config, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and compare agent configurations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("experiments") / "routing_ablation.json",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("data/evaluations"))
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--no-store", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    report = asyncio.run(run_experiment(config, base_dir=Path.cwd()))
    if not args.no_store:
        output = ExperimentStore(args.results_dir).save(report)
    else:
        output = None

    summary = {
        "experiment_id": report.experiment_id,
        "experiment": report.experiment_name,
        "winner": report.winner,
        "dataset_fingerprint": report.dataset_fingerprint,
        "report_path": str(output) if output else None,
        "variants": [
            {
                "name": item.variant.name,
                "quality_score": round(item.quality_score, 4),
                "pass_rate": round(item.pass_rate, 4),
                "p50_latency_ms": round(item.p50_latency_ms, 3),
                "p95_latency_ms": round(item.p95_latency_ms, 3),
                "total_cost_usd": round(item.total_cost_usd, 6),
                "pareto_optimal": item.pareto_optimal,
                "failures": item.failure_categories,
            }
            for item in report.reports
        ],
    }
    print(json.dumps(summary, indent=2))
    threshold = args.min_score if args.min_score is not None else config.pass_threshold
    winner = next(item for item in report.reports if item.variant.name == report.winner)
    return 0 if winner.quality_score >= threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
