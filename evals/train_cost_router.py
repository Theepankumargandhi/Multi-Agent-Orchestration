"""Train and evaluate the adaptive model router from a real experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.adaptive_router import (
    RoutingObservation,
    evaluate_router,
    train_router,
    tune_threshold,
)
from evals.platform import ExperimentReport


def observations_from_report(
    report: ExperimentReport,
    small_variant: str,
    strong_variant: str,
) -> list[RoutingObservation]:
    by_name = {item.variant.name: item for item in report.reports}
    if small_variant not in by_name or strong_variant not in by_name:
        raise ValueError("small and strong variants must both exist in the report")
    small_cases = {item.case_id: item for item in by_name[small_variant].cases}
    strong_cases = {item.case_id: item for item in by_name[strong_variant].cases}
    if set(small_cases) != set(strong_cases):
        raise ValueError("variants were not evaluated on identical case IDs")
    observations = []
    for case_id, small in small_cases.items():
        strong = strong_cases[case_id]
        observations.append(
            RoutingObservation(
                id=case_id,
                query=small.input,
                small_success=small.passed,
                strong_success=strong.passed,
                small_cost_usd=small.actual.estimated_cost_usd,
                strong_cost_usd=strong.actual.estimated_cost_usd,
                split=small.split,
                high_risk=bool({"safety", "adversarial"} & set(small.tags)),
            )
        )
    if not any(item.strong_cost_usd > item.small_cost_usd for item in observations):
        raise ValueError(
            "report has no measurable strong-vs-small cost difference; configure per-million-token prices"
        )
    return observations


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit a cost-aware router from an experiment report.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--small-variant", required=True)
    parser.add_argument("--strong-variant", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/evaluations/cost_router.json"))
    parser.add_argument("--minimum-success", type=float, default=0.9)
    args = parser.parse_args()

    report = ExperimentReport.model_validate_json(args.report.read_text(encoding="utf-8"))
    observations = observations_from_report(report, args.small_variant, args.strong_variant)
    router = train_router(observations)
    validation = [item for item in observations if item.split == "validation"]
    test = [item for item in observations if item.split == "test"]
    if not validation or not test:
        raise ValueError("the report must contain both validation and test cases")
    validation_report = tune_threshold(router, validation, args.minimum_success)
    test_report = evaluate_router(router, test)
    router.save(args.output)
    payload = {
        "model_path": str(args.output),
        "training_fingerprint": router.training_fingerprint,
        "validation": validation_report.model_dump(),
        "held_out_test": test_report.model_dump(),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
