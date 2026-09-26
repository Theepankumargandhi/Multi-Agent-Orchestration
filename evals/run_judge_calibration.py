"""Run position-bias and human-agreement checks for an LLM-as-judge."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from evals.calibration import PairwiseCase, build_structured_judge, calibrate_pairwise_judge


def _load_cases(path: Path) -> list[PairwiseCase]:
    cases = [
        PairwiseCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("pairwise dataset is empty")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a structured pairwise LLM judge.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--provider", choices=["openai", "groq"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/evaluations/judge_calibration.json"))
    parser.add_argument("--minimum-reviewed", type=int, default=20)
    parser.add_argument("--minimum-accuracy", type=float, default=0.8)
    parser.add_argument("--minimum-position-consistency", type=float, default=0.9)
    args = parser.parse_args()

    cases = _load_cases(args.dataset)
    report = asyncio.run(
        calibrate_pairwise_judge(build_structured_judge(args.provider, args.model), cases)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(report.model_dump_json(indent=2))
    enough_labels = report.human_reviewed_cases >= args.minimum_reviewed
    accurate = report.accuracy is not None and report.accuracy >= args.minimum_accuracy
    consistent = report.position_consistency >= args.minimum_position_consistency
    return 0 if enough_labels and accurate and consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
