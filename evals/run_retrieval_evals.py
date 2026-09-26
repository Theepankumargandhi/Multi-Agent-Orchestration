"""Compare lexical, vector, hybrid, reranked, or graph retrieval rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.retrieval_metrics import RetrievalCase, evaluate_rankings


def _load_cases(path: Path) -> list[RetrievalCase]:
    cases = [
        RetrievalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("retrieval dataset is empty")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Score and compare saved retrieval rankings.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--ranking",
        action="append",
        required=True,
        help="Variant and JSON ranking file as name=path; JSON maps case IDs to ordered document IDs.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/evaluations/retrieval_report.json"))
    args = parser.parse_args()

    cases = _load_cases(args.dataset)
    reports = {}
    for specification in args.ranking:
        name, separator, raw_path = specification.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError("ranking arguments must use name=path")
        rankings = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        reports[name] = evaluate_rankings(cases, rankings, k=args.k).model_dump()
    baseline_name = next(iter(reports))
    baseline = reports[baseline_name]
    payload = {
        "dataset": str(args.dataset),
        "k": args.k,
        "baseline": baseline_name,
        "variants": {
            name: {
                **report,
                "delta_recall_at_k": report["recall_at_k"] - baseline["recall_at_k"],
                "delta_mrr": report["mrr"] - baseline["mrr"],
            }
            for name, report in reports.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
