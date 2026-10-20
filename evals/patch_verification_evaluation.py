"""Run the complete authored-control benchmark using actual Docker and no model calls."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from code_agent.patch_inputs import read_bounded, strict_json
from code_agent.patch_verify import reserve_private_directory, write_private_json
from code_agent.regression_benchmark import RegressionCorpus, freeze_corpus, run_benchmark

ROOT = Path(__file__).resolve().parent.parent


async def evaluate():
    corpus = RegressionCorpus.model_validate(strict_json(read_bounded(ROOT / "evals/datasets/patch_regression_controls.json", 40_000)))
    frozen = freeze_corpus(corpus, ROOT)
    report = await run_benchmark(frozen, ROOT)
    expected = {
        ("fee-regression", "tests_only"): "no_alarm", ("fee-regression", "contracts"): "no_alarm", ("fee-regression", "generated"): "alarm",
        ("discount-regression", "tests_only"): "no_alarm", ("discount-regression", "contracts"): "alarm", ("discount-regression", "generated"): "alarm",
    }
    for name in ("fee-correct", "discount-correct"):
        for mode in ("tests_only", "contracts", "generated"):
            expected[name, mode] = "no_alarm"
    checks = [{"case_id": row["case_id"], "mode": row["mode"], "expected": expected[row["case_id"], row["mode"]],
               "observed": row["decision"], "passed": row["decision"] == expected[row["case_id"], row["mode"]]}
              for row in report["rows"]]
    return {"evidence_kind": "authored_controls_actual_docker", "independently_reviewed_real_cases": 0,
            "real_maintainer_acceptances": 0, "live_provider_calls": 0, "total": len(checks),
            "passed": sum(row["passed"] for row in checks), "checks": checks, "freeze": frozen, "benchmark": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = reserve_private_directory(args.output_dir, "regression-benchmark")
    result = asyncio.run(evaluate())
    write_private_json(output / "controls.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"checks", "freeze", "benchmark"}}, indent=2))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
