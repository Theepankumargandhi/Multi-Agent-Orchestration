"""Private-reference and mutation-sensitivity controls; no provider quality or activation claim."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from code_agent.models import CodeTask, OracleCalibrationPolicy
from code_agent.oracle_calibration import calibrate_suite, reference_fingerprint, verify_calibration_report
from code_agent.regression_challenges import BehavioralProbeSuite, fingerprint
from code_agent.sandbox import DockerSandbox, DockerUnavailableError
from evals.regression_challenge_evaluation import FIXTURE, authored_suite

REFERENCE = Path(__file__).parent / "fixtures" / "oracle_reference"


async def run_controls(*, sandbox_factory=DockerSandbox, execution_mode="docker") -> dict:
    if execution_mode not in {"docker", "authored_test_double"} or (sandbox_factory is not DockerSandbox and execution_mode != "authored_test_double"):
        raise ValueError("test doubles must disclose authored execution")
    policy = OracleCalibrationPolicy(reference_sha256=reference_fingerprint(REFERENCE))
    task = CodeTask(repository="regression_challenges", model="no-provider", issue="Preserve inclusive upper and lower clamp bounds.")
    suites = {
        "strong_suite": authored_suite(),
        "weak_suite": BehavioralProbeSuite(probes=[authored_suite().probes[0]]),
        "wrong_oracle": BehavioralProbeSuite(probes=[authored_suite().probes[0].model_copy(update={"expected": 99})]),
    }
    arms = []
    for name, suite in suites.items():
        receipt = await calibrate_suite(task, FIXTURE.resolve(), suite, policy, REFERENCE.resolve(), sandbox_factory=sandbox_factory)
        if any("DockerUnavailableError" in reason for reason in receipt.blocking_reasons):
            raise DockerUnavailableError("reference/mutation execution is unavailable")
        arms.append({"name": name, "receipt_valid": verify_calibration_report(receipt), "receipt": receipt.model_dump(mode="json")})
    receipts = [row["receipt"] for row in arms]
    report = {"schema_version": "1.0", "kind": "authored_oracle_and_mutation_controls", "execution_mode": execution_mode,
        "decision": "held", "production_activation": False, "model_calls": 0, "arms": arms,
        "controls_passed": all(row["receipt_valid"] and row["receipt"]["reference_unchanged"] for row in arms)
            and [receipt["eligible"] for receipt in receipts] == [True, False, False]
            and receipts[0]["mutation_score"] == 1 and receipts[1]["mutation_score"] == 0.5
            and "probe_expectations_disagree_with_reference" in receipts[2]["blocking_reasons"],
        "limitations": ["One authored fixture/reference with four bounded AST faults; not a generated-test or live-model accuracy result.",
                        "Reference agreement depends on the operator reference being correct; surviving mutants may be equivalent.",
                        "Greedy probe cover is diagnostic only; no suite reduction, source release, or serving activation."]}
    report["fingerprint"] = fingerprint(report)
    return report


def markdown(report: dict) -> str:
    lines = ["# Oracle calibration and mutation sensitivity", "", f"Execution: {report['execution_mode']}; controls passed: {report['controls_passed']}.",
             "Decision: held. No provider calls or activation.", "", "| Suite | Reference agrees | Killed / generated | Mutation score | Eligible |",
             "|---|---|---|---|---|"]
    for row in report["arms"]:
        receipt = row["receipt"]
        reference = receipt["reference_execution"]
        agrees = bool(reference and all(status == "matched" for status in reference["statuses"]))
        lines.append(f"| {row['name']} | {agrees} | {receipt['killed_mutants']} / {len(receipt['mutants'])} | {receipt['mutation_score']:.2f} | {receipt['eligible']} |")
    return "\n".join(lines + ["", "## Limits", ""] + [f"- {item}" for item in report["limitations"]] + [""])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/evaluations/oracle-calibration/controls.json"))
    parser.add_argument("--require-gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        target, root = args.output.resolve(), Path(__file__).resolve().parents[1]
        card = target.with_suffix(".md")
        if target == card or any(path.exists() or (path.is_relative_to(root)
                and path.relative_to(root).parts[:2] != ("data", "evaluations")) for path in (target, card)):
            raise ValueError("fresh protected output path required")
        report = asyncio.run(run_controls())
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with card.open("x", encoding="utf-8") as handle:
            handle.write(markdown(report))
        print(json.dumps({"controls_passed": report["controls_passed"], "decision": "held", "fingerprint": report["fingerprint"]}))
        return 1 if args.require_gate or not report["controls_passed"] else 0
    except (OSError, ValueError, DockerUnavailableError):
        print("Oracle calibration unavailable or invalid: Docker/image and fresh protected paths are required.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
