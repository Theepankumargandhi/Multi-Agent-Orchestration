"""Docker-backed weak-patch ablation with authored probes; never a live-model result."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from code_agent.models import CodeTask, RegressionChallengePolicy, RegressionProbeEvidence
from code_agent.regression_challenges import (
    BehavioralProbe,
    BehavioralProbeSuite,
    baseline_is_discriminating,
    blocking_safely,
    candidate_matches,
    execute_suite,
    fingerprint,
    validate_suite,
)
from code_agent.sandbox import DockerSandbox, DockerUnavailableError
from code_agent.verification import repository_fingerprint

FIXTURE = Path(__file__).parent / "fixtures" / "regression_challenges"
VARIANTS = {
    "baseline": None,
    "weak_patch": "def clamp(value, lower=0, upper=10):\n    if value == -1:\n        return lower\n    return min(value, upper)\n",
    "full_fix": "def clamp(value, lower=0, upper=10):\n    return max(lower, min(value, upper))\n",
}


def authored_suite() -> BehavioralProbeSuite:
    return BehavioralProbeSuite(probes=[
        BehavioralProbe(target="app.clamp", args=[-1], expected=0, rationale="Just below the lower bound."),
        BehavioralProbe(target="app.clamp", args=[-100], expected=0, rationale="Far below the lower bound; detect example overfit."),
        BehavioralProbe(target="app.clamp", args=[12], expected=10, rationale="Preserve the upper-bound behavior."),
        BehavioralProbe(target="app.clamp", args=[-1], second_args=[-100], relation="same_result",
                        rationale="Two below-bound inputs must clamp to the same lower value."),
    ])


async def run_controls(*, sandbox_factory=DockerSandbox, execution_mode="docker") -> dict:
    if execution_mode not in {"docker", "authored_test_double"}:
        raise ValueError("invalid execution mode")
    if sandbox_factory is not DockerSandbox and execution_mode != "authored_test_double":
        raise ValueError("test doubles must disclose authored execution")
    task = CodeTask(repository="regression_challenges", model="no-provider",
                    issue="Clamp numeric values to both inclusive bounds and preserve the existing contract.")
    source_hash = repository_fingerprint(FIXTURE, task.policy.max_files, task.policy.max_repository_bytes)
    suite = authored_suite()
    validate_suite(suite, RegressionChallengePolicy(allowed_targets=["app.clamp"]))
    rows = []
    for name, patch in VARIANTS.items():
        sandbox = sandbox_factory(FIXTURE, task.policy)
        try:
            await blocking_safely(sandbox.start)
            if patch is not None:
                await blocking_safely(lambda: sandbox.workspace.write_file("app.py", patch))
            owner = await blocking_safely(lambda: sandbox.run(task.test_command))
            evidence = await execute_suite(task, sandbox, suite)
            rows.append({"variant": name, "owner_tests_passed": owner.exit_code == 0 and not owner.timed_out,
                         "probes_matched": candidate_matches(evidence), "evidence": evidence.model_dump(mode="json")})
        finally:
            await blocking_safely(sandbox.close)
    unchanged = source_hash == repository_fingerprint(FIXTURE, task.policy.max_files, task.policy.max_repository_bytes)
    report = {"schema_version": "1.0", "kind": "authored_regression_probe_ablation", "execution_mode": execution_mode,
        "decision": "held", "production_activation": False, "model_calls": 0,
        "suite": suite.model_dump(mode="json"), "suite_sha256": fingerprint(suite.model_dump(mode="json")),
        "source_sha256": source_hash, "source_unchanged": unchanged, "arms": rows,
        "controls_passed": unchanged and all(row["owner_tests_passed"] for row in rows)
            and baseline_is_discriminating(RegressionProbeEvidence.model_validate(rows[0]["evidence"]))
            and [row["probes_matched"] for row in rows] == [False, False, True],
        "limitations": ["Authored probes and patches on one toy fixture; no generated oracle or provider quality measured.",
                        "Passing probes does not prove semantic correctness; model expectations require owner review.",
                        "No production activation, live-model resolution improvement, or generalized overfitting claim."]}
    report["fingerprint"] = fingerprint(report)
    return report


def markdown(report: dict) -> str:
    lines = ["# Regression probe ablation", "", f"Execution: {report['execution_mode']}; controls passed: {report['controls_passed']}.",
             "Decision: held. No provider calls or activation.", "", "| Arm | Existing tests | All behavioral probes |",
             "|---|---|---|"]
    lines += [f"| {row['variant']} | {row['owner_tests_passed']} | {row['probes_matched']} |" for row in report["arms"]]
    return "\n".join(lines + ["", "## Limits", ""] + [f"- {item}" for item in report["limitations"]] + [""])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/evaluations/regression-challenges/controls.json"))
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
        print("Regression ablation unavailable or invalid: Docker/image and fresh protected output paths are required.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
