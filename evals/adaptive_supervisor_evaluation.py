"""Three-way authored controls; not a live-model performance benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from code_agent.models import AdaptiveDispatchPolicy, CodeTask, RepairTournamentPolicy, SandboxPolicy
from code_agent.repair_tournament import SharedModelBudget, digest, verify_tournament_report
from evals.repair_tournament_evaluation import FIXTURE, SCENARIOS, AuthoredSandbox, solver

EXPECTED = {
    "focused_fix": (True, True, True),
    "wrong_first_fix": (False, True, True),
    "unsafe_first_fix": (False, True, False),
    "first_provider_outage": (False, True, False),
    "challenge_veto": (True, False, False),
    "deadline": (False, False, False),
    "shared_call_budget": (False, False, False),
}


async def run_controls() -> dict:
    task = CodeTask(repository="fixture", issue="Change VALUE from one to two and preserve the existing regression test.",
                    model="authored-control", policy=SandboxPolicy(max_repair_rounds=0))
    controls = []
    for scenario in SCENARIOS:
        arms = {}
        for index, name in enumerate(("verified_pr", "repair_tournament", "adaptive_supervisor")):
            template = solver(scenario, 3 if index else 1)
            if index == 0:
                budget = SharedModelBudget(template.policy)
                sandbox = AuthoredSandbox(FIXTURE, task.policy)
                sandbox.start()
                try:
                    result = await asyncio.wait_for(template.candidate_factory("candidate-1", "", budget).solve(task, sandbox),
                                                    timeout=template.policy.candidate_timeout_seconds)
                    passed = result.status == "completed"
                except TimeoutError:
                    result, passed = None, False
                finally:
                    sandbox.close()
                calls, valid = budget.calls, True
                receipt = None
            else:
                if index == 2:
                    values = template.policy.model_dump()
                    values.update(max_parallel=1, max_model_calls=2 if scenario == "shared_call_budget" else 64,
                                  adaptive_dispatch=AdaptiveDispatchPolicy(verification_seconds=0.01 if scenario == "deadline" else 1))
                    template.policy = RepairTournamentPolicy.model_validate(values)
                result = await template.solve(task, FIXTURE)
                receipt = result.tournament
                passed, calls, valid = result.status == "completed", result.model_calls, verify_tournament_report(receipt)
            arms[name] = {"completed": passed, "expected_completed": EXPECTED[scenario][index],
                "control_passed": passed == EXPECTED[scenario][index] and valid, "receipt_valid": valid,
                "logical_model_calls": calls, "usage_accounting_complete": False, "cost_per_resolved_usd": None,
                "stop_reason": receipt.stop_reason if receipt else "not_applicable",
                "dispatched_perspectives": [decision.perspective for decision in receipt.dispatch_decisions] if receipt else [],
                "candidate_diagnoses": [row.diagnosis.category for row in receipt.candidates if row.diagnosis] if receipt else []}
        controls.append({"scenario": scenario, "arms": arms})
    report = {"schema_version": "1.0", "kind": "authored_adaptive_dispatch_controls", "decision": "held",
              "production_activation": False, "controls": controls,
              "controls_passed": all(arm["control_passed"] for row in controls for arm in row["arms"].values()),
              "limitations": ["Scripted models and authored command outcomes; no Docker, provider, or fixture-code host execution.",
                              "Control outcomes and logical calls do not establish real-world resolution, cost, or latency gains.",
                              "Safety and provider holds deliberately stop adaptive escalation; this differs from fixed tournaments.",
                              "The verified baseline has no challenger and uses its own iteration limits; per-team resources differ."]}
    report["report_fingerprint"] = digest(report)
    return report


def markdown(report: dict) -> str:
    lines = ["# Adaptive coding supervisor controls", "", f"Controls passed: {report['controls_passed']}",
             "Decision: held. No activation or live-model quality claim.", "",
             "| Scenario | Verified PR / calls | Fixed tournament / calls | Adaptive / calls / stop |", "|---|---|---|---|"]
    for row in report["controls"]:
        a, b, c = (row["arms"][name] for name in ("verified_pr", "repair_tournament", "adaptive_supervisor"))
        lines.append(f"| {row['scenario']} | {a['completed']} / {a['logical_model_calls']} | "
                     f"{b['completed']} / {b['logical_model_calls']} | {c['completed']} / {c['logical_model_calls']} / {c['stop_reason']} |")
    return "\n".join(lines + ["", "## Limits", ""] + [f"- {limit}" for limit in report["limitations"]] + [""])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/evaluations/adaptive-supervisor/controls.json"))
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
        print(json.dumps({"controls_passed": report["controls_passed"], "decision": report["decision"]}))
        return 1 if args.require_gate or not report["controls_passed"] else 0
    except (OSError, ValueError):
        print("Adaptive supervisor controls failed: invalid input or protected output path.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
