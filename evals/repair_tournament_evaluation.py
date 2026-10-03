"""Credential-free orchestration controls, not a coding-model quality benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from code_agent.models import (
    AnalysisPlan,
    CodeAction,
    CodeTask,
    RepairTournamentPolicy,
    ReviewFinding,
    ReviewVerdict,
    SandboxCommandResult,
    SandboxPolicy,
)
from code_agent.repair_tournament import BudgetedModel, RepairTournament, digest, verify_tournament_report
from code_agent.verification import repository_fingerprint
from code_agent.verified_pr import VerifiedPRAgent
from code_agent.workspace import EphemeralWorkspace

FIXTURE = Path(__file__).parent / "fixtures" / "repair_tournament"
SCENARIOS = {
    "focused_fix": (True, True),
    "wrong_first_fix": (False, True),
    "unsafe_first_fix": (False, True),
    "first_provider_outage": (False, True),
    "challenge_veto": (False, False),
    "deadline": (False, False),
    "shared_call_budget": (False, False),
}


class AuthoredSandbox:
    """Use real isolated files, but never execute fixture code on the host."""

    def __init__(self, repository, policy):
        self.repository, self.policy = repository, policy
        self.workspace = EphemeralWorkspace(repository, policy)

    def start(self):
        self.workspace.prepare()
        return self

    def run(self, command):
        # This is an explicitly authored outcome oracle, not actual pytest/Ruff execution.
        approved_test = command == ["python", "-m", "pytest", "-q"]
        passed = not approved_test or self.workspace.read_file("app.py").strip() == "VALUE = 2"
        return SandboxCommandResult(command=command, exit_code=0 if passed else 1,
                                    stdout="authored command outcome (not an executed test)", duration_ms=0)

    def close(self):
        self.workspace.cleanup()


class AuthoredModel:
    def __init__(self, values, *, fail=False, delay=0):
        self.values, self.fail, self.delay = list(values), fail, delay
        self.position = 0

    async def ainvoke(self, messages, config=None):
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("authored provider outage")
        value = self.values[min(self.position, len(self.values) - 1)]
        self.position += 1
        return value


def solver(scenario: str, count: int) -> RepairTournament:
    def factory(candidate_id, instructions, budget):
        first = candidate_id == "candidate-1"
        content = "VALUE = 3\n" if scenario == "wrong_first_fix" and first else "VALUE = 2\n"
        if scenario == "unsafe_first_fix" and first:
            content = "import os\nVALUE = os.system('unsafe')\n"
        actions = AuthoredModel([CodeAction(kind="write", path="app.py", content=content),
                                CodeAction(kind="test"), CodeAction(kind="finish")],
                                fail=scenario == "first_provider_outage" and first,
                                delay=0.1 if scenario == "deadline" else 0)
        return VerifiedPRAgent(
            BudgetedModel(AuthoredModel([AnalysisPlan(summary="Make the focused value change and retain the owner test.")]),
                          budget, candidate_id, instructions),
            BudgetedModel(actions, budget, candidate_id, instructions),
            BudgetedModel(AuthoredModel([ReviewVerdict(approved=True, summary="Authored review control.")]), budget, candidate_id),
        )

    findings = [ReviewFinding(severity="high", category="testing", message="Authored missing-boundary control.")] if scenario == "challenge_veto" else []
    return RepairTournament(factory, AuthoredModel([ReviewVerdict(approved=True, summary="Authored challenge control.", findings=findings)]),
        policy=RepairTournamentPolicy(candidates=count, max_parallel=min(count, 2),
            max_model_calls=1 if scenario == "shared_call_budget" else 64,
            candidate_timeout_seconds=0.025 if scenario == "deadline" else 30), sandbox_factory=AuthoredSandbox)


async def run_controls() -> dict:
    task = CodeTask(repository="fixture", issue="Change VALUE from one to two and preserve the existing regression test.",
                    model="authored-control", policy=SandboxPolicy(max_repair_rounds=0))
    rows = []
    for scenario, expected in SCENARIOS.items():
        arms = {}
        for name, count, expected_success in zip(("one_team", "three_teams"), (1, 3), expected, strict=True):
            result = await solver(scenario, count).solve(task, FIXTURE)
            receipt = result.tournament
            passed = result.status == "completed"
            arms[name] = {"status": result.status, "winner_id": receipt.winner_id,
                          "expected_completed": expected_success, "control_passed": passed == expected_success,
                          "model_calls": receipt.model_calls, "max_model_calls": receipt.policy.max_model_calls,
                          "unique_eligible_patches": receipt.unique_eligible_patches,
                          "source_unchanged": receipt.source_unchanged, "receipt_valid": verify_tournament_report(receipt),
                          "provider_usage_complete": receipt.usage_accounting_complete,
                          "eligible_candidates": sum(candidate.eligible for candidate in receipt.candidates),
                          "candidate_reasons": {candidate.candidate_id: candidate.reasons for candidate in receipt.candidates}}
        rows.append({"scenario": scenario, "arms": arms})
    report = {"schema_version": "1.0", "kind": "authored_orchestration_controls", "decision": "held",
              "production_activation": False, "controls": rows,
              "source_fingerprint": repository_fingerprint(FIXTURE, task.policy.max_files, task.policy.max_repository_bytes),
              "controls_passed": all(arm["control_passed"] and arm["receipt_valid"] and arm["source_unchanged"]
                                     and arm["model_calls"] <= arm["max_model_calls"]
                                     for row in rows for arm in row["arms"].values()),
              "limitations": ["Scripted model responses and authored command outcomes; no provider or Docker execution.",
                              "These seven scenarios test orchestration contracts, not model accuracy, diversity, or real-world resolution lift.",
                              "One/three-team arms share the global write/call budget, not equal per-team compute.",
                              "Token usage, dollar cost, and latency gains are not measured by these controls."]}
    report["report_fingerprint"] = digest(report)
    return report


def markdown(report: dict) -> str:
    lines = ["# Multi-agent repair tournament controls", "", f"Orchestration controls passed: {report['controls_passed']}",
             "Decision: held; no activation or real-world quality claim.", "",
             "| Authored scenario | One-team status / calls | Three-team status / calls |", "|---|---|---|"]
    for row in report["controls"]:
        base, multi = row["arms"]["one_team"], row["arms"]["three_teams"]
        lines.append(f"| {row['scenario']} | {base['status']} / {base['model_calls']} | {multi['status']} / {multi['model_calls']} |")
    lines.extend(["", "## Evidence limits", ""] + [f"- {limit}" for limit in report["limitations"]] + [""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/evaluations/repair-tournament/controls.json"))
    parser.add_argument("--require-gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        target, workspace = args.output.resolve(), Path(__file__).resolve().parents[1]
        card = target.with_suffix(".md")
        if target == card or any(path.exists() or (path.is_relative_to(workspace)
                and path.relative_to(workspace).parts[:2] != ("data", "evaluations")) for path in (target, card)):
            raise ValueError("fresh protected output path required")
        report = asyncio.run(run_controls())
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with card.open("x", encoding="utf-8") as handle:
            handle.write(markdown(report))
        print(json.dumps({"controls_passed": report["controls_passed"], "decision": report["decision"],
                          "report_fingerprint": report["report_fingerprint"]}))
        return 1 if args.require_gate or not report["controls_passed"] else 0
    except (OSError, ValueError):
        print("Tournament controls failed: invalid inputs or protected output path.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
