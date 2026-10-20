"""Credential-free scripted coding-loop comparisons, never executing fixture code."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from code_agent.agent import CodingAgent
from code_agent.models import CodeAction, CodeTask, SandboxCommandResult, SandboxPolicy
from code_agent.workspace import EphemeralWorkspace


class ScriptedModel:
    def __init__(self, actions):
        self.actions, self.calls, self.messages = actions, 0, []

    async def ainvoke(self, messages, config=None):
        self.messages.append(messages)
        action = self.actions[self.calls]
        self.calls += 1
        return action


class ScriptedSandbox:
    """Static predicate on literal fixture text; no subprocess/import/eval."""
    def __init__(self, root, policy):
        self.workspace = EphemeralWorkspace(root, policy)
        self.workspace.prepare()
        self.runs = 0

    def run(self, command):
        self.runs += 1
        passed = self.workspace.read_file("app.py") == "VALUE = 3\n"
        return SandboxCommandResult(command=command, exit_code=0 if passed else 1,
            stdout=f"1 {'passed' if passed else 'failed'} in {self.runs / 10:.2f}s", duration_ms=1)


def scenarios():
    read = CodeAction(kind="read", path="app.py")
    search = CodeAction(kind="search", pattern="VALUE")
    test = CodeAction(kind="test")
    def write(value):
        return CodeAction(kind="write", path="app.py", content=f"VALUE = {value}\n")
    return {
        "repeated-read": [read] * 12,
        "read-search-cycle": [read, search] * 6,
        "edit-oscillation": [write(1), write(2)] * 6,
        "distinct-edits": [write(1), write(2), write(3), test],
        "same-failure-distinct-source": [item for value in range(10, 16) for item in (write(value), test)],
        "warning-escape": [read, read, write(3), test],
        "repeated-policy-rejection": [CodeAction(kind="write", path=".env", content="not-a-secret")] * 12,
        "repeated-test": [test] * 12,
    }


async def evaluate():
    rows = []
    for name, actions in scenarios().items():
        arms = {}
        for mode in ("off", "shadow", "enforce"):
            policy = SandboxPolicy(loop_guard_mode=mode, max_iterations=len(actions), max_writes=20)
            with tempfile.TemporaryDirectory(prefix="af-loop-controls-") as temporary:
                root = Path(temporary)
                (root / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
                sandbox = ScriptedSandbox(root, policy)
                model = ScriptedModel(actions)
                try:
                    result = await CodingAgent(model, capture_telemetry=False).solve(
                        CodeTask(repository="authored", issue="Fix the authored VALUE from zero to three.", policy=policy), sandbox)
                    arms[mode] = {"status": result.status, "scripted_model_calls": result.model_calls,
                        "scripted_owner_runs": sandbox.runs, "writes": result.writes,
                        "detections": [item.disposition for item in result.loop_guard]}
                finally:
                    sandbox.workspace.cleanup()
        stop = name in {"repeated-read", "read-search-cycle", "edit-oscillation", "repeated-policy-rejection", "repeated-test"}
        matched = (arms["off"]["scripted_model_calls"] == arms["shadow"]["scripted_model_calls"]
                   and arms["off"]["status"] == arms["shadow"]["status"]
                   and (arms["enforce"]["scripted_model_calls"] < arms["off"]["scripted_model_calls"] if stop else
                        arms["enforce"]["scripted_model_calls"] == arms["off"]["scripted_model_calls"])
                   and ("stopped" in arms["enforce"]["detections"]) == stop)
        rows.append({"scenario": name, "passed": matched, "arms": arms})
    return {"schema_version": "1.0", "evidence_kind": "authored-scripted-coding-loop-comparison",
            "provider_calls": 0, "repository_code_executed": False, "live_quality_measured": False,
            "total": len(rows), "passed": sum(row["passed"] for row in rows), "scenarios": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(evaluate())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["total"] == report["passed"] else 1)


if __name__ == "__main__":
    main()
