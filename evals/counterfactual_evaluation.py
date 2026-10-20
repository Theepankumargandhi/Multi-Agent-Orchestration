"""Authored outcome controls and an optional real-container, no-model drill."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from code_agent.counterfactual import assess_trials, fingerprint, verify_counterfactual
from code_agent.models import CodeTask, CounterfactualTrial, SandboxPolicy
from code_agent.workspace import EphemeralWorkspace


def _trials(original="assertion", patched="passed"):
    return [CounterfactualTrial(arm=arm, repeat=repeat, outcomes={fingerprint("test"): status},
        source_sha256=fingerprint(arm), manifest_sha256=fingerprint("manifest"), workspace_unchanged=True)
        for arm, status in [("patched", patched), ("original", original)] for repeat in range(2)]


def evaluate():
    checks = {"stable_bug_discrimination": assess_trials(_trials())[0] == "stable_discrimination",
              "tautology_blocked": assess_trials(_trials(original="passed"))[0] == "no_discriminating_test",
              "dependency_error_not_bug_evidence": assess_trials(_trials(original="error"))[0] == "original_errors_or_skips",
              "skips_not_bug_evidence": assess_trials(_trials(original="skipped"))[0] == "original_errors_or_skips",
              "failing_patch_blocked": assess_trials(_trials(patched="assertion"))[0] == "patched_tests_not_all_passed",
              "incomplete_execution_blocked": assess_trials(_trials()[:3])[0] == "incomplete_execution"}
    trials = _trials()
    trials[-1].outcomes = {fingerprint("test"): "passed"}
    checks["flaky_outcome_blocked"] = assess_trials(trials)[0] == "unstable_or_mismatched_execution"
    trials = _trials()
    trials[0].workspace_unchanged = False
    checks["workspace_mutation_blocked"] = assess_trials(trials)[0] == "workspace_or_receipt_invalid"
    trials = _trials()
    trials[0].outcomes = trials[1].outcomes = {fingerprint("unrelated-test"): "passed"}
    checks["unrelated_test_identity_blocked"] = assess_trials(trials)[0] == "unstable_or_mismatched_execution"
    return {"schema_version": "1.0", "evidence_kind": "authored-counterfactual-outcome-controls",
        "provider_calls": 0, "repository_execution_measured": False, "live_coding_resolution_measured": False,
        "total": len(checks), "passed": sum(checks.values()), "checks": checks}


async def docker_drill():
    with tempfile.TemporaryDirectory(prefix="af-counterfactual-") as directory:
        root = Path(directory)
        (root / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
        policy = SandboxPolicy(counterfactual_verification_enabled=True)
        workspace = EphemeralWorkspace(root, policy)
        workspace.prepare()
        try:
            workspace.write_file("app.py", "def value():\n    return 2\n")
            workspace.write_file("tests/test_regression.py", "from app import value\ndef test_value():\n    assert value() == 2\n")
            sandbox = SimpleNamespace(repository=root, workspace=workspace)
            report = await verify_counterfactual(CodeTask(repository="authored-fixture",
                issue="Correct the authored value and verify the regression test.", policy=policy), sandbox, workspace.changed_files())
            return report.model_dump(mode="json")
        finally:
            workspace.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--docker", action="store_true", help="Run one authored real-container drill; never fall back to host execution")
    args = parser.parse_args()
    report = evaluate()
    if args.docker:
        report["docker_drill"] = asyncio.run(docker_drill())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["passed"] != report["total"] or (args.docker and report["docker_drill"]["status"] != "passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
