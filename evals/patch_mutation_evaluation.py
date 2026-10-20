"""Authored acceptance controls and optional Docker-only patch-mutation drill."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from code_agent.models import CodeTask, MutationTestTrial, SandboxPolicy
from code_agent.patch_mutation import pair_status, source_mutants, test_patch_mutations
from code_agent.regression_challenges import fingerprint
from code_agent.verification import repository_fingerprint
from code_agent.workspace import EphemeralWorkspace


def authored_pair(status="assertion", *, drift=False):
    return [MutationTestTrial(phase="mutant", mutant_id=fingerprint("mutant"), repeat=repeat,
        source_sha256=fingerprint("source"), manifest_sha256=fingerprint("manifest"), receipt_sha256=fingerprint([status, drift and repeat]),
        outcomes={fingerprint("test"): "passed" if drift and repeat else status}, exit_code=int(status in {"assertion", "error"} and not (drift and repeat)), duration_ms=1)
        for repeat in (0, 1)]


def evaluate():
    killed = authored_pair()
    changed = authored_pair()
    for row in changed:
        row.outcomes = {fingerprint("other-test"): "assertion"}
    source = "# preserved comment\ndef allowed(value):\n    return value >= 18  # retain formatting\n"
    mutant = source_mutants("app.py", source, {3}, 1)[0]
    checks = {
        "stable_assertions_kill": pair_status(killed)[0] == "killed",
        "all_passing_tests_survive": pair_status(authored_pair("passed"))[0] == "survived",
        "errors_not_kills": pair_status(authored_pair("error"))[0] == "invalid",
        "skips_not_kills": pair_status(authored_pair("skipped"))[0] == "invalid",
        "flaky_outcomes_not_kills": pair_status(authored_pair(drift=True))[0] == "invalid",
        "changed_identities_not_kills": pair_status(changed, killed[0].outcomes)[0] == "invalid",
        "incomplete_repeats_not_kills": pair_status(killed[:1])[0] == "invalid",
        "single_token_changed_line_mutation": mutant.source == source.replace(">=", ">") and mutant.evidence.line == 3,
        "unchanged_line_excluded": source_mutants("app.py", source, {1}, 1) == [],
    }
    return {"schema_version": "1.0", "evidence_kind": "authored-patch-mutation-acceptance-controls",
            "provider_calls": 0, "repository_code_executed": False, "live_quality_measured": False,
            "total": len(checks), "passed": sum(checks.values()), "checks": checks}


async def docker_drill():
    with tempfile.TemporaryDirectory(prefix="af-mutation-drill-") as temporary:
        root = Path(temporary)
        (root / "app.py").write_text("def allowed(value):\n    return value > 18\n", encoding="utf-8")
        policy = SandboxPolicy(mutation_testing_mode="enforce", command_timeout_seconds=30)
        workspace = EphemeralWorkspace(root, policy)
        workspace.prepare()
        try:
            workspace.write_file("app.py", "def allowed(value):\n    return value >= 18\n")
            workspace.write_file("tests/test_allowed.py", "from app import allowed\ndef test_allowed():\n    assert allowed(18)\n")
            sandbox = SimpleNamespace(repository=root, workspace=workspace)
            task = CodeTask(repository="authored", issue="Allow the boundary value eighteen with regression coverage.", policy=policy)
            report = await test_patch_mutations(task, sandbox, workspace.unified_diff(), workspace.changed_files(),
                original_sha256=repository_fingerprint(root, policy.max_files, policy.max_repository_bytes), deadline=time.perf_counter() + 180)
            return report.model_dump(mode="json")
        finally:
            workspace.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--docker", action="store_true", help="Additional authored real-container drill; no model or host fallback")
    args = parser.parse_args()
    report = evaluate()
    if args.docker:
        report["docker_drill"] = asyncio.run(docker_drill())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    success = report["total"] == report["passed"] and (not args.docker or report["docker_drill"]["status"] == "measured")
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
