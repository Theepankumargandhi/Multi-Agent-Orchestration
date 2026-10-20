"""Authored ablation acceptance controls, with an optional no-model Docker drill."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from code_agent.models import CodeTask, PatchAblationTrial, SandboxPolicy
from code_agent.patch_minimization import matching_passes, minimize_patch, receipt_data
from code_agent.regression_challenges import fingerprint
from code_agent.verification import repository_fingerprint
from code_agent.workspace import EphemeralWorkspace


def _pair(status="passed"):
    outcomes = {fingerprint("authored-test"): status}
    source = fingerprint("authored-source")
    code = int(status in {"failed", "error"})
    return [PatchAblationTrial(phase="ablation", removed_paths=["support.py"], repeat=repeat,
        source_sha256=source, receipt_sha256=fingerprint(receipt_data(source, outcomes, code)),
        outcomes=outcomes, exit_code=code, duration_ms=1) for repeat in range(2)]


def evaluate():
    expected = _pair()[0].outcomes
    changed = _pair()
    changed[-1].outcomes = {fingerprint("different-test"): "passed"}
    changed[-1].receipt_sha256 = fingerprint(receipt_data(changed[-1].source_sha256, changed[-1].outcomes, 0))
    drift = _pair()
    drift[-1].source_sha256 = fingerprint("different-source")
    checks = {
        "clean_matching_tests_accepted": matching_passes(_pair(), expected),
        "essential_change_reversion_rejected": not matching_passes(_pair("failed"), expected),
        "skipped_tests_rejected": not matching_passes(_pair("skipped")),
        "fixture_errors_rejected": not matching_passes(_pair("error")),
        "changed_test_collection_rejected": not matching_passes(changed, expected),
        "source_repeat_drift_rejected": not matching_passes(drift, expected),
        "incomplete_repeat_rejected": not matching_passes(_pair()[:1], expected),
        "empty_evidence_rejected": not matching_passes([], expected),
    }
    return {"schema_version": "1.0", "evidence_kind": "authored-file-ablation-controls", "provider_calls": 0,
            "repository_execution_measured": False, "live_coding_resolution_measured": False,
            "total": len(checks), "passed": sum(checks.values()), "checks": checks}


async def docker_drill():
    with tempfile.TemporaryDirectory(prefix="af-minimize-") as directory:
        root = Path(directory)
        (root / "app.py").write_text("def value():\n    return 1\n", newline="")
        (root / "support.py").write_text("FLAG = False\n", newline="")
        (root / "test_app.py").write_text("from app import value\ndef test_value():\n    assert value() == 2\n", newline="")
        policy = SandboxPolicy(patch_minimization_enabled=True)
        workspace = EphemeralWorkspace(root, policy)
        workspace.prepare()
        try:
            workspace.write_file("app.py", "def value():\n    return 2\n")
            workspace.write_file("support.py", "FLAG = True\n")
            before = repository_fingerprint(root, policy.max_files, policy.max_repository_bytes)
            report = await minimize_patch(CodeTask(repository="authored-fixture", issue="Fix value without unrelated edits.", policy=policy),
                SimpleNamespace(repository=root, workspace=workspace), remaining_writes=6,
                original_sha256=before, deadline=time.perf_counter() + 300)
            return report.model_dump(mode="json")
        finally:
            workspace.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--docker", action="store_true", help="Authored real-container drill; no user-repository host fallback")
    args = parser.parse_args()
    result = evaluate()
    if args.docker:
        result["docker_drill"] = asyncio.run(docker_drill())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["passed"] != result["total"] or (args.docker and result["docker_drill"]["status"] != "minimized"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
