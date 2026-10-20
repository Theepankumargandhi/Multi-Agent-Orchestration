"""Authored frozen-memory leak controls; no paid provider or coding-quality claim."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from code_agent.evaluation_models import CodeBenchmarkCase
from code_agent.memory_snapshot import FrozenRepairMemory, SourceLineage, audit_snapshot, freeze_memory
from code_agent.models import CodeTask
from code_agent.repair_memory import RepairMemoryStore, scope_hash
from evals.repair_memory_evaluation import _scripted_evidence


def evaluate():
    with tempfile.TemporaryDirectory(prefix="af-snapshot-controls-") as directory:
        store = RepairMemoryStore(Path(directory) / "memory.db")
        identifier = store.approve(_scripted_evidence(), project="control", namespace="alice", repository="repo",
            problem="Incorrect tax rounding precision", lesson="Preserve decimal precision and verify boundary amounts.",
            applicable_paths=["tax.py"], reviewer="authored-owner", approve=True, now=1000)
        origin = SourceLineage(case_id="training-case", family_id="training-family", dataset_sha256="a" * 64,
            issue="A currency tax adjustment loses precision for fractional monetary values.", completed_at=999)
        snapshot = freeze_memory(store, project="control", namespace="alice", repositories=["repo"],
                                 lineage={identifier: origin}, frozen_at=1001)
        case = CodeBenchmarkCase(id="heldout-case", repository="repo", metadata={"family_id": "refunds"},
            issue="Tax rounding must preserve precision for negative refundable amounts.", test_command=["python", "-m", "pytest", "-q"])
        checks = {"independent_selection_admitted": audit_snapshot(snapshot, [case], now=1002)["passed"]}
        for name, changes in [
            ("case_overlap_blocked", {"id": origin.case_id}),
            ("family_overlap_blocked", {"metadata": {"family_id": origin.family_id}}),
            ("exact_issue_blocked", {"issue": origin.issue}),
            ("near_duplicate_blocked", {"issue": origin.issue + " Additional boundary checks are required."}),
            ("lesson_text_overlap_blocked", {"issue": "Preserve decimal precision and verify boundary amounts."}),
        ]:
            checks[name] = not audit_snapshot(snapshot, [case.model_copy(update=changes)], now=1002)["passed"]
        frozen = FrozenRepairMemory.prepare(snapshot, [case], now=1002)
        store.withdraw(identifier, scope_hash("control", "alice", "repo"), delete=True)

        class Workspace:
            def list_files(self, limit):
                return ["tax.py"]

        context, receipt = frozen.retrieve(CodeTask(repository="repo", issue=case.issue, benchmark_case_id=case.id),
                                           Workspace(), stage="implementation", iteration=0)
        checks["frozen_context_survives_live_store_deletion"] = bool(context) and identifier in receipt.lesson_scores
        checks["snapshot_receipt_bound"] = receipt.snapshot_sha256 == snapshot.fingerprint
    return {"schema_version": "1.0", "evidence_kind": "authored-memory-snapshot-controls",
            "provider_calls": 0, "coding_resolution_measured": False, "semantic_independence_proven": False,
            "total": len(checks), "passed": sum(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
