"""Authored repair-memory retrieval/lifecycle controls, not model-quality scores."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from code_agent.models import (
    AnalysisPlan,
    CodeAgentResult,
    QualityGateResult,
    ReviewVerdict,
    SandboxCommandResult,
    SecuritySummary,
    VerificationReport,
    VerificationRound,
)
from code_agent.repair_memory import RepairMemoryStore, scope_hash


def _scripted_evidence():
    return CodeAgentResult(status="completed", summary="Authored memory-control evidence, not a real repair",
        patch="--- a/tax.py\n+++ b/tax.py\n-1\n+2\n", changed_files=["tax.py"],
        final_test=SandboxCommandResult(command=["python", "-m", "pytest", "-q"], exit_code=0, duration_ms=1),
        sandbox={"original_repository_unchanged": True}, security=SecuritySummary(policy_version="authored-control"),
        verification=VerificationReport(analysis=AnalysisPlan(summary="Authored control"), final_decision="verified",
            rounds=[VerificationRound(round=0, verdict=ReviewVerdict(approved=True, summary="Scripted review"),
                quality_gates=[QualityGateResult(name=name, status="passed", summary="Scripted gate")
                               for name in ("mandatory_tests", "patch_scope", "secret_scan", "unsafe_code_scan", "source_repository_integrity")])]))


def evaluate():
    checks = {}
    with tempfile.TemporaryDirectory(prefix="af-memory-controls-") as directory:
        store = RepairMemoryStore(Path(directory) / "memory.db")
        args = dict(project="control-project", namespace="alice", repository="repo", reviewer="control-reviewer",
                    problem="Incorrect tax rounding precision", lesson="Preserve two decimal places and verify tax rounding at boundary amounts.",
                    applicable_paths=["tax.py"], approve=True, now=1000, ttl_days=1)
        identifier = store.approve(_scripted_evidence(), **args)
        scope = scope_hash("control-project", "alice", "repo")

        def query(**kwargs):
            values = dict(scope=scope, query="Fix the tax rounding precision", known_paths={"tax.py"}, now=1001)
            values.update(kwargs)
            return store.retrieve(**values)

        context, scores, _, _ = query()
        checks["reviewed_lesson_retrieved"] = identifier in scores and bool(context)
        checks["tenant_isolation"] = not query(scope=scope_hash("control-project", "bob", "repo"))[0]
        checks["repository_isolation"] = not query(scope=scope_hash("control-project", "alice", "other"))[0]
        checks["project_isolation"] = not query(scope=scope_hash("other-project", "alice", "repo"))[0]
        checks["expired_lesson_abstained"] = not query(now=1000 + 86400)[0]
        checks["unrelated_query_abstained"] = not query(query="Concurrent worker leases expire unexpectedly")[0]
        checks["absent_source_path_abstained"] = not query(known_paths={"other.py"})[0]
        try:
            store.approve(_scripted_evidence(), **{**args, "approve": False})
            checks["human_approval_required"] = False
        except ValueError:
            checks["human_approval_required"] = True
        with closing(sqlite3.connect(store.path)) as connection, connection:
            body = connection.execute("SELECT body FROM lessons").fetchone()[0]
            modified = json.loads(body)
            modified["lesson"] = "Unreviewed tax precision changes were injected into this stored lesson."
            connection.execute("UPDATE lessons SET body=?", (json.dumps(modified),))
        checks["modified_body_rejected"] = not query()[0] and query()[3] == 1
        with closing(sqlite3.connect(store.path)) as connection, connection:
            connection.execute("UPDATE lessons SET body=?", (body,))
        store.withdraw(identifier, scope)
        checks["revocation_effective"] = not query()[0]
        store.withdraw(identifier, scope, delete=True)
        with closing(sqlite3.connect(store.path)) as connection, connection:
            checks["scoped_delete_effective"] = connection.execute("SELECT COUNT(*) FROM lessons").fetchone()[0] == 0
    return {"schema_version": "1.0", "evidence_kind": "authored-reviewed-memory-controls",
            "coding_resolution_measured": False, "provider_calls": 0,
            "total": len(checks), "passed": sum(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate()
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
