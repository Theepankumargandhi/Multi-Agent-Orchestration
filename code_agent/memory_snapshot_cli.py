"""Freeze private reviewed lessons or audit them against a held-out selection."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from code_agent.evaluation import load_code_cases
from code_agent.memory_snapshot import MemorySnapshot, SourceLineage, audit_snapshot, freeze_memory
from code_agent.repair_memory import RepairMemoryStore


def _read(path):
    if path.stat().st_size > 50_000_000:
        raise ValueError("snapshot input exceeds the read bound")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--store", type=Path, default=Path("data/code-agent/repair-memory.db"))
    freeze.add_argument("--project", required=True)
    freeze.add_argument("--namespace", required=True)
    freeze.add_argument("--repositories", nargs="+", required=True)
    freeze.add_argument("--lineage", type=Path, required=True, help="Private lesson-ID mapping to reviewed source task lineage")
    freeze.add_argument("--output", type=Path, required=True, help="New private snapshot file; existing files are never overwritten")
    audit = sub.add_parser("audit")
    audit.add_argument("snapshot", type=Path)
    audit.add_argument("dataset", type=Path)
    audit.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            raw = _read(args.lineage)
            if not isinstance(raw, dict) or len(raw) > 5000:
                raise ValueError("lineage must be a bounded lesson-ID mapping")
            lineage = {key: SourceLineage.model_validate(value) for key, value in raw.items()}
            snapshot = freeze_memory(RepairMemoryStore(args.store), project=args.project, namespace=args.namespace,
                                     repositories=args.repositories, lineage=lineage)
            _write_new(args.output, snapshot.model_dump(mode="json"))
            print(json.dumps({"snapshot_sha256": snapshot.fingerprint, "repositories": len(snapshot.repositories)}))
        else:
            snapshot = MemorySnapshot.model_validate(_read(args.snapshot))
            report = audit_snapshot(snapshot, load_code_cases(args.dataset))
            if args.output:
                _write_new(args.output, report)
            print(json.dumps(report, indent=2))
            if not report["passed"]:
                raise SystemExit(1)
    except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Memory snapshot operation rejected ({type(exc).__name__}); review lineage, scope, and file bindings.\n")


if __name__ == "__main__":
    main()
