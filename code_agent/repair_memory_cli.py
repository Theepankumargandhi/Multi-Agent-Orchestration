"""Operator-only management of reviewed repair lessons, never an agent tool."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from code_agent.models import CodeAgentResult
from code_agent.repair_memory import RepairMemoryStore, scope_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/code-agent/repair-memory.db"))
    parser.add_argument("--project", required=True)
    parser.add_argument("--namespace", required=True, help="Authenticated tenant identity, not a model-generated label")
    parser.add_argument("--repository", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    approve = sub.add_parser("approve")
    approve.add_argument("--trajectory", type=Path, required=True)
    approve.add_argument("--problem", required=True)
    approve.add_argument("--lesson", required=True)
    approve.add_argument("--paths", nargs="+", required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--approve-lesson", action="store_true", help="Explicit attestation that you reviewed this lesson and its evidence")
    approve.add_argument("--ttl-days", type=int, default=30)
    for command in ("revoke", "delete"):
        sub.add_parser(command).add_argument("--lesson-id", required=True)
    args = parser.parse_args()
    try:
        _execute(args, parser)
    except (ValueError, OSError, sqlite3.Error, TypeError) as exc:
        # Pydantic/SQLite exception text can include raw evidence or lesson content.
        parser.exit(1, f"Repair memory operation rejected ({type(exc).__name__}); review evidence, scope, and approval inputs.\n")


def _execute(args, parser):
    store = RepairMemoryStore(args.store)
    scope = scope_hash(args.project, args.namespace, args.repository)
    if args.command == "approve":
        if not args.approve_lesson:
            parser.error("--approve-lesson is required after human review")
        if args.trajectory.stat().st_size > 10_000_000:
            parser.error("trajectory exceeds the review read bound")
        payload = json.loads(args.trajectory.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            parser.error("trajectory must be a JSON object")
        request = payload.get("case") or payload.get("request") or {}
        if not isinstance(request, dict):
            parser.error("trajectory must include a repository-scoped case or request")
        repository = request.get("repository")
        if repository != args.repository:
            parser.error("trajectory repository does not match the requested scope")
        if "user_id" in payload and payload["user_id"] != args.namespace:
            parser.error("job tenant does not match the requested namespace")
        result = CodeAgentResult.model_validate(payload.get("result"))
        identifier = store.approve(result, project=args.project, namespace=args.namespace,
            repository=args.repository, problem=args.problem, lesson=args.lesson,
            applicable_paths=args.paths, reviewer=args.reviewer, approve=args.approve_lesson,
            ttl_days=args.ttl_days)
        print(json.dumps({"lesson_id": identifier, "scope_sha256": scope, "status": "approved"}))
    else:
        changed = store.withdraw(args.lesson_id, scope, delete=args.command == "delete")
        print(json.dumps({"lesson_id": args.lesson_id, "changed": changed, "status": args.command}))
        if not changed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
