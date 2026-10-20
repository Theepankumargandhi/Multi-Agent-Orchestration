"""Local owner CLI for UpgradePilot. No pull requests, merges or provider calls by default."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from code_agent.upgrade_inputs import CounterexamplePolicy
from code_agent.upgrade_models import MigrationSpec
from code_agent.upgrade_plan import plan_migration
from code_agent.upgrade_relations import RelationPolicy
from code_agent.upgrade_runner import run_migration
from code_agent.upgrade_schema import SchemaPolicy
from code_agent.upgrade_sessions import SessionPolicy


def private_run_path(destination: Path):
    root = (Path.cwd() / "data/upgradepilot").resolve()
    destination = destination.absolute()
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("private run paths cannot contain symlinks")
    resolved = destination.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("private run artifacts must be below data/upgradepilot/ with a new run directory")
    return destination


def reserve_run(destination: Path):
    destination = private_run_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def save_run(report, destination: Path):
    payload = {key: value for key, value in report.items() if key != "patch"}
    with (destination / "report.json").open("x", encoding="utf-8", newline="") as output:
        output.write(json.dumps(payload, indent=2) + "\n")
        output.flush()
        os.fsync(output.fileno())
    if report.get("patch"):
        with (destination / "migration.diff").open("x", encoding="utf-8", newline="") as output:
            output.write(report["patch"])
            output.flush()
            os.fsync(output.fileno())
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "run"))
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--model", help="Explicit provider:model; requires separate request authorization")
    parser.add_argument("--authorize-provider", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--counterexamples", action="store_true", help="Enable bounded generated discovery and an independent final gate")
    parser.add_argument("--counterexample-seed", type=int, default=1729)
    parser.add_argument("--schema-policy", type=Path, help="Owner allowlist of model/contract mappings; implies counterexamples")
    parser.add_argument("--session-policy", type=Path, help="Owner-authored stateful discovery and withheld final scenarios")
    parser.add_argument("--relation-policy", type=Path, help="Owner-declared workflow laws and bounded generated variants")
    args = parser.parse_args()
    if args.spec.stat().st_size > 40000:
        parser.error("specification exceeds the byte budget")
    spec = MigrationSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
    if args.operation == "plan":
        if args.model or args.authorize_provider or args.counterexamples or args.schema_policy or args.session_policy or args.relation_policy:
            parser.error("planning never accepts inference authorization or execution flags")
        print(plan_migration(args.repository, spec).model_dump_json(indent=2))
        return
    if args.authorize_provider and not args.model:
        parser.error("provider authorization requires an explicit model")
    if args.model and not args.authorize_provider:
        parser.error("model requests require --authorize-provider")
    # Read private environment only after explicit live authorization; never persist it.
    if args.authorize_provider:
        from dotenv import load_dotenv
        load_dotenv()
    destination = args.output_dir or Path("data/upgradepilot") / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    reserve_run(destination)  # Refuse overwrites or an unsafe output path before any possible inference.
    policy = CounterexamplePolicy(seed=args.counterexample_seed) if args.counterexamples or args.schema_policy else None
    if args.schema_policy and args.schema_policy.stat().st_size > 8000:
        parser.error("schema policy exceeds its byte budget")
    schema = SchemaPolicy.model_validate_json(args.schema_policy.read_bytes()) if args.schema_policy else None
    if args.session_policy and args.session_policy.stat().st_size > 40000:
        parser.error("session policy exceeds its byte budget")
    if args.session_policy and args.session_policy.resolve().is_relative_to(args.repository.resolve()):
        parser.error("keep withheld session policies outside the repository snapshot visible to repair agents")
    sessions = SessionPolicy.model_validate_json(args.session_policy.read_bytes()) if args.session_policy else None
    if args.relation_policy and (args.relation_policy.stat().st_size > 40000
            or args.relation_policy.resolve().is_relative_to(args.repository.resolve())):
        parser.error("keep bounded relation policies outside the repository snapshot visible to repair agents")
    relations = RelationPolicy.model_validate_json(args.relation_policy.read_bytes()) if args.relation_policy else None
    report = asyncio.run(run_migration(args.repository, spec, model=args.model, authorize_provider=args.authorize_provider,
        counterexample_policy=policy, schema_policy=schema, session_policy=sessions, relation_policy=relations))
    saved = save_run(report, destination)
    print(json.dumps({"status": report["status"], "reason": report["reason"], "private_artifacts": str(saved),
        "model_calls": report["model_calls"], "source_unchanged": report["source_unchanged"], "owner_approval_required": True}))
    if report["status"] != "ready_for_owner_review":
        raise SystemExit(2)  # A valid quality hold is not a successful migration.


if __name__ == "__main__":
    main()
