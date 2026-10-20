"""Freeze and compare owner-supplied migration snapshots; never clone, push or activate a patch."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path

from pydantic import Field, model_validator

from code_agent.benchmark_checkpoint import checkpoint_scope
from code_agent.models import SandboxPolicy
from code_agent.upgrade_cli import private_run_path, reserve_run, save_run
from code_agent.upgrade_inputs import (
    GENERATOR_VERSION,
    REDUCER_VERSION,
    CounterexamplePolicy,
    generate_campaign,
)
from code_agent.upgrade_models import MigrationSpec, StrictModel, fingerprint, safe_relative
from code_agent.upgrade_recovery import RECOVERY_PROTOCOL, MigrationJournal, StudyRunBudget, attempt_key
from code_agent.upgrade_relations import RelationPolicy, generate_relations, validate_relations
from code_agent.upgrade_runner import resolve_image, run_migration
from code_agent.upgrade_schema import SCHEMA_GENERATOR, SchemaPolicy
from code_agent.upgrade_schema_runtime import prepare_schema_campaign
from code_agent.upgrade_sessions import SessionPolicy, session_commitment, validate_sessions
from code_agent.verification import repository_fingerprint

ARMS = ("codemod_only", "existing_repair", "counterexample_repair")


def arm_metrics(results, declared_tasks):
    metrics = {}
    for arm in ARMS:
        rows = [row for row in results if row["arm"] == arm]
        ready = sum(row["status"] == "ready_for_owner_review" for row in rows)
        measured = [row["final_evaluation"] for row in rows if row["final_evaluation"] is not None]
        sessions = [row["final_session_evaluation"] for row in rows if row.get("final_session_evaluation") is not None]
        relations = [row["final_relation_evaluation"] for row in rows if row.get("final_relation_evaluation") is not None]
        complete_cost = len(rows) == declared_tasks and all(row["billed_cost_usd"] is not None for row in rows)
        total_cost = sum(row["billed_cost_usd"] for row in rows) if complete_cost else None
        metrics[arm] = {"declared_tasks": declared_tasks, "recorded_attempts": len(rows), "ready_for_owner_review": ready,
            "ready_for_owner_review_rate": ready / declared_tasks, "final_evaluations": len(measured),
            "final_case_regressions": sum(value["regressions"] for value in measured) if measured else None,
            "final_session_evaluations": len(sessions),
            "final_session_regressions": sum(value["regressions"] for value in sessions) if sessions else None,
            "final_relation_evaluations": len(relations),
            "final_relation_violations": sum(value["violations"] for value in relations) if relations else None,
            "mean_recorded_latency_ms": sum(row["duration_ms"] for row in rows) / len(rows) if rows else None,
            "model_calls": sum(row["model_calls"] for row in rows),
            "usage_complete_attempts": sum(row["usage_complete"] for row in rows), "total_billed_cost_usd": total_cost,
            "billed_cost_per_review_ready_task_usd": total_cost / ready if ready and total_cost is not None else None}
    return metrics


class StudyTask(StrictModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    repository: str
    upstream_url: str = Field(pattern=r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    upstream_commit: str = Field(pattern=r"^[a-f0-9]{40,64}$")
    license_file: str
    license_identifier: str = Field(min_length=1, max_length=100)
    owner_authorized: bool
    spec: MigrationSpec
    schema_policy: SchemaPolicy | None = None
    session_policy: SessionPolicy | None = None
    relation_policy: RelationPolicy | None = None

    @model_validator(mode="after")
    def private_snapshot(self):
        safe_relative(self.repository)
        safe_relative(self.license_file)
        if not self.owner_authorized or not self.repository.startswith("repositories/") or self.spec.task_id != self.task_id:
            raise ValueError("study snapshots must be under repositories/ with matching task IDs")
        return self


class StudyRequest(StrictModel):
    study_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    model: str = Field(pattern=r"^(?:groq|openai):[^\s]{1,120}$")
    policy: CounterexamplePolicy = Field(default_factory=CounterexamplePolicy)
    tasks: list[StudyTask] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def unique_tasks(self):
        if len({row.task_id for row in self.tasks}) != len(self.tasks):
            raise ValueError("study task IDs must be unique")
        return self


def _snapshot(root, task):
    private_root = (root / "repositories").resolve()
    repository = (root / task.repository).resolve()
    if (not private_root.is_relative_to(root.resolve()) or repository == private_root
            or not repository.is_relative_to(private_root) or not repository.is_dir()):
        raise ValueError("snapshot must be a confined private repository directory")
    license_path = repository / task.license_file
    if (license_path.is_symlink() or not license_path.resolve().is_relative_to(repository)
            or not license_path.is_file() or not 0 < license_path.stat().st_size <= 64000):
        raise ValueError("bounded license evidence is required")
    return repository, {"source_fingerprint": repository_fingerprint(repository, 5000, 50000000),
        "license_sha256": hashlib.sha256(license_path.read_bytes()).hexdigest()}


def freeze_study(request: StudyRequest, root: Path):
    snapshots, tasks = {}, []
    for task in request.tasks:
        repository, snapshots[task.task_id] = _snapshot(root, task)
        # Bind the resolved image, rather than a mutable local tag, before any arm runs.
        spec = task.spec.model_copy(update={"image": resolve_image(task.spec.image)})
        spec = MigrationSpec.model_validate(spec.model_dump())
        if task.schema_policy:
            campaign, schema_receipt = prepare_schema_campaign(repository, spec, task.schema_policy, request.policy)
            snapshots[task.task_id]["source_schema_fingerprint"] = schema_receipt["fingerprint"]
        else:
            campaign = generate_campaign(spec, request.policy)
        tasks.append({**task.model_dump(), "spec": spec.model_dump()})
        snapshots[task.task_id]["campaign_commitment"] = campaign.commitment
        if task.session_policy:
            validate_sessions(repository, spec, task.session_policy)
            snapshots[task.task_id]["session_commitment"] = session_commitment(task.session_policy)
        if task.relation_policy:
            validate_relations(repository, spec, task.relation_policy)
            snapshots[task.task_id]["relation_commitment"] = generate_relations(spec, task.relation_policy).commitment
    payload = {"protocol": "upgrade-migration-study-v1", "request": {**request.model_dump(), "tasks": tasks},
        "snapshots": snapshots, "arms": list(ARMS), "generator": GENERATOR_VERSION, "reducer": REDUCER_VERSION,
        "upstream_and_license_metadata": "owner_attested_not_independently_verified"}
    if any(task.schema_policy for task in request.tasks):
        payload["schema_generator"] = SCHEMA_GENERATOR
    return {**payload, "fingerprint": fingerprint(payload)}


async def run_study(frozen, root: Path, output: Path, *, authorize_provider=False, resume=False,
                    retry_interrupted=False, budget: StudyRunBudget | None = None):
    if not authorize_provider:
        raise ValueError("a three-arm live study requires explicit provider authorization")
    if retry_interrupted and not resume:
        raise ValueError("--retry-interrupted requires --resume")
    if (frozen.get("protocol") != "upgrade-migration-study-v1" or frozen.get("arms") != list(ARMS)
            or frozen.get("generator") != GENERATOR_VERSION or frozen.get("reducer") != REDUCER_VERSION
            or fingerprint({key: value for key, value in frozen.items() if key != "fingerprint"}) != frozen.get("fingerprint")):
        raise ValueError("study manifest integrity mismatch")
    request = StudyRequest.model_validate(frozen["request"])
    if any(task.schema_policy for task in request.tasks) and frozen.get("schema_generator") != SCHEMA_GENERATOR:
        raise ValueError("schema generator version changed")
    # Check every snapshot before the first inference, then again at each admission.
    def validate(task):
        repository, observed = _snapshot(root, task)
        if resolve_image(task.spec.image) != task.spec.image:
            raise ValueError("frozen local image identity changed")
        expected = frozen["snapshots"][task.task_id]
        if task.relation_policy:
            validate_relations(repository, task.spec, task.relation_policy)
            if generate_relations(task.spec, task.relation_policy).commitment != expected.get("relation_commitment"):
                raise ValueError("relation campaign changed since study freeze")
        if task.session_policy:
            validate_sessions(repository, task.spec, task.session_policy)
            if session_commitment(task.session_policy) != expected.get("session_commitment"):
                raise ValueError("session campaign changed since study freeze")
        if task.schema_policy:
            campaign, receipt = prepare_schema_campaign(repository, task.spec, task.schema_policy, request.policy)
            if receipt["fingerprint"] != expected.get("source_schema_fingerprint"):
                raise ValueError("source schema changed since study freeze")
        else:
            campaign = generate_campaign(task.spec, request.policy)
        if (any(observed[key] != expected[key] for key in observed)
                or campaign.commitment != expected["campaign_commitment"] or not re.fullmatch(r"sha256:[a-f0-9]{64}", task.spec.image)):
            raise ValueError("study source, license, campaign or pinned image changed")
        return repository
    for task in request.tasks:
        validate(task)
    directory = private_run_path(output)
    entries = {attempt_key(task.task_id, arm): (task, arm, 0 if arm == "codemod_only" else task.spec.max_model_calls)
        for task in request.tasks for arm in ARMS}
    scope = checkpoint_scope(request.tasks, root.resolve(), SandboxPolicy(), frozen["fingerprint"])
    default_budget = StudyRunBudget(max_attempts=len(entries),
        max_provider_calls=sum(reservation for _task, _arm, reservation in entries.values()))
    with MigrationJournal(directory, resume=resume) as journal:
        selected_budget = budget if budget is not None else (None if resume else default_budget)
        journal.bind(scope, frozen["fingerprint"], initial_metadata=None if resume else {
            "migration_protocol": RECOVERY_PROTOCOL, "run_budget": selected_budget.model_dump(), "revoked": False})
        journal.configure(entries, selected_budget)
        dispositions = {}
        failure_type = None
        try:
            # Validate/recover EVERY prior arm before permitting a new admission.
            # An unresolved older interruption must not be skipped in favor of later work.
            for key in entries:
                _record, disposition = journal.recover(key, retry_interrupted=retry_interrupted)
                dispositions[key] = disposition
            if journal.state.get("revoked"):
                raise ValueError("study admissions revoked after recovered source-integrity failure")
            for key, (task, arm, _reservation) in entries.items():
                if journal.state["attempts"].get(key, [{}])[-1].get("status") == "complete":
                    continue
                repository = validate(task)
                artifacts = journal.admit(key)
                report = await run_migration(repository, task.spec,
                    model=None if arm == "codemod_only" else request.model,
                    authorize_provider=arm != "codemod_only", counterexample_policy=request.policy,
                    discovery_feedback=arm == "counterexample_repair", schema_policy=task.schema_policy,
                    expected_campaign_commitment=frozen["snapshots"][task.task_id]["campaign_commitment"],
                    session_policy=task.session_policy, session_feedback=arm == "counterexample_repair",
                    expected_session_commitment=frozen["snapshots"][task.task_id].get("session_commitment"),
                    relation_policy=task.relation_policy, relation_feedback=arm == "counterexample_repair",
                    expected_relation_commitment=frozen["snapshots"][task.task_id].get("relation_commitment"))
                save_run(report, artifacts)
                journal.finish_report(key, report)
                if not report["source_unchanged"]:
                    raise ValueError("source changed; remaining study admissions revoked")
        except BaseException as exc:
            failure_type = type(exc).__name__
            raise
        finally:
            results = [attempts[-1]["outcome"] for key in entries
                if (attempts := journal.state["attempts"].get(key)) and attempts[-1]["status"] == "complete"]
            totals = journal.accounting()
            metrics = arm_metrics(results, len(request.tasks))
            for arm in ARMS:
                resources = journal.accounting(arm)
                metrics[arm].update(resources)
                metrics[arm]["quality_metrics_valid"] = (len([row for row in results if row["arm"] == arm]) == len(request.tasks)
                    and not resources["interrupted_attempts"] and all(row["source_unchanged"] for row in results if row["arm"] == arm))
                if resources["interrupted_attempts"]:
                    metrics[arm].update(total_billed_cost_usd=None, billed_cost_per_review_ready_task_usd=None)
            summary = {"study_fingerprint": frozen["fingerprint"], "expected_attempts": len(entries),
                "recorded_attempts": len(results), "complete": len(results) == len(entries),
                "outcomes": results, "owner_acceptance_measured": False, "production_impact_measured": False,
                "incomplete_attempt_usage": "unknown_not_zero" if totals["interrupted_attempts"] else None,
                "quality_metrics_valid": len(results) == len(entries) and not totals["interrupted_attempts"] and not journal.state.get("revoked"),
                "admissions_revoked": bool(journal.state.get("revoked")),
                "run_budget": journal.budget.model_dump(), "resource_accounting": totals,
                "recovery_dispositions": dispositions, "stopped_error_type": failure_type,
                "arm_metrics": metrics}
            journal.write_summary(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "run"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorize-provider", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse a matching study checkpoint without repeating completed arms")
    parser.add_argument("--retry-interrupted", action="store_true", help="Explicitly permit a new arm attempt; prior call reservations remain consumed")
    parser.add_argument("--max-attempts", type=int, help="Reviewed total attempt cap, including interrupted history")
    parser.add_argument("--max-provider-calls", type=int, help="Admitted request cap, not a dollar or account billing limit")
    args = parser.parse_args()
    if args.manifest.stat().st_size > 500000:
        parser.error("study manifest exceeds its byte budget")
    payload = json.loads(args.manifest.read_bytes())
    if args.operation == "freeze":
        if args.authorize_provider or args.resume or args.retry_interrupted or args.max_attempts is not None or args.max_provider_calls is not None:
            parser.error("freezing never authorizes inference")
        frozen = freeze_study(StudyRequest.model_validate(payload), Path.cwd())
        directory = reserve_run(args.output_dir)
        with (directory / "manifest.json").open("x", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps(frozen, indent=2) + "\n")
        print(json.dumps({"frozen": True, "tasks": len(frozen["request"]["tasks"]), "provider_calls": 0}))
    else:
        if not args.authorize_provider:
            parser.error("running a live comparison requires --authorize-provider and an approved account budget")
        if (args.max_attempts is None) != (args.max_provider_calls is None):
            parser.error("provide both reviewed run budget limits, or neither")
        selected_budget = StudyRunBudget(max_attempts=args.max_attempts, max_provider_calls=args.max_provider_calls) if args.max_attempts is not None else None
        from dotenv import load_dotenv
        load_dotenv()
        summary = asyncio.run(run_study(payload, Path.cwd(), args.output_dir, authorize_provider=True,
            resume=args.resume, retry_interrupted=args.retry_interrupted, budget=selected_budget))
        print(json.dumps({key: value for key, value in summary.items() if key != "outcomes"}))


if __name__ == "__main__":
    main()
