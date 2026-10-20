"""Offline old/new verification and optional budgeted repair through existing coding teams."""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from pathlib import Path

from code_agent.delivery import digest, manifest
from code_agent.models import AnalysisPlan, CodeAction, CodeTask, ReviewVerdict, SandboxPolicy
from code_agent.provider_reliability import ProviderCircuitOpen, ProviderFailure, classify_provider_error
from code_agent.sandbox import DockerSandbox
from code_agent.upgrade_contracts import compare_contracts, parse_probe, validate_source_contracts
from code_agent.upgrade_differential import settled_call, verify_generated
from code_agent.upgrade_inputs import (
    GENERATOR_VERSION,
    REDUCER_VERSION,
    CounterexamplePolicy,
    generate_campaign,
    input_key,
)
from code_agent.upgrade_models import (
    BASELINE_VERSION,
    CONTRACT_PATH,
    PROBE_PATH,
    SOURCE_VERSION,
    TARGET_VERSION,
    V1_PYTHON,
    V2_PYTHON,
    MigrationSpec,
    fingerprint,
)
from code_agent.upgrade_plan import RULES, plan_migration, source_files
from code_agent.upgrade_relation_runtime import verify_relations
from code_agent.upgrade_relations import (
    RELATION_GENERATOR,
    RelationPolicy,
    generate_relations,
    relation_inventory,
    relation_reserve,
    validate_relations,
)
from code_agent.upgrade_schema import (
    SCHEMA_CONFIG,
    SCHEMA_GENERATOR,
    SchemaPolicy,
    declaration_changes,
    schema_campaign,
    validate_schema_targets,
)
from code_agent.upgrade_schema_runtime import read_schemas
from code_agent.upgrade_session_runtime import verify_sessions
from code_agent.upgrade_sessions import (
    SessionPolicy,
    session_commitment,
    validate_sessions,
    verification_reserve,
)
from code_agent.verification import is_test_path, repository_fingerprint, run_quality_gates
from code_agent.verified_pr import VerifiedPRAgent
from code_agent.workspace import EphemeralWorkspace

CODEMOD = "/opt/codemod/bin/bump-pydantic"


async def _owned_operation(function):
    """Settle resource creation before cancellation cleanup can inspect ownership."""
    return await settled_call(function)


class BudgetModel:
    """One admission ledger across analysis/actions/review, reserving the final review call."""
    def __init__(self, runnable, budget, stage):
        self.runnable, self.budget, self.stage = runnable, budget, stage

    async def ainvoke(self, messages, config=None):
        stage = self.stage if self.stage in {"analysis", "review"} else "implementation"
        if self.budget.get("failure"):
            raise ProviderCircuitOpen(self.budget["failure"])
        cost = sum(len(str(message.content).encode()) for message in messages) + 4096
        reserve_calls, reserve_tokens = (0, 0) if self.stage == "review" else (1, 24000)
        if (self.budget["calls"] + 1 + reserve_calls > self.budget["max_calls"]
                or self.budget["reserved_tokens"] + cost + reserve_tokens > self.budget["max_reserved_tokens"]
                or time.monotonic() >= self.budget["deadline"]):
            failure = ProviderFailure(category="invalid_request", stage=stage)
            raise ProviderCircuitOpen(failure)
        self.budget["calls"] += 1
        self.budget["reserved_tokens"] += cost
        try:
            from langsmith import tracing_context
            # Inference consent does not authorize an additional raw-source trace export.
            with tracing_context(enabled=False):
                return await asyncio.wait_for(self.runnable.ainvoke(messages, config=config), min(31, self.budget["deadline"] - time.monotonic()))
        except Exception as exc:
            # No workflow-level retry after an admitted request: its spend can be unknown.
            failure = classify_provider_error(exc, stage=stage)
            self.budget["failure"] = failure
            raise ProviderCircuitOpen(failure) from None


def _policy(spec, image_id):
    return SandboxPolicy(image=image_id, allowed_images={image_id}, max_output_chars=100000,
        max_iterations=spec.max_model_calls, max_writes=spec.max_writes, max_workflow_writes=spec.max_writes,
        max_repair_rounds=0, task_timeout_seconds=spec.task_timeout_seconds, command_timeout_seconds=spec.command_timeout_seconds,
        targeted_edits_only=True, require_specialist_completion=True, delivery_verification_enabled=True,
        allowed_test_executables={"python", V1_PYTHON, V2_PYTHON, CODEMOD, "/opt/codemod/bin/python"})


def resolve_image(image):
    result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", "--", image],
        capture_output=True, text=True, timeout=10, check=True)
    value = result.stdout.strip()
    if not value.startswith("sha256:") or len(value) != 71:
        raise ValueError("a local content-addressed image is required")
    return value


def _source_pin(text):
    old = f"pydantic=={SOURCE_VERSION}"
    lines = text.splitlines(keepends=True)
    if sum(line.strip() == old for line in lines) != 1:
        raise ValueError("dependency file must contain exactly the supported source-version pin")
    # Reject a conflicting second Pydantic requirement or an indirect constraints override.
    if any((line.strip().lower().startswith("pydantic") and line.strip() != old) or line.strip().startswith(("-r", "-c", "--")) for line in lines):
        raise ValueError("dependency overrides require a different operator-reviewed recipe")
    return "".join(line.replace(old, f"pydantic=={TARGET_VERSION}", 1) if line.strip() == old else line for line in lines)


def _scope_valid(changed, spec, *, allow_dependency=True):
    return all((allow_dependency and name == spec.dependency_file) or any(name == root or name.startswith(root.rstrip("/") + "/")
        for root in spec.source_paths) for name in changed)


def _native_target(workspace, spec):
    # Conservative anti-shortcut checks; not a proof against adversarial patches.
    for path in source_files(workspace.root, spec):
        text = path.read_text(encoding="utf-8")
        if "pydantic.v1" in text or path.name == "pydantic.py" or "pydantic" in path.relative_to(workspace.root).parts[:-1]:
            return False
    return not any(row.rule_id in {"validators", "config", "field-options", "v1-namespace"}
                   for row in plan_migration(workspace.root, spec).findings)


async def _run(sandbox, command, report, stage):
    before = digest(manifest(sandbox.workspace))
    result = await asyncio.to_thread(sandbox.run, command)
    after = digest(manifest(sandbox.workspace))
    report["executions"].append({"stage": stage, "exit_code": result.exit_code, "timed_out": result.timed_out,
        "duration_ms": result.duration_ms, "workspace_unchanged": before == after})
    if result.timed_out or before != after:
        raise ValueError("execution timed out or modified source/configuration")
    return result


async def _probes(sandbox, spec, report, version):
    interpreter = V1_PYTHON if version == SOURCE_VERSION else V2_PYTHON
    values = []
    for repeat in range(2):
        result = await _run(sandbox, [interpreter, "-I", PROBE_PATH, "/workspace/" + CONTRACT_PATH, version], report,
                            f"contracts:{version}:{repeat}")
        if result.exit_code != 0:
            raise ValueError("contract execution unavailable")
        values.append(parse_probe(result.stdout, spec, version))
    if values[0] != values[1]:
        raise ValueError("contract behavior was unstable across fresh processes")
    return values[0]


async def _attest(sandbox, report):
    for interpreter, package, version in ((V1_PYTHON, "pydantic", SOURCE_VERSION),
        (V2_PYTHON, "pydantic", TARGET_VERSION), ("/opt/codemod/bin/python", "bump-pydantic", BASELINE_VERSION)):
        code = f"import importlib.metadata as m;print(m.version({package!r}))"
        result = await _run(sandbox, [interpreter, "-I", "-c", code], report, "attest:" + package + ":" + version)
        if result.exit_code != 0 or result.stdout.strip() != version:
            raise ValueError("sandbox dependency attestation failed")
    expected = hashlib.sha256(Path(__file__).with_name("upgrade_probe.py").read_bytes()).hexdigest()
    result = await _run(sandbox, [V2_PYTHON, "-I", "-c", f"import hashlib;print(hashlib.sha256(open({PROBE_PATH!r},'rb').read()).hexdigest())"], report, "attest:probe")
    if result.exit_code or result.stdout.strip() != expected:
        raise ValueError("image probe differs from the reviewed harness; rebuild the migration image")


def _record_campaign(report, campaign, policy, feedback, *, generator=GENERATOR_VERSION):
    report["input_campaign"] = {"generator": generator, "reducer": REDUCER_VERSION, "commitment": campaign.commitment,
        "discovery_fingerprint": campaign.discovery_fingerprint, "final_fingerprint": campaign.final_fingerprint,
        "discovery_cases": len(campaign.discovery), "final_cases": len(campaign.final),
        "discovery_feedback": feedback, "policy": policy.model_dump(), "final_inputs_exposed_to_repair": False}


async def _schema_target(sandbox, schema_policy, source_schemas):
    schemas, receipt = await settled_call(read_schemas, sandbox, schema_policy, TARGET_VERSION)
    return {**receipt, "available": True, "declarations": {name: declaration_changes(source_schemas[name], schema)
        for name, schema in schemas.items()}}


async def run_migration(repository: Path, spec: MigrationSpec, *, model=None, authorize_provider=False, scripted_models=None,
                        counterexample_policy: CounterexamplePolicy | None = None, discovery_feedback=True,
                        schema_policy: SchemaPolicy | None = None, schema_discovery=True, expected_campaign_commitment=None,
                        session_policy: SessionPolicy | None = None, session_feedback=True, expected_session_commitment=None,
                        relation_policy: RelationPolicy | None = None, relation_feedback=True, expected_relation_commitment=None):
    started = time.monotonic()
    report = {"protocol": spec.protocol, "task_id": spec.task_id, "family": spec.family, "status": "held",
        "reason": "not_verified", "spec_fingerprint": fingerprint(spec.model_dump()), "executions": [], "baseline": None,
        "repair": None, "model_calls": 0, "live_provider_calls": 0, "usage_complete": False,
        "provider_input_tokens": None, "provider_output_tokens": None, "billed_cost_usd": None,
        "inference_kind": "none", "source_unchanged": False, "owner_approval_required": True,
        "external_repository_writes": False, "patch": ""}
    view, baseline, candidate = None, None, None
    source_before, budget = None, None
    report["phase"] = "preflight"
    try:
        if schema_policy and not counterexample_policy:
            raise ValueError("schema guidance requires counterexample verification")
        if scripted_models is not None and (model or authorize_provider):
            raise ValueError("control and live execution cannot be combined")
        if model and (not authorize_provider or not model.startswith(("groq:", "openai:"))):
            raise ValueError("live inference needs explicit provider identity and authorization")
        image_id = await asyncio.to_thread(resolve_image, spec.image)
        report["image_id"] = image_id
        policy = _policy(spec, image_id)
        source_before = await asyncio.to_thread(repository_fingerprint, repository, policy.max_files, policy.max_repository_bytes)
        if expected_session_commitment is not None and (session_policy is None
                or session_commitment(session_policy) != expected_session_commitment):
            raise ValueError("session campaign changed since study freeze")
        relations = generate_relations(spec, relation_policy) if relation_policy else None
        if expected_relation_commitment is not None and (relations is None or relations.commitment != expected_relation_commitment):
            raise ValueError("relation campaign changed since study freeze")
        view = EphemeralWorkspace(repository, policy)
        await _owned_operation(view.prepare)
        if (view.root / CONTRACT_PATH).exists():
            raise ValueError("reserved owner contract path already exists")
        view.write_file(CONTRACT_PATH, spec.model_dump_json())
        if schema_policy:
            if (view.root / SCHEMA_CONFIG).exists():
                raise ValueError("reserved schema policy path already exists")
            validate_schema_targets(view.root, spec, schema_policy)
            view.write_file(SCHEMA_CONFIG, schema_policy.model_dump_json())
        report["phase"] = "impact_analysis"
        plan = plan_migration(view.root, spec)
        report["plan"] = plan.model_dump()
        if plan.status != "planned":
            raise ValueError("impact planning held unsupported source")
        if session_policy:
            validate_sessions(view.root, spec, session_policy)
            report["session_campaign"] = {"commitment": session_commitment(session_policy),
                "discovery_scenarios": len(session_policy.discovery), "final_scenarios": len(session_policy.final),
                "final_fingerprint": fingerprint([row.model_dump() for row in session_policy.final]),
                "discovery_feedback": session_feedback, "final_inputs_exposed_to_repair": False,
                "origin": "owner_authored_not_generated_or_production_traffic"}
        if relations:
            validate_relations(view.root, spec, relation_policy)
            report["relation_campaign"] = {"generator": RELATION_GENERATOR, "commitment": relations.commitment,
                "discovery_cases": len(relations.discovery), "final_cases": len(relations.final),
                "discovery_properties": relation_inventory(relations.discovery), "final_properties": relation_inventory(relations.final),
                "final_fingerprint": relations.final_fingerprint, "discovery_feedback": relation_feedback,
                "final_variants_exposed_to_repair": False, "laws": "owner_declared_not_inferred"}
        campaign = generate_campaign(spec, counterexample_policy) if counterexample_policy and not schema_policy else None
        if expected_campaign_commitment is not None and not schema_policy and (campaign is None or campaign.commitment != expected_campaign_commitment):
            raise ValueError("campaign changed since study freeze")
        if campaign:
            # Commit both sets before baseline execution or any possible model call.
            _record_campaign(report, campaign, counterexample_policy, discovery_feedback)
        discovery_cases = campaign.discovery if campaign else ()
        dependency = view.read_file(spec.dependency_file)
        replacement = _source_pin(dependency)
        baseline = DockerSandbox(view.root, policy)
        await _owned_operation(baseline.start)
        async with asyncio.timeout(max(1, spec.task_timeout_seconds - (time.monotonic() - started))):
            report["phase"] = "image_attestation"
            await _attest(baseline, report)
            if schema_policy:
                report["phase"] = "source_schema_extraction"
                source_schemas, receipt = await settled_call(read_schemas, baseline, schema_policy, SOURCE_VERSION)
                report["schema_intelligence"] = {"policy": schema_policy.model_dump(), "source": receipt,
                    "discovery_mode": "schema_guided" if schema_discovery else "blind_ablation"}
                campaign = schema_campaign(spec, counterexample_policy, schema_policy, source_schemas)
                if expected_campaign_commitment is not None and campaign.commitment != expected_campaign_commitment:
                    raise ValueError("schema campaign changed since study freeze")
                _record_campaign(report, campaign, counterexample_policy, discovery_feedback, generator=SCHEMA_GENERATOR)
                discovery_cases = campaign.discovery
                if not schema_discovery:
                    blind = generate_campaign(spec, counterexample_policy,
                        excluded_inputs=frozenset(input_key(case) for case in campaign.final))
                    discovery_cases = blind.discovery
                    report["input_campaign"]["active_discovery_fingerprint"] = blind.discovery_fingerprint
            report["phase"] = "source_tests_and_contracts"
            old_tests = await _run(baseline, [V1_PYTHON, *spec.test_command], report, "source_tests")
            if old_tests.exit_code != 0:
                raise ValueError("source tests must pass before a migration comparison")
            source = await _probes(baseline, spec, report, SOURCE_VERSION)
            if not validate_source_contracts(source, spec):
                raise ValueError("source behavior does not satisfy the owner's contract kinds")
            if relations:
                report["phase"] = "metamorphic_source_calibration"
                calibration = await verify_relations(view.root, spec, policy, (*relations.discovery, *relations.final), SOURCE_VERSION)
                report["relation_source_calibration"] = calibration
                if not calibration["passed"]:
                    # Contradictory owner laws or existing bugs need an owner decision, not AI repair.
                    raise ValueError("source violates declared workflow laws or their positive-evidence preconditions")
            baseline.workspace.write_file(spec.dependency_file, replacement)
            report["phase"] = "codemod_baseline"
            for source_path in spec.source_paths:
                result = await asyncio.to_thread(baseline.run, [CODEMOD, "--log-file", "/tmp/upgradepilot-codemod.log", source_path])
                report["executions"].append({"stage": "bump-pydantic", "exit_code": result.exit_code,
                    "timed_out": result.timed_out, "duration_ms": result.duration_ms})
                if result.exit_code or result.timed_out:
                    raise ValueError("codemod baseline failed")
            if not _scope_valid(baseline.workspace.changed_files(), spec):
                raise ValueError("codemod changed files outside owner source/dependency scope")
            protected = [name for name in baseline.workspace.list_files(50000)
                if name != spec.dependency_file and (is_test_path(name) or name == CONTRACT_PATH
                    or not _scope_valid([name], spec, allow_dependency=False) or name.endswith((".toml", ".ini", ".cfg")))]
            if len(protected) > 90:
                raise ValueError("owner-protected file inventory exceeds the first-release policy budget")
            policy.protected_paths = sorted(set(policy.protected_paths + protected))
            policy.controller_managed_paths = [spec.dependency_file]
            task = CodeTask(repository=spec.task_id, issue="Migrate native Pydantic V1 to V2 while preserving the frozen owner contracts.",
                model=model or "scripted-control", test_command=[V2_PYTHON, *spec.test_command], policy=policy)
            before = repository_fingerprint(view.root, policy.max_files, policy.max_repository_bytes)
            report["phase"] = "baseline_verification"
            baseline_result = await _verify(task, baseline, spec, source, before, report)
            report["baseline"] = baseline_result
            if schema_policy:
                report["phase"] = "target_schema_analysis"
                try:
                    report["schema_intelligence"]["baseline_target"] = await _schema_target(baseline, schema_policy, source_schemas)
                except ValueError:
                    report["schema_intelligence"]["baseline_target"] = {"available": False}
            if campaign and discovery_feedback:
                report["phase"] = "counterexample_discovery"
                discovery = await verify_generated(view.root, baseline.workspace.root, spec, policy, discovery_cases,
                    policy=counterexample_policy, shrink=True,
                    excluded_inputs=frozenset(input_key(case) for case in (*campaign.final, *spec.contracts)))
                report["discovery"] = discovery
                if not discovery["passed"]:
                    report["reason"] = "generated_discovery_regression"
                baseline_result["owner_gates_passed"] = baseline_result["passed"]
                baseline_result["passed"] = baseline_result["passed"] and discovery["passed"]
            if session_policy and session_feedback:
                report["phase"] = "session_discovery"
                sessions = await verify_sessions(view.root, baseline.workspace.root, spec, policy, session_policy.discovery,
                    policy=session_policy, shrink=True)
                report["session_discovery"] = sessions
                baseline_result["pre_session_gates_passed"] = baseline_result["passed"]
                baseline_result["passed"] = baseline_result["passed"] and sessions["passed"]
                if not sessions["passed"]:
                    report["reason"] = "session_discovery_regression"
            if relations and relation_feedback:
                report["phase"] = "metamorphic_discovery"
                evidence = await verify_relations(baseline.workspace.root, spec, policy, relations.discovery, TARGET_VERSION,
                    feedback_limit=relation_policy.max_feedback_cases)
                report["relation_discovery"] = evidence
                baseline_result["pre_relation_gates_passed"] = baseline_result["passed"]
                baseline_result["passed"] = baseline_result["passed"] and evidence["passed"]
                if not evidence["passed"]:
                    report["reason"] = "metamorphic_discovery_violation"
            winner = baseline
            if baseline_result["passed"]:
                report.update(status="ready_for_owner_review", reason="codemod_verified_without_inference")
            elif model or scripted_models is not None:
                report["phase"] = "budgeted_repair"
                candidate = DockerSandbox(view.root, policy)
                await _owned_operation(candidate.start)
                # Reapply the baseline bytes without executing any repository module on the host.
                for name in baseline.workspace.changed_files():
                    candidate.workspace.write_file(name, baseline.workspace.read_file(name))
                budget = {"calls": 0, "reserved_tokens": 0, "max_calls": spec.max_model_calls,
                    "max_reserved_tokens": spec.max_reserved_tokens, "deadline": started + spec.task_timeout_seconds -
                        (30 + 8 * (len(campaign.final) + (len(campaign.discovery) if discovery_feedback else 0)) if campaign else 30)
                        - (verification_reserve(session_policy) if session_policy else 0)
                        - (relation_reserve((*relations.final, *(relations.discovery if relation_feedback else ()))) if relations else 0)}
                report["inference_kind"] = "scripted_control" if scripted_models is not None else "live_provider"
                if scripted_models is None:
                    from code_agent.providers import build_provider
                    adapter = build_provider(model, max_completion_tokens=2048, timeout_seconds=30)
                    scripted_models = (adapter.with_structured_output(AnalysisPlan), adapter.with_structured_output(CodeAction), adapter.with_structured_output(ReviewVerdict))
                models = [BudgetModel(value, budget, stage) for value, stage in zip(scripted_models, ("analysis", "action", "review"), strict=True)]
                issue = {"goal": task.issue, "impact_plan": plan.model_dump(), "rules": RULES,
                         "baseline_failures": baseline_result, "constraints": "Do not edit owner tests, dependency pins or contract files. No compatibility namespace or test suppression."}
                if schema_policy:
                    issue["schema_declaration_findings"] = report["schema_intelligence"]["baseline_target"]
                if campaign and discovery_feedback:
                    # Raw observations remain private local artifacts. Bound source values in prompts.
                    issue["discovered_counterexamples"] = [{key: value for key, value in example.items()
                        if key not in {"source", "target"}} for example in report["discovery"]["counterexamples"]]
                if session_policy and session_feedback:
                    issue["stateful_counterexamples"] = json.loads(json.dumps(report["session_discovery"]["counterexamples"]))
                if relations and relation_feedback:
                    issue["metamorphic_counterexamples"] = json.loads(json.dumps(report["relation_discovery"]["counterexamples"]))
                # Retain valid structured evidence; never cut JSON halfway through a counterexample.
                while len(json.dumps(issue).encode()) > 40000:
                    if issue["impact_plan"]["findings"]:
                        issue["impact_plan"]["findings"].pop()
                    elif issue.get("discovered_counterexamples"):
                        issue["discovered_counterexamples"].pop()
                    elif issue.get("stateful_counterexamples"):
                        issue["stateful_counterexamples"].pop()
                    elif issue.get("metamorphic_counterexamples"):
                        issue["metamorphic_counterexamples"].pop()
                    else:
                        raise ValueError("repair context exceeds its admission budget")
                if campaign and discovery_feedback:
                    report["repair_counterexamples_provided"] = len(issue["discovered_counterexamples"])
                if session_policy and session_feedback:
                    report["repair_session_counterexamples_provided"] = len(issue["stateful_counterexamples"])
                if relations and relation_feedback:
                    report["repair_relation_counterexamples_provided"] = len(issue["metamorphic_counterexamples"])
                repair_task = task.model_copy(update={"issue": json.dumps(issue)})
                result = await VerifiedPRAgent(*models).solve(repair_task, candidate)
                report.update(model_calls=budget["calls"], live_provider_calls=budget["calls"] if report["inference_kind"] == "live_provider" else 0,
                    usage_complete=result.usage_accounting_complete, provider_input_tokens=result.prompt_tokens if result.usage_accounting_complete else None,
                    provider_output_tokens=result.completion_tokens if result.usage_accounting_complete else None)
                report["repair_workflow_status"] = result.status
                report["specialist_stages"] = result.verification.stage_outcomes if result.verification else {}
                report["phase"] = "repair_verification"
                repaired = await _verify(task, candidate, spec, source, before, report)
                report["repair"] = repaired
                if campaign and discovery_feedback:
                    report["phase"] = "counterexample_recheck"
                    recheck = await verify_generated(view.root, candidate.workspace.root, spec, policy, discovery_cases,
                        policy=counterexample_policy)
                    report["discovery_recheck"] = recheck
                    repaired["owner_gates_passed"] = repaired["passed"]
                    repaired["passed"] = repaired["passed"] and recheck["passed"]
                if session_policy and session_feedback:
                    report["phase"] = "session_recheck"
                    recheck = await verify_sessions(view.root, candidate.workspace.root, spec, policy, session_policy.discovery,
                        policy=session_policy)
                    report["session_recheck"] = recheck
                    repaired["passed"] = repaired["passed"] and recheck["passed"]
                if relations and relation_feedback:
                    report["phase"] = "metamorphic_recheck"
                    recheck = await verify_relations(candidate.workspace.root, spec, policy, relations.discovery, TARGET_VERSION)
                    report["relation_recheck"] = recheck
                    repaired["passed"] = repaired["passed"] and recheck["passed"]
                if result.status == "completed" and repaired["passed"]:
                    winner = candidate
                    report.update(status="ready_for_owner_review", reason="repair_verified_not_merged_or_deployed")
            if report["status"] == "ready_for_owner_review":
                if schema_policy:
                    report["phase"] = "final_schema_analysis"
                    report["schema_intelligence"]["final_target"] = await _schema_target(winner, schema_policy, source_schemas)
                patch = winner.workspace.unified_diff()
                patch_sha256 = hashlib.sha256(patch.encode()).hexdigest()
                if campaign:
                    report["phase"] = "frozen_final_evaluation"
                    # Terminal gate: no model invocation or repair branch follows this point.
                    final = await verify_generated(view.root, winner.workspace.root, spec, policy, campaign.final,
                        policy=counterexample_policy)
                    report["final_evaluation"] = {**final, "patch_sha256": patch_sha256,
                        "input_fingerprint": campaign.final_fingerprint, "repair_after_evaluation": False}
                    if hashlib.sha256(winner.workspace.unified_diff().encode()).hexdigest() != patch_sha256:
                        raise ValueError("candidate changed after final evaluation")
                    if not final["passed"]:
                        report.update(status="held", reason="frozen_final_contract_regression")
                if session_policy:
                    report["phase"] = "frozen_final_session_evaluation"
                    # Terminal evaluation: no repair agent can see these scenarios or run again.
                    final = await verify_sessions(view.root, winner.workspace.root, spec, policy, session_policy.final,
                        policy=session_policy)
                    report["final_session_evaluation"] = {**final, "patch_sha256": patch_sha256,
                        "input_fingerprint": report["session_campaign"]["final_fingerprint"], "repair_after_evaluation": False}
                    if hashlib.sha256(winner.workspace.unified_diff().encode()).hexdigest() != patch_sha256:
                        raise ValueError("candidate changed after final session evaluation")
                    if not final["passed"]:
                        report.update(status="held", reason="frozen_final_session_regression")
                if relations:
                    report["phase"] = "frozen_final_metamorphic_evaluation"
                    final = await verify_relations(winner.workspace.root, spec, policy, relations.final, TARGET_VERSION)
                    report["final_relation_evaluation"] = {**final, "patch_sha256": patch_sha256,
                        "input_fingerprint": relations.final_fingerprint, "repair_after_evaluation": False}
                    if hashlib.sha256(winner.workspace.unified_diff().encode()).hexdigest() != patch_sha256:
                        raise ValueError("candidate changed after final metamorphic evaluation")
                    if not final["passed"]:
                        report.update(status="held", reason="frozen_final_metamorphic_violation")
                if report["status"] == "ready_for_owner_review":
                    report["patch"] = patch
                    report["patch_sha256"] = patch_sha256
    except Exception as exc:
        report.update(status="held", reason="hold_at:" + report["phase"], error_type=type(exc).__name__)
    finally:
        if budget:
            report.update(model_calls=budget["calls"], live_provider_calls=budget["calls"] if report["inference_kind"] == "live_provider" else 0,
                reserved_tokens=budget["reserved_tokens"])
            if budget.get("failure"):
                report["provider_failure"] = budget["failure"].model_dump()
        for sandbox in (candidate, baseline):
            if sandbox:
                try:
                    await asyncio.to_thread(sandbox.close)
                except (OSError, RuntimeError):
                    report.update(status="held", reason="sandbox_cleanup_failed", patch="")
        if view:
            try:
                view.cleanup()
            except OSError:
                report.update(status="held", reason="workspace_cleanup_failed", patch="")
        if source_before:
            try:
                report["source_unchanged"] = source_before == repository_fingerprint(repository, 5000, 50000000)
            except (OSError, ValueError):
                pass
        if not report["source_unchanged"]:
            report.update(status="held", reason="source_integrity_unavailable_or_changed", patch="")
        report["duration_ms"] = (time.monotonic() - started) * 1000
        report["fingerprint"] = fingerprint({key: value for key, value in report.items() if key != "fingerprint"})
    return report


async def _verify(task, sandbox, spec, source, before, report):
    patch = sandbox.workspace.unified_diff()
    gates, tests = await run_quality_gates(task, sandbox, patch=patch, changed_files=sandbox.workspace.changed_files(),
        source_fingerprint_before=before, baseline_lint=None)
    native = _native_target(sandbox.workspace, spec)
    immutable = all(sandbox.workspace.read_file(name) == (sandbox.repository / name).read_bytes().decode("utf-8")
        for name in task.policy.protected_paths if "*" not in name and name != spec.dependency_file and (sandbox.repository / name).is_file())
    target = None
    try:
        target = compare_contracts(source, await _probes(sandbox, spec, report, TARGET_VERSION), spec)
    except (ValueError, OSError):
        pass
    scope = _scope_valid(sandbox.workspace.changed_files(), spec)
    pin = sandbox.workspace.read_file(spec.dependency_file) == _source_pin((sandbox.repository / spec.dependency_file).read_bytes().decode("utf-8"))
    passed = (tests.exit_code == 0 and not tests.timed_out and all(gate.status != "failed" for gate in gates)
              and native and immutable and scope and pin and target is not None and target["passed"])
    return {"passed": passed, "tests_exit_code": tests.exit_code, "native_v2": native, "owner_files_unchanged": immutable,
        "scope_valid": scope, "target_pin_valid": pin, "contracts": target,
        "gates": [{"name": gate.name, "status": gate.status} for gate in gates]}
