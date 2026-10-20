"""Portable, provider-free patch verification with fresh repeated Docker execution."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from code_agent.models import SandboxPolicy
from code_agent.patch_inputs import (
    MAX_PATCH_BYTES,
    PROTOCOL,
    PatchVerifySpec,
    apply_file_patch,
    generated_cases,
    import_prediction,
    parse_patch,
    read_bounded,
    strict_json,
    test_source_path,
)
from code_agent.patch_json import campaign_fingerprint, complexity, generate_json_cases
from code_agent.patch_reduction import reduce_counterexample
from code_agent.sandbox import DockerSandbox, DockerUnavailableError
from code_agent.upgrade_differential import settled_call
from code_agent.upgrade_models import ContractCase, fingerprint
from code_agent.upgrade_runner import resolve_image
from code_agent.workspace import EXCLUDED_PARTS, _is_secret_or_excluded

RESERVED = (".agentforge-verify-input.json", ".agentforge-verify-probe.py", ".agentforge-verify-receipt.json")
MODES = ("tests_only", "contracts", "generated")


def tree_fingerprint(root: Path) -> str:
    """Bounded regular-file snapshot; links/reparse points fail rather than being followed."""
    if not root.is_dir():
        raise ValueError("repository directory is missing")
    files, size, entries = {}, 0, 0
    for parent, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in EXCLUDED_PARTS)
        for name in [*directories, *sorted(names)]:
            path = Path(parent) / name
            entries += 1
            if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
                raise ValueError("repository links and reparse points are unsupported")
            if entries > 10_000:
                raise ValueError("repository entry budget exceeded")
            relative = path.relative_to(root)
            if path.is_file() and not _is_secret_or_excluded(relative):
                data = read_bounded(path, 1_000_000) if path.stat().st_size else b""
                size += len(data)
                files[relative.as_posix()] = hashlib.sha256(data).hexdigest()
                if len(files) > 5000 or size > 50_000_000:
                    raise ValueError("repository snapshot budget exceeded")
    return fingerprint(files)


def implementation_fingerprint():
    return fingerprint({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("patch_verify.py", "patch_inputs.py", "patch_probe.py", "sandbox.py", "workspace.py",
                                     "models.py", "bounded_process.py", "provider_reliability.py", "secret_patterns.py", "patch_json.py", "patch_reduction.py", "upgrade_models.py", "verification.py",
                                     "upgrade_differential.py", "upgrade_runner.py")})


def private_directory(path: Path, namespace="patch-verify") -> Path:
    root = (Path.cwd() / "data" / namespace).absolute()
    destination = path.absolute()
    for item in (destination, *destination.parents):
        if item.is_symlink() or (item.exists() and getattr(item.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("private artifact links/reparse points are not supported")
    if destination.resolve() == root.resolve() or not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"private artifacts must be below data/{namespace}/")
    return destination


def write_private_json(path: Path, payload):
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def protect_private_directory(path: Path):
    """Also protect artifacts when the installed CLI runs outside this checkout."""
    path.chmod(0o700)
    with (path / ".gitignore").open("x", encoding="utf-8", newline="") as stream:
        stream.write("*\n")
        stream.flush()
        os.fsync(stream.fileno())


def reserve_private_directory(path: Path, namespace="patch-verify"):
    destination = private_directory(path, namespace)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    protect_private_directory(destination)
    return destination


def _decode(raw, binding, mode, cases):
    receipt = strict_json(raw)
    if (not isinstance(receipt, dict) or set(receipt) != {"binding", "mode", "result"}
            or receipt["binding"] != binding or receipt["mode"] != mode or not isinstance(receipt["result"], dict)):
        raise ValueError("execution receipt binding is invalid")
    result = receipt["result"]
    if mode == "tests":
        if (set(result) != {"passed", "exit_code", "collected", "node_ids_sha256", "collection_errors"}
                or type(result["passed"]) is not bool or type(result["exit_code"]) is not int
                or type(result["collected"]) is not int or not 0 <= result["collected"] <= 500
                or type(result["collection_errors"]) is not int or result["collection_errors"] < 0
                or not re.fullmatch(r"[a-f0-9]{64}", result["node_ids_sha256"])
                or (result["passed"] and (result["exit_code"] != 0 or result["collected"] == 0 or result["collection_errors"]))):
            raise ValueError("test receipt is invalid")
    else:
        if set(result) != {row.case_id for row in cases}:
            raise ValueError("behavior case identities changed")
        for value in result.values():
            if (not isinstance(value, dict) or set(value) != {"sha256", "kind", "type"}
                    or not isinstance(value["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", value["sha256"])
                    or value["kind"] not in {"value", "error", "harness_error"}
                    or (value["kind"] == "value" and value["type"] is not None)
                    or (value["kind"] != "value" and (not isinstance(value["type"], str)
                        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,100}", value["type"])) )):
                raise ValueError("behavior outcome is invalid")
    return result, fingerprint(receipt)


async def _command(sandbox, mode, request, cases):
    workspace = sandbox.workspace
    before = tree_fingerprint(workspace.root)
    workspace.write_file(RESERVED[0], json.dumps(request, allow_nan=False))
    workspace.write_file(RESERVED[1], Path(__file__).with_name("patch_probe.py").read_text(encoding="utf-8"))
    result = await settled_call(sandbox.run, ["python", "-I", "-B", RESERVED[1], mode])
    if result.exit_code != 0 or result.timed_out:
        raise ValueError("Docker harness did not complete")
    raw = read_bounded(workspace.resolve(RESERVED[2]), 100_000)
    value, receipt_hash = _decode(raw, request["binding"], mode, cases)
    for name in RESERVED:
        workspace.delete_file(name)
    if tree_fingerprint(workspace.root) != before:
        raise ValueError("execution changed its repository snapshot")
    return value, {"mode": mode, "sha256": receipt_hash, "duration_ms": result.duration_ms}


async def _trial(repository, spec, edits, cases, arm, repeat, policy, factory):
    sandbox = factory(repository, policy)
    try:
        await settled_call(sandbox.start)
        if any((sandbox.workspace.root / name).exists() for name in RESERVED):
            raise ValueError("reserved verifier file already exists")
        if arm == "candidate":
            for edit in edits:
                sandbox.workspace.check_agent_edit_path(edit.path)
                original = sandbox.workspace.read_file(edit.path)
                sandbox.workspace.write_file(edit.path, apply_file_patch(original, edit))
        snapshot = tree_fingerprint(sandbox.workspace.root)
        binding = fingerprint({"spec": spec.model_dump(), "snapshot": snapshot, "arm": arm,
                               "repeat": repeat, "nonce": uuid4().hex})
        request = {"binding": binding, "test_paths": spec.test_paths,
                   "cases": [row.model_dump() for row in cases]}
        tests, receipt = await _command(sandbox, "tests", request, cases)
        trial = {"arm": arm, "repeat": repeat, "snapshot_sha256": snapshot,
                 "tests": tests, "behavior": {}, "receipts": [receipt]}
        if cases:
            trial["behavior"], receipt = await _command(sandbox, "behavior", request, cases)
            trial["receipts"].append(receipt)
        return trial
    finally:
        await settled_call(sandbox.close)


def assess(trials, cases, authorized):
    if len(trials) != 4 or {(row["arm"], row["repeat"]) for row in trials} != {(arm, n) for arm in ("original", "candidate") for n in range(2)}:
        return "inconclusive", "incomplete_execution", []
    original = [row for row in trials if row["arm"] == "original"]
    candidate = [row for row in trials if row["arm"] == "candidate"]
    for pair in (original, candidate):
        if any(pair[0][key] != pair[1][key] for key in ("snapshot_sha256", "tests", "behavior")):
            return "inconclusive", "unstable_execution", []
    if any(not row["tests"]["passed"] for row in original):
        return "inconclusive", "original_tests_not_all_passed", []
    if any(row["tests"]["node_ids_sha256"] != original[0]["tests"]["node_ids_sha256"] for row in candidate):
        return "inconclusive", "test_population_changed", []
    if any(not row["tests"]["passed"] for row in candidate):
        return "regression_found", "existing_test_regression", []
    if not cases or not authorized:
        return "inconclusive", "no_owner_authorized_behavioral_scope", []
    for case in cases:
        observed = original[0]["behavior"][case.case_id]
        if observed["kind"] != case.expected_kind or (case.expected_error and observed["type"] != case.expected_error):
            return "inconclusive", "original_behavior_does_not_match_owner_scope", []
    if any(value["kind"] == "harness_error" for row in candidate for value in row["behavior"].values()):
        return "inconclusive", "candidate_behavior_harness_error", []
    changed = [case.case_id for case in cases
               if original[0]["behavior"][case.case_id] != candidate[0]["behavior"][case.case_id]]
    return ("regression_found", "owner_preservation_violation", changed) if changed else ("checks_passed", "declared_checks_passed_not_a_proof", [])


async def verify_patch(repository: Path, patch: str, spec: PatchVerifySpec, *, mode="generated", sandbox_factory=None, image_resolver=None):
    if mode not in MODES:
        raise ValueError("unsupported verification strategy")
    # Revalidate even typed caller input; model_copy is not a validation boundary.
    spec = PatchVerifySpec.model_validate(spec.model_dump())
    report = {"protocol": PROTOCOL, "task_id": spec.task_id, "mode": mode, "status": "inconclusive",
              "reason": "not_executed", "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
              "spec_sha256": fingerprint(spec.model_dump()), "implementation_sha256": implementation_fingerprint(),
              "source_sha256": "", "source_unchanged": False, "image_id": "", "trials": [], "case_count": 0,
              "generated_case_count": 0, "counterexamples": [], "live_provider_calls": 0,
              "json_campaign_sha256": None, "minimizations": [],
              "provider_cost_usd": 0, "execution_mode": "docker", "duration_ms": 0.0,
              "limitations": "Owner-scoped repeated execution, not proof, general semantic equivalence or hostile-repository attestation."}
    started, before = time.perf_counter(), ""
    try:
        read_bounded(Path(__file__), 100_000)  # Fixed harness/controller inputs are bounded too.
        for item in (repository.absolute(), *repository.absolute().parents):
            if item.is_symlink() or (item.exists() and getattr(item.lstat(), "st_file_attributes", 0) & 0x400):
                raise ValueError("repository root traverses a link")
        repository = repository.resolve()
        before = tree_fingerprint(repository)
        report["source_sha256"] = before
        if spec.reference_source_sha256 is not None and spec.reference_source_sha256 != before:
            raise ValueError("reproduction source snapshot has changed")
        if spec.reference_patch_sha256 is not None and spec.reference_patch_sha256 != report["patch_sha256"]:
            raise ValueError("reproduction patch has changed")
        edits = parse_patch(patch, spec)
        for name in spec.test_paths:
            path = repository / test_source_path(name)
            if not path.exists() or not path.resolve().is_relative_to(repository):
                raise ValueError("owner test scope is missing")
        for edit in edits:
            apply_file_patch(read_bounded(repository / edit.path, 1_000_000).decode("utf-8"), edit)
        if spec.contract_review is not None:
            if spec.contract_review.patch_sha256 != report["patch_sha256"]:
                raise ValueError("reviewed proposal belongs to another patch")
            for name, expected in spec.contract_review.source_files.items():
                if hashlib.sha256(read_bounded(repository / name, 1_000_000)).hexdigest() != expected:
                    raise ValueError("reviewed proposal source has changed")
        cases = [] if mode == "tests_only" else list(spec.contracts)
        generated = generated_cases(spec) if mode == "generated" else []
        cases.extend(generated)
        if cases and not spec.preservation_authorized:
            raise ValueError("behavioral preservation requires owner authorization")
        report.update(case_count=len(cases), generated_case_count=len(generated))
        if mode == "generated" and spec.json_domains:
            report["json_campaign_sha256"] = campaign_fingerprint(spec.json_domains, generated, spec.seed)
        image = await settled_call(image_resolver or resolve_image, spec.image)
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
            raise ValueError("Docker image could not be pinned")
        report["image_id"] = image
        policy = SandboxPolicy(image=image, allowed_images={image}, command_timeout_seconds=spec.command_timeout_seconds,
                               allowed_test_executables={"python"}, max_output_chars=1000)
        for arm in ("original", "candidate"):
            for repeat in range(2):
                report["trials"].append(await _trial(repository, spec, edits, cases, arm, repeat, policy, sandbox_factory or DockerSandbox))
        report["status"], report["reason"], changed = assess(report["trials"], cases, spec.preservation_authorized)
        report["counterexamples"] = [row.model_dump() for row in cases if row.case_id in changed]
        structured = [(domain, case) for domain in spec.json_domains for case in generate_json_cases(domain, spec.seed)
                      if mode == "generated" and case.case_id in changed]
        if structured:
            # Confirm/reduce one bounded counterexample, choosing the largest
            # observed input so extra fields do not obscure the useful failure.
            domain, parent = max(structured, key=lambda pair: complexity(pair[0].value(pair[1])))

            async def evaluate(selected, deadline):
                trials = []
                try:
                    for arm in ("original", "candidate"):
                        for repeat in range(2):
                            remaining = deadline - time.monotonic()
                            if remaining < 5:
                                return {"status": "inconclusive", "reason": "reduction_budget_exhausted", "changed": [], "trials": trials}
                            bounded = policy.model_copy(update={"command_timeout_seconds": max(1, min(policy.command_timeout_seconds, int(remaining / 2)))})
                            trials.append(await _trial(repository, spec, edits, selected, arm, repeat, bounded, sandbox_factory or DockerSandbox))
                    status, reason, hits = assess(trials, selected, True)
                    return {"status": status, "reason": reason, "changed": hits, "trials": trials}
                except (OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError):
                    return {"status": "inconclusive", "reason": "reduction_execution_invalid", "changed": [], "trials": trials}

            reduction = await reduce_counterexample(domain, parent, spec.reduction, evaluate)
            report["minimizations"].append(reduction)
            # Batch-only differences may depend on previous calls. Do not report
            # those as independently reproducible structured counterexamples.
            structured_ids = {case.case_id for _, case in structured}
            report["counterexamples"] = [row for row in report["counterexamples"] if row["case_id"] not in structured_ids]
            if reduction["confirmed_case"] is not None:
                report["counterexamples"].append(reduction["confirmed_case"])
            elif not report["counterexamples"]:
                report["status"], report["reason"] = "inconclusive", "structured_difference_not_isolated"
    except DockerUnavailableError:
        report["reason"] = "docker_unavailable"
    except (OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError):
        # Repository-controlled exceptions, outputs and provider credentials never enter reports.
        report["status"], report["reason"] = "inconclusive", "admission_or_execution_invalid"
    finally:
        if before:
            try:
                report["source_unchanged"] = tree_fingerprint(repository) == before
            except (OSError, ValueError):
                pass
            if not report["source_unchanged"]:
                report["status"], report["reason"] = "inconclusive", "source_integrity_changed"
        report["duration_ms"] = (time.perf_counter() - started) * 1000
        report["fingerprint"] = fingerprint(report)
    return report


def reproducer_spec(spec: PatchVerifySpec, report: dict) -> PatchVerifySpec | None:
    """Export only an isolated, confirmed failure, pinned to its execution inputs.

    Hashes detect accidental edits; they do not authenticate a hostile controller.
    """
    spec = PatchVerifySpec.model_validate(spec.model_dump())
    if (report.get("fingerprint") != fingerprint({key: value for key, value in report.items() if key != "fingerprint"})
            or report.get("spec_sha256") != fingerprint(spec.model_dump())):
        raise ValueError("reproducer report binding is invalid")
    if report["status"] != "regression_found" or not report["source_unchanged"]:
        return None
    confirmed = next((row["confirmed_case"] for row in report.get("minimizations", [])
                      if row.get("confirmed_case") is not None), None)
    if confirmed is None:
        return None
    case = ContractCase.model_validate(confirmed)
    case = case.model_copy(update={"case_id": f"replay.{fingerprint(case.model_dump())[:20]}"})
    value = spec.model_dump()
    value.update(contracts=[case.model_dump()], integer_domains=[], json_domains=[],
                 image=report["image_id"], reference_source_sha256=report["source_sha256"],
                 reference_patch_sha256=report["patch_sha256"],
                 reduction={**spec.reduction.model_dump(), "enabled": False})
    return PatchVerifySpec.model_validate(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--patch", type=Path)
    source.add_argument("--predictions", type=Path)
    parser.add_argument("--instance-id")
    parser.add_argument("--mode", choices=MODES, default="generated")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = PatchVerifySpec.model_validate(strict_json(read_bounded(args.spec, 40_000)))
    if args.predictions:
        if not args.instance_id:
            parser.error("prediction import requires an explicit --instance-id")
        patch, origin = import_prediction(read_bounded(args.predictions, 2_000_000), args.instance_id)
    else:
        if args.instance_id:
            parser.error("--instance-id only applies to predictions")
        patch, origin = read_bounded(args.patch, MAX_PATCH_BYTES).decode("utf-8"), {"adapter": "unified_diff", "producer_authenticated": False}
    output = reserve_private_directory(args.output_dir)
    report = asyncio.run(verify_patch(args.repository, patch, spec, mode=args.mode))
    # Persist provenance outside the execution fingerprint, with a second enclosing hash.
    evidence = {"report": report, "origin": origin}
    evidence["fingerprint"] = fingerprint(evidence)
    write_private_json(output / "report.json", evidence)
    replay = reproducer_spec(spec, report)
    if replay is not None:
        write_private_json(output / "reproducer.spec.json", replay.model_dump())
    print(json.dumps({"status": report["status"], "reason": report["reason"], "case_count": report["case_count"],
                      "counterexamples": len(report["counterexamples"]), "reproducer_available": replay is not None,
                      "private_artifacts": str(output), "live_provider_calls": 0}))
    if report["status"] != "checks_passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
